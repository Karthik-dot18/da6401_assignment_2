"""Training entrypoint — DA6401 Assignment 2."""

import os
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import f1_score
import wandb

from models.vgg11 import VGG11
from models.localization import LocalizationModel
from models.segmentation import UNet
from losses.iou_loss import IoULoss
from data.pets_dataset import (PetClassificationDataset,
                                PetDetectionDataset,
                                PetSegmentationDataset)

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224


# ── helpers ────────────────────────────────────────────────────

def save_ckpt(model, name):
    """Save to checkpoints/ and mirror to Drive if mounted."""
    Path("checkpoints").mkdir(exist_ok=True)
    local = f"checkpoints/{name}"
    torch.save(model.state_dict(), local)
    drive = f"/content/drive/MyDrive/da6401_a2/checkpoints/{name}"
    if os.path.exists("/content/drive/MyDrive"):
        os.makedirs(os.path.dirname(drive), exist_ok=True)
        torch.save(model.state_dict(), drive)
    print(f"saved {local}")

def cosine_sched(opt, epochs):
    return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

# segmentation losses
class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
    def forward(self, logits, targets):
        nc    = logits.shape[1]
        probs = F.softmax(logits, dim=1)
        oh    = F.one_hot(targets, nc).permute(0,3,1,2).float()
        inter = (probs * oh).sum(dim=(0,2,3))
        denom = (probs + oh).sum(dim=(0,2,3))
        return 1 - ((2*inter + self.eps)/(denom + self.eps)).mean()

class SegLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.ce    = nn.CrossEntropyLoss()
        self.dice  = DiceLoss()
    def forward(self, logits, targets):
        return (1-self.alpha)*self.ce(logits,targets) + self.alpha*self.dice(logits,targets)

def dice_score(logits, targets, nc=3, eps=1e-6):
    pred = logits.argmax(1)
    s = []
    for c in range(nc):
        p = (pred==c).float(); t = (targets==c).float()
        s.append(((2*(p*t).sum()+eps)/(p.sum()+t.sum()+eps)).item())
    return sum(s)/len(s)

def iou_metric(pred, target, eps=1e-7):
    def xyxy(b):
        cx,cy,w,h = b.unbind(-1)
        return torch.stack([cx-w/2,cy-h/2,cx+w/2,cy+h/2],-1)
    p,t = xyxy(pred), xyxy(target)
    inter = ((torch.min(p[...,2],t[...,2])-torch.max(p[...,0],t[...,0])).clamp(0)*
             (torch.min(p[...,3],t[...,3])-torch.max(p[...,1],t[...,1])).clamp(0))
    ap = (p[...,2]-p[...,0]).clamp(0)*(p[...,3]-p[...,1]).clamp(0)
    at = (t[...,2]-t[...,0]).clamp(0)*(t[...,3]-t[...,1]).clamp(0)
    return inter/(ap+at-inter+eps)


# ── Task 1: Classification ──────────────────────────────────────

def train_task1(args):
    wandb.init(project="da6401-a2", name=f"task1-p{args.dropout_p}", config=vars(args))

    ds    = PetClassificationDataset(args.data_root)
    n_val = int(0.15 * len(ds))
    tr, vl = random_split(ds, [len(ds)-n_val, n_val])
    tr_ld  = DataLoader(tr, args.bs, shuffle=True,  num_workers=2, pin_memory=True)
    vl_ld  = DataLoader(vl, args.bs, shuffle=False, num_workers=2, pin_memory=True)

    model = VGG11(num_classes=37, p=args.dropout_p).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    crit  = nn.CrossEntropyLoss()
    sched = cosine_sched(opt, args.epochs)
    best  = 0.0

    for ep in range(1, args.epochs+1):
        model.train()
        tr_loss = tr_correct = tr_n = 0
        for imgs, labels in tr_ld:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            opt.zero_grad()
            out  = model(imgs)
            loss = crit(out, labels)
            loss.backward()
            opt.step()
            tr_loss    += loss.item() * len(imgs)
            tr_correct += (out.argmax(1)==labels).sum().item()
            tr_n       += len(imgs)
        sched.step()

        model.eval()
        vl_loss = vl_correct = vl_n = 0
        all_p, all_l = [], []
        with torch.no_grad():
            for imgs, labels in vl_ld:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                out      = model(imgs)
                vl_loss += crit(out, labels).item() * len(imgs)
                preds    = out.argmax(1)
                vl_correct += (preds==labels).sum().item()
                vl_n    += len(imgs)
                all_p.extend(preds.cpu().numpy())
                all_l.extend(labels.cpu().numpy())

        vl_acc = vl_correct / vl_n
        vl_f1  = f1_score(all_l, all_p, average="macro", zero_division=0)

        wandb.log({"epoch":ep,
                   "train/loss": tr_loss/tr_n, "train/acc": tr_correct/tr_n,
                   "val/loss":   vl_loss/vl_n, "val/acc":   vl_acc,
                   "val/f1":     vl_f1,
                   "lr": sched.get_last_lr()[0]})
        print(f"[T1] ep {ep:3d} | tr_acc {tr_correct/tr_n:.3f} | vl_acc {vl_acc:.3f} | f1 {vl_f1:.3f}")

        if vl_acc > best:
            best = vl_acc
            save_ckpt(model, "classifier.pth")

    wandb.finish()
    print(f"Task1 done. best val acc: {best:.3f}")


# ── Task 2: Localisation ────────────────────────────────────────

def train_task2(args):
    wandb.init(project="da6401-a2", name="task2-bbox", config=vars(args))

    ds    = PetDetectionDataset(args.data_root)
    n_val = int(0.15 * len(ds))
    tr, vl = random_split(ds, [len(ds)-n_val, n_val])
    tr_ld  = DataLoader(tr, args.bs, shuffle=True,  num_workers=2, pin_memory=True)
    vl_ld  = DataLoader(vl, args.bs, shuffle=False, num_workers=2, pin_memory=True)

    # load backbone from classifier checkpoint
    backbone = VGG11(num_classes=37)
    if args.ckpt and os.path.exists(args.ckpt):
        backbone.load_state_dict(torch.load(args.ckpt, map_location="cpu"), strict=False)
        print(f"loaded backbone from {args.ckpt}")

    model  = LocalizationModel(backbone=backbone).to(DEVICE)
    opt    = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    iou_fn = IoULoss(reduction="mean")
    mse_fn = nn.MSELoss()
    sched  = cosine_sched(opt, args.epochs)
    best   = 0.0

    for ep in range(1, args.epochs+1):
        model.train()
        for imgs, boxes in tr_ld:
            imgs, boxes = imgs.to(DEVICE), boxes.to(DEVICE)
            opt.zero_grad()
            pred = model(imgs)
            # MSE + IoU loss as instructed
            loss = mse_fn(pred, boxes) + iou_fn(pred, boxes)
            loss.backward()
            opt.step()
        sched.step()

        model.eval()
        ious = []
        with torch.no_grad():
            for imgs, boxes in vl_ld:
                imgs, boxes = imgs.to(DEVICE), boxes.to(DEVICE)
                ious.extend(iou_metric(model(imgs), boxes).cpu().tolist())

        mean_iou = sum(ious) / len(ious)
        wandb.log({"epoch":ep, "val/iou": mean_iou})
        print(f"[T2] ep {ep:3d} | val iou {mean_iou:.4f}")

        if mean_iou > best:
            best = mean_iou
            save_ckpt(model, "localizer.pth")

    wandb.finish()
    print(f"Task2 done. best iou: {best:.4f}")


# ── Task 3: Segmentation ────────────────────────────────────────

def train_task3(args, freeze="full_finetune"):
    wandb.init(project="da6401-a2", name=f"task3-{freeze}",
               config={**vars(args), "freeze": freeze})

    ds    = PetSegmentationDataset(args.data_root)
    n_val = int(0.15 * len(ds))
    tr, vl = random_split(ds, [len(ds)-n_val, n_val])
    tr_ld  = DataLoader(tr, args.bs, shuffle=True,  num_workers=2, pin_memory=True)
    vl_ld  = DataLoader(vl, args.bs, shuffle=False, num_workers=2, pin_memory=True)

    backbone = VGG11(num_classes=37)
    if args.ckpt and os.path.exists(args.ckpt):
        backbone.load_state_dict(torch.load(args.ckpt, map_location="cpu"), strict=False)
        print(f"loaded backbone from {args.ckpt}")

    model = UNet(backbone=backbone, num_classes=3).to(DEVICE)

    if freeze == "frozen":
        for p in model.encoder.parameters():
            p.requires_grad_(False)
    elif freeze == "partial":
        for blk in [model.encoder.b1, model.encoder.b2, model.encoder.b3]:
            for p in blk.parameters():
                p.requires_grad_(False)

    params = filter(lambda p: p.requires_grad, model.parameters())
    opt    = torch.optim.Adam(params, lr=args.lr, weight_decay=1e-4)
    crit   = SegLoss(alpha=0.5)
    sched  = cosine_sched(opt, args.epochs)
    best   = 0.0

    for ep in range(1, args.epochs+1):
        model.train()
        for imgs, masks in tr_ld:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            opt.zero_grad()
            crit(model(imgs), masks).backward()
            opt.step()
        sched.step()

        model.eval()
        vl_dice = vl_px = vl_n = 0
        with torch.no_grad():
            for imgs, masks in vl_ld:
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                out      = model(imgs)
                vl_dice += dice_score(out, masks) * len(imgs)
                vl_px   += (out.argmax(1)==masks).float().mean().item() * len(imgs)
                vl_n    += len(imgs)

        d = vl_dice / vl_n
        wandb.log({"epoch":ep, "val/dice": d, "val/px_acc": vl_px/vl_n})
        print(f"[T3/{freeze}] ep {ep:3d} | dice {d:.4f} | px_acc {vl_px/vl_n:.4f}")

        if d > best:
            best = d
            save_ckpt(model, "unet.pth")

    wandb.finish()
    print(f"Task3 ({freeze}) done. best dice: {best:.4f}")


# ── argparse entrypoint ─────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--task",       type=int,   required=True, choices=[1,2,3])
    p.add_argument("--data_root",  type=str,   required=True)
    p.add_argument("--epochs",     type=int,   default=50)
    p.add_argument("--bs",         type=int,   default=32)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--ckpt",       type=str,   default=None,
                   help="classifier.pth path for backbone init in tasks 2/3")
    p.add_argument("--dropout_p",  type=float, default=0.5)
    p.add_argument("--freeze",     type=str,   default="full_finetune",
                   choices=["frozen","partial","full_finetune"])
    args = p.parse_args()

    if   args.task == 1: train_task1(args)
    elif args.task == 2: train_task2(args)
    elif args.task == 3: train_task3(args, freeze=args.freeze)
