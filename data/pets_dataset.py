import os
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

MEAN = (0.485, 0.456, 0.406)
STD  = (0.229, 0.224, 0.225)
IMG_SIZE = 224  # fixed as per VGG11 paper


def train_tfms():
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.4),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format="yolo", label_fields=["cls"]))


def val_tfms():
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format="yolo", label_fields=["cls"]))


def seg_tfms(train=True):
    augs = [A.Resize(IMG_SIZE, IMG_SIZE)]
    if train:
        augs += [A.HorizontalFlip(p=0.5), A.ColorJitter(p=0.3)]
    augs += [A.Normalize(mean=MEAN, std=STD), ToTensorV2()]
    return A.Compose(augs)


class PetClassificationDataset(Dataset):
    """image + breed label (0-indexed, 37 classes)."""
    def __init__(self, root, split="trainval", tfms=None):
        self.img_dir = Path(root) / "images"
        self.tfms    = tfms or (train_tfms() if split == "trainval" else val_tfms())
        self.samples = []
        with open(Path(root) / "annotations" / "list.txt") as f:
            for line in f:
                line = line.strip()
                if line.startswith("#") or not line:
                    continue
                parts = line.split()
                self.samples.append((parts[0], int(parts[1]) - 1))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        name, label = self.samples[idx]
        img = np.array(Image.open(self.img_dir / f"{name}.jpg").convert("RGB"))
        img = self.tfms(image=img, bboxes=[], cls=[])["image"]
        return img, label


class PetDetectionDataset(Dataset):
    """
    image + bbox [cx, cy, w, h] in PIXEL space (not normalised).
    Labels are raw pixel coordinates at IMG_SIZE resolution.
    """
    def __init__(self, root, split="trainval", tfms=None):
        self.img_dir = Path(root) / "images"
        self.xml_dir = Path(root) / "annotations" / "xmls"
        self.tfms    = tfms or (train_tfms() if split == "trainval" else val_tfms())
        self.samples = [f.stem for f in sorted(self.xml_dir.glob("*.xml"))]

    def _parse_xml(self, path, orig_w, orig_h):
        root = ET.parse(path).getroot()
        bb = root.find("object").find("bndbox")
        x1 = float(bb.find("xmin").text)
        y1 = float(bb.find("ymin").text)
        x2 = float(bb.find("xmax").text)
        y2 = float(bb.find("ymax").text)
        # convert to cx,cy,w,h in pixel space at IMG_SIZE
        scale_x = IMG_SIZE / orig_w
        scale_y = IMG_SIZE / orig_h
        cx = ((x1 + x2) / 2) * scale_x
        cy = ((y1 + y2) / 2) * scale_y
        w  = (x2 - x1) * scale_x
        h  = (y2 - y1) * scale_y
        return [cx, cy, w, h]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        name = self.samples[idx]
        img  = np.array(Image.open(self.img_dir / f"{name}.jpg").convert("RGB"))
        oh, ow = img.shape[:2]
        box  = self._parse_xml(self.xml_dir / f"{name}.xml", ow, oh)

        # apply image transforms (no bbox transform needed since we scale manually)
        out = self.tfms(image=img, bboxes=[], cls=[])
        return out["image"], torch.tensor(box, dtype=torch.float32)


class PetSegmentationDataset(Dataset):
    """
    image + trimap mask.
    Trimap 1,2,3 → class 0,1,2 (foreground, background, boundary)
    """
    def __init__(self, root, split="trainval"):
        self.img_dir  = Path(root) / "images"
        self.mask_dir = Path(root) / "annotations" / "trimaps"
        self.tfms     = seg_tfms(train=(split == "trainval"))
        self.samples  = []
        with open(Path(root) / "annotations" / "list.txt") as f:
            for line in f:
                line = line.strip()
                if line.startswith("#") or not line:
                    continue
                name = line.split()[0]
                if (self.mask_dir / f"{name}.png").exists():
                    self.samples.append(name)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        name = self.samples[idx]
        img  = np.array(Image.open(self.img_dir  / f"{name}.jpg").convert("RGB"))
        mask = np.array(Image.open(self.mask_dir / f"{name}.png"))

        cls_mask = np.zeros_like(mask, dtype=np.uint8)
        cls_mask[mask == 1] = 0
        cls_mask[mask == 2] = 1
        cls_mask[mask == 3] = 2

        img = self.tfms(image=img)["image"]
        cls_mask = np.array(
            Image.fromarray(cls_mask).resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)
        )
        return img, torch.from_numpy(cls_mask).long()
