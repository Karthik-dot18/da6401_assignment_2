"""Inference and evaluation — DA6401 Assignment 2."""

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import gdown

from multitask import MultiTaskPerceptionModel

MEAN     = np.array([0.485, 0.456, 0.406])
STD      = np.array([0.229, 0.224, 0.225])
IMG_SIZE = 224
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRIMAP_COLORS = np.array([
    [255,  87,  51],   # foreground
    [ 52, 152, 219],   # background
    [255, 215,   0],   # boundary
], dtype=np.uint8)


def download_checkpoints(cls_id, loc_id, seg_id):
    """Download checkpoints from Google Drive using gdown."""
    import os
    os.makedirs("checkpoints", exist_ok=True)
    gdown.download(id=cls_id, output="checkpoints/classifier.pth", quiet=False)
    gdown.download(id=loc_id, output="checkpoints/localizer.pth",  quiet=False)
    gdown.download(id=seg_id, output="checkpoints/unet.pth",       quiet=False)


def preprocess(img_path):
    img = np.array(Image.open(img_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE)))
    img = (img / 255.0 - MEAN) / STD
    return torch.tensor(img.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0)


def run_pipeline(img_path):
    model = MultiTaskPerceptionModel().to(DEVICE)
    model.eval()

    img_t = preprocess(img_path).to(DEVICE)
    with torch.no_grad():
        cls_out, bbox_out, seg_out = model(img_t)

    breed_idx = cls_out.argmax(1).item()
    bbox      = bbox_out[0].cpu().numpy()   # [cx, cy, w, h] pixels
    seg_mask  = seg_out[0].argmax(0).cpu().numpy()

    orig = np.array(Image.open(img_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE)))
    cx, cy, w, h = bbox
    x1, y1 = int(cx - w/2), int(cy - h/2)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(orig)
    axes[0].add_patch(patches.Rectangle(
        (x1, y1), int(w), int(h), lw=2, edgecolor="red", facecolor="none"
    ))
    axes[0].set_title(f"Localisation (class {breed_idx})")
    axes[1].imshow(orig)
    axes[1].set_title("Original")
    axes[2].imshow(TRIMAP_COLORS[seg_mask])
    axes[2].set_title("Segmentation")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("pipeline_output.png", dpi=150)
    plt.show()

    return cls_out, bbox, seg_mask


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--img", required=True)
    args = p.parse_args()
    run_pipeline(args.img)
