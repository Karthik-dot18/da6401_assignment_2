import torch
import torch.nn as nn
from models.vgg11 import VGG11
from models.localization import LocalizationModel
from models.segmentation import UNet
from models.layers import CustomDropout


class MultiTaskPerceptionModel(nn.Module):
    """
    Unified multi-task model.
    Loads weights from:
      checkpoints/classifier.pth
      checkpoints/localizer.pth
      checkpoints/unet.pth

    Single forward pass → (cls_logits, bbox_pixels, seg_mask)
    """
    def __init__(self,
                 num_classes=37,
                 num_seg=3,
                 cls_ckpt="checkpoints/classifier.pth",
                 loc_ckpt="checkpoints/localizer.pth",
                 seg_ckpt="checkpoints/unet.pth"):
        super().__init__()

        # shared backbone
        self.backbone = VGG11(num_classes=num_classes)

        # load classifier weights into backbone if checkpoint exists
        self._load_cls(cls_ckpt)

        # classification head (reuse backbone's classifier)
        self.cls_head = self.backbone.classifier

        # localization head
        self.loc_model = LocalizationModel(backbone=self.backbone)
        self._load_ckpt(self.loc_model, loc_ckpt)

        # segmentation decoder
        self.seg_model = UNet(backbone=self.backbone, num_classes=num_seg)
        self._load_ckpt(self.seg_model, seg_ckpt)

    def _load_cls(self, path):
        try:
            sd = torch.load(path, map_location="cpu")
            self.backbone.load_state_dict(sd, strict=False)
            print(f"loaded classifier weights from {path}")
        except Exception:
            print(f"no classifier checkpoint at {path}, using random weights")

    def _load_ckpt(self, model, path):
        try:
            sd = torch.load(path, map_location="cpu")
            model.load_state_dict(sd, strict=False)
            print(f"loaded {path}")
        except Exception:
            print(f"no checkpoint at {path}, using random weights")

    def forward(self, x):
        # classification
        s1 = self.backbone.b1(x)
        s2 = self.backbone.b2(s1)
        s3 = self.backbone.b3(s2)
        s4 = self.backbone.b4(s3)
        s5 = self.backbone.b5(s4)

        pool = self.backbone.pool(s5)
        cls  = self.cls_head(torch.flatten(pool, 1))

        # bbox (pixel space)
        bbox = self.loc_model.head(s5)

        # segmentation
        d = self.seg_model.dec5(torch.cat([self.seg_model.up5(s5), s4], dim=1))
        d = self.seg_model.dec4(torch.cat([self.seg_model.up4(d),  s3], dim=1))
        d = self.seg_model.dec3(torch.cat([self.seg_model.up3(d),  s2], dim=1))
        d = self.seg_model.dec2(torch.cat([self.seg_model.up2(d),  s1], dim=1))
        d = self.seg_model.dec1(self.seg_model.up1(d))
        seg = self.seg_model.head(d)

        return cls, bbox, seg
