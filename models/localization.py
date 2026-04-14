import torch
import torch.nn as nn
from models.layers import CustomDropout


class LocalizationModel(nn.Module):
    """
    VGG11 encoder + regression head.
    Output: [x_center, y_center, width, height] in PIXEL space (not normalised).
    Loss: MSE + IoULoss (see losses/iou_loss.py)

    Full backbone fine-tuning: pet head bbox is domain-specific,
    frozen features don't localise well enough.
    """
    def __init__(self, backbone=None, img_size=224):
        super().__init__()
        self.img_size = img_size
        self.encoder  = backbone

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            CustomDropout(0.3),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 4),
            nn.ReLU()   # output >= 0, pixel coords are non-negative
        )

    def forward(self, x):
        # get bottleneck features from backbone
        s1 = self.encoder.b1(x)
        s2 = self.encoder.b2(s1)
        s3 = self.encoder.b3(s2)
        s4 = self.encoder.b4(s3)
        s5 = self.encoder.b5(s4)
        return self.head(s5)   # (B, 4) in pixel space
