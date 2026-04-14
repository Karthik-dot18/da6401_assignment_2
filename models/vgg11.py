import torch
import torch.nn as nn
from models.layers import CustomDropout


def make_conv(in_c, out_c):
    """3x3 Conv + BN + ReLU. BN before ReLU stabilises training."""
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True)
    )


class VGG11(nn.Module):
    """
    VGG11 with BatchNorm and CustomDropout for 37-class pet classification.
    Architecture follows the original paper exactly:
    cfg: 64,M,128,M,256,256,M,512,512,M,512,512,M

    BatchNorm: placed after each Conv, before ReLU — stabilises
               activations and allows higher learning rates.
    CustomDropout: only on FC layers (not conv) — conv already
                   regularised by BN; FC layers have 25M+ params
                   so dropout(0.5) acts as strong ensemble regulariser.
    """
    def __init__(self, num_classes=37, p=0.5):
        super().__init__()
        # conv backbone — 5 blocks
        self.b1 = nn.Sequential(make_conv(3,   64),  nn.MaxPool2d(2, 2))
        self.b2 = nn.Sequential(make_conv(64,  128), nn.MaxPool2d(2, 2))
        self.b3 = nn.Sequential(make_conv(128, 256), make_conv(256, 256), nn.MaxPool2d(2, 2))
        self.b4 = nn.Sequential(make_conv(256, 512), make_conv(512, 512), nn.MaxPool2d(2, 2))
        self.b5 = nn.Sequential(make_conv(512, 512), make_conv(512, 512), nn.MaxPool2d(2, 2))

        self.pool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(p),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(p),
            nn.Linear(4096, num_classes)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def forward_skips(self, x):
        """For U-Net skip connections."""
        s1 = self.b1(x)
        s2 = self.b2(s1)
        s3 = self.b3(s2)
        s4 = self.b4(s3)
        s5 = self.b5(s4)
        return s1, s2, s3, s4, s5

    def get_backbone(self):
        """Return just the conv blocks for use as encoder."""
        return nn.ModuleList([self.b1, self.b2, self.b3, self.b4, self.b5])
