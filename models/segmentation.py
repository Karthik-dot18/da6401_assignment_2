import torch
import torch.nn as nn


def dec_block(in_c, skip_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c + skip_c, out_c, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True)
    )


class UNet(nn.Module):
    """
    U-Net with VGG11 encoder for trimap segmentation (3 classes).

    Upsampling: ConvTranspose2d strictly — no bilinear/unpooling.
    Skip connections: channel-wise concatenation at each stage.
    Loss: CE + Dice (background dominates trimap, plain CE is insufficient).
    """
    def __init__(self, backbone=None, num_classes=3):
        super().__init__()
        self.encoder = backbone

        self.up5  = nn.ConvTranspose2d(512, 512, 2, stride=2)
        self.dec5 = dec_block(512, 512, 512)

        self.up4  = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4 = dec_block(256, 256, 256)

        self.up3  = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = dec_block(128, 128, 128)

        self.up2  = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = dec_block(64, 64, 64)

        self.up1  = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True)
        )
        self.head = nn.Conv2d(32, num_classes, 1)

    def forward(self, x):
        s1, s2, s3, s4, s5 = self.encoder.forward_skips(x)
        d = self.dec5(torch.cat([self.up5(s5), s4], dim=1))
        d = self.dec4(torch.cat([self.up4(d),  s3], dim=1))
        d = self.dec3(torch.cat([self.up3(d),  s2], dim=1))
        d = self.dec2(torch.cat([self.up2(d),  s1], dim=1))
        d = self.dec1(self.up1(d))
        return self.head(d)
