import torch
import torch.nn as nn


class IoULoss(nn.Module):
    """
    Differentiable IoU loss for bbox regression.
    Input: [cx, cy, w, h] in pixel space.
    Loss = 1 - IoU, always in range [0, 1].

    reduction: "mean" (default) or "sum"
    """
    def __init__(self, reduction="mean", eps=1e-7):
        super().__init__()
        assert reduction in ("mean", "sum"), f"reduction must be 'mean' or 'sum', got {reduction}"
        self.reduction = reduction
        self.eps       = eps

    def _to_xyxy(self, b):
        cx, cy, w, h = b.unbind(-1)
        return torch.stack([cx - w/2, cy - h/2,
                            cx + w/2, cy + h/2], dim=-1)

    def forward(self, pred, target):
        p = self._to_xyxy(pred)
        t = self._to_xyxy(target)

        ix1 = torch.max(p[..., 0], t[..., 0])
        iy1 = torch.max(p[..., 1], t[..., 1])
        ix2 = torch.min(p[..., 2], t[..., 2])
        iy2 = torch.min(p[..., 3], t[..., 3])

        inter = (ix2 - ix1).clamp(0) * (iy2 - iy1).clamp(0)
        ap = (p[...,2]-p[...,0]).clamp(0) * (p[...,3]-p[...,1]).clamp(0)
        at = (t[...,2]-t[...,0]).clamp(0) * (t[...,3]-t[...,1]).clamp(0)

        iou  = inter / (ap + at - inter + self.eps)
        loss = 1 - iou  # in [0, 1]

        if self.reduction == "mean":
            return loss.mean()
        return loss.sum()
