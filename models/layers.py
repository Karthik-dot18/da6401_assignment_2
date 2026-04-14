import torch
import torch.nn as nn


class CustomDropout(nn.Module):
    """
    Inverted dropout from scratch. No nn.Dropout used.
    - train: zero neurons randomly, scale survivors by 1/(1-p)
    - eval:  pass through unchanged
    """
    def __init__(self, p=0.5):
        super().__init__()
        assert 0.0 <= p < 1.0, f"p must be in [0,1), got {p}"
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        keep = 1.0 - self.p
        mask = torch.empty_like(x).bernoulli_(keep)
        return x * mask / keep

    def extra_repr(self):
        return f"p={self.p}"
