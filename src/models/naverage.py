#!/usr/bin/env python3
import torch

class NaiveAverage(torch.nn.Module):  # noqa
    def __init__(self):
        """Returns prediction consisting of repeating last frame."""
        super(NaiveAverage, self).__init__()

    def forward(self, x: torch.Tensor, *args, **kwargs):
        x = torch.mean(x, dim=1)
        x = torch.unsqueeze(x, dim=1)
        x = torch.repeat_interleave(x, repeats=6, dim=1)
        return x
