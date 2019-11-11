# !/usr/bin/env python
# -*- coding: utf-8 -*-

# It's written by https://github.com/yumatsuoka

import torch
import torch.nn as nn


class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.1, dim=1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.dim = dim

    def forward(self, pred, target):
        classes = pred.size(self.dim)
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (classes - 1))
            true_dist.scatter_(self.dim, target.data.unsqueeze(1), self.confidence)
        return nn.functional.binary_cross_entropy_with_logits(pred, true_dist)


if __name__ == "__main__":
    lsl = LabelSmoothingLoss()

    sample_input = torch.randn((3, 3), requires_grad=True)
    sample_target = torch.empty(3, dtype=torch.long).random_(2)
    loss = lsl(sample_input, sample_target)
    print("loss=", loss.item())
    loss.backward()
