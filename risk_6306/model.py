import torch
from torch import nn
import math


class ConvLSTM(nn.Module):
    """One recurrent spatial layer; output logits for each fixed area's occupancy."""
    def __init__(self, channels, hidden=16, prevalence=0.001):
        super().__init__()
        self.hidden = hidden
        self.gates = nn.Conv2d(channels+hidden, 4*hidden, 3, padding=1)
        self.head = nn.Conv2d(hidden, 1, 1)
        nn.init.normal_(self.head.weight, std=0.01)
        p = min(max(float(prevalence), 1e-7), 1-1e-7)
        nn.init.constant_(self.head.bias, math.log(p/(1-p)))

    def forward(self, x):
        b, t, _, h, w = x.shape
        hx = x.new_zeros((b, self.hidden, h, w))
        cx = torch.zeros_like(hx)
        for j in range(t):
            i, f, o, g = self.gates(torch.cat([x[:, j], hx], 1)).chunk(4, 1)
            cx = torch.sigmoid(f)*cx+torch.sigmoid(i)*torch.tanh(g)
            hx = torch.sigmoid(o)*torch.tanh(cx)
        return self.head(hx)[:, 0]
