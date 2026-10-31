"""ConvLSTM with nested marginal exceedance probabilities, no softmax."""
import math
import torch
from torch import nn


def nested_logits(raw):
    """p5=sigmoid(a)*sigmoid(b); stable marginal logits for BCEWithLogits."""
    a, b = raw[:, 0], raw[:, 1]
    high = a + b - torch.logsumexp(torch.stack((torch.zeros_like(a), a, b)), dim=0)
    return torch.stack((a, high), dim=1)


class ConvLSTM(nn.Module):
    def __init__(self, channels, hidden=16, prevalence=(0.001, 0.0001), task='dual'):
        super().__init__()
        if task not in ('dual', 'low', 'high'):
            raise ValueError(task)
        self.hidden, self.task = hidden, task
        self.gates = nn.Conv2d(channels + hidden, 4 * hidden, 3, padding=1)
        self.head = nn.Conv2d(hidden, 2 if task == 'dual' else 1, 1)
        nn.init.normal_(self.head.weight, std=0.01)
        p4, p5 = map(float, prevalence)
        values = [p4, p5 / max(p4, 1e-12)] if task == 'dual' else [p4 if task == 'low' else p5]
        with torch.no_grad():
            for i, p in enumerate(values):
                p = min(max(p, 1e-7), 1-1e-7)
                self.head.bias[i] = math.log(p/(1-p))

    def forward(self, x):
        b, t, _, h, w = x.shape
        hx = x.new_zeros((b, self.hidden, h, w))
        cx = torch.zeros_like(hx)
        for j in range(t):
            i, f, o, g = self.gates(torch.cat((x[:, j], hx), dim=1)).chunk(4, 1)
            cx = torch.sigmoid(f)*cx + torch.sigmoid(i)*torch.tanh(g)
            hx = torch.sigmoid(o)*torch.tanh(cx)
        raw = self.head(hx)
        return nested_logits(raw) if self.task == 'dual' else raw
