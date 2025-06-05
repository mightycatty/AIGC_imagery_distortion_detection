import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, dim):
        super(Attention, self).__init__()
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.attend = nn.Softmax(dim=-1)
        self.to_out = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        attn = self.attend(torch.matmul(q, k.transpose(-2, -1)) / (x.shape[-1] ** 0.5))
        out = torch.matmul(attn, v)
        return self.to_out(out)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super(FeedForward, self).__init__()
        self.ff = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):
        return self.ff(x)

class Residual(nn.Module):
    def __init__(self, module):
        super(Residual, self).__init__()
        self.module = module

    def forward(self, x):
        return x + self.module(x)