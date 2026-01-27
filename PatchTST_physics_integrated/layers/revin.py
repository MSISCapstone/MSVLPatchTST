import torch
import torch.nn as nn

class RevIN(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, 1, num_features))
        self.beta  = nn.Parameter(torch.zeros(1, 1, num_features))

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        std  = x.std(dim=1, keepdim=True) + self.eps
        return (x - mean) / std * self.gamma + self.beta

