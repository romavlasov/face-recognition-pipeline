import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduce = reduce

    def forward(self, inputs, targets):
        ce = F.cross_entropy(inputs, targets, reduction='none')
        p = torch.exp(-ce)
        loss = self.alpha * (1 - p) ** self.gamma * ce

        if self.reduce:
            return torch.mean(loss)
        return loss
        
        
def focal(*argv, **kwargs):
    return FocalLoss(*argv, **kwargs)
