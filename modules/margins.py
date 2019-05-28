import math

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['arc_margin', 'cos_margin']


class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=64.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, target):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        cosine.clamp(-1, 1)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, target.view(-1, 1).long(), 1)

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s
        return output


class AddMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.40):
        super(AddMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, target):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        cosine.clamp(-1, 1)
        phi = cosine - self.m

        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, target.view(-1, 1).long(), 1)

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output


def arc_margin(in_features, out_features, s=64.0, m=0.50, easy_margin=False, device='cpu'):
    margin = ArcMarginProduct(in_features, out_features, s, m, easy_margin)
    return margin.to(device)


def add_margin(in_features, out_features, s=30.0, m=0.40, device='cpu'):
    margin = ArcMarginProduct(in_features, out_features, s, m)
    return margin.to(device)
