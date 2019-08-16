import torch
import torch.nn as nn
import torch.nn.functional as F

import losses


class KDLoss(nn.Module):
    def __init__(self, t=1.0, alpha=0.9, reduce=True):
        super(KDLoss, self).__init__()
        self.t = t
        self.alpha = alpha
        self.focal = losses.focal()
        self.reduce = reduce

    def forward(self, teacher_output, teacher_cosine, teacher_feature, 
                      student_output, student_cosine, student_feature, 
                      targets):

        soft_log_probs = F.log_softmax(student_cosine / self.t, dim=1)
        soft_targets = F.softmax(teacher_cosine / self.t, dim=1)

        kl = F.kl_div(soft_log_probs, soft_targets, reduction='batchmean') * (self.t ** 2) * self.alpha

        focal = self.focal(student_output, targets) * (1 - self.alpha)

        mse = F.mse_loss(student_feature, teacher_feature, reduction='mean')

        return kl, focal, mse

        # loss = kl + ce + mse

        # if self.reduce:
        #     return torch.mean(loss)
        # return loss
        
        
def kd(*argv, **kwargs):
    return KDLoss(*argv, **kwargs)
