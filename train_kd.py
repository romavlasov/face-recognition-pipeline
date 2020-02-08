import os
import datetime
import argparse
import yaml

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from tqdm import tqdm

import losses

from models import encoders
from modules import margins

from data.datasets import insightface
from data.transform import Transforms

from metrics.classification import accuracy

from utils.handlers import AverageMeter
from utils.handlers import MetaData

from utils.storage import save_weights
from utils.storage import load_weights


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class BaseModel(nn.Module):
    def __init__(self, model_name, margin_name, features, num_classes, prefix):
        super(BaseModel, self).__init__()
        self.prefix = prefix

        self.model = getattr(encoders, model_name)(out_features=features, 
                                                   device=device)

        self.margin = getattr(margins, margin_name)(in_features=features, 
                                                    out_features=num_classes, 
                                                    device=device)

    def load_model(self, prefix, epoch):
        load_weights(self.model, prefix, 'model', epoch)

    def load_margin(self, prefix, epoch):
        load_weights(self.margin, prefix, 'margin', epoch)

    def load_weight(self, epoch):
        self.load_model(self.prefix, epoch)
        self.load_margin(self.prefix, epoch)

    def save_weights(self, epoch, parallel=True):
        save_weights(self.model, self.prefix, 'model', epoch, parallel)
        #save_weights(self.margin, self.prefix, 'margin', epoch, parallel)

    def parallel(self):
        self.model = nn.DataParallel(self.model)
        self.margin = nn.DataParallel(self.margin)

    def set_grad(self, model=True, margin=True):
        for param in self.model.parameters():
            param.requires_grad = model
        for param in self.margin.parameters():
            param.requires_grad = margin

    def forward(self, x, target):
        feature = self.model(x)
        output, cosine = self.margin(feature, target)
        return output, cosine, feature


def main(config):
    teacher = BaseModel(**config['teacher'])
    teacher.load_weight(25)
    teacher.set_grad(model=False, margin=False)

    student = BaseModel(**config['student'])

    if config['model_transfer']:
        student.load_model('mobilenet_256d', 30)

    if config['margin_transfer']:
        student.load_margin(config['teacher']['prefix'], 25)

    student.set_grad(model=config['model_grad'], margin=config['margin_grad'])

    if config['snapshot']['use']:
        student.load_weight(config['snapshot']['epoch'])
    
    if torch.cuda.is_available() and config['parallel']:
        teacher.parallel()
        student.parallel()

    criterion = getattr(losses, config['criterion'])()
        
    optimizer = optim.SGD(student.parameters(),
                          lr=config['learning_rate'],
                          momentum=config['momentum'],
                          weight_decay=config['weight_decay'])
    
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                  milestones=config['milestones'],
                                                  gamma=0.1)
    
    transforms = Transforms(input_size=config['input_size'], train=True)
    data_loader = DataLoader(insightface.Train(folder=config['train']['folder'], 
                                               dataset=config['train']['dataset'],
                                               transforms=transforms),
                             batch_size=config['batch_size'], 
                             num_workers=config['num_workers'],
                             shuffle=True)
    
    for epoch in range(config['num_epochs']):
        print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        
        train(data_loader, teacher, student, criterion, optimizer, epoch, config)
        lr_scheduler.step()
        
        if (epoch + 1) % config['save_freq'] == 0:
            student.save_weights(epoch + 1, config['parallel'])
        

def train(data_loader, 
          teacher, 
          student, 
          criterion, optimizer, epoch, config):
    
    top_1 = AverageMeter()
    top_n = AverageMeter()
#     kl_losses = AverageMeter()
    ce_losses = AverageMeter()
    mse_losses = AverageMeter()
    
    teacher.eval()
    student.train()

    tq = tqdm(total=len(data_loader) * config['batch_size'])
    
    for i, (image, target) in enumerate(data_loader):
        image = image.to(device)
        target = target.to(device)
        
        teacher_output, teacher_cosine, teacher_feature = teacher(image, target)
        student_output, student_cosine, student_feature = student(image, target)

        ce_loss, mse_loss = criterion(teacher_output, teacher_cosine, teacher_feature, 
                                      student_output, student_cosine, student_feature, 
                                      target)

#         kl_losses.update(kl_loss.item())
        ce_losses.update(ce_loss.item())
        mse_losses.update(mse_loss.item())

        loss = ce_loss * config['ce_weight'] + mse_loss * config['mse_weight']
        
        acc_1, acc_n = accuracy(student_cosine, target, topk=(1, 10))
        top_1.update(acc_1.item())
        top_n.update(acc_n.item())

        loss.backward()
        if (i + 1) % config['step'] == 0:
            optimizer.step()
            optimizer.zero_grad()

        current_lr = get_learning_rate(optimizer)

        tq.set_description('Epoch {}, lr {:.2e}'.format(epoch + 1, current_lr))
        tq.set_postfix(ce_loss='{:.4f}'.format(ce_losses.avg),
#                        kl_loss='{:.4f}'.format(kl_losses.avg),
                       mse_loss='{:.4f}'.format(mse_losses.avg),
                       top_1='{:.4f}'.format(top_1.avg),
                       top_n='{:.4f}'.format(top_n.avg))
        tq.update(config['batch_size'])

    tq.close()


def validation(data_loader, model):
    pass


def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train code')
    parser.add_argument('--config', required=True, help='configuration file')
    args = parser.parse_args()
    
    config = yaml.load(open(args.config), Loader=yaml.FullLoader)
    main(config)
