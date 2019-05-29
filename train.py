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

from models import encoders
from modules import margins

from data.datasets import insightface
from data.transform import Transforms

from losses.focal_loss import FocalLoss

from metrics.classification import accuracy

from utils.handlers import AverageMeter
from utils.handlers import MetaData

from utils.storage import save_weights
from utils.storage import load_weights


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main(config):   
    model = getattr(encoders, config['model_name'])(out_features=config['features'],
                                                    device=device)
    
    margin = getattr(margins, config['margin_name'])(in_features=config['features'], 
                                                     out_features=config['num_classes'], 
                                                     device=device)
    
    if config['snapshot']['use']:
        load_weights(model, config['prefix'], 'model', config['snapshot']['epoch'])
        load_weights(margin, config['prefix'], 'margin', config['snapshot']['epoch'])
    
    if torch.cuda.is_available() and config['parallel']:
        model = nn.DataParallel(model)
        margin = nn.DataParallel(margin)

    if config['criterion'] == 'FocalLoss':
        criterion = FocalLoss(gamma=2)
    elif config['criterion'] == 'CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss()
        
    if config['optimizer'] == 'SGD':
        optimizer = optim.SGD([{'params': model.parameters()}, 
                               {'params': margin.parameters()}],
                               lr=config['learning_rate'],
                               momentum=config['momentum'],
                               weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'Adam':
        optimizer = optim.Adam([{'params': model.parameters()}, 
                                {'params': margin.parameters()}],
                               lr=config['learning_rate'],
                               weight_decay=config['weight_decay'])
    
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                  milestones=config['milestones'],
                                                  gamma=0.1)
    
    transforms = Transforms(input_size=config['input_size'], train=True)
    data_loader = DataLoader(insightface.Train(folder=config['train']['folder'],
                                               transforms=transforms),
                             batch_size=config['batch_size'], 
                             num_workers=config['num_workers'],
                             shuffle=True)
    
    for epoch in range(config['num_epochs']):
        lr_scheduler.step()
        train(data_loader, model, margin, criterion, optimizer, epoch)
        
        if (epoch + 1) % config['save_freq'] == 0:
            save_weights(model, config['prefix'], 'model', epoch + 1, config['parallel'])
            save_weights(margin, config['prefix'], 'margin', epoch + 1, config['parallel'])
        

def train(data_loader, model, margin, criterion, optimizer, epoch):
    top_1 = AverageMeter()
    top_n = AverageMeter()
    losses = AverageMeter()
    
    model.train()
    margin.train()
    
    for i, (image, target) in enumerate(data_loader):
        image = image.to(device)
        target = target.to(device)
        
        optimizer.zero_grad()
        
        feature = model(image)
        output = margin(feature, target)

        loss = criterion(output, target)
        losses.update(loss.item(), image.size(0))
        
        acc_1, acc_n = accuracy(output, target, topk=(1, 10))
        top_1.update(acc_1[0], image.size(0))
        top_n.update(acc_n[0], image.size(0))
        
        loss.backward()
        optimizer.step()
        
        current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        current_lr = get_learning_rate(optimizer)
        iter_ = i + (epoch * len(data_loader)) + 1
        
        if (iter_ + 1) % 100 == 0:
            print(' Time: {}'
                  ' Lr: {:.2e}'
                  ' Epoch: {} -'
                  ' Iter: {} -'
                  ' Loss: {:.4f} -'
                  ' Top1: {:.4f} -'
                  ' Top10: {:.4f}'.format(current_time, current_lr,
                                          epoch + 1, iter_ + 1, 
                                          losses.val, top_1.val, top_n.val))


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
