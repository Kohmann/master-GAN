import torch
import torch.nn as nn


def rnn_weight_init(module):
    with torch.no_grad():
        for name, param in module.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                torch.nn.init.xavier_uniform_(param.data)
            elif 'bias_ih' in name:
                param.data.fill_(1)
            elif 'bias_hh' in name:
                param.data.fill_(0)


def linear_weight_init(module):
    with torch.no_grad():
        for name, param in module.named_parameters():
            if 'weight' in name:
                torch.nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                param.data.fill_(0)


def global_weight_init(module):
    with torch.no_grad():
        for name, param in module.named_parameters():
            if 'weight' in name:
                torch.nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                param.data.fill_(0)


def weight_init(module):
    for m in module:
        if isinstance(m, nn.Conv2d):
            # nn.init.normal_(m.weight, 0, 0.02)
            nn.init.xavier_uniform_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.xavier_uniform_(m.weight)
            # nn.init.normal_(m.weight, 0, 0.02)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            # nn.init.normal_(m.weight, 0, 0.02)
            nn.init.constant_(m.bias, 0)
