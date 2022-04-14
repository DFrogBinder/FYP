from . import regnet
import numpy as np
import torch
import json
from easydict import EasyDict
from prettytable import PrettyTable
from torch.nn import init

class StopCriterion(object):
    def __init__(self, stop_std=0.001, query_len=100, num_min_iter=200):
        self.query_len = query_len
        self.stop_std = stop_std
        self.loss_list = []
        self.loss_min = 1.
        self.num_min_iter = num_min_iter

    def add(self, loss):
        self.loss_list.append(loss)
        if loss < self.loss_min:
            self.loss_min = loss
            self.loss_min_i = len(self.loss_list)

    def stop(self):
        # return True if the stop creteria are met
        query_list = self.loss_list[-self.query_len:]
        query_std = np.std(query_list)
        if query_std < self.stop_std and self.loss_list[-1] - self.loss_min < self.stop_std / 3. and len(
                self.loss_list) > self.loss_min_i and len(self.loss_list) > self.num_min_iter:
            return True
        else:
            return False

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

