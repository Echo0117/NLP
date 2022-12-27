# -*- coding:utf-8 -*-
"""
@file: optimizer.py
@time: 21/11/2022 21:24
@desc: 
@author: Echo
"""
import torch


class Optimizer:
    def __init__(self):
        pass

    def adam_optimizer(self, model, lr):
        return torch.optim.Adam(model.parameters(), lr=lr)

    def SGD_optimizer(self, model, lr):
        return torch.optim.SGD(model.parameters(), lr=lr)