# -*- coding:utf-8 -*-
"""
@file: losses.py
@time: 21/11/2022 20:54
@desc: 
@author: Echo
"""
import torch.nn as nn


class losses():
    def __int__(self):
        pass

    def BCEWithLogitsLoss(self, reduction='mean'):
        return nn.BCEWithLogitsLoss(reduction=reduction)

    def NLLLoss(self):
        return nn.NLLLoss()