# -*- coding:utf-8 -*-
"""
@file: metrics.py
@time: 16/11/2022 14:38
@desc: 
@author: Echo
"""
from sklearn.metrics import accuracy_score


class Metrics:
    def __int__(self):
        pass

    def accuracy(self, y_true, y_pred):
        return accuracy_score(list(y_pred), list(y_true))