# -*- coding:utf-8 -*-
"""
@file: save_load_models.py
@time: 22/11/2022 09:52
@desc: 
@author: Echo
"""
import torch


def save_models_pt(model, path):
    torch.save(model, path)


def save_state_dict():
    pass


def load_models(path):
    return torch.load(path)
