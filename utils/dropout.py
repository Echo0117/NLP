# -*- coding:utf-8 -*-
"""
@file: dropout.py
@time: 27/12/2022 10:34
@desc: 
@author: Echo
"""
import torch
import torch.nn as nn


class VariationalDropout(nn.Module):
    def __init__(self, dropout: float):
        super().__init__()
        self.dropout = dropout
        if dropout < 0 or dropout > 1:
            print("Warning: Dropout should be >0 and <1")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.dropout <= 0. or self.dropout > 1.:
            return x
        batch_size, num_words, emb_dim = x.size()
        m = x.new_empty(batch_size, 1, emb_dim, requires_grad=False).bernoulli_(1 - self.dropout)
        x = (x * m) / (1 - self.dropout)
        return x
