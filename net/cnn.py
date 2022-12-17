# -*- coding:utf-8 -*-
"""
@file: cnn.py
@time: 16/11/2022 14:07
@desc: 
@author: Echo
"""

import torch.nn as nn
import torch


class CNN(nn.Module):
    def __init__(self, input_shape, kernel_size, window_size, embedding_dim, hidden_dim, dropout) -> None:
        super(CNN, self).__init__()
        self.input_shape = input_shape
        self.kernal_size = kernel_size
        self.window_size = window_size
        self.kernals = torch.randn(self.kernal_size, self.window_size)
        self.biases = torch.randn(self.input_shape[1])
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.proj = nn.Linear(self.window_size * self.embedding_dim, hidden_dim)  # input size and output size
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        batch_size, words, embedding_size = input.size()
        z = torch.zeros((batch_size, words - self.window_size + 1, self.window_size * self.embedding_dim))
        for step in range(words - self.window_size + 1):
            z[:, step, :] = input[:, step:self.window_size + step, :].reshape(z[:, step, :].size())
        z = self.proj(z)
        z, _ = torch.max(z, dim=1)
        z = nn.ReLU()(z)  # hidden representation of a sentence
        output = self.dropout(z)
        return output  # remove unnecessary dimension
