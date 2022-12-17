# -*- coding:utf-8 -*-
"""
@file: cbow_cnn.py
@time: 16/11/2022 14:29
@desc: 
@author: Echo
"""
from net.cbow import CBOW_classifier
import torch.nn as nn
import torch
from net.cnn import CNN


class CbowCNNClassifier(CBOW_classifier):
    def __init__(self, vocab_size, embedding_dim, kernel_size=2, window_size=2, hidden=False, hidden_dim=10, dropout=0.2):
        super(CBOW_classifier, self).__init__()
        # To create an embedding table: https://pytorch.org/docs/stable/nn.html#torch.nn.Embedding
        self.emb_table = torch.nn.Embedding(vocab_size, embedding_dim)
        self.cnn = CNN([vocab_size, embedding_dim], kernel_size, window_size, embedding_dim, hidden_dim, dropout)
        if not hidden:  # no hidden
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, 1)
            )
        else:  # hidden layer
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )

    def forward(self, inputs):
        word_emb = self.emb_table(inputs)
        mask = (inputs!=0).to(torch.int).unsqueeze(dim=2).expand(word_emb.size())
        word_emb *= mask
        # sentence_emb = word_emb.mean(dim=1)
        cnn_out = self.cnn(word_emb)
        output = self.classifier(cnn_out)  # output is a scalar
        return output

    def predict(self, inputs):
        label = self(inputs).detach().numpy()
        label[label < 0] = 0
        label[label > 0] = 1
        return label
