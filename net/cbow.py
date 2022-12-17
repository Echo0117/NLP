# -*- coding:utf-8 -*-
"""
@file: cbow.py
@time: 16/11/2022 14:07
@desc: 
@author: Echo
"""
import torch.nn as nn
import torch


class CBOW_classifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, kernel_size=2, window_size=2, hidden=False, hidden_dim=10):
        super(CBOW_classifier, self).__init__()
        # To create an embedding table: https://pytorch.org/docs/stable/nn.html#torch.nn.Embedding
        self.emb_table = torch.nn.Embedding(vocab_size, embedding_dim)
        if not hidden:  # no hidden
            self.classifier = nn.Sequential(
                nn.Linear(embedding_dim, 1)
            )
        else:  # hidden layer
            self.classifier = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim),
                nn.ReLU(),
                nn.Linear(embedding_dim, 1)
            )

    def forward(self, inputs):
        word_emb = self.emb_table(inputs)
        mask = (inputs != 0).to(torch.int).unsqueeze(dim=2).expand(word_emb.size())
        word_emb *= mask
        sentence_emb = word_emb.mean(dim=1)
        output = self.classifier(sentence_emb)  # output is a scalar
        return output
