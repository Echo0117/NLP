# -*- coding:utf-8 -*-
"""
@file: n_gram.py
@time: 26/12/2022 11:51
@desc: 
@author: Echo
"""
import torch.nn as nn
import torch.nn.functional as F


class nGramModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, batch_size, n_grams=2, hidden_size=10):
        super(nGramModel, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.n_grams = n_grams
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.emb_table = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.linear1 = nn.Linear(self.n_grams * self.embedding_dim, self.hidden_size)
        self.linear2 = nn.Linear(self.hidden_size, self.vocab_size)

    def forward(self, inputs):
        embeds = self.emb_table(inputs).view(self.batch_size, -1)
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs