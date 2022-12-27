# -*- coding:utf-8 -*-
"""
@file: lstm.py
@time: 26/12/2022 11:52
@desc: 
@author: Echo
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim=10, hidden_size=10, context_size=3, num_layers=1, dropout=0.0):
        super(LSTM, self).__init__()
        # To create an embedding table: https://pytorch.org/docs/stable/nn.html#torch.nn.Embedding
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.context_size = context_size
        self.hidden_dim = hidden_dim
        self.hidden_size = hidden_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.emb_table = nn.Embedding(
            self.vocab_size+1, self.embedding_dim, padding_idx=self.vocab_size)
        self.lstm = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_size,
                            num_layers=num_layers, batch_first=True).to(self.device)  # lstm
        self.mlp = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.vocab_size),
        ).to(self.device)

    def forward(self, inputs):
        w = torch.empty(len(inputs), max(len(s) for s in inputs), dtype=torch.long).to(self.device)
        w.fill_(self.emb_table.padding_idx)
        for i,input in enumerate(inputs):
            w[i,:input.shape[0]] = input
        embs = self.emb_table(w)
        self.lstm.flatten_parameters()   # to avoid a warning
        output, _ = self.lstm(embs)
        output = self.dropout(output)
        logits = self.mlp(output)
        log_probs = F.log_softmax(logits, dim=1)
        list_log_probs = []
        for i in range(log_probs.shape[0]):
            list_log_probs.append(log_probs[i, :(inputs[i].shape[0])])
        return list_log_probs