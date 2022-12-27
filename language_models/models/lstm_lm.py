# -*- coding:utf-8 -*-
"""
@file: lstm_lm.py
@time: 26/12/2022 12:39
@desc: 
@author: Echo
"""
import torch
import torch.nn.functional as F
from net.lstm import LSTM
from utils.dropout import VariationalDropout


class LSTMLM(LSTM):
    def __init__(self, vocab_size, embedding_dim, hidden_dim=10, hidden_size=10, context_size=3, num_layers=1,
                 dropout=0.0):
        super(LSTMLM, self).__init__(vocab_size, embedding_dim)
        self.dropout = VariationalDropout(dropout)

    def forward(self, inputs):
        w = torch.empty(len(inputs), max(len(s) for s in inputs), dtype=torch.long).to(self.device)
        w.fill_(self.emb_table.padding_idx)
        for i, input in enumerate(inputs):
            w[i, :input.shape[0]] = input
        embs = self.emb_table(w)
        embs = self.dropout(embs)
        self.lstm.flatten_parameters()  # to avoid a warning
        output, _ = self.lstm(embs)
        output = self.dropout(output)
        logits = self.mlp(output)
        log_probs = F.log_softmax(logits, dim=1)
        list_log_probs = []
        for i in range(log_probs.shape[0]):
            list_log_probs.append(log_probs[i, :(inputs[i].shape[0])])
        return list_log_probs
