# -*- coding:utf-8 -*-
"""
@file: multi_stack_bilstm_pos_tagging_example.py
@time: 28/12/2022 11:50
@desc: 
@author: Echo
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence


class Embedding(nn.Module):
    def __init__(self, emb_table: torch.Tensor):
        super(Embedding, self).__init__()
        self.emb_table = emb_table
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, input):
        emb = []
        for sentence in input:
            emb.append(self.emb_table[sentence].to(self.device))
        return emb


class MultiStackBiLSTMPOSTagging(nn.Module):
    def __init__(self, output_dim, stack_size=2, embedding_dim=300, hidden_dim=10, hidden_size=10, num_layers=1,
                 dropout=0.0):
        super(MultiStackBiLSTMPOSTagging, self).__init__()
        assert stack_size > 0
        self.output_dim = output_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.hidden_size = hidden_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lstms = nn.ModuleList(
            nn.LSTM(input_size=self.hidden_size * 2, hidden_size=self.hidden_size,
                    num_layers=num_layers, batch_first=True, bidirectional=True).to(self.device)
            for _ in range(stack_size - 1)
        )
        self.lstms.insert(0, nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_size,
                                     num_layers=num_layers, batch_first=True, bidirectional=True).to(self.device))

        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim),
        ).to(self.device)

    def forward(self, emb):
        # pad the sequence
        lengths = list(map(len, emb))
        padded_emb = pad_sequence(emb, batch_first=True)
        packed_emb = pack_padded_sequence(padded_emb, batch_first=True, lengths=lengths, enforce_sorted=False)
        # forward through LSTMs
        packed_output = packed_emb
        for lstm in self.lstms:
            packed_output, _ = lstm(packed_output)
        unpacked_output, unpacked_lengths = pad_packed_sequence(packed_output, batch_first=True)
        stacked_output = unpacked_output.view(-1, unpacked_output.size(2))
        # create the mask
        mask = torch.zeros(size=unpacked_output.shape[:2]).to(self.device)
        for i, length in enumerate(unpacked_lengths):
            mask[i, :length] = 1
        mask = mask.view(-1).to(bool)
        # forward through MLP
        output = stacked_output[mask]
        logits = self.mlp(output)
        log_probs = F.log_softmax(logits, dim=1)
        return log_probs

    def predict(self, emb, batch_size=256):
        self.eval()
        y_pred = []
        for i in range(0, len(emb), batch_size):
            emb_batch = emb[i:i + batch_size]
            log_probs = self.forward(emb_batch)
            y_pred.append(log_probs.argmax(dim=1))
        return torch.cat(y_pred)

    def compute_accuracy(self, emb, y_gold, batch_size=256):
        self.eval()
        gold_batch = torch.cat([torch.tensor(y) for y in y_gold]).to(self.device)
        y_pred = self.predict(emb, batch_size=batch_size)
        acc = (gold_batch == y_pred).sum() / y_pred.size(0)
        return acc.item()


# tagger = Tagger(output_dim=len(en_pos_to_id)).to(device)
# print(tagger.compute_accuracy(en_emb_table(en_x_train[:10]), en_y_train[:10]))
# tagger.predict(emb_example)