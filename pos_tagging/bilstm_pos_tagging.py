# -*- coding:utf-8 -*-
"""
@file: bilstm_pos_tagging.py
@time: 28/12/2022 11:30
@desc: 
@author: Echo
"""
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from data_processing.data_loader import read_embedding_pos_en_fr


class WordEmbedding(nn.Module):
    def __init__(self, embedding_path, max_len):
        super(WordEmbedding, self).__init__()
        origin_embedding, self.word_to_id, self.id_to_word = read_embedding_pos_en_fr(embedding_path)
        self.max_len = max_len
        self.embedding_dim = len(origin_embedding[0])
        self.embedding_mean = origin_embedding.mean(dim=0).reshape(1, self.embedding_dim)
        self.embedding = torch.cat((origin_embedding, self.embedding_mean), dim=0)

    def forward(self, inputs):
        # Get word embeddings
        embeddings = torch.zeros([len(inputs), self.max_len, self.embedding_dim])
        for i, sentence in enumerate(inputs):
            sentence = sentence.split()[:self.max_len]
            sentence_embeddings = torch.zeros([self.max_len, self.embedding_dim])
            for j, word in enumerate(sentence):
                try:
                    index = self.word_to_id[word]
                    sentence_embeddings[j] = self.embedding[index]
                except:
                    sentence_embeddings[j] = self.embedding_mean
            embeddings[i] = sentence_embeddings
        return embeddings


class POSTagging(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, tagset_size, max_len, batch_size, device):
        super(POSTagging, self).__init__()

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)
        self.device = device
        self.mlp = nn.Linear(hidden_dim * 2, tagset_size)
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size

    def forward(self, inputs, lengths):
        # Get context-sensitive representations from LSTM
        packed_inputs = pack_padded_sequence(inputs, lengths, batch_first=True)
        packed_output, (hidden, cell) = self.lstm(packed_inputs)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        # Use MLP to predict POS tags
        tags = self.mlp(output)
        res = F.log_softmax(tags, dim=1)
        return res

    def predict(self, inputs, lengths):
        label = self(inputs, lengths)
        # find max for each input
        _, max_indices = torch.max(label, dim=2)
        return max_indices

    def evaluate(self, inputs, lengths, actual_tags):
        predicted_tags = self.predict(inputs, lengths)
        acc_list = []
        for i, actual_tag in enumerate(actual_tags):
            non_zeros = actual_tag[actual_tag != 0]
            len_non_zeros = len(non_zeros)
            accuracy = (predicted_tags[i][:len_non_zeros] == actual_tag[:len_non_zeros]).float().mean()
            acc_list.append(accuracy.item())
        return sum(acc_list) / len(acc_list)


