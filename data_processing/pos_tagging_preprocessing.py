# -*- coding:utf-8 -*-
"""
@file: pos_tagging_preprocessing.py
@time: 28/12/2022 11:07
@desc: 
@author: Echo
"""
import torch
from config import config
from data_processing.preprocessing import Preprocessing
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence


class POSTaagingPreprocessing(Preprocessing):
    def __init__(self):
        self.max_len = config["modelTraining"]["maxLen"]

    def pos_tag_to_idx(self, data, pos_tag_dict, max_len):
        tags_tensor = pad_sequence([torch.tensor([pos_tag_dict[word] for word in sentence]) for sentence in data],
                                   batch_first=True)
        if tags_tensor.shape[1] > self.max_len:
            return tags_tensor[:, :self.max_len]
        else:
            return torch.nn.functional.pad(tags_tensor, (0, max_len - tags_tensor.shape[1], 0, 0))

    def get_pos_idx(self, pos):
        pos_to_id = {}
        id_to_pos = {}
        for sentence in pos:
            for p in sentence:
                if p not in pos_to_id:
                    index = len(pos_to_id)
                    pos_to_id[p] = index
                    id_to_pos[index] = p
        return pos_to_id, id_to_pos

    def encoder(self, txt, pos, word_to_id, pos_to_id):
        def word_encoder(word):
            if word not in word_to_id and word[-1] == 's' and word[:-1] in word_to_id:
                word = word[:-1]
            elif word not in word_to_id:
                word = '<unk>'
            return word_to_id[word]

        sentence_encoder = lambda sentence: list(map(word_encoder, sentence))
        pos_encoder = lambda sentence: list(map(lambda p: pos_to_id[p], sentence))

        txt_encoded = list(map(sentence_encoder, txt))
        pos_encoded = list(map(pos_encoder, pos))

        return txt_encoded, pos_encoded