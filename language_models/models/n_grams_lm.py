# -*- coding:utf-8 -*-
"""
@file: n_grams_lm.py
@time: 26/12/2022 12:38
@desc: 
@author: Echo
"""
from net.n_gram import nGramModel


class nGramModelLM(nGramModel):
    def __init__(self, vocab_size, embedding_dim, batch_size, n_grams, hidden_size):
        super(nGramModelLM, self).__init__(vocab_size, embedding_dim, batch_size)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.n_grams = n_grams
        self.hidden_size = hidden_size
        self.batch_size = batch_size