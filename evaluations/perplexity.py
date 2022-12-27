# -*- coding:utf-8 -*-
"""
@file: perplexity.py
@time: 26/12/2022 11:49
@desc: 
@author: Echo
"""
import torch


class Perplexity:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.log_probs = torch.tensor([]).to(self.device)

    def reset(self):
        self.log_probs = torch.tensor([]).to(self.device)

    def add_sentence(self, log_probs):
        # log_probs: vector of log probabilities of words in a sentence
        self.log_probs = torch.cat([self.log_probs, log_probs])

    def compute_perplexity(self):
        return 2**((-1/self.log_probs.size(0))*self.log_probs.sum())