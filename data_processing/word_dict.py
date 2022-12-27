# -*- coding:utf-8 -*-
"""
@file: word_dict.py
@time: 27/12/2022 10:18
@desc: 
@author: Echo
"""


class WordDict:
    def __init__(self):
        pass

    # return the integer associated with a word
    def word_to_id(self, word):
        pass

    # return the word associated with an integer
    def id_to_word(self, idx):
        pass

    # number of word in the dictionnary
    def __len__(self):
        pass


class WordDictPTB(WordDict):
    # constructor, words must be a set containing all words
    def __init__(self, words):
        assert type(words) == set
        self.words_dict = dict(zip(words, range(len(words))))
        self.index_dict = dict(zip(self.words_dict.values(), self.words_dict.keys()))

    # return the integer associated with a word
    def word_to_id(self, word):
        assert type(word) == str
        return self.words_dict[word]

    # return the word associated with an integer
    def id_to_word(self, idx):
        assert type(idx) == int
        return self.index_dict[idx]

    # number of word in the dictionnary
    def __len__(self):
        return len(self.words_dict)
