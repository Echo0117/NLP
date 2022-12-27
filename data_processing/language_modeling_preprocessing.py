# -*- coding:utf-8 -*-
"""
@file: language_modeling_preprocessing.py
@time: 26/12/2022 11:54
@desc: 
@author: Echo
"""
from sklearn.model_selection import train_test_split
import torch
from config import config
from data_processing.preprocessing import Preprocessing


class LanguageModellingPreprocessing(Preprocessing):
    def __init__(self):
        self.max_len = config["modelTraining"]["maxLen"]

    def ngram_processing(self, data, n_grams, word_dict):
        output = []
        bos_id = word_dict.word_to_id("<bos>")
        for sentence in data:
            sentence = [bos_id, bos_id] + sentence
            sentence_dict = {}
            for i, val in enumerate(sentence[:len(sentence) - n_grams - 1]):
                sentence_dict["input"] = sentence[i: i + n_grams]
                sentence_dict["output"] = sentence[i + n_grams + 1]
                output.append(sentence_dict)
                sentence_dict = {}
        return output

    def to_list_tensors(self, data):
        return list(map(lambda x: torch.LongTensor(x), data))

    def sentence_to_id(self, word_dict):
        return lambda sentence: list(map(word_dict.word_to_id, sentence))

    def train_dev_test_dataset_split(self, x_data, y_data, dev_ratio: float, test_ratio: float, shuffle=True):
        x_train_dev, x_test, y_train_dev, y_test = train_test_split(x_data, y_data, test_size=test_ratio,
                                                                    random_state=42)

        """As we need to devide the train/dev dataset based on original dev_ratio, 
        we need to divide them with the correct ratio"""
        x_train, x_dev, y_train, y_dev = train_test_split(x_train_dev, y_train_dev,
                                                          test_size=dev_ratio / (1 - test_ratio),
                                                          random_state=42)
        if shuffle:
            x_train, y_train = zip(*sorted(zip(x_train, y_train)))
            x_dev, y_dev = zip(*sorted(zip(x_dev, y_dev)))
            x_test, y_test = zip(*sorted(zip(x_test, y_test)))

        return x_train, y_train, x_dev, y_dev, x_test, y_test
