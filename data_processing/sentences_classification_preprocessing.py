# -*- coding:utf-8 -*-
"""
@file: sentences_classification_preprocessing.py
@time: 21/11/2022 19:07
@desc: 
@author: Echo
"""
from sklearn.model_selection import train_test_split

from config import config
from data_processing.preprocessing import Preprocessing
import numpy as np
import torch


class SentencesClassificationPreprocessing(Preprocessing):
    def __init__(self):
        self.max_len = config["modelTraining"]["maxLen"]

    def train_dev_test_dataset_split(self, postxt, negtxt, dev_ratio: float, test_ratio: float, shuffle=True) -> (
            list, list, list, list, list, list):
        x_train_dev_pos, x_test_pos, y_train_dev_pos, y_test_pos = train_test_split(postxt, ['pos'] * len(postxt),
                                                                                    test_size=test_ratio,
                                                                                    random_state=42)

        """As we need to devide the train/dev dataset based on original dev_ratio, 
        we need to divide them with the correct ratio"""
        x_train_pos, x_dev_pos, y_train_pos, y_dev_pos = train_test_split(x_train_dev_pos,
                                                                          ['pos'] * len(y_train_dev_pos),
                                                                          test_size=dev_ratio / (1 - test_ratio),
                                                                          random_state=42)

        x_train_dev_neg, x_test_neg, y_train_dev_neg, y_test_neg = train_test_split(negtxt, ['neg'] * len(negtxt),
                                                                                    test_size=test_ratio,
                                                                                    random_state=42)

        x_train_neg, x_dev_neg, y_train_neg, y_dev_neg = train_test_split(x_train_dev_neg,
                                                                          ['neg'] * len(y_train_dev_neg),
                                                                          test_size=dev_ratio / (1 - test_ratio),
                                                                          random_state=42)
        if shuffle:
            x_train, y_train = zip(
                *sorted(zip(np.array(x_train_pos + x_train_neg), np.array(y_train_pos + y_train_neg))))
            x_dev, y_dev = zip(*sorted(zip(np.array(x_dev_pos + x_dev_neg), np.array(y_dev_pos + y_dev_neg))))
            x_test, y_test = zip(*sorted(zip(np.array(x_test_pos + x_test_neg), np.array(y_test_pos + y_test_neg))))

        return x_train, y_train, x_dev, y_dev, x_test, y_test

    def shuffle_two_arrays(self, a, b):
        assert len(a) == len(b)
        p = np.random.permutation(len(a))  # shuffle all the indices
        return np.array(a, dtype=object)[p].tolist(), np.array(b, dtype=object)[p].tolist()

    def get_unique_words_indices(self, data_list):
        flatten_list = [item for sublist in data_list for item in sublist]
        unique_words = list(set(flatten_list))
        words_dict = dict(zip(unique_words, range(len(unique_words))))
        words_dict["unknownToken"] = len(unique_words)
        return words_dict

    def convert_words_to_tensor(self, data_list, words_dict):
        words_tensor = torch.zeros([len(data_list), self.max_len], dtype=torch.int64)
        for i, data in enumerate(data_list):
            for j, word in enumerate(data):
                try:
                    words_tensor[i][j] = words_dict[word]
                except:
                    words_tensor[i][j] = words_dict["unknownToken"]
        return torch.LongTensor(words_tensor)

    def conver_labels_to_tensor(self, labels):
        numerical_labels = np.array(list(map(lambda x: 1 if x == "pos" else 0, labels))).reshape([len(labels), 1])
        torsor_labels = torch.tensor(numerical_labels, dtype=torch.float32, requires_grad=False)
        return torsor_labels
