# -*- coding:utf-8 -*-
"""
@file: test_sentences_classification_preprocessing.py
@time: 23/11/2022 10:51
@desc: 
@author: Echo
"""
import pytest

from data_processing.sentences_classification_preprocessing import SentencesClassificationPreprocessing


class TestSentencesClassificationPreprocessing(object):

    def test_convert_words_to_tensor(self):
        data_list = [["I", "like", "pizzas"], ["I", "like", "dogs", "very", "much"]]
        words_dict = {"I": 0, "like": 1, "pizzas": 2, "dogs": 3, "very": 4, "much":5, "cats":6}
        result = SentencesClassificationPreprocessing().convert_words_to_tensor(data_list, words_dict)
        print(result)


if __name__ == "__main__":
    pytest.main()
