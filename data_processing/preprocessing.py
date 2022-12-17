# -*- coding:utf-8 -*-
"""
@file: preprocessing.py
@time: 16/11/2022 14:44
@desc: 
@author: Echo
"""
import re
from config import config


class Preprocessing:
    def __init__(self):
        self.max_len = config["modelTraining"]["maxLen"]


    def clean_str(self, string, tolower=True):
        """
        Tokenization/string cleaning.
        Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
        """
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        if tolower:
            string = string.lower()
        return string.strip()

