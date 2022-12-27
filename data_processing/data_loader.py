# -*- coding:utf-8 -*-
"""
@file: data_loader.py
@time: 16/11/2022 14:45
@desc: 
@author: Echo
"""
from data_processing.preprocessing import Preprocessing


# reads the content of the file passed as an argument.
# if limit > 0, this function will return only the first "limit" sentences in the file.
def read_file_imdb(filename, limit=-1):
    dataset = []
    with open(filename) as f:
        line = f.readline()
        cpt = 1
        skip = 0
        while line:
            cleanline = Preprocessing().clean_str(f.readline()).split()
            if cleanline:
                dataset.append(cleanline)
            else:
                line = f.readline()
                skip += 1
                continue
            if limit > 0 and cpt >= limit:
                break
            line = f.readline()
            cpt += 1

        print("Load ", cpt, " lines from ", filename, " / ", skip, " lines discarded")
    return dataset


def read_file_ptb(path):
    data = list()
    with open(path) as inf:
        for line in inf:
            line = line.strip()
            if len(line) == 0:
                continue
            data.append({"text": line.split()})
    return data