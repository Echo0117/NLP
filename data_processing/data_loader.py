# -*- coding:utf-8 -*-
"""
@file: data_loader.py
@time: 16/11/2022 14:45
@desc: 
@author: Echo
"""
from data_processing.preprocessing import Preprocessing
import torch


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


def read_file_pos_en_fr(path):
    sentences, upos_list, sentence, upos = [], [], [], []
    with open(path) as inf:
        for line in inf:
            line = line.strip()
            if line == "":
                sentences.append(" ".join(sentence))
                upos_list.append(upos)
                sentence, upos = [], []
            elif line[0].isdigit():
                line_list = line.split()
                sentence.append(line_list[1].lower())
                upos.append(line_list[3].lower())
    return sentences, upos_list


def read_embedding_pos_en_fr(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        embedding_dim = len(lines[0].split()) - 1
        embedding = torch.empty(size=(len(lines), embedding_dim))
        word_to_id = {}
        id_to_word = {}
        for i, line in enumerate(lines):
            splitted_lines = line.split()
            word = splitted_lines[0]
            word_to_id[word] = i
            id_to_word[i] = word
            embedding[i] = torch.tensor(list(map(float, splitted_lines[-embedding_dim:])))  # mistake line 13334

        return embedding, word_to_id, id_to_word


def read_file_pos_en_fr_multistack(in_file, lowercase=True, max_example=None):
    list_txt = []
    list_pos = []
    with open(in_file) as f:
        word, pos = [], []
        for line in f.readlines():
            sp = line.strip().split('\t')
            if len(sp) == 10:
                if sp[0].isdigit():
                    word.append(sp[1].lower() if lowercase else sp[1])
                    pos.append(sp[3])
            elif len(word) > 0:
                list_txt.append(word)
                list_pos.append(pos)
                word, pos = [], []
                if (max_example is not None) and (len(list_txt) == max_example):
                    break
        if len(word) > 0:
            list_txt.append(word)
            list_pos.append(pos)
    return list_txt, list_pos


def read_embedding_pos_en_fr_multistack(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        emb = torch.empty(size=(len(lines) + 1, 300))
        word_to_id = {}
        id_to_word = {}
        i = len(lines)
        word_to_id['<unk>'] = i
        id_to_word[i] = '<unk>'
        emb[i].zero_()
        for i, line in enumerate(lines):
            splitted_lines = line.split()
            word = splitted_lines[0]
            word_to_id[word] = i
            id_to_word[i] = word
            emb[i] = torch.tensor(list(map(float, splitted_lines[-300:])), requires_grad=False)  # mistake line 13334
        return emb, word_to_id, id_to_word
