# -*- coding:utf-8 -*-
"""
@file: lstm_lm_example.py
@time: 27/12/2022 10:37
@desc: 
@author: Echo
"""
import os
import torch
from config import config
from data_processing.data_loader import read_file_ptb
from data_processing.language_modeling_preprocessing import LanguageModellingPreprocessing
from data_processing.word_dict import WordDictPTB
from language_models.models.lstm_lm import LSTMLM
from utils.losses import losses
from utils.optimizer import Optimizer
from utils.trainer_lm import TrainerLSTMLM

root_path = os.path.dirname(os.path.realpath(__file__))
project_path = os.path.dirname(os.path.dirname(root_path))

if __name__ == '__main__':
    """set up parameters"""
    embedding_dim = config["modelTraining"]["embeddingDim"]
    batch_size = config["modelTraining"]["batchSize"]
    n_epoch = config["modelTraining"]["epoch"]
    lr = config["modelTraining"]["lr"]
    context_size = config["modelTraining"]["contextSize"]
    loss_function = losses().NLLLoss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    """data processing"""
    train_data = read_file_ptb(project_path + config["dataset"]["ptbPath"]["ptbTrain"])
    dev_data = read_file_ptb(project_path + config["dataset"]["ptbPath"]["ptbValid"])
    test_data = read_file_ptb(project_path + config["dataset"]["ptbPath"]["ptbTrain"])

    train_words = set()
    for sentence in train_data:
        train_words.update(sentence["text"])
    train_words.update(["<bos>", "<eos>"])
    word_dict = WordDictPTB(train_words)
    vocab_size = len(word_dict)

    preprocessing = LanguageModellingPreprocessing()
    sentence_to_id = preprocessing.sentence_to_id(word_dict)
    process_data = lambda data: list(map(sentence_to_id, [sentence['text'] for sentence in data]))
    train_data_idx = process_data(train_data)
    dev_data_idx = process_data(dev_data)
    test_data_idx = process_data(test_data)

    x_train = preprocessing.to_list_tensors(train_data_idx)
    x_dev = preprocessing.to_list_tensors(dev_data_idx)
    x_test = preprocessing.to_list_tensors(test_data_idx)

    """initialize the model"""
    lstm_model = LSTMLM(vocab_size, embedding_dim, 2, batch_size, device)
    optimizer = Optimizer().adam_optimizer(lstm_model, lr)

    """training"""
    _, _, model = TrainerLSTMLM(n_epoch, batch_size, lr).trainer(lstm_model, x_train, x_dev, loss_function, optimizer,
                                                   is_clip_grad=True, max_norm=5, draw_image=True)
