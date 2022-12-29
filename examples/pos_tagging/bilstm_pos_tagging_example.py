# -*- coding:utf-8 -*-
"""
@file: bilstm_pos_tagging_example.py
@time: 28/12/2022 10:45
@desc: 
@author: Echo
"""
import os
import torch
from config import config
from data_processing.data_loader import read_embedding_pos_en_fr, read_file_pos_en_fr
from data_processing.pos_tagging_preprocessing import POSTaagingPreprocessing
from pos_tagging.bilstm_pos_tagging import WordEmbedding, POSTagging
from utils.losses import losses
from utils.optimizer import Optimizer
from utils.save_load_models import save_models_pt
from utils.trainer_bilstm_pos_tagging import TrainerPOSTagging

root_path = os.path.dirname(os.path.realpath(__file__))
project_path = os.path.dirname(os.path.dirname(root_path))

if __name__ == '__main__':
    """set up parameters"""
    embedding_dim = 300
    batch_size = config["modelTraining"]["batchSize"]
    n_epoch = config["modelTraining"]["epoch"]
    lr = config["modelTraining"]["lr"]
    context_size = config["modelTraining"]["contextSize"]
    hidden_size = config["modelTraining"]["hiddenSize"]
    hidden_dim = config["modelTraining"]["hiddenDim"]
    n_grams = config["modelTraining"]["nGrams"]
    max_len = config["modelTraining"]["maxLen"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loss_function = losses().NLLLoss()

    embedding_en, word_to_id, id_to_word = read_embedding_pos_en_fr(project_path + config["dataset"]["posEnFrPath"]["posEmbeddingEn"])
    x_train_words, y_train_pos = read_file_pos_en_fr(project_path + config["dataset"]["posEnFrPath"]["posEnTrain"])
    x_dev_words, y_dev_pos = read_file_pos_en_fr(project_path + config["dataset"]["posEnFrPath"]["posEnValid"])
    x_test_words, y_test_pos = read_file_pos_en_fr(project_path + config["dataset"]["posEnFrPath"]["posEnTest"])

    pos_tag_list = set([val[0] for val in y_train_pos])
    pos_tag_dict = {val: i + 1 for i, val in enumerate(pos_tag_list)}
    preprocessing = POSTaagingPreprocessing()
    y_train = preprocessing.pos_tag_to_idx(y_train_pos, pos_tag_dict, max_len)
    y_dev = preprocessing.pos_tag_to_idx(y_dev_pos, pos_tag_dict, max_len)
    y_test = preprocessing.pos_tag_to_idx(y_test_pos, pos_tag_dict, max_len)

    vocab_size = len(word_to_id)
    word_embedding_en = WordEmbedding(project_path + config["dataset"]["posEnFrPath"]["posEmbeddingEn"], max_len)
    x_train = word_embedding_en(x_train_words)
    x_dev = word_embedding_en(x_dev_words)
    x_test = word_embedding_en(x_test_words)

    tagset_size = len(pos_tag_list) + 1
    pos_model = POSTagging(embedding_dim, hidden_dim, tagset_size, max_len, batch_size, device)
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    x_dev = x_dev.to(device)
    y_dev = y_dev.to(device)

    optimizer = Optimizer().adam_optimizer(pos_model, lr)
    _, _, model = TrainerPOSTagging(n_epoch, batch_size, lr).trainer(pos_model, x_train, y_train, x_dev, y_dev, loss_function,
                                                           optimizer, is_clip_grad=True, max_norm=5, draw_image=True)

    save_models_pt(model, project_path + config["modelsPath"]["posTagging"] + "pos_tagging_bilstm.pt")
