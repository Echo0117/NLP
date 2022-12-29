# -*- coding:utf-8 -*-
"""
@file: multi_stack_bilstm_pos_tagging_example.py
@time: 28/12/2022 11:53
@desc: 
@author: Echo
"""
import os
import torch
from config import config
from data_processing.data_loader import read_embedding_pos_en_fr_multistack, read_file_pos_en_fr_multistack
from data_processing.pos_tagging_preprocessing import POSTaagingPreprocessing
from pos_tagging.multi_stack_bilstm_pos_tagging import MultiStackBiLSTMPOSTagging, Embedding
from utils.losses import losses
from utils.optimizer import Optimizer
from utils.save_load_models import save_models_pt
from utils.trainer_multi_stack_bilstm_pos_tagging import TrainerMultiStackBilstmPOSTagging

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

    embedding_en, en_word_to_id, id_to_word = read_embedding_pos_en_fr_multistack(
        project_path + config["dataset"]["posEnFrPath"]["posEmbeddingEn"])
    x_train_words, y_train_pos = read_file_pos_en_fr_multistack(project_path + config["dataset"]["posEnFrPath"]["posEnTrain"])
    x_dev_words, y_dev_pos = read_file_pos_en_fr_multistack(project_path + config["dataset"]["posEnFrPath"]["posEnValid"])
    x_test_words, y_test_pos = read_file_pos_en_fr_multistack(project_path + config["dataset"]["posEnFrPath"]["posEnTest"])

    embedding_fr, fr_word_to_id, fr_id_to_word = read_embedding_pos_en_fr_multistack(
        project_path + config["dataset"]["posEnFrPath"]["posEmbeddingFr"])
    x_test_words_fr, y_test_pos_fr = read_file_pos_en_fr_multistack(project_path + config["dataset"]["posEnFrPath"]["posFrTest"])
    x_test_words_en, y_test_pos_en = read_file_pos_en_fr_multistack(
        project_path + config["dataset"]["posEnFrPath"]["posEnPudTest"])

    preprocessing = POSTaagingPreprocessing()
    en_pos_to_id, en_id_to_pos = preprocessing.get_pos_idx(y_train_pos)
    en_word_to_id['<unk>'] = len(en_word_to_id) - 1
    id_to_word[len(en_word_to_id)] = '<unk>'
    fr_word_to_id['<unk>'] = len(fr_word_to_id) - 1
    fr_id_to_word[len(fr_word_to_id)] = '<unk>'

    en_x_train, en_y_train = preprocessing.encoder(x_train_words, y_train_pos, en_word_to_id, en_pos_to_id)
    en_x_dev, en_y_dev = preprocessing.encoder(x_dev_words, y_dev_pos, en_word_to_id, en_pos_to_id)
    en_x_test, en_y_test = preprocessing.encoder(x_test_words, y_test_pos, en_word_to_id, en_pos_to_id)

    # fr_x_test, fr_y_test = preprocessing.encoder(x_test_words_fr, y_test_pos_fr, fr_word_to_id, fr_id_to_word)
    pud_x_test, pud_y_test = preprocessing.encoder(x_test_words_en, y_test_pos_en, en_word_to_id, en_pos_to_id)

    pos_model = MultiStackBiLSTMPOSTagging(output_dim=len(en_pos_to_id)).to(device)
    en_emb_table = Embedding(embedding_en)
    optimizer = Optimizer().adam_optimizer(pos_model, lr)
    model, _, _, _, _ = TrainerMultiStackBilstmPOSTagging(n_epoch, batch_size, lr).trainer(pos_model, en_x_train,
                                                                                           en_y_train,
                                                                                           en_x_dev, en_y_dev,
                                                                                           loss_function,
                                                                                           optimizer, en_emb_table)

    save_models_pt(model, project_path + config["modelsPath"]["posTagging"] + "pos_tagging_multistack_bilstm.pt")
