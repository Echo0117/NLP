# -*- coding:utf-8 -*-
"""
@file: cbow_cnn_example.py
@time: 16/11/2022 14:43
@desc: 
@author: Echo
"""
from data_processing.data_loader import read_file_imdb
from config import config
from data_processing.sentences_classification_preprocessing import SentencesClassificationPreprocessing
from sentence_classification.models.cbow_cnn import CbowCNNClassifier
from utils.losses import losses
from utils.optimizer import Optimizer
from utils.save_load_models import save_models_pt
import os

from utils.trainer_cbow_cnn import TrainerCBOWCNN

root_path = os.path.dirname(os.path.realpath(__file__))
project_path = os.path.dirname(os.path.dirname(root_path))

if __name__ == '__main__':

    """set up parameters"""
    LIM = 5000
    embedding_dim = config["modelTraining"]["embeddingDim"]
    batch_size = config["modelTraining"]["batchSize"]
    n_epoch = config["modelTraining"]["epoch"]
    max_len = config["modelTraining"]["maxLen"]
    lr = config["modelTraining"]["lr"]
    loss_function = losses().BCEWithLogitsLoss()

    """data processing"""
    postxt = read_file_imdb(project_path + config["dataset"]["imdbPath"]["posSentencesPath"], limit=LIM)
    negtxt = read_file_imdb(project_path + config["dataset"]["imdbPath"]["negSentencesPath"], limit=LIM)

    preprocessing = SentencesClassificationPreprocessing()
    txt_train, label_train, txt_dev, label_dev, txt_test, label_test = preprocessing.train_dev_test_dataset_split(postxt, negtxt, 0.2, 0.2)

    train_label = preprocessing.conver_labels_to_tensor(label_train)
    dev_label = preprocessing.conver_labels_to_tensor(label_dev)
    test_label = preprocessing.conver_labels_to_tensor(label_test)

    words_dict = preprocessing.get_unique_words_indices(txt_train)
    train_set = preprocessing.convert_words_to_tensor(txt_train, words_dict)
    dev_set = preprocessing.convert_words_to_tensor(txt_dev, words_dict)
    test_set = preprocessing.convert_words_to_tensor(txt_test, words_dict)

    """initialize the model"""
    model = CbowCNNClassifier(len(words_dict), embedding_dim)
    optimizer = Optimizer().adam_optimizer(model, lr)

    """training"""
    _, _, model = TrainerCBOWCNN(n_epoch, batch_size, lr).trainer(model, train_set, train_label, test_set, test_label, loss_function, optimizer)

    save_models_pt(model, project_path + config["modelsPath"]["sentenceClassification"] + "sentence_classification.pt")