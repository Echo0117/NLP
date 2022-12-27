# -*- coding:utf-8 -*-
"""
@file: n_gram_lm_example.py
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
from language_models.models.n_grams_lm import nGramModelLM
from utils.losses import losses
from utils.optimizer import Optimizer
from utils.save_load_models import save_models_pt
from utils.trainer_lm import TrainerNGramLM

root_path = os.path.dirname(os.path.realpath(__file__))
project_path = os.path.dirname(os.path.dirname(root_path))

if __name__ == '__main__':

    """set up parameters"""
    embedding_dim = config["modelTraining"]["embeddingDim"]
    batch_size = config["modelTraining"]["batchSize"]
    n_epoch = config["modelTraining"]["epoch"]
    lr = config["modelTraining"]["lr"]
    context_size = config["modelTraining"]["contextSize"]
    hidden_size = config["modelTraining"]["hiddenSize"]
    n_grams = config["modelTraining"]["nGrams"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loss_function = losses().NLLLoss()

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
    x_train = list(map(sentence_to_id, [sentence['text'] for sentence in train_data]))

    processed_train_data = preprocessing.ngram_processing(x_train, n_grams, word_dict)

    words = [val["input"] for val in processed_train_data]
    next_word = [val["output"] for val in processed_train_data]

    x_train, y_train, x_dev, y_dev, x_test, y_test = preprocessing.train_dev_test_dataset_split(words, next_word, 0.1,
                                                                                                0.1)
    x_train, y_train, x_dev, y_dev, x_test, y_test = tuple(
        map(lambda x: torch.LongTensor(x), [x_train, y_train, x_dev, y_dev, x_test, y_test]))

    x_train = x_train.to(device)
    y_train = y_train.to(device)
    x_dev = x_dev.to(device)
    y_dev = y_dev.to(device)

    """initialize the model"""
    n_gram_model = nGramModelLM(vocab_size, embedding_dim, batch_size, context_size, hidden_size).to(device)
    optimizer = Optimizer().adam_optimizer(n_gram_model, lr)

    """training"""
    _, _, model = TrainerNGramLM(n_epoch, batch_size, lr).trainer(n_gram_model, x_train, y_train, x_dev, y_dev, loss_function,
                                                    optimizer, is_clip_grad=True, max_norm=5, draw_image=True)

    save_models_pt(model, project_path + config["modelsPath"]["languageModels"] + "language_model_ngram.pt")