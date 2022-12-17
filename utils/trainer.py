# -*- coding:utf-8 -*-
"""
@file: trainer.py
@time: 16/11/2022 14:23
@desc: 
@author: Echo
"""
import torch.nn as nn
import torch
from matplotlib import pyplot as plt


class Trainer:
    def __init__(self, n_epoch, batch_size, lr):
        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.lr = lr

    def trainer(self, model, x_train, y_train, x_dev, y_dev, loss_function, optimizer, is_clip_grad=True, max_norm=5, draw_image=True):
        losses_train = []
        losses_dev = []

        for _ in range(self.n_epoch):
            loss_train = []
            for i in range(0, x_train.shape[0], self.batch_size):
                optimizer.zero_grad()
                batch = x_train[i:i + self.batch_size]
                labels = y_train[i:i + self.batch_size]
                logits = model(batch)
                loss = loss_function(logits, labels.reshape([len(labels), 1]))
                loss_train.append(loss.item())
                loss.backward()
                if is_clip_grad:
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                optimizer.step()
            mean_train_loss = sum(loss_train) / len(loss_train)
            losses_train.append(mean_train_loss)

            """evaluation"""
            loss_dev = []
            with torch.no_grad():
                for i in range(0, x_dev.shape[0], self.batch_size):
                    dev_batch = x_dev[i:i + self.batch_size]
                    dev_labels = y_dev[i:i + self.batch_size]
                    dev_loss = loss_function(model(dev_batch), dev_labels)
                    loss_dev.append(dev_loss.item())
            mean_dev_loss = sum(loss_dev) / len(loss_dev)
            losses_dev.append(mean_dev_loss)
            print("training loss: {} evaluation loss: {}".format(mean_train_loss, mean_dev_loss))

        if draw_image:
            self.draw_image(losses_train, losses_dev)
        return losses_train, losses_dev, model

    def trainer_cross_validation(self):
        pass

    def evaluator(self, model, loss_function, x_dev, y_dev):
        loss_dev = []
        for _ in range(self.n_epoch):
            loss_dev.append(loss_function(model(x_dev), y_dev).item())
        return loss_dev

    def draw_image(self, training_loss, dev_loss):
        plt.plot(training_loss, label="training loss")
        plt.plot(dev_loss, label="evaluation loss")
        plt.legend()
        plt.show()
