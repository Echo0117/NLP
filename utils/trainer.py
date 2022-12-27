# -*- coding:utf-8 -*-
"""
@file: trainer.py
@time: 16/11/2022 14:23
@desc: 
@author: Echo
"""
from matplotlib import pyplot as plt


class Trainer:
    def __init__(self, n_epoch, batch_size, lr):
        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.lr = lr

    def trainer(self):
        pass

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
