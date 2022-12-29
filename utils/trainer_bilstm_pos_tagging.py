# -*- coding:utf-8 -*-
"""
@file: trainer_bilstm_pos_tagging.py
@time: 28/12/2022 11:43
@desc: 
@author: Echo
"""
import torch.nn as nn
import torch
from utils.trainer import Trainer
from matplotlib import pyplot as plt


class TrainerPOSTagging(Trainer):
    def __init__(self, n_epoch, batch_size, lr):
        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.lr = lr

    def trainer(self, model, x_train, y_train, x_dev, y_dev, loss_function, optimizer, tagset_size=19,
                is_clip_grad=True, max_norm=5, draw_image=True):
        losses_train = []
        losses_dev = []
        accuracies_train = []
        accuracies_dev = []
        for _ in range(self.n_epoch):
            loss_train = []
            acc_train = []
            for i in range(0, len(x_train), self.batch_size):
                optimizer.zero_grad()
                batch = x_train[i:i + self.batch_size]
                labels = y_train[i:i + self.batch_size]

                lengths = [len(sentence) for sentence in batch]
                logits = model(batch, lengths)
                labels = torch.tensor(labels, dtype=torch.long)
                loss = loss_function(logits.view(-1, tagset_size), labels.view(-1))
                loss_train.append(loss.item())
                loss.backward()
                if is_clip_grad:
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                optimizer.step()

            mean_train_loss = sum(loss_train) / len(loss_train)
            losses_train.append(mean_train_loss)

            train_acc = model.evaluate(batch, lengths, labels)
            accuracies_train.append(train_acc)

            """evaluation"""
            loss_dev = []
            with torch.no_grad():
                for i in range(0, len(x_dev), self.batch_size):
                    try:
                        dev_batch = x_dev[i:i + self.batch_size]
                        dev_labels = y_dev[i:i + self.batch_size]
                        dev_lengths = [len(sentence) for sentence in dev_batch]
                        dev_loss = loss_function(model(dev_batch, dev_lengths).view(-1, tagset_size),
                                                 dev_labels.view(-1))
                        loss_dev.append(dev_loss.item())
                    except Exception as e:
                        print(e)
                        continue

            dev_acc = model.evaluate(dev_batch, dev_lengths, dev_labels)
            accuracies_dev.append(dev_acc)

            mean_dev_loss = sum(loss_dev) / len(loss_dev)
            losses_dev.append(mean_dev_loss)

            print("training loss: {} evaluation loss: {} training acc: {} dev acc: {}".format(mean_train_loss,
                                                                                              mean_dev_loss, train_acc,
                                                                                              dev_acc))

        if draw_image:
            self.draw_image(losses_train, losses_dev, accuracies_train, accuracies_dev)
        return losses_train, losses_dev, model

    def draw_image(self, training_loss, dev_loss, accuracies_train, accuracies_dev):
        plt.plot(training_loss, label="training loss")
        plt.plot(dev_loss, label="evaluation loss")
        plt.legend()
        plt.show()

        plt.plot(accuracies_train, label="training acc")
        plt.plot(accuracies_dev, label="evaluation acc")
        plt.legend()
        plt.show()