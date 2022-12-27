# -*- coding:utf-8 -*-
"""
@file: trainer_lm.py
@time: 27/12/2022 11:22
@desc: 
@author: Echo
"""
import torch.nn as nn
import torch
from utils.trainer import Trainer


class TrainerNGramLM(Trainer):
    def __init__(self, n_epoch, batch_size, lr):
        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.lr = lr

    def trainer(self, model, x_train, y_train, x_dev, y_dev, loss_function, optimizer, is_clip_grad=True, max_norm=5,
                draw_image=True):
        losses_train = []
        losses_dev = []
        for _ in range(self.n_epoch):
            loss_train = []
            for i in range(0, len(x_train), self.batch_size):
                optimizer.zero_grad()
                batch = x_train[i:i + self.batch_size]
                labels = y_train[i:i + self.batch_size]
                try:
                    logits = model(batch)
                    labels = torch.tensor(labels, dtype=torch.long)

                    loss = loss_function(logits, labels)
                    loss_train.append(loss.item())
                    loss.backward()
                    if is_clip_grad:
                        nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                    optimizer.step()
                except:
                    print("batch", batch.shape)
                    continue
            mean_train_loss = sum(loss_train) / len(loss_train)
            losses_train.append(mean_train_loss)

            """evaluation"""
            loss_dev = []
            with torch.no_grad():
                for i in range(0, x_dev.shape[0], self.batch_size):
                    try:
                        dev_batch = x_dev[i:i + self.batch_size]
                        dev_labels = y_dev[i:i + self.batch_size]
                        dev_loss = loss_function(model(dev_batch), dev_labels)
                        loss_dev.append(dev_loss.item())
                    except:
                        print("batch", batch.shape)
                        continue

            mean_dev_loss = sum(loss_dev) / len(loss_dev)
            losses_dev.append(mean_dev_loss)
            print("training loss: {} evaluation loss: {}".format(mean_train_loss, mean_dev_loss))

        if draw_image:
            self.draw_image(losses_train, losses_dev)
        return losses_train, losses_dev, model


class TrainerLSTMLM(Trainer):
    def __init__(self, n_epoch, batch_size, lr):
        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.lr = lr

    def trainer(self, model, x_train, x_dev, loss_function, optimizer, is_clip_grad=True, max_norm=5,
                draw_image=True):
        losses_train = []
        losses_dev = []
        for _ in range(self.n_epoch):
            print("training")
            model.train()
            for i in range(0, len(x_train), self.batch_size):
                optimizer.zero_grad()
                batch = x_train[i:i+self.batch_size]
                list_log_probs = model(batch)
                log_probs = torch.cat(list_log_probs)
                gold = torch.cat(batch)
                loss = loss_function(log_probs, gold)
                losses_train.append(loss.item())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
            mean_train_loss = sum(losses_train)/len(losses_train)
            losses_train.append(mean_train_loss)

            model.eval()
            loss_dev = []
            with torch.no_grad():
                for i in range(0, len(x_dev), self.batch_size):
                    try:
                        batch = x_dev[i:i+self.batch_size]
                        list_log_probs = model(batch)
                        log_probs = torch.cat(list_log_probs)
                        dev_labels = torch.cat(batch)
                        dev_loss = loss_function(log_probs, dev_labels)
                        loss_dev.append(dev_loss.item())
                    except:
                        print("batch", len(batch))
                        continue

            mean_dev_loss = sum(loss_dev) / len(loss_dev)
            losses_dev.append(mean_dev_loss)

            print("training loss: {} evaluation loss: {}".format(mean_train_loss, mean_dev_loss))

        if draw_image:
            self.draw_image(losses_train, losses_dev)
        return losses_train, losses_dev, model

