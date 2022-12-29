# -*- coding:utf-8 -*-
"""
@file: trainer_multi_stack_bilstm_pos_tagging.py
@time: 28/12/2022 11:55
@desc: 
@author: Echo
"""
import torch
from utils.trainer import Trainer


class TrainerMultiStackBilstmPOSTagging(Trainer):
    def __init__(self, n_epoch, batch_size, lr):
        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.lr = lr
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def trainer(self, model,  x_train, y_train, x_dev, y_dev, loss_function, optimizer, emb_table, early_stopping=True, early_stopping_limit=3):
        early_stopping_limit = early_stopping_limit
        dev_loss_up = 0
        loss_train = []
        loss_dev = []
        acc_train = []
        acc_dev = []
        for epoch in range(self.n_epoch):
            model.train()
            losses_train = list()
            batch_counter = 0
            for i in range(0, len(x_train), self.batch_size):
                optimizer.zero_grad()
                gold = torch.cat([torch.tensor(y) for y in y_train[i:i + self.batch_size]]).to(self.device)
                batch = x_train[i:i + self.batch_size]
                emb_batch = emb_table(batch)
                log_probs = model(emb_batch)
                loss = loss_function(log_probs, gold)
                losses_train.append(loss.item())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
                print(f"batch {batch_counter}/{int(len(x_train) / self.batch_size)}", end='\r', flush=True)
                batch_counter += 1
            mean_train_loss = sum(losses_train) / len(losses_train)
            loss_train.append(mean_train_loss)

            model.eval()
            log_probs = model(emb_table(x_dev))
            dev_labels = torch.cat([torch.tensor(y) for y in y_dev]).to(self.device)
            dev_loss = loss_function(log_probs, dev_labels)
            loss_dev.append(dev_loss.item())

            acc_train.append(model.compute_accuracy(emb_table(x_train), y_train))
            acc_dev.append(model.compute_accuracy(emb_table(x_dev), y_dev))

            print(
                f"{epoch + 1}) train_loss: {mean_train_loss:.3f}  dev_loss: {loss_dev[-1]:.3f} train_acc:{acc_train[-1]:.3f} dev_acc:{acc_dev[-1]:.3f}")

            if early_stopping:
                if len(loss_dev) > 5 and loss_dev[-1] >= loss_dev[-2]:
                    dev_loss_up += 1
                else:
                    dev_loss_up = 0
                if dev_loss_up == early_stopping_limit:
                    print(f"Preformed {epoch + 1} epochs")
                    break

        return model, loss_train, loss_dev, acc_train, acc_dev
