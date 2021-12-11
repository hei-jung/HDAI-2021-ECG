import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, recall_score, precision_score, \
    f1_score
from utils.ECGDataset import ECGDataset


class HyperParamConfig:
    def __init__(self, hyper_params):
        self.n_epoch = hyper_params['n_epoch']
        self.batch_size = hyper_params['batch_size']
        self.lr = hyper_params['lr']


class Train:
    def __init__(self, model, config_dict):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = model

        self.config = HyperParamConfig(config_dict)
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)

        self.y_pred_list = []

    def train_epoch(self, train_loader):
        self.model.train()
        avg_train_loss = 0
        for iter, batch in enumerate(train_loader):
            X = batch['X'].to(self.device)
            y_target = batch['y_target'].to(self.device)
            self.optimizer.zero_grad()  # optimizer reset

            y_pred = self.model(X)
            loss = self.criterion(y_pred, y_target)
            loss.backward()
            self.optimizer.step()
            avg_train_loss += loss.item() / len(train_loader)
        return avg_train_loss

    def valid_epoch(self, test_loader):
        self.model.eval()
        avg_test_loss = 0
        with torch.no_grad():  # no need to compute gradient
            self.y_pred_list = []
            for iter, batch in enumerate(test_loader):
                X = batch['X'].to(self.device)
                y_target = batch['y_target'].to(self.device)
                y_pred = self.model(X)
                self.y_pred_list.append(y_pred)  #####
                loss = self.criterion(y_pred, y_target)
                avg_test_loss += loss.item() / len(test_loader)
        return avg_test_loss

    def train_model(self):
        self.model.to(self.device)

        train_loader = DataLoader(ECGDataset(data_type='train'), batch_size=self.config.batch_size,
                                  num_workers=8, shuffle=True)
        # num_workers: cpu 여러개로 학습을 시켜라(8)
        test_loader = DataLoader(ECGDataset(data_type='val'), batch_size=self.config.batch_size, num_workers=8,
                                 shuffle=False)

        # trainer
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        loss_plot, val_loss_plot = [], []

        for n_epoch in range(1, self.config.n_epoch + 1):
            print(f'Train Epoch {n_epoch} start!')
            avg_train_loss = self.train_epoch(train_loader)
            print('[Train] loss: {:.3f}'.format(avg_train_loss))
            loss_plot.append(avg_train_loss)

            # validation
            avg_test_loss = self.valid_epoch(test_loader)
            print('[Test]  loss: {:.3f}\n'.format(avg_test_loss))
            val_loss_plot.append(avg_test_loss)
            y_predicted = torch.cat([*self.y_pred_list], dim=0)

            y_target = ECGDataset(data_type='val').y

            y_predicted = (y_predicted >= 0.5).float()  # threshold (round values)

            print("Area Under the Curve(AUC): ",
                  roc_auc_score(y_target, y_predicted))  # Compute Area Under the (ROC AUC) from prediction scores.
            print("Average Precision: ",
                  average_precision_score(y_target,
                                          y_predicted))  # Compute average precision (AP) from prediction scores.

            print(accuracy_score(y_target, y_predicted))
            print(recall_score(y_target, y_predicted))
            print(precision_score(y_target, y_predicted))
            print(f1_score(y_target, y_predicted))
            print()

        end.record()
        torch.cuda.synchronize()
        print(start.elapsed_time(end), "ms")
        return self.model
