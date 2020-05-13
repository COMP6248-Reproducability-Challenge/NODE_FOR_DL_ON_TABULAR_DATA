import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import torchbearer
from torchbearer import Trial
import numpy as np
import matplotlib.pyplot as plt


class eval_model():
    def __init__(self, model, loss_f, device, metrics, opt=optim.Adam,opt_params={}, **kwargs):  
        self.model = model
        self.loss_f = loss_f
        self.opt = opt(list(self.model.parameters()), **opt_params)
        self.metrics = metrics
        self.device = device
        self.trial = Trial(self.model,self.opt,self.loss_f,metrics=self.metrics).to(self.device)
    

    def run(self, epoch, tr_loader,val_loader,t_loader=None):
        self.trial.with_generators(tr_loader, test_generator=t_loader,val_generator=val_loader)
        return self.trial.run(epoch)

    def evaluate(self,data_loader):
        self.model.eval()
        return self.trial.evaluate(data_loader)


    def plot_fig(self,out,epoch,title,y_label, para=['loss','val_loss']):
        tem =  [ [0]*epoch for i in range(len(para))]
        for idx, label in enumerate(para):
            for i in range(epoch):
                tem[idx][i] = out[i][label]

        plt.figure()
        ax = plt.axes()
        ax.set_title(title)
        ax.set_ylabel(y_label)
        ax.set_xlabel("Epoch")
        for i, item in enumerate(tem):
            plt.plot(range(epoch),item, label=para[i])
        plt.legend()
        plt.show()