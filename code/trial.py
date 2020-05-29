import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

import torchbearer
from torchbearer import Trial#
from torchbearer.callbacks.decorators import on_backward
from torchbearer.callbacks.decorators import on_end_validation
from torchbearer.callbacks.decorators import on_start_validation
from torch.utils.tensorboard import SummaryWriter
from matplotlib.ticker import MaxNLocator
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

'''
using torchbearer framwork to trail our model for imporvements.
'''

class eval_model(nn.Module):
    def __init__(self, model, loss_f, device, metrics, trial_name, opt=optim.Adam,opt_params={}, **kwargs):  
        super().__init__()
        self.model = model
        self.loss_f = loss_f
        self.opt = opt(list(self.model.parameters()), **opt_params)
        self.metrics = metrics
        self.device = device
        self.trial_path = os.path.join('logs/', trial_name) 
        self.call_backs = None
        self.best_step = 0  

    def init_trial(self):   
        #initial call_back functions for the trail.
        if self.call_backs:
            self.trial = Trial(self.model,self.opt,self.loss_f,metrics=self.metrics,callbacks=self.call_backs).to(self.device)
        else:
            self.trial = Trial(self.model,self.opt,self.loss_f,metrics=self.metrics).to(self.device)
            

    def run(self, epoch, tr_loader,val_loader,t_loader=None,val_steps=None):
        self.trial.with_generators(tr_loader, test_generator=t_loader,val_generator=val_loader,val_steps=val_steps)
        return self.trial.run(epoch)
    

    def save_model(self, suffix, path=None, mkdir=True, **kwargs):
        if path is None:
            path = os.path.join(self.trial_path, "model_{}.mf".format(suffix))
        if mkdir:
            os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(OrderedDict([
            ('model', self.model.state_dict(**kwargs)),
            ('trial', self.trial.state_dict()),
            ('opt',   self.opt.state_dict()),
            ('best_step',  self.best_step),      
            ]), path)

        return path
    
    def load_model(self, suffix, path=None, **kwargs):
       
        if path is None:
            path = os.path.join(self.trial_path, "model_{}.mf".format(suffix))
        checkpoint = torch.load(path)

        self.trial.load_state_dict(checkpoint['trial'])
        self.model.load_state_dict(checkpoint['model'], **kwargs)
        self.opt.load_state_dict(checkpoint['opt'])
        self.best_step = int(checkpoint['best_step'])
        


    def plot_loss(self, loss, title, xlabel, x_all=False):    
        plt.figure()
        ax = plt.axes()
        ax.set_title(title)
        ax.set_ylabel("loss")
        ax.set_xlabel(xlabel)
        x = range(len(loss))
        plt.plot(x,loss)
        if x_all:
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.xticks(x)
        plt.legend()
        plt.show()

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