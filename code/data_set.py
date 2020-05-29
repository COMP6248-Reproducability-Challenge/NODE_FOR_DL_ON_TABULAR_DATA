import torch
from torch.utils.data import Dataset, TensorDataset
from torch.utils.data import DataLoader
import numpy as np
import random
from sklearn.preprocessing import QuantileTransformer

'''
custome data set class to convert the data frame object to tensors,
and also provied functions for nomorlisation.
'''


class MyDataset(Dataset):
    def __init__(self, data):
        self.X_tr = data['X_train']
        self.y_tr = data['y_train']
        self.X_val = data['X_valid']
        self.y_val = data['y_valid']
        self.X_t = data['X_test']
        self.y_t = data['y_test']
        
    def to_tensor(self,input):
        return torch.from_numpy(input).float()
    
    def normalised(self):
        mu, std = self.y_tr.mean(), self.y_tr.std()
        normalize = lambda x: ((x - mu) / std).astype(np.float32)
        self.y_tr, self.y_val, self.y_t = map(normalize, [self.y_tr,self.y_val, self.y_t])
    
    def quantile_transform(self, noise,random_state, distribution='normal'):
        np.random.seed(random_state)
        torch.manual_seed(random_state)
        random.seed(random_state)
        quantile_train = np.copy(self.X_tr)
        if noise:
            stds = np.std(quantile_train, axis=0, keepdims=True)
            noise_std = noise / np.maximum(stds, noise)
            quantile_train += noise_std * np.random.randn(*quantile_train.shape)

        qt = QuantileTransformer(random_state=random_state, output_distribution=distribution).fit(quantile_train)
        self.X_tr  = qt.transform(self.X_tr)
        self.X_val = qt.transform(self.X_val)
        self.X_t   = qt.transform(self.X_t)

    def get_loader(self, flag, params=None):
        if flag not in ['train','vaild','test']:
            print('invalid require!')
            return None
        data_set = None
        if flag == 'train':data_set = TensorDataset(self.to_tensor(self.X_tr),self.to_tensor(self.y_tr))         
        if flag == 'vaild':data_set = TensorDataset(self.to_tensor(self.X_val),self.to_tensor(self.y_val))
        if flag == 'test': data_set = TensorDataset(self.to_tensor(self.X_t),self.to_tensor(self.y_t))
        
        if params is None:
            return DataLoader(data_set)
        return DataLoader(data_set,**params)

    # def save(path):
    #     torch.save()
