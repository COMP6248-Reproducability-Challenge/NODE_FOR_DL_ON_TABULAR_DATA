import torch
import torch.nn as nn
import torch.nn.functional as F
from odst import ODST


'''
The DNN structure
adopted from https://github.com/Qwicen/node
'''


class DenseBlock(nn.Sequential):
    def __init__(self, input_dim, layer_dim, num_layers, tree_dim=1, max_features=None,
                 dropout=0.0, flatten=True, module=ODST, **kwargs):
       
        layers = []

        for n in range(num_layers):
            oddt = module(input_dim, layer_dim, tree_dim=tree_dim, flatten=True, **kwargs)
            input_dim = min(input_dim + layer_dim * tree_dim, max_features or float('inf'))
            layers.append(oddt)

        super().__init__(*layers)
        self.num_layers, self.layer_dim, self.tree_dim = num_layers, layer_dim, tree_dim
        self.max_features, self.flatten = max_features, flatten
        self.dropout = dropout

        

    def forward(self, x):
        initial_features = x.shape[-1]
        for layer in self:
            L_input = x
            if self.max_features is not None:
                tail_features = min(self.max_features, L_input.shape[-1]) - initial_features
                if tail_features != 0:
                    L_input = torch.cat([L_input[..., :initial_features], L_input[..., -tail_features:]], dim=-1)
            
            if self.training and self.dropout:
                L_input = F.dropout(L_input, self.dropout)

            h = layer(L_input)
            x = torch.cat([x, h], dim=-1)

        outputs = x[..., initial_features:]

        if not self.flatten:
            outputs = outputs.view(*outputs.shape[:-1], self.num_layers * self.layer_dim, self.tree_dim)
        return outputs
