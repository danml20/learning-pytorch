import torch
import torch.nn as nn
import torch.optim as optim

class MLP(nn.Module):

    def __init__(self, layer_dims): # layer dims includes input dimensions

        super(MLP, self).__init__()

        layers = []

        for i in range(len(layer_dims) - 1): 
            layers.append(nn.Linear(layer_dims[i], layer_dims[i+1]))

            if i < len(layer_dims) - 2: # last linear layer does not pass outputs through ReLU
                layers.append(nn.ReLU())

        self.network = nn.Sequential(*layers)

    def forward(self, x):

        return self.network(x)