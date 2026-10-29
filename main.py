'''Entry point to choose which model to train: setup device - load data - initialise model, loss, optimiser - train and eval'''
import torch
import torch.nn as nn
import torch.optim as optim

from models.mlp import MLP
from dataset import get_loaders
from train import train_epoch, evaluate_classification


# configuration

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

SELECTED_MODEL = None
SELECTED_DATASET = None

# hyperparameters

batch_size = 64
learning_rate = 0.001
num_epochs = 1000

# load corresponding data

train_loader, test_loader = get_loaders(SELECTED_DATASET, batch_size)

# initialise corresponding model

layer_dims = [] # relate to data
model = MLP(layer_dims)

# training loop

criterion = nn.CrossEntropyLoss()
optimiser = optim.Adam(model.parameters(), learning_rate)

for epoch in num_epochs:
    train_loss = train_epoch(model, train_loader, criterion, optimiser, device)
    # do validation set too
test_accuracy = evaluate_classification(model, test_loader, device)