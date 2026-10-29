'''Entry point to choose which model to train: setup device - load data - initialise model, loss, optimiser - train and eval'''
import torch
import torch.nn as nn
import torch.optim as optim

from models.mlp import MLP
from dataset import get_loaders
from train import train_epoch, evaluate_classification

import time


# configuration

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

SELECTED_MODEL = None
SELECTED_DATASET = "mnist"

# hyperparameters

batch_size = 64
learning_rate = 0.001
num_epochs = 10

# load corresponding data

train_loader, test_loader = get_loaders(SELECTED_DATASET, batch_size)

# initialise corresponding model

layer_dims_mnist = [28*28, 128, 128, 10] 
model = MLP(layer_dims_mnist)

# training loop

criterion = nn.CrossEntropyLoss()
optimiser = optim.Adam(model.parameters(), learning_rate)

start_time = time.time()

for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader, criterion, optimiser, device)
    print(f"train loss: {train_loss}")
    # do validation set too

end_time = time.time()

elapsed_time = end_time - start_time
print(f"train time: {elapsed_time} seconds")

test_accuracy = evaluate_classification(model, test_loader, device)

print(f"test accuracy: {test_accuracy}")