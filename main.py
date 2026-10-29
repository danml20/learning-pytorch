'''Entry point to choose which model to train: setup device - load data - initialise model, loss, optimiser - train and eval'''
import torch
import torch.nn as nn
import torch.optim as optim

from models.mlp import MLP
from models.cnn import CNN
from dataset import get_loaders
from train import train_epoch, evaluate_classification

import time


# configuration

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

SELECTED_MODEL = "mlp"
SELECTED_DATASET = "mnist"

# hyperparameters

batch_size = 64
learning_rate = 0.001

# load corresponding data

train_loader, val_loader, test_loader = get_loaders(SELECTED_DATASET, batch_size)

# initialise corresponding model

if SELECTED_MODEL == "mlp" and SELECTED_DATASET == "mnist":
    layer_dims_mnist = [28*28, 128, 128, 10] 
    model = MLP(layer_dims_mnist)
    path = 'mlp_params.pth'

elif SELECTED_MODEL == "cnn" and SELECTED_DATASET == "cifar10":
    pass
elif SELECTED_MODEL == "cnn" and SELECTED_DATASET == "cifar10":
    channels = []
    kernel_size = 3
    model = CNN(channels, kernel_size)
    path = 'cnn_params.pth'

# training loop

criterion = nn.CrossEntropyLoss()
optimiser = optim.Adam(model.parameters(), learning_rate)

num_epochs = 20
best_val_acc = 0.0

start_time = time.time()

for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader, criterion, optimiser, device)
    print(f"train loss: {train_loss}")

    val_accuracy = evaluate_classification(model, val_loader, device)
    if val_accuracy > best_val_acc:
        best_val_acc = val_accuracy
        torch.save(model.state_dict(), path)
        print("saved new best model params")



end_time = time.time()

elapsed_time = end_time - start_time
print(f"train time: {elapsed_time} seconds")


torch.save(model.state_dict(), path)
print(f"{SELECTED_MODEL} saved to {path}")

test_accuracy = evaluate_classification(model, test_loader, device)

print(f"test accuracy: {test_accuracy}")