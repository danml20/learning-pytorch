'''General and model-specific training loops'''

import torch
import numpy as np

def train_epoch(model, dataloader, criterion, optimiser, device):
    '''general training loop for one epoch'''

    model.train() # set model to training mode
    running_loss = 0.0 # track total loss of epoch 

    for inputs, labels in dataloader:

        inputs = inputs.to(device)
        labels = labels.to(device)

        optimiser.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, labels)

        loss.backward() # backward pass 
        optimiser.step() # updates

        running_loss += loss.item() # tensor -> float (detach from computational graph)

    return running_loss / len(dataloader) # epoch average per batch 

def evaluate_classification(model, dataloader, criterion, device):
    '''general evaluation loop to compute accuracy on a test or validation set'''

    model.eval() # eval mode: turn off dropout, batch norm, etc

    loss = 0
    correct = 0
    total = 0

    with torch.no_grad(): # temporarily turn off gradient calculation in this block
        for inputs, labels in dataloader:

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs) # raw logits 

            loss += criterion(outputs, labels).items()

            _, predicted = torch.max(outputs.data, dim=1) # on raw data (doesnt carry gradient history), max returns (vals, idxs)

            total += labels.size(0) # batch size
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    loss = loss / len(dataloader)
    
    return accuracy, loss

class EarlyStopping:

    def __init__(self, path, patience=5, delta=0.001):
        '''args: 
            patience (int): number of epochs to wait after last time val loss improved
            delta (float): minimum change in given quantity to qualify as an improvement
        '''
        self.patience = patience
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_loss = np.inf
        self.early_stop = False

    def __call__(self, val_loss, model): # call instance directly (like a func)
        if val_loss < self.best_loss - self.delta: # improvement
            self.best_loss = val_loss
            self.save_checkpoint(model)
            self.counter = 0
        else: # no improvement
            self.counter += 1
            if self.counter >= self.patience:
                        self.early_stop = True


    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path)