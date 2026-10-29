'''General and model-specific training loops'''

import torch

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

def evaluate_classification(model, dataloader, device):
    '''general evaluation loop to compute accuracy on a test or validation set'''

    model.eval() # eval mode: turn off dropout, batch norm, etc

    correct = 0
    total = 0

    with torch.no_grad(): # temporarily turn off gradient calculation in this block
        for inputs, labels in dataloader:

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs) # probabiliites or raw logits 

            _, predicted = torch.max(outputs.data, dim=1) # on raw data (doesnt carry gradient history), max returns (vals, idxs)

            total += labels.size(0) # batch size
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy