'''data loaders for: MNIST, CIFAR'''

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_loaders(dataset_name, batch_size=64):
    
    if dataset_name == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(), # convert to pytorch tensors
            transforms.Normalize((0.5,), (0.5,)) # centre around 0 with sd 1
        ])
        full_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        
    elif dataset_name == "cifar10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # 3 channels
        ])
        full_trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

        train_size = 45000
        val_size = 5000 # take from train
        train_set, val_set = random_split(full_trainset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size, shuffle = True)
    test_loader = DataLoader(test_set, batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

