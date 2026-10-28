'''data loaders for: MNIST, CIFAR'''

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_mnist_loaders(batch_size=64):
    
    transform = transforms.Compose([
        transforms.ToTensor(), # convert image data to pytorch tensors
        transforms.Normalize((0.5,), (0.5,)) # centre around 0 with standard deviation 1
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader

