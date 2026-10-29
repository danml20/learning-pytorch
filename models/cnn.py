import torch
import torch.nn as nn

class CNN(nn.Module):

    def __init__(self, channels, kernel_size):
        super(CNN, self).__init__()

        layers = []
        for i in range(len(channels) - 1):
            layers.extend((
                nn.Conv2d(channels[i], channels[i+1], kernel_size),
                nn.ReLU(),
                nn.MaxPool2d(2, 2)
            ))

        self.features = nn.Sequential(*layers)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(128), # automatically infer flattened input size
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x