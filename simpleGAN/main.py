"""
This is a simple implementation of a GAN on MNIST.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets


class Discriminator(nn.Module):
    """
    Discriminator class.
    """

    def __init__(self, input_size, hidden_size, output_size):
        """
        Initialize the Discriminator class.
        """
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Forward pass of the discriminator.
        """
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Generator(nn.Module):
    """
    Generator class.
    """

    def __init__(self, input_size, output_size):
        """
        Initialize the Generator class.
        """
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(input_size, 10)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(10, output_size)

    def forward(self, x):
        """
        Forward pass of the generator.
        """
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        return x


def train(dataloader, discriminator, generator, criterion, optimizer_d, optimizer_g, num_epochs):
    """
    Train the GAN.
    """
    for epoch in range(num_epochs):
        pass


if __name__ == '__main__':
    print("fisk")
    mnist = datasets.MNIST('./mnist_data', train=True, download=False)
    dataloader = torch.utils.data.DataLoader(mnist, batch_size=64, shuffle=True)
    print("downloaded")

