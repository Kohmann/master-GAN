import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, latent_size):
        super().__init__()

        self.latent_size = latent_size

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2)

        self.mean = nn.Linear(32 * 6 * 6, self.latent_size)
        self.logvar = nn.Linear(32 * 6 * 6, self.latent_size)

        self.linear1 = nn.Linear(self.latent_size, 32 * 6 * 6)
        self.deconv1 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2)  # scale up to 16x13x13
        self.deconv2 = nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2)  # scale up to 8x27x27
        self.deconv3 = nn.ConvTranspose2d(8, 1, kernel_size=2, stride=1)  # scale up to 8x28x28

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        mean = self.mean(x)
        logvar = self.logvar(x)

        z = self.reparameterize(mean, logvar)
        x = self.linear1(z)
        x = x.view(-1, 32, 6, 6)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.sigmoid(self.deconv3(x))
        x = x.view(-1, 28, 28)

        return x, mean, logvar

    def decode(self, z):
        x = self.linear1(z)
        x = x.view(-1, 32, 6, 6)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.sigmoid(self.deconv3(x))
        x = x.view(-1, 28, 28)
        return x


    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn(self.latent_size)
        return mean + eps * std


def kullbeck_leibler_divergence(mean, logvar):
    return -0.5 * torch.sum(mean ** 2 + torch.exp(logvar) - logvar - 1)


def reconstruction_loss(target, preds):
    return F.binary_cross_entropy(preds, target, reduction="sum")


def loss_fun(targets, preds, mean, logvar, beta=0.0001):
    return reconstruction_loss(targets, preds) + beta * kullbeck_leibler_divergence(mean, logvar)

