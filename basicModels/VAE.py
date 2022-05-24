import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim

        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=2)  # -> [1, 8, 13, 13]
        self.conv1_bn = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=2)  # -> [1, 16, 6, 6]
        self.conv2_bn = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=2)  # -> [1, 32, 2, 2]
        self.conv3_bn = nn.BatchNorm2d(32)

        self.mean = nn.Linear(32 * 2 * 2, self.latent_dim)
        self.logvar = nn.Linear(32 * 2 * 2, self.latent_dim)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = F.relu(self.conv2_bn(self.conv2(x)))
        x = F.relu(self.conv3_bn(self.conv3(x)))
        x = x.view(x.size(0), -1)
        mean = self.mean(x)
        logvar = self.logvar(x)
        return mean, logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim

        self.linear1 = nn.Linear(self.latent_dim, 32 * 2 * 2)

        self.deconv1 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=3)  # -> [1, 16, 6, 6]
        self.deconv1_bn = nn.BatchNorm2d(16)
        self.deconv2 = nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=0)  # -> [1, 8, 13, 13]
        self.deconv2_bn = nn.BatchNorm2d(8)
        self.deconv3 = nn.ConvTranspose2d(8, 1, kernel_size=3, stride=2, padding=0, output_padding=1)  # -> [1, 1, 28, 28]

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = x.view(-1, 32, 2, 2)
        x = F.relu(self.deconv1_bn(self.deconv1(x)))
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.sigmoid(self.deconv3(x))

        return x.view(-1, 28, 28)


class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dim):
        super(VariationalAutoencoder, self).__init__()

        self.latent_dim = latent_dim

        self.encoder = Encoder(self.latent_dim)
        self.decoder = Decoder(self.latent_dim)

    def forward(self, x):
        mean, logvar = self.encoder(x)
        z = self.reparameterize(mean, logvar)

        return self.decoder(z), mean, logvar

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn(self.latent_dim)
        return eps.mul(std).add_(mean)

    def decode(self, z):
        return self.decoder(z)


def kullbeck_leibler_divergence(mean, logvar):
    return 0.5 * torch.sum(mean ** 2 + torch.exp(logvar) - logvar - 1)


def reconstruction_loss(target, preds):
    return F.binary_cross_entropy(preds, target, reduction="sum")


def loss_fun(targets, preds, mean, logvar, beta=0.0001):
    return reconstruction_loss(targets, preds) + beta * kullbeck_leibler_divergence(mean, logvar)

