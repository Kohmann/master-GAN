"""
This is an implementation of the RGAN model proposed by paper:
     Real-valued (Medical) Time Series Generation with Recurrent Conditional GANs (2017)
"""
import gc
import glob
from PIL import Image
import numpy as np
import datetime

import torch
import torchvision
import torch.nn as nn
from torch import autograd
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("runs/RGAN/" + datetime.datetime.now().strftime("%d-%m-%Y--%H:%M:%S"))

# GLOBAL VARIABLES
device = torch.device("mps" if torch.has_mps else"cpu")

K = 5  # number of times to train the discriminator
BATCH_SIZE = 32  # 32
SEQUENCE_LENGTH = 30
EPOCHS = 40

HIDDEN_SIZE = 30
NUM_LAYERS = 1

G_LRATE = 0.00002
D_LRATE = 0.00002


class Discriminator(nn.Module):
    """
    Discriminator class.
    """

    def __init__(self, input_size, hidden_size, num_layers):
        """
        Initialize the Discriminator class.
        """
        super(Discriminator, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,  # number of layers in the LSTM, more than 1 becomes a stacked LSTM
                            batch_first=False)  # input & output will have shape (1, batch_size, hidden_size)
        self.linear = nn.Linear(hidden_size, 1)
        # self.sigmoid = nn.Sigmoid() # Using BCE with logits instead. Sigmoid baked in
        self.hidden = None
        self._init_weights()

    def _init_weights(self):
        """
        Initialize the weights of the discriminator.
        """
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.Linear)):  # , nn.LSTM
                nn.init.normal_(m.weight, 0, 0.02)
                nn.init.constant_(m.bias, 0)

    def _init_hidden(self, batch_size):
        self.hidden = (torch.zeros(self.num_layers, batch_size, self.hidden_size, dtype=torch.float32, device=device),
                       torch.zeros(self.num_layers, batch_size, self.hidden_size, dtype=torch.float32, device=device))

    def forward(self, inputs):
        self._init_hidden(inputs.size(1))

        pred, self.hidden = self.lstm(inputs, self.hidden)
        pred = self.linear(pred).squeeze()  # squeeze from [1, batch-size, 1] to [batch-size]
        # pred = self.sigmoid(pred) # Using BCE with logits instead. Sigmoid baked in
        return pred


class Generator(nn.Module):
    """
    Generator class.
    """

    def __init__(self, input_size, hidden_size, num_layers, output_size):
        """
        Initialize the Generator class.
        """
        super(Generator, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,  # number of layers in the LSTM, more than 1 becomes a stacked LSTM
                            batch_first=False)  # input & output will have shape (batch_size, seq_len, feature_dim)

        self.linear = nn.Linear(hidden_size, output_size)
        self.tanh = nn.Tanh()

        self.hidden = None
        self._init_weights()

    def _init_weights(self):
        """
        Initialize the weights of the generator.
        """
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.Linear)):  # , nn.LSTM
                nn.init.normal_(m.weight, 0, 0.02)
                nn.init.constant_(m.bias, 0)

    def _init_hidden(self, batch_size):
        self.hidden = (torch.zeros(self.num_layers, batch_size, self.hidden_size, dtype=torch.float32, device=device),
                       torch.zeros(self.num_layers, batch_size, self.hidden_size, dtype=torch.float32, device=device))

    def forward(self, x):
        self._init_hidden(x.size(1))
        out, _ = self.lstm(x, self.hidden)
        out = self.linear(out)
        out = self.tanh(out)
        return out


def train(dataloader, discriminator, generator, loss_func, optimizer_d, optimizer_g, num_epochs):
    """
    Train the GAN using vanilla loss
    """

    # Fixed random vector for visualization purposes
    fixed_z = torch.randn(1, BATCH_SIZE, generator.input_size, device=device).detach_()
    step = 0

    for epoch in range(num_epochs):
        for i, (real, target) in enumerate(dataloader):
            batchsize = real.size(0)
            real = real.unsqueeze(0)
            real_data = real.to(device)
            discriminator.train()

            accu_loss_d = []
            accu_loss_g = []

            # --------------------------
            # Train the Discriminator
            # --------------------------
            for _ in range(K):
                z = torch.randn(1, batchsize, SEQUENCE_LENGTH, dtype=torch.float32, device=device)
                with torch.no_grad():  # create some fake data and don't track gradients
                    fake_data = generator(z)

                real_labels = torch.distributions.uniform.Uniform(0.7, 1.0).sample((batchsize,))
                fake_labels = torch.distributions.uniform.Uniform(0, 0.3).sample((batchsize,))

                real_loss = loss_func(discriminator(real_data).to(device), real_labels.to(device))
                fake_loss = loss_func(discriminator(fake_data).to(device), fake_labels.to(device))

                loss_d = (real_loss + fake_loss)  # * 0.5
                accu_loss_d.append(loss_d.item())

                discriminator.zero_grad()
                loss_d.backward(retain_graph=True)
                optimizer_d.step()
                optimizer_d.zero_grad()

            # --------------------------
            # Train the Generator
            # --------------------------

            # real_labels = torch.distributions.uniform.Uniform(0.7, 1.2).sample((batchsize,)) * torch.ones(batchsize)
            real_labels = torch.distributions.uniform.Uniform(0.7, 1.0).sample((batchsize,))
            fake_data = generator(z)
            loss_g = loss_func(discriminator(fake_data).to(device), real_labels.to(device))

            accu_loss_g.append(loss_g.item())
            generator.zero_grad()
            loss_g.backward()
            optimizer_g.step()
            optimizer_g.zero_grad()

            current_iteration = epoch * len(dataloader) + i

            if current_iteration % 200 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}]: Batch: [{i}/{len(dataloader)}],  ", end="")
                print("Loss D: %-8.7f, Loss G: %-8.7f" % (
                    torch.asarray(accu_loss_d).mean(), torch.asarray(accu_loss_g).mean()))

                writer.add_scalars('Loss', {'discriminator': torch.asarray(accu_loss_d).mean(),
                                            'generator': torch.asarray(accu_loss_g).mean()}, current_iteration)

                with torch.no_grad():
                    fake_data = generator(fixed_z).squeeze(0)

                    x_axis = np.linspace(0, 1, SEQUENCE_LENGTH)
                    fig, axs = plt.subplots(3, 3, figsize=(10, 10))

                    for x in range(3):
                        for y in range(3):
                            axs[x, y].plot(x_axis, fake_data[x * 3 + y].cpu().numpy())
                            axs[x, y].set_ylim([-1, 1])
                            axs[x, y].set_yticklabels([])

                    fig.suptitle(f"Generation: {step}", fontsize=14)
                    fig.savefig('./images/sinwave_at_epoch_{:04d}.png'.format(step))
                    writer.add_figure('Generated sinwaves', fig, step)
                    step += 1
    writer.close()


def create_gif(fp_in="images/image_*.png", fp_out="images/mnist_generation.gif"):
    imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]
    # extract first image from iterator
    imgs[0].save(fp=fp_out, append_images=imgs,
                 save_all=True, duration=100, loop=0)
    print("Gif created")


def create_sinwaves(n_samples, wave_length, freq_range, amp_range, phase_range):
    """
    Create a dataset of sin waves with random frequencies, phases and amplitudes.
    """
    X = np.zeros((n_samples, wave_length))
    freqs = np.random.uniform(freq_range[0], freq_range[1], n_samples)
    phases = np.random.uniform(phase_range[0], phase_range[1], n_samples)
    amplitudes = np.random.uniform(amp_range[0], amp_range[1], n_samples)

    for i in range(n_samples):
        X[i, :] = amplitudes[i] * np.sin(2 * np.pi * freqs[i] * np.arange(wave_length) / wave_length + phases[i])
    return X


if __name__ == "__main__":
    transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0, 1)])

    num_samples = 10_000  # number of samples to be generated
    sinwaves = create_sinwaves(n_samples=num_samples, wave_length=SEQUENCE_LENGTH, freq_range=(1, 4),
                               amp_range=(0.8, 0.9), phase_range=(0.0, 0.1))

    inputs = transforms(sinwaves).squeeze(0).float()

    train_data = TensorDataset(torch.asarray(inputs), torch.ones(num_samples))

    train_dl = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    print("Number of training data: {}".format(num_samples))

    sample, _ = next(iter(train_dl))
    print("Sample statistics: ", sample.shape, sample.max(), sample.min(), end="\n")
    print("Training on: %s" % device)

    # Create the discriminator and generator
    discriminator = Discriminator(input_size=SEQUENCE_LENGTH,
                                  hidden_size=HIDDEN_SIZE,
                                  num_layers=NUM_LAYERS).to(device)
    generator = Generator(input_size=SEQUENCE_LENGTH,
                          hidden_size=HIDDEN_SIZE,
                          num_layers=NUM_LAYERS,
                          output_size=SEQUENCE_LENGTH).to(device)

    optimizer_d = optim.Adam(discriminator.parameters(), lr=D_LRATE, betas=(0.5, 0.999))
    optimizer_g = optim.Adam(generator.parameters(), lr=G_LRATE, betas=(0.5, 0.999))

    # Train the GAN
    train(train_dl, discriminator, generator, nn.BCEWithLogitsLoss(), optimizer_d, optimizer_g, num_epochs=EPOCHS)
    create_gif(fp_in="images/sinwave_*.png", fp_out="images/sinewave_generation.gif")
