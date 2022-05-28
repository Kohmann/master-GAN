"""
This is a simple implementation of the original GAN on MNIST data.
"""
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("runs/mnist/gan")

# GLOBAL VARIABLES

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

K = 1 # number of times to train the discriminator
BATCH_SIZE = 128
GENERATOR_LATENT_DIM = 100
EPOCHS = 15




class Discriminator(nn.Module):
    """
    Discriminator class.
    """

    def __init__(self):
        """
        Initialize the Discriminator class.
        """
        super(Discriminator, self).__init__()

        self.conv1 = nn.Conv2d(1, 16*2, kernel_size=4, stride=2, padding=1, bias=True)  # -> (batch_size, 16, 16, 16)
        #self.conv1_bn = nn.BatchNorm2d(16)
        self.leaky_relu1 = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(16*2, 32*4, kernel_size=4, stride=2, padding=1, bias=False)  # -> (batch_size, 32, 8, 8)
        self.conv2_bn = nn.BatchNorm2d(32*4)
        self.leaky_relu2 = nn.LeakyReLU(0.2)
        self.conv3 = nn.Conv2d(32*4, 64*8, kernel_size=4, stride=2, padding=1, bias=False)  # -> [batch_size, 64, 4, 4]
        self.conv3_bn = nn.BatchNorm2d(64*8)
        self.leaky_relu3 = nn.LeakyReLU(0.2)

        self.conv4 = nn.Conv2d(64*8, 1, kernel_size=4, stride=2, padding=0, bias=False)  # -> [batch_size, 1, 1, 1]
        self.sigmoid = nn.Sigmoid()

        self._init_weights()

    def _init_weights(self):
        """
        Initialize the weights of the discriminator.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 0, 0.02)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.02)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.leaky_relu1(self.conv1(x))
        x = self.leaky_relu2(self.conv2_bn(self.conv2(x)))
        x = self.leaky_relu3(self.conv3_bn(self.conv3(x)))
        x = self.conv4(x)
        out = self.sigmoid(x)

        return out


class Generator(nn.Module):
    """
    Generator class.
    """

    def __init__(self, latent_dim):
        """
        Initialize the Generator class.
        """
        super(Generator, self).__init__()
        self.latent_dim = latent_dim

        self.deconv1 = nn.ConvTranspose2d(self.latent_dim, 64*8, kernel_size=4, stride=2, padding=0, bias=False)  # -> [1, 32, 4, 4]
        self.deconv1_bn = nn.BatchNorm2d(64*8)
        self.deconv2 = nn.ConvTranspose2d(64*8, 32*4, kernel_size=4, stride=2, padding=1, bias=False)  # -> [1, 16, 16, 16]
        self.deconv2_bn = nn.BatchNorm2d(32*4)
        self.deconv3 = nn.ConvTranspose2d(32*4, 16*2, kernel_size=4, stride=2, padding=1)  # -> [1, 1, 32, 32]
        self.deconv3_bn = nn.BatchNorm2d(16*2)
        self.deconv4 = nn.ConvTranspose2d(16*2, 1, kernel_size=4, stride=2, padding=1)
        self.tanh = nn.Tanh()

        self._init_weights()

    def _init_weights(self):
        """
        Initialize the weights of the generator.
        """
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, 0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.02)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = F.relu(self.deconv1_bn(self.deconv1(x)))
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = self.deconv4(x)
        x = self.tanh(x)

        return x



def train(dataloader, discriminator, generator, loss_func, optimizer_d, optimizer_g, num_epochs):
    """
    Train the GAN using vanilla loss
    """

    fixed_z = torch.randn(16, generator.latent_dim, 1, 1).to(device)
    step = 0

    for epoch in range(num_epochs):
        for i, (real, _) in enumerate(dataloader):

            real_data = real.to(device)
            batchsize = real.size(0)

            #discriminator.train()
            #generator.eval()

            accu_loss_d = []
            accu_loss_g = []
            # train the discriminator
            for _ in range(K):
                # get the real data
                z = torch.randn(batchsize, GENERATOR_LATENT_DIM, 1, 1).to(device)
                fake_data = generator(z)

                real_labels = torch.distributions.uniform.Uniform(0.7, 1.2).sample((batchsize,)) * torch.ones(batchsize)
                fake_labels = torch.distributions.uniform.Uniform(0.0, 0.3).sample((batchsize,)) * torch.zeros(batchsize)
                #real_loss = loss_func(discriminator(real_data).reshape(-1).to(device), 0.9*torch.ones(real_data.size(0)).to(device))
                #fake_loss = loss_func(discriminator(fake_data).reshape(-1).to(device), torch.zeros(fake_data.size(0)).to(device))

                real_loss = loss_func(discriminator(real_data).reshape(-1).to(device), real_labels.to(device))
                fake_loss = loss_func(discriminator(fake_data).reshape(-1).to(device), fake_labels.to(device))

                loss_d = real_loss + fake_loss
                accu_loss_d.append(loss_d.item())

                discriminator.zero_grad()
                loss_d.backward(retain_graph=True)
                optimizer_d.step()
                optimizer_d.zero_grad()

            # train the generator
            #discriminator.eval()
            #generator.train()

            #fake_data = generator.sample(BATCH_SIZE)
            real_labels = torch.distributions.uniform.Uniform(0.7, 1.2).sample((batchsize,)) * torch.ones(batchsize)
            loss_g = loss_func(discriminator(fake_data).reshape(-1).to(device), real_labels.to(device))
            # loss_g = -1.0 * torch.mean(torch.log(discriminator(fake_data)))
            accu_loss_g.append(loss_g.item())
            generator.zero_grad()
            loss_g.backward()
            optimizer_g.step()
            optimizer_g.zero_grad()

            #writer.add_scalar('Loss/discriminator', torch.asarray(accu_loss_d).mean(), epoch)
            #writer.add_scalar('Loss/generator', torch.asarray(accu_loss_g).mean(), epoch)

            current_iteration = epoch * len(dataloader) + i

            if i % 100 == 0:
                #print('Epoch: {}, Loss D: {}, Loss G: {}'.format(current_iteration, torch.asarray(accu_loss_d).mean(),
                #                                                        torch.asarray(accu_loss_g).mean()))
                print(f"Epoch [{epoch+1}/{num_epochs}]: Batch: [{i}/{len(dataloader)}],  ", end="")
                print("Loss D: %-8.7f, Loss G: %-8.7f" % (torch.asarray(accu_loss_d).mean(), torch.asarray(accu_loss_g).mean()))

                # torch.asarray(accu_loss_d).mean(),torch.asarray(accu_loss_g).mean()))
                writer.add_scalars('COLAB/Loss', {'discriminator': torch.asarray(accu_loss_d).mean(),
                                            'generator': torch.asarray(accu_loss_g).mean()}, current_iteration)
                #generator.eval()
                with torch.no_grad():
                    fake_data = generator(fixed_z)
                    image_grid = torchvision.utils.make_grid(fake_data, nrow=4, normalize=True)
                    writer.add_image('Generated Images', image_grid, step)
                    torchvision.utils.save_image(image_grid, './images/image_at_epoch_{:04d}.png'.format(step))
                    step += 1



import glob
from PIL import Image


def create_gif(fp_in="images/image_*.png", fp_out="images/mnist_generation.gif"):
    imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]
    # extract first image from iterator
    imgs[0].save(fp=fp_out, append_images=imgs,
                 save_all=True, duration=100, loop=0)


if __name__ == "__main__":

    transforms = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize((32, 32)),
                                    transforms.Normalize((0.5,), (0.5,)),
                                    ])
    mnist = datasets.MNIST('../mnist_data', train=True, download=True, transform=transforms)
    ones = mnist.targets == 0

    #mnist_normalized = (mnist.data.view(-1, 1, 28, 28) - 127.5) / 127.5
    #train_data = TensorDataset(mnist_normalized[ones], torch.ones(ones.sum()))
    #train_data = TensorDataset(mnist)
    train_dl = DataLoader(mnist, batch_size=BATCH_SIZE, shuffle=True)
    print("number of training data: {}".format(len(train_dl) * BATCH_SIZE))

    image_sample, _ = next(iter(train_dl))
    print("image sample statistics: ", image_sample.shape, image_sample.max(), image_sample.min(), end="\n\n")

    # Create the discriminator and generator
    discriminator = Discriminator().to(device)
    generator = Generator(GENERATOR_LATENT_DIM).to(device)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_g = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Train the GAN
    train(train_dl, discriminator, generator, nn.BCELoss(), optimizer_d, optimizer_g, num_epochs=EPOCHS)
    create_gif()

