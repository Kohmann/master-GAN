# -*- coding: utf-8 -*-


import torch
import torch.nn as nn

ID = "RTSGAN"


def toggle_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)


class Encoder(nn.Module):

    def __init__(self, feature_dim, hidden_dim, num_layers, max_seq_len):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len

        self.emb_rnn = nn.GRU(
            input_size=self.feature_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True
        )
        self.fc1 = nn.Linear(self.hidden_dim * 3, self.hidden_dim)  # [H_mean,H_max,H_last] -> hidden_dim
        self.activation = nn.LeakyReLU(0.2)
        self.fc2 = nn.Linear(self.hidden_dim * self.num_layers, self.hidden_dim * self.num_layers)

    def forward(self, X):
        batchsize, max_len, _ = X.size()
        X_packed = nn.utils.rnn.pack_padded_sequence(
            input=X,
            lengths=torch.ones(batchsize, dtype=torch.int64) * self.max_seq_len,
            batch_first=True,
            enforce_sorted=False
        )
        H_o, H_t = self.emb_rnn(X_packed)
        H_o, T = nn.utils.rnn.pad_packed_sequence(
            sequence=H_o,
            batch_first=True,
            padding_value=0.0,
            total_length=self.max_seq_len
        )


        H_mean = torch.mean(H_o, dim=1)
        H_max, _ = torch.max(H_o, dim=1)
        all_hidden = H_t.view(self.num_layers, -1, batchsize, self.hidden_dim)
        H_last = all_hidden[-1].view(batchsize, -1)  # last H_t state

        glob = torch.cat([H_mean, H_max, H_last], dim=-1)
        glob = self.activation(self.fc1(glob))

        lasth = all_hidden.view(-1, batchsize, self.hidden_dim)
        lasth = lasth.permute(1, 0, 2).contiguous().view(batchsize, -1)
        lasth = self.activation(self.fc2(lasth))
        hidden = torch.cat([glob, lasth], dim=-1)

        return hidden


class Decoder(nn.Module):

    def __init__(self, feature_dim, hidden_dim, num_layers, max_seq_len):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len

        self.emb_rnn = nn.GRU(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True
        )
        self.fc1 = nn.Linear(self.hidden_dim, self.feature_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X):
        glob, hidden = X[:, :self.hidden_dim], X[:, self.hidden_dim:]  # decomposing of X = torch.concat[glob, lasth]
        batchsize = X.size(0)
        # print("\nglob shape", glob.size())
        # print("hidden shape raw",hidden.shape)
        hidden = hidden.view(batchsize, self.num_layers, -1).permute(1, 0, 2).contiguous()
        # print("hidden shape after permute", hidden.shape)
        # hidden, finh = hidden[:-1], hidden[-1:]
        # print("hidden shape", hidden.shape,"finh shape", finh.shape)

        glob_expand = glob.unsqueeze(1).expand(-1, self.max_seq_len, -1)
        # print("current input: glob_expand", glob_expand.size())
        glob_packed = nn.utils.rnn.pack_padded_sequence(
            input=glob_expand,
            lengths=torch.ones(batchsize, dtype=torch.int64) * self.max_seq_len,
            batch_first=True,
            enforce_sorted=False
        )
        # print("current input: glob_packed", glob_packed.data.size())
        H_o, H_t = self.emb_rnn(glob_packed, hidden)
        H_o, T = nn.utils.rnn.pad_packed_sequence(
            sequence=H_o,
            batch_first=True,
            padding_value=0.0,
            total_length=self.max_seq_len
        )
        out = self.sigmoid(self.fc1(H_o))

        return out


class Generator(torch.nn.Module):
    """The generator network (encoder) for TimeGAN
    """

    def __init__(self, Z_dim, hidden_dim, layers):
        super(Generator, self).__init__()

        def block(inp, out):
            return nn.Sequential(
                nn.Linear(inp, out),
                nn.LayerNorm(out),
                nn.LeakyReLU(0.2),
            )

        self.block_0 = block(Z_dim, Z_dim)
        self.block_1 = block(Z_dim, Z_dim)
        self.block_2 = block(Z_dim, Z_dim)
        self.block_3 = block(Z_dim, Z_dim)
        self.block_4 = nn.Linear(Z_dim, hidden_dim * (layers + 1))
        self.final = nn.LeakyReLU(0.2)

    def forward(self, Z):
        out = self.block_0(Z) + Z
        out = self.block_1(out) + out
        out = self.block_2(out) + out
        out = self.block_3(out) + out
        return self.final(self.block_4(out))


class Discriminator(nn.Module):
    def __init__(self, input_dim, SN=False):
        super(Discriminator, self).__init__()
        if SN:
            self.model = nn.Sequential(
                nn.utils.parametrizations.spectral_norm(nn.Linear(input_dim, (2 * input_dim) // 3)),
                nn.LeakyReLU(0.2),
                nn.utils.parametrizations.spectral_norm(nn.Linear((2 * input_dim) // 3, input_dim // 3)),
                nn.LeakyReLU(0.2),
                nn.Linear(input_dim // 3, 1),
            )
        else:
            self.model = nn.Sequential(
                nn.Linear(input_dim, (2 * input_dim) // 3),
                nn.LeakyReLU(0.2),
                nn.Linear((2 * input_dim) // 3, input_dim // 3),
                nn.LeakyReLU(0.2),
                nn.Linear(input_dim // 3, 1),
            )

    def forward(self, x):
        return self.model(x)


class RTSGAN(torch.nn.Module):

    def __init__(self, params):
        super(RTSGAN, self).__init__()
        self.device = params['device']
        self.feature_dim = params['feature_dim']
        self.Z_dim = params['Z_dim']
        self.hidden_dim = params['hidden_dim']
        self.num_layers = params['num_layers']
        self.max_seq_len = params['max_seq_len']
        self.batch_size = params['batch_size']
        self.use_spectral_norm = params['spectralnorm']

        # Networks
        self.encoder = Encoder(self.feature_dim, self.hidden_dim, self.num_layers, self.max_seq_len)
        self.decoder = Decoder(self.feature_dim, self.hidden_dim, self.num_layers, self.max_seq_len)
        self.generator = Generator(self.Z_dim, self.hidden_dim, self.num_layers)
        disc_input_dim = self.hidden_dim * (self.num_layers + 1)
        self.discriminator = Discriminator(disc_input_dim, SN=self.use_spectral_norm)

    def load_ae(self):
        self.encoder.load_state_dict(torch.load("rtsgan_encoder" +str(self.max_seq_len)+ ".pt"))
        self.decoder.load_state_dict(torch.load("rtsgan_decoder" +str(self.max_seq_len)+ ".pt"))
        print("Pretrained autoencoder loaded.")

    def _autoencoder_forward(self, X):
        # Forward Pass
        H = self.encoder(X)
        X_hat = self.decoder(H)

        # Reconstruction Loss
        loss = torch.nn.functional.mse_loss(X_hat, X)
        return loss

    def _discriminator_forward(self, X, Z, gamma=10):
        toggle_grad(self.generator, False)
        toggle_grad(self.discriminator, True)

        with torch.no_grad():  # turn off gradient calculation for these networks
            H_real = self.encoder(X)
            H_fake = self.generator(Z)
        H_fake.requires_grad_(True)
        H_real.requires_grad_(True)

        # Discriminator Loss
        D_real = self.discriminator(H_real)
        D_fake = self.discriminator(H_fake)
        # Wasserstein Loss
        loss = torch.mean(D_fake) - torch.mean(D_real)

        if not self.use_spectral_norm:  # if not using spectral normalization
            # Gradient Penalty
            alpha = torch.rand(self.batch_size, 1, 1, device=self.device)
            H_hat = (alpha * H_real + (1 - alpha) * H_fake).requires_grad_(True)
            D_hat = self.discriminator(H_hat)
            gradients = torch.autograd.grad(
                outputs=D_hat,
                inputs=H_hat,
                grad_outputs=torch.ones_like(D_hat),
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]
            gradient_penalty = gamma * ((gradients.norm(2, dim=1) - 1) ** 2).mean()
            loss += gradient_penalty
        return loss

    def _generator_forward(self, Z):
        # Forward Pass
        toggle_grad(self.generator, True)
        toggle_grad(self.discriminator, False)
        H_fake = self.generator(Z)

        D_fake = self.discriminator(H_fake)
        # Generator Loss
        loss = -torch.mean(D_fake)
        return loss

    def _inference(self, Z):
        # Generator Forward Pass
        H_fake = self.generator(Z)

        # Synthetic data generated
        X_hat = self.decoder(H_fake)
        return X_hat

    def forward(self, X, Z, obj):
        """
        Args:
            - X: the input features (B, H, F)
            - Z: the sampled noise (B, H, Z)
            - obj: the network to be trained (`autoencoder`, `supervisor`, `generator`, `discriminator`)
            - gamma: loss hyperparameter
        Returns:
            - loss: The loss for the forward pass
            - X_hat: The generated data
        """
        if obj != "inference":
            if X is not None:
                # raise ValueError("`X` should be given")
                X = torch.FloatTensor(X)
                X = X.to(self.device)

        # if Z is not None:
        #    Z = torch.FloatTensor(Z)
        #    Z = Z.to(self.device)

        if obj == "autoencoder":
            # Embedder & Recovery
            loss = self._autoencoder_forward(X)

        elif obj == "generator":
            if Z is None:
                raise ValueError("`Z` is not given")

            # Generator
            loss = self._generator_forward(Z)

        elif obj == "discriminator":
            if Z is None:
                raise ValueError("`Z` is not given")

            # Discriminator
            loss = self._discriminator_forward(X, Z)

            return loss

        elif obj == "inference":

            X_hat = self._inference(Z)
            X_hat = X_hat.cpu().detach()

            return X_hat

        else:
            raise ValueError("`obj` should be either `autoencoder`, `supervisor`, `generator`, or `discriminator`")

        return loss
