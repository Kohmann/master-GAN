# -*- coding: utf-8 -*-
"""TimeGAN_cnn.ipynb

"""

import torch
import torch.nn as nn
from .weight_inits import rnn_weight_init, linear_weight_init, weight_init

ID = "TimeGAN_G"

class EmbeddingNetwork(nn.Module):

    def __init__(self, feature_dim, hidden_dim, num_layers, padding_value, max_seq_len):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.padding_value = padding_value
        self.max_seq_len = max_seq_len

        self.emb_rnn = nn.GRU(
            input_size=self.feature_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
        )

        self.emb_linear = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.emb_sigmoid = nn.Sigmoid()

        rnn_weight_init(self.emb_rnn)
        linear_weight_init(self.emb_linear)

    def forward(self, X, T):
        X_packed = nn.utils.rnn.pack_padded_sequence(
            input=X,
            lengths=T,
            batch_first=True,
            enforce_sorted=False
        )

        H_o, H_t = self.emb_rnn(X_packed)

        H_o, T = nn.utils.rnn.pad_packed_sequence(
            sequence=H_o,
            batch_first=True,
            padding_value=self.padding_value,
            total_length=self.max_seq_len
        )

        logits = self.emb_linear(H_o)
        H = self.emb_sigmoid(logits)
        return H


class RecoveryNetwork(nn.Module):
    """The recovery network (decoder) for TimeGAN
    """

    def __init__(self, feature_dim, hidden_dim, num_layers, padding_value, max_seq_len):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.padding_value = padding_value
        self.max_seq_len = max_seq_len

        self.rec_rnn = nn.GRU(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=False
        )

        self.rec_linear = torch.nn.Linear(self.hidden_dim, self.feature_dim)
        rnn_weight_init(self.rec_rnn)
        linear_weight_init(self.rec_linear)

    def forward(self, H, T):
        H_packed = torch.nn.utils.rnn.pack_padded_sequence(
            input=H,
            lengths=T,
            batch_first=True,
            enforce_sorted=False
        )

        # 128 x 100 x 10
        H_o, H_t = self.rec_rnn(H_packed)

        # Pad RNN output back to sequence length
        H_o, T = torch.nn.utils.rnn.pad_packed_sequence(
            sequence=H_o,
            batch_first=True,
            padding_value=self.padding_value,
            total_length=self.max_seq_len
        )

        # 128 x 100 x 71
        X_tilde = self.rec_linear(H_o)
        return X_tilde


class SupervisorNetwork(torch.nn.Module):
    """The Supervisor network (decoder) for TimeGAN
    """

    def __init__(self, hidden_dim, num_layers, padding_value, max_seq_len):
        super(SupervisorNetwork, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers - 1 if num_layers > 1 else num_layers
        self.padding_value = padding_value
        self.max_seq_len = max_seq_len

        # Supervisor Architecture
        self.sup_rnn = torch.nn.GRU(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            # bidirectional=True
        )
        self.sup_linear = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
        self.sup_sigmoid = torch.nn.Sigmoid()
        rnn_weight_init(self.sup_rnn)
        linear_weight_init(self.sup_linear)

    def forward(self, H, T):
        """Forward pass for the supervisor for predicting next step
        Args:
            - H: latent representation (B x S x E)
            - T: input temporal information (B)
        Returns:
            - H_hat: predicted next step data (B x S x E)
        """
        # Dynamic RNN input for ignoring paddings
        H_packed = torch.nn.utils.rnn.pack_padded_sequence(
            input=H,
            lengths=T,
            batch_first=True,
            enforce_sorted=False
        )

        # 128 x 100 x 10
        H_o, H_t = self.sup_rnn(H_packed)

        # Pad RNN output back to sequence length
        H_o, T = torch.nn.utils.rnn.pad_packed_sequence(
            sequence=H_o,
            batch_first=True,
            padding_value=self.padding_value,
            total_length=self.max_seq_len
        )

        # 128 x 100 x 10
        logits = self.sup_linear(H_o)
        # 128 x 100 x 10
        H_hat = self.sup_sigmoid(logits)
        return H_hat


class GeneratorNetwork(torch.nn.Module):
    """The generator network (encoder) for TimeGAN
    """

    def __init__(self, Z_dim, hidden_dim, num_layers, padding_value, max_seq_len):
        super(GeneratorNetwork, self).__init__()
        self.Z_dim = Z_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.padding_value = padding_value
        self.max_seq_len = max_seq_len

        # Generator Architecture

        self.gen_conv = nn.Sequential(
            nn.Conv1d(in_channels=100, out_channels=100, kernel_size=(6), stride=2, bias=False)
            , nn.BatchNorm1d(100)
            , nn.LeakyReLU()
            , nn.Conv1d(in_channels=100, out_channels=100, kernel_size=(9), stride=2, bias=True)
        )

        self.gen_sigmoid = torch.nn.Sigmoid()  # x in range [0, 1]
        weight_init(self.gen_conv)

    def forward(self, Z, T):
        """Takes in random noise (features) and generates synthetic features within the latent space
        Args:
            - Z: input random noise (B x S x Z)
            - T: input temporal information
        Returns:
            - H: embeddings (B x S x E)
        """

        logits = self.gen_conv(Z)
        logits = logits.view(-1, self.max_seq_len, self.hidden_dim)

        H = self.gen_sigmoid(logits)
        return H


class DiscriminatorNetwork(torch.nn.Module):
    """The Discriminator network (decoder) for TimeGAN
    """

    def __init__(self, hidden_dim, num_layers, padding_value, max_seq_len):
        super(DiscriminatorNetwork, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.padding_value = padding_value
        self.max_seq_len = max_seq_len

        # Discriminator Architecture
        self.dis_rnn = torch.nn.GRU(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            #bidirectional=True
        )
        self.dis_linear = torch.nn.Linear(self.hidden_dim, 1)
        rnn_weight_init(self.dis_rnn)
        linear_weight_init(self.dis_linear)

    def forward(self, H, T):
        """Forward pass for predicting if the data is real or synthetic
        Args:
            - H: latent representation (B x S x E)
            - T: input temporal information
        Returns:
            - logits: predicted logits (B x S x 1)
        """
        # Dynamic RNN input for ignoring paddings
        H_packed = torch.nn.utils.rnn.pack_padded_sequence(
            input=H,
            lengths=T,
            batch_first=True,
            enforce_sorted=False
        )

        # 128 x 100 x 10
        H_o, H_t = self.dis_rnn(H_packed)

        # Pad RNN output back to sequence length
        H_o, T = torch.nn.utils.rnn.pad_packed_sequence(
            sequence=H_o,
            batch_first=True,
            padding_value=self.padding_value,
            total_length=self.max_seq_len
        )

        # 128 x 100
        logits = self.dis_linear(H_o).squeeze(-1)
        return logits


class TimeGAN(torch.nn.Module):
    """Implementation of TimeGAN (Yoon et al., 2019) using PyTorch\n
    Reference:
    - https://papers.nips.cc/paper/2019/hash/c9efe5f26cd17ba6216bbe2a7d26d490-Abstract.html
    - https://github.com/jsyoon0823/TimeGAN
    """

    def __init__(self, params):
        super(TimeGAN, self).__init__()
        self.device = params['device']
        self.feature_dim = params['feature_dim']
        self.Z_dim = params['Z_dim']
        self.hidden_dim = params['hidden_dim']
        self.num_layers = params['num_layers']
        self.padding_value = 0.0
        self.max_seq_len = params['max_seq_len']
        self.batch_size = params['batch_size']

        # Networks
        self.embedder = EmbeddingNetwork(self.feature_dim, self.hidden_dim, self.num_layers, self.padding_value,
                                         self.max_seq_len)
        self.recovery = RecoveryNetwork(self.feature_dim, self.hidden_dim, self.num_layers, self.padding_value,
                                        self.max_seq_len)
        self.supervisor = SupervisorNetwork(self.hidden_dim, self.num_layers, self.padding_value, self.max_seq_len)
        self.generator = GeneratorNetwork(self.Z_dim, self.hidden_dim, self.num_layers, self.padding_value,
                                          self.max_seq_len)
        self.discriminator = DiscriminatorNetwork(self.hidden_dim, self.num_layers, self.padding_value,
                                                  self.max_seq_len)

    def _recovery_forward(self, X, T):
        """The embedding network forward pass and the embedder network loss
        Args:
            - X: the original input features
            - T: the temporal information
        Returns:
            - E_loss: the reconstruction loss
            - X_tilde: the reconstructed features
        """
        # Forward Pass
        H = self.embedder(X, T)
        X_tilde = self.recovery(H, T)

        # For Joint training
        H_hat_supervise = self.supervisor(H, T)
        G_loss_S = torch.nn.functional.mse_loss(
            H_hat_supervise[:, :-1, :],
            H[:, 1:, :]
        )  # Teacher forcing next output

        # Reconstruction Loss
        E_loss_T0 = torch.nn.functional.mse_loss(X_tilde, X)
        E_loss0 = torch.sqrt(E_loss_T0) * 10
        E_loss = E_loss0 + 0.1 * G_loss_S
        return E_loss, E_loss0, E_loss_T0

    def _supervisor_forward(self, X, T):
        """The supervisor training forward pass
        Args:
            - X: the original feature input
        Returns:
            - S_loss: the supervisor's loss
        """
        # Supervision Forward Pass
        H = self.embedder(X, T)
        H_hat_supervise = self.supervisor(H, T)

        # Supervised loss
        S_loss = torch.nn.functional.mse_loss(H_hat_supervise[:, :-1, :], H[:, 1:, :])  # Teacher forcing next output
        return S_loss

    def _discriminator_forward(self, X, T, Z, gamma=1):
        """The discriminator forward pass and adversarial loss
        Args:
            - X: the input features
            - T: the temporal information
            - Z: the input noise
        Returns:
            - D_loss: the adversarial loss
        """
        # Real
        H = self.embedder(X, T).detach()

        # Generator
        E_hat = self.generator(Z, T).detach()
        H_hat = self.supervisor(E_hat, T).detach()

        # Forward Pass
        Y_real = self.discriminator(H, T)  # Encoded original data
        Y_fake = self.discriminator(H_hat, T)  # Output of generator + supervisor
        Y_fake_e = self.discriminator(E_hat, T)  # Output of generator

        smooth_label_real = torch.ones_like(Y_real)  # * 0.9
        D_loss_real = torch.nn.functional.binary_cross_entropy_with_logits(Y_real, smooth_label_real)
        D_loss_fake = torch.nn.functional.binary_cross_entropy_with_logits(Y_fake, torch.zeros_like(Y_fake))
        D_loss_fake_e = torch.nn.functional.binary_cross_entropy_with_logits(Y_fake_e, torch.zeros_like(Y_fake_e))

        D_loss = D_loss_real + D_loss_fake + gamma * D_loss_fake_e

        return D_loss

    def _generator_forward(self, X, T, Z, gamma=1):
        """The generator forward pass
        Args:
            - X: the original feature input
            - T: the temporal information
            - Z: the noise for generator input
        Returns:
            - G_loss: the generator's loss
        """

        # Supervisor Forward Pass
        H = self.embedder(X, T)
        H_hat_supervise = self.supervisor(H, T)

        # Generator Forward Pass
        E_hat = self.generator(Z, T)
        H_hat = self.supervisor(E_hat, T)
        # Synthetic data generated
        X_hat = self.recovery(H_hat, T)

        # Generator Loss
        # 1. Adversarial loss
        Y_fake = self.discriminator(H_hat, T)  # Output of supervisor
        Y_fake_e = self.discriminator(E_hat, T)  # Output of generator
        # Using max E[log(D(G(z)))]
        smooth_labels_L = torch.ones_like(Y_fake)  # * 0.9  # torch.tensor(np.random.uniform(0.7, 0.9, Y_fake.size()))
        smooth_labels_U = torch.ones_like(Y_fake)  # * 0.9
        G_loss_U = torch.nn.functional.binary_cross_entropy_with_logits(Y_fake, smooth_labels_U)
        G_loss_U_e = torch.nn.functional.binary_cross_entropy_with_logits(Y_fake_e, smooth_labels_L)

        # 2. Supervised loss
        G_loss_S = torch.nn.functional.mse_loss(H_hat_supervise[:, :-1, :], H[:, 1:, :])  # Teacher forcing next output

        # 3. Two Momments
        G_loss_V1 = torch.mean(torch.abs(
            torch.sqrt(X_hat.var(dim=0, unbiased=False) + 1e-6) - torch.sqrt(X.var(dim=0, unbiased=False) + 1e-6)))
        G_loss_V2 = torch.mean(torch.abs((X_hat.mean(dim=0)) - (X.mean(dim=0))))

        G_loss_V = G_loss_V1 + G_loss_V2

        # 4. Summation
        G_loss = G_loss_U + gamma * G_loss_U_e + 100 * torch.sqrt(G_loss_S) + 100 * G_loss_V

        return G_loss

    def _inference(self, Z, T):
        """Inference for generating synthetic data
        Args:
            - Z: the input noise
            - T: the temporal information
        Returns:
            - X_hat: the generated data
        """
        # Generator Forward Pass
        E_hat = self.generator(Z, T)
        H_hat = self.supervisor(E_hat, T)

        # Synthetic data generated
        X_hat = self.recovery(H_hat, T)
        return X_hat

    def forward(self, X, T, Z, obj, gamma=1):
        """
        Args:
            - X: the input features (B, H, F)
            - T: the temporal information (B)
            - Z: the sampled noise (B, H, Z)
            - obj: the network to be trained (`autoencoder`, `supervisor`, `generator`, `discriminator`)
            - gamma: loss hyperparameter
        Returns:
            - loss: The loss for the forward pass
            - X_hat: The generated data
        """
        if obj != "inference":
            if X is None:
                raise ValueError("`X` should be given")

            X = torch.FloatTensor(X)
            X = X.to(self.device)

        if Z is not None:
            Z = torch.FloatTensor(Z)
            Z = Z.to(self.device)

        if obj == "autoencoder":
            # Embedder & Recovery
            loss = self._recovery_forward(X, T)

        elif obj == "supervisor":
            # Supervisor
            loss = self._supervisor_forward(X, T)

        elif obj == "generator":
            if Z is None:
                raise ValueError("`Z` is not given")

            # Generator
            loss = self._generator_forward(X, T, Z)

        elif obj == "discriminator":
            if Z is None:
                raise ValueError("`Z` is not given")

            # Discriminator
            loss = self._discriminator_forward(X, T, Z)

            return loss

        elif obj == "inference":

            X_hat = self._inference(Z, T)
            X_hat = X_hat.cpu().detach()

            return X_hat

        else:
            raise ValueError("`obj` should be either `autoencoder`, `supervisor`, `generator`, or `discriminator`")

        return loss
