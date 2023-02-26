import torch.nn as nn

from cost_utils import *

class SinusDiscriminator(nn.Module):
    def __init__(self, args):
        super(SinusDiscriminator, self).__init__()
        # Basic parameters
        self.device = args["device"]

        self.hidden_dim = args["hidden_dim"]
        self.dis_rnn_hidden_dim = args["dis_rnn_hidden_dim"]
        self.dis_rnn_num_layers = args["dis_rnn_num_layers"]
        self.feature_dim = args["feature_dim"]
        self.J_dim = args["J_dim"]
        self.max_seq_len = args["max_seq_len"]
        self.rnn_type = args["rnn_type"] # GRU or LSTM
        self.use_bn = args["use_bn"]

        # Discriminator Architecture

        self.dis_cnn = list()
        self.dis_cnn.append(nn.Conv1d(in_channels=self.feature_dim,
                                      out_channels=self.hidden_dim,
                                      kernel_size=5,
                                      stride=1,))
        if self.use_bn:
            self.dis_cnn.append(nn.BatchNorm1d(self.hidden_dim))
        self.dis_cnn.append(nn.LeakyReLU())
        self.dis_cnn.append(nn.Conv1d(in_channels=self.hidden_dim,
                                      out_channels=self.hidden_dim*2,
                                      kernel_size=5,
                                      stride=1,))
        if self.use_bn:
            self.dis_cnn.append(nn.BatchNorm1d(self.hidden_dim*2))
        self.dis_cnn.append(nn.LeakyReLU())
        self.dis_cnn = nn.Sequential(*self.dis_cnn)

        self.dis_rnn_2 = None
        input_rnn_dim = self.hidden_dim * 2
        if self.dis_rnn_num_layers-1 > 0:
            self.dis_rnn_2 = nn.GRU(input_size=input_rnn_dim,
                   hidden_size=self.dis_rnn_hidden_dim,
                   num_layers=self.dis_rnn_num_layers-1,
                   batch_first=True)
            input_rnn_dim = self.dis_rnn_hidden_dim

        self.dis_rnn = nn.GRU(input_size=input_rnn_dim,
                              hidden_size=self.feature_dim,
                              num_layers=1,
                              batch_first=True)

        """if self.rnn_type == "GRU":
            self.dis_rnn = nn.GRU(input_size=self.hidden_dim*2,
                                  hidden_size=self.feature_dim,
                                  num_layers=self.dis_rnn_num_layers,
                                  batch_first=True)
        elif self.rnn_type == "LSTM":
            self.dis_rnn = nn.LSTM(input_size=self.hidden_dim*2,
                                  hidden_size=self.feature_dim,
                                  num_layers=self.dis_rnn_num_layers,
                                  batch_first=True)
        else:
            raise NotImplementedError"""

    def forward(self, x):
        # x: B x S x F
        x = x.permute(0, 2, 1) # B x F x S
        x = self.dis_cnn(x)
        x = x.permute(0, 2, 1) # B x S x F
        if self.dis_rnn_2 is not None:
            H, _ = self.dis_rnn_2(x)
            x = H
        H, _ = self.dis_rnn(x)

        #logits = torch.sigmoid(H)
        return H

class SinusGenerator(nn.Module):
    def __init__(self, args):
        super(SinusGenerator, self).__init__()
        # Basic parameters
        self.device = args["device"]
        self.Z_dim = args["Z_dim"]
        self.hidden_dim = args["hidden_dim"]
        self.num_hidden_layers = args["num_hidden_layers"]
        self.gen_rnn_hidden_dim = args["gen_rnn_hidden_dim"]
        self.gen_rnn_num_layers = args["gen_rnn_num_layers"]
        #self.use_batch_norm = args["use_batch_norm"]
        self.feature_dim = args["feature_dim"]
        self.max_seq_len = args["max_seq_len"]
        self.rnn_type = args["rnn_type"]

        # Generator Architecture
        if self.rnn_type == "GRU":
            self.gen_rnn = nn.GRU(input_size=self.Z_dim,
                                 hidden_size=self.gen_rnn_hidden_dim,
                                 num_layers=self.gen_rnn_num_layers,
                                 batch_first=True)
        elif self.rnn_type == "LSTM":
            self.gen_rnn = nn.LSTM(input_size=self.Z_dim,
                                 hidden_size=self.gen_rnn_hidden_dim,
                                 num_layers=self.gen_rnn_num_layers,
                                 batch_first=True)
        else:
            raise NotImplementedError

        """self.gen_FC = nn.Sequential(
            nn.Linear(self.gen_rnn_hidden_dim, self.hidden_dim),
            #nn.BatchNorm1d(self.Z_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.feature_dim),
            nn.Sigmoid()
        )"""

        input_hidden = self.gen_rnn_hidden_dim
        self.gen_FC = list()
        for _ in range(self.num_hidden_layers-1):
            self.gen_FC.append(nn.Linear(input_hidden, self.hidden_dim))
            #if self.use_batch_norm:
            #    self.gen_FC.append(nn.BatchNorm1d(self.hidden_dim))
            self.gen_FC.append(nn.ReLU())
            input_hidden = self.hidden_dim

        self.gen_FC.append(nn.Linear(input_hidden, self.feature_dim))
        self.gen_FC.append(nn.Sigmoid())


        self.gen_FC = nn.Sequential(*self.gen_FC)

    def forward(self, z):
        # (B x S x Z)
        H, H_t = self.gen_rnn(z)
        # B x F x S
        H = H.squeeze(-1)
        # B x F x S
        out = self.gen_FC(H)
        return out



class SolitonDiscriminator(nn.Module):
    def __init__(self, args):
        super(SolitonDiscriminator, self).__init__()
        # Basic parameters
        self.device = args["device"]
        self.batch_size = args["batch_size"]

        self.hidden_dim = args["hidden_dim"]
        self.dis_rnn_hidden_dim = args["dis_rnn_hidden_dim"]
        self.dis_rnn_num_layers = args["dis_rnn_num_layers"]
        self.feature_dim = args["feature_dim"]
        self.J_dim = args["J_dim"]
        self.max_seq_len = args["max_seq_len"]
        self.rnn_type = args["rnn_type"] # GRU or LSTM
        self.use_bn = args["use_bn"]

        # Discriminator Architecture
        # input = (B, S, D), e.g. (32, 100, 25)

        self.dis_cnn = list()
        self.dis_cnn.append(nn.Conv1d(in_channels = 1,
                                      out_channels=self.hidden_dim,
                                      kernel_size=5,
                                      stride=1,
                                      padding=2))
        if self.use_bn:
            self.dis_cnn.append(nn.BatchNorm1d(self.hidden_dim))
        self.dis_cnn.append(nn.LeakyReLU())
        self.dis_cnn.append(nn.Conv1d(in_channels =self.hidden_dim,
                                      out_channels=self.hidden_dim * 2,
                                      kernel_size=5,
                                      stride=1,
                                      padding=2))
        if self.use_bn:
            self.dis_cnn.append(nn.BatchNorm1d(self.hidden_dim * 2))
        self.dis_cnn.append(nn.LeakyReLU())
        self.dis_cnn = nn.Sequential(*self.dis_cnn)
        #input_dim = self.hidden_dim * 2
        self.dis_rnn = nn.GRU(input_size = self.hidden_dim * 2 * self.feature_dim,
                                hidden_size=self.dis_rnn_hidden_dim,
                                num_layers=1,
                                batch_first=True)

        self.dis_rnn_2 = nn.GRU(input_size=self.dis_rnn_hidden_dim,
                              hidden_size=self.J_dim,
                              num_layers=1,
                              batch_first=True)

    def forward(self, x):
        # (B x S x D)
        #print(f"input shape: {x.shape}")
        x = x.view(self.batch_size * self.max_seq_len, 1, -1)
        #print(f"input shape after reshape: {x.shape}")
        # (B*S x D)
        x = self.dis_cnn(x)
        #print(f"after cnn shape: {x.shape}")
        # (B*S x D)
        x = x.view(self.batch_size, self.max_seq_len, -1)
        #print(f"after view shape: {x.shape}")
        # (B x S x D)
        x, _ = self.dis_rnn(x)
        #print(f"after rnn shape: {x.shape}")
        # (B x S x dis_rnn_hidden_dim)
        x, _ = self.dis_rnn_2(x)
        #print(f"after rnn_2 shape: {x.shape}")
        # (B x S x J)
        return x

class SolitonGenerator(nn.Module):
    def __init__(self, args):
        super(SolitonGenerator, self).__init__()
        # Basic parameters
        self.device = args["device"]
        self.batch_size = args["batch_size"]
        self.Z_dim = args["Z_dim"]

        self.hidden_dim = args["hidden_dim"]
        self.gen_rnn_hidden_dim = args["gen_rnn_hidden_dim"]
        self.gen_rnn_num_layers = args["gen_rnn_num_layers"]
        self.num_hidden_layers = args["num_hidden_layers"]
        self.feature_dim = args["feature_dim"]
        self.max_seq_len = args["max_seq_len"]
        self.rnn_type = args["rnn_type"] # GRU or LSTM
        self.use_bn = args["use_bn"]

        # Generator Architecture
        # input = (B, S, D), e.g. (32, 100, 25)

        self.gen_rnn = nn.GRU(input_size=self.Z_dim,
                                hidden_size=self.gen_rnn_hidden_dim,
                                num_layers=1,
                                batch_first=True)
        # For FC layers
        input_hidden = self.gen_rnn_hidden_dim
        if self.gen_rnn_num_layers > 1:
            input_hidden = self.gen_rnn_hidden_dim * 2
            self.gen_rnn2 = nn.GRU(input_size=self.gen_rnn_hidden_dim,
                                hidden_size=self.gen_rnn_hidden_dim * 2,
                                num_layers=1,
                                batch_first=True)

        self.gen_FC = list()
        for _ in range(self.num_hidden_layers - 1):
            self.gen_FC.append(nn.Linear(input_hidden, self.hidden_dim))
            if self.use_batch_norm:
                self.gen_FC.append(nn.BatchNorm1d(self.hidden_dim))
            self.gen_FC.append(nn.LeakyReLU())
            input_hidden = self.hidden_dim

        self.gen_FC.append(nn.Linear(input_hidden, self.feature_dim))
        self.gen_FC.append(nn.Sigmoid())
        self.gen_FC = nn.Sequential(*self.gen_FC)

    def forward(self, z):
        # (B x S x Z)
        x, _ = self.gen_rnn(z)
        # (B x S x gen_rnn_hidden_dim)
        if self.gen_rnn_num_layers > 1:
            x, _ = self.gen_rnn2(x)
        # (B x S x gen_rnn_hidden_dim * 2)
        out = self.gen_FC(x)
        return out

class COTGAN(nn.Module):
    def __init__(self, args):
        super(COTGAN, self).__init__()
        self.Z_dim = args["Z_dim"]
        self.max_seq_len = args["max_seq_len"]
        self.device = args["device"]
        self.batch_size = args["batch_size"]
        self.sinkhorn_eps = args["sinkhorn_eps"]
        self.sinkhorn_l = args["sinkhorn_l"]
        self.reg_lam = args["reg_lam"]
        if "sinus" in args["dataset"]:
            self.generator = SinusGenerator(args=args)
            self.discriminator_h = SinusDiscriminator(args=args)
            self.discriminator_m = SinusDiscriminator(args=args)
        elif "soliton" in args["dataset"]:
            self.generator = SolitonGenerator(args=args)
            self.discriminator_h = SolitonDiscriminator(args=args)
            self.discriminator_m = SolitonDiscriminator(args=args)

    def __discriminator_loss(self, real_data, real_data_p, z1, z2):
        fake_data = self.generator(z1).detach()
        fake_data_p = self.generator(z2).detach()

        # h_real = self.discriminator_h(real_data)
        h_fake = self.discriminator_h(fake_data)

        m_real = self.discriminator_m(real_data)
        m_fake = self.discriminator_m(fake_data)

        h_real_p = self.discriminator_h(real_data_p)
        h_fake_p = self.discriminator_h(fake_data_p)

        m_real_p = self.discriminator_m(real_data_p)
        # m_fake_p = self.discriminator_m(fake_data_p)

        mixed_sinkhorn_loss = compute_mixed_sinkhorn_loss(real_data, fake_data, m_real, m_fake, h_fake,
                                                          real_data_p, fake_data_p, m_real_p, h_real_p, h_fake_p,
                                                          self.sinkhorn_eps, self.sinkhorn_l)

        pm = scale_invariante_martingale_regularization(m_real, reg_lam=self.reg_lam)

        return -mixed_sinkhorn_loss + pm

    def __generator_loss(self, real_data, real_data_p, z1, z2):
        fake_data = self.generator(z1)
        fake_data_p = self.generator(z2)

        # h_real = self.discriminator_h(real_data)
        h_fake = self.discriminator_h(fake_data)

        m_real = self.discriminator_m(real_data)
        m_fake = self.discriminator_m(fake_data)

        h_real_p = self.discriminator_h(real_data_p)
        h_fake_p = self.discriminator_h(fake_data_p)

        m_real_p = self.discriminator_m(real_data_p)
        # m_fake_p = self.discriminator_m(fake_data_p)

        mixed_sinkhorn_loss = compute_mixed_sinkhorn_loss(real_data, fake_data, m_real, m_fake, h_fake,
                                                          real_data_p, fake_data_p, m_real_p, h_real_p, h_fake_p,
                                                          self.sinkhorn_eps, self.sinkhorn_l)
        return mixed_sinkhorn_loss

    def generate(self, N):
        Z = torch.randn(N, self.max_seq_len, self.Z_dim, device=self.device)
        X_hat = self.generator(Z)
        return X_hat.cpu().detach()
    def forward(self, Z, X, obj="inference"):
        if obj == "generator":
            return self.__generator_loss(real_data=X[:self.batch_size], real_data_p=X[self.batch_size:],
                                         z1=Z[:self.batch_size], z2=Z[self.batch_size:])
        elif obj == "discriminator":
            return self.__discriminator_loss(real_data=X[:self.batch_size], real_data_p=X[self.batch_size:],
                                         z1=Z[:self.batch_size], z2=Z[self.batch_size:])
        elif obj == "inference":
            X_hat = self.generator(Z)
            return X_hat.detach()
        else:
            raise ValueError("Invalid obj description. Must be in (generator, discriminator, inference)")

######### TIMEGAN

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
            #dropout=0.5
            #bidirectional=True
        )

        self.emb_linear = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.emb_sigmoid = nn.Sigmoid()
    def forward(self, X):
        H_o, H_t = self.emb_rnn(X)
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

    def forward(self, H):
        H_o, H_t = self.rec_rnn(H)
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
            #bidirectional=True
        )
        self.sup_linear = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
        self.sup_sigmoid = torch.nn.Sigmoid()
    def forward(self, H):
        H_o, H_t = self.sup_rnn(H)
        logits = self.sup_linear(H_o)
        H_hat = self.sup_sigmoid(logits)
        return H_hat
class GeneratorNetwork(torch.nn.Module):
    def __init__(self, Z_dim, hidden_dim, num_layers, padding_value, max_seq_len):
        super(GeneratorNetwork, self).__init__()
        self.Z_dim = Z_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.padding_value = padding_value
        self.max_seq_len = max_seq_len

        # Generator Architecture
        self.gen_rnn = torch.nn.GRU(
            input_size=self.Z_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
        )
        self.gen_linear = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
        self.gen_sigmoid = torch.nn.Sigmoid()

    def forward(self, Z):
        H_o, H_t = self.gen_rnn(Z)
        logits = self.gen_linear(H_o)
        H = self.gen_sigmoid(logits)
        return H
class DiscriminatorNetwork(torch.nn.Module):
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
            bidirectional=False
        )
        self.dis_linear = torch.nn.Linear(self.hidden_dim, 1)

    def forward(self, H):
        H_o, H_t = self.dis_rnn(H)
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

    def _recovery_forward(self, X):
        """The embedding network forward pass and the embedder network loss
        Args:
            - X: the original input features
            - T: the temporal information
        Returns:
            - E_loss: the reconstruction loss
            - X_tilde: the reconstructed features
        """
        # Forward Pass
        H = self.embedder(X)
        X_tilde = self.recovery(H)

        # For Joint training
        H_hat_supervise = self.supervisor(H)
        G_loss_S = torch.nn.functional.mse_loss(
            H_hat_supervise[:, :-1, :],
            H[:, 1:, :]
        )  # Teacher forcing next output

        # Reconstruction Loss
        E_loss_T0 = torch.nn.functional.mse_loss(X_tilde, X)
        E_loss0 = torch.sqrt(E_loss_T0) * 10
        E_loss = E_loss0 + 0.1 * G_loss_S
        return E_loss, E_loss0, E_loss_T0

    def _supervisor_forward(self, X):
        """The supervisor training forward pass
        Args:
            - X: the original feature input
        Returns:
            - S_loss: the supervisor's loss
        """
        # Supervision Forward Pass
        H = self.embedder(X)
        H_hat_supervise = self.supervisor(H)

        # Supervised loss
        S_loss = torch.nn.functional.mse_loss(H_hat_supervise[:, :-1, :], H[:, 1:, :])  # Teacher forcing next output
        return S_loss

    def _discriminator_forward(self, X, Z, gamma=1):
        """The discriminator forward pass and adversarial loss
        Args:
            - X: the input features
            - T: the temporal information
            - Z: the input noise
        Returns:
            - D_loss: the adversarial loss
        """
        # Real
        H = self.embedder(X).detach()

        # Generator
        E_hat = self.generator(Z).detach()
        H_hat = self.supervisor(E_hat).detach()

        # Forward Pass
        Y_real = self.discriminator(H)  # Encoded original data
        Y_fake = self.discriminator(H_hat)  # Output of generator + supervisor
        Y_fake_e = self.discriminator(E_hat)  # Output of generator

        smooth_label_real = torch.ones_like(Y_real)  # * 0.9
        D_loss_real = torch.nn.functional.binary_cross_entropy_with_logits(Y_real, smooth_label_real)
        D_loss_fake = torch.nn.functional.binary_cross_entropy_with_logits(Y_fake, torch.zeros_like(Y_fake))
        D_loss_fake_e = torch.nn.functional.binary_cross_entropy_with_logits(Y_fake_e, torch.zeros_like(Y_fake_e))

        D_loss = D_loss_real + D_loss_fake + gamma * D_loss_fake_e

        return D_loss

    def _generator_forward(self, X, Z, gamma=1):
        """The generator forward pass
        Args:
            - X: the original feature input
            - T: the temporal information
            - Z: the noise for generator input
        Returns:
            - G_loss: the generator's loss
        """
        # Supervisor Forward Pass
        H = self.embedder(X)
        H_hat_supervise = self.supervisor(H)

        # Generator Forward Pass
        E_hat = self.generator(Z)
        H_hat = self.supervisor(E_hat)

        # Synthetic data generated
        X_hat = self.recovery(H_hat)

        # Generator Loss
        # 1. Adversarial loss
        Y_fake = self.discriminator(H_hat)  # Output of supervisor
        Y_fake_e = self.discriminator(E_hat)  # Output of generator

        # Using max E[log(D(G(z)))]
        smooth_labels_L = torch.ones_like(Y_fake)  # * 0.9  # torch.tensor(np.random.uniform(0.7, 0.9, Y_fake.size()))
        smooth_labels_U = torch.ones_like(Y_fake_e)  # * 0.9
        G_loss_U = torch.nn.functional.binary_cross_entropy_with_logits(Y_fake, smooth_labels_U)
        G_loss_U_e = torch.nn.functional.binary_cross_entropy_with_logits(Y_fake_e, smooth_labels_L)

        # 2. Supervised loss
        G_loss_S = torch.nn.functional.mse_loss(H_hat_supervise[:, :-1, :], H[:, 1:, :])  # Teacher forcing next output

        # 3. Two Momments
        G_loss_V1 = torch.mean(torch.abs(torch.sqrt(X_hat.var(dim=0, unbiased=False) + 1e-6)
                                         - torch.sqrt(X.var(dim=0, unbiased=False) + 1e-6)))
        G_loss_V2 = torch.mean(torch.abs((X_hat.mean(dim=0)) - (X.mean(dim=0))))

        G_loss_V = G_loss_V1 + G_loss_V2

        # 4. Summation
        G_loss = G_loss_U + gamma * G_loss_U_e + 100 * torch.sqrt(G_loss_S) + 100 * G_loss_V

        return G_loss

    def _inference(self, Z):
        """Inference for generating synthetic data
        Args:
            - Z: the input noise
            - T: the temporal information
        Returns:
            - X_hat: the generated data
        """
        # Generator Forward Pass
        E_hat = self.generator(Z)
        H_hat = self.supervisor(E_hat)

        # Synthetic data generated
        X_hat = self.recovery(H_hat)
        return X_hat

    def generate(self, N):
        Z = torch.randn(N, self.max_seq_len, self.Z_dim, device=self.device)
        X_hat = self._inference(Z)
        return X_hat.detach()

    def forward(self, X, Z, obj, gamma=1):
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

            #X = torch.FloatTensor(X)
            X = X.to(self.device)

        if Z is not None:
            Z = torch.FloatTensor(Z)
            Z = Z.to(self.device)

        if obj == "autoencoder":
            # Embedder & Recovery
            loss = self._recovery_forward(X)

        elif obj == "supervisor":
            # Supervisor
            loss = self._supervisor_forward(X)

        elif obj == "generator":
            if Z is None:
                raise ValueError("`Z` is not given")

            # Generator
            loss = self._generator_forward(X, Z)

        elif obj == "discriminator":
            if Z is None:
                raise ValueError("`Z` is not given")

            # Discriminator
            loss = self._discriminator_forward(X, Z)

            return loss

        elif obj == "inference":
            X_hat = self._inference(Z)
            X_hat = X_hat.detach()

            return X_hat

        else:
            raise ValueError("`obj` should be either `autoencoder`, `supervisor`, `generator`, or `discriminator`")

        return loss






if __name__ == "__main__":

    x = torch.randn(16, 100, 5)
    z = torch.randn(16, 100, 100)

    args = {
        "gen_rnn_hidden_dim": 4,
        "gen_rnn_num_layers": 2,
        "dis_rnn_hidden_dim": 4,
        "dis_rnn_num_layers": 2,
        "num_hidden_layers": 2,
        "use_batch_norm": False,
        "Z_dim": z.size(-1),
        "max_seq_len": x.size(1),
        "hidden_dim": 10,
        "feature_dim": x.size(-1),
        "device": "cpu",
        "scaling_coef": 1,
        "sinkhorn_eps": 0.1,
        "sinkhorn_l": 10,
        "reg_lam": 0.1

    }

    gen = SinusGenerator(args)
    dis = SinusDiscriminator(args)
    model = COTGAN(args)
    print(gen(z).size())
    #print(dis(x))
    #print("Generator loss:", model(z, z, x, x, "generator"))
    #print("Discriminator loss:", model(z, z, x, x, "discriminator"))


