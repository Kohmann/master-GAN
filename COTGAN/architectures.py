import torch.nn as nn
from cost_utils import *
ID = "COTGAN"
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

        self.dis_cnn = list()
        self.dis_cnn.append(nn.Conv1d(in_channels = 1,
                                      out_channels=self.hidden_dim,
                                      kernel_size=5,
                                      stride=1,
                                      padding="same"))
        if self.use_bn:
            self.dis_cnn.append(nn.BatchNorm1d(self.hidden_dim))
        self.dis_cnn.append(nn.LeakyReLU())
        self.dis_cnn.append(nn.Conv1d(in_channels =self.hidden_dim,
                                      out_channels=self.hidden_dim * 2,
                                      kernel_size=5,
                                      stride=1,
                                      padding="same"))
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

        self.gen_rnn = nn.GRU(input_size=self.Z_dim,
                                hidden_size=self.gen_rnn_hidden_dim,
                                num_layers=self.gen_rnn_num_layers-1 if self.gen_rnn_num_layers > 1 else 1,
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
            if self.use_bn:
                self.gen_FC.append(nn.LayerNorm(self.hidden_dim))
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
        self.Z_distribution = args["Z_distribution"]


        """if "sinus" in args["dataset"]:
            self.generator = SinusGenerator(args=args)
            self.discriminator_h = SinusDiscriminator(args=args)
            self.discriminator_m = SinusDiscriminator(args=args)
        elif "soliton" in args["dataset"]:
            self.generator = SolitonGenerator(args=args)
            self.discriminator_h = SolitonDiscriminator(args=args)
            self.discriminator_m = SolitonDiscriminator(args=args)
            """

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
        if self.Z_distribution == "uniform":
            Z = torch.rand(N, self.max_seq_len, self.Z_dim, device=self.device)*2.0 - 1.0 # Uniform in [-1, 1]
        elif self.Z_distribution == "normal":
            Z = torch.randn(N, self.max_seq_len, self.Z_dim, device=self.device)
        else:
            raise ValueError("Invalid Z_distribution. Must be in (uniform, normal)")
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


if __name__ == "__main__":

    x = torch.randn(16, 100, 3)
    z = torch.randn(16, 100, 100)


    args = {
        "n_samples":32*4,
        "gen_rnn_hidden_dim": 32,
        "gen_rnn_num_layers": 2,
        "dis_rnn_hidden_dim": 32,
        "dis_rnn_num_layers": 2,
        "num_hidden_layers": 2,
        "use_batch_norm": False,
        "Z_dim": 100,
        "max_seq_len": 100,
        "hidden_dim": 20,
        "feature_dim": 3,
        "device": "cpu",
        "scaling_coef": 1,
        "sinkhorn_eps": 0.8,
        "sinkhorn_l": 10,
        "reg_lam": 0.1,
        "rnn_type": "GRU",
        "batch_size": 16,
        "use_bn": True,
        "J_dim": 10,
        "dataset": "sinus",
        "alpha": 0.7,
        "noise": 0.,

    }
    from utils import DatasetSinus
    dataset = DatasetSinus(args["n_samples"], seq_len=args["max_seq_len"],
                            alpha=args["alpha"], noise=args["noise"], device=args["device"])

    gen = SolitonGenerator(args)
    dis = SolitonDiscriminator(args)
    model = COTGAN(args)
    print(f"GENERATOR: {gen}")
    print("------------------")
    print(f"DISCRIMINATOR: {dis}")
    print("------------------")

    x_real = dataset[:args["batch_size"]]
    z = torch.randn(args["batch_size"], args["max_seq_len"], args["Z_dim"], device=args["device"])
    x_fake = gen(z)
    print(f"X_FAKE: {x_fake.shape}")
    disc_fake = dis(x_fake)
    disc_real = dis(x_real)
    print(f"DISC_FAKE: {disc_fake.shape}")
    print(f"DISC_REAL: {disc_real.shape}")


