import torch.nn as nn

from cost_utils import *

class SinusDiscriminator(nn.Module):
    def __init__(self, args):
        super(SinusDiscriminator, self).__init__()
        # Basic parameters
        self.device = args["device"]

        self.hidden_dim = args["hidden_dim"]
        #self.dis_rnn_hidden_dim = args["dis_rnn_hidden_dim"]
        self.dis_rnn_num_layers = args["dis_rnn_num_layers"]
        self.feature_dim = args["feature_dim"]
        self.max_seq_len = args["max_seq_len"]

        # Discriminator Architecture
        """self.dis_cnn = nn.Sequential(
            nn.Conv1d(in_channels=self.feature_dim,
                      out_channels=self.hidden_dim,
                      kernel_size=5,
                      stride=1,),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.hidden_dim,
                      out_channels=self.hidden_dim*2,
                      kernel_size=5,
                      stride=1,),
            nn.ReLU(),
        )"""

        self.dis_cnn = list()
        self.dis_cnn.append(nn.Conv1d(in_channels=self.feature_dim,
                                      out_channels=self.hidden_dim,
                                      kernel_size=5,
                                      stride=2,))
        self.dis_cnn.append(nn.BatchNorm1d(self.hidden_dim))
        self.dis_cnn.append(nn.LeakyReLU())
        self.dis_cnn.append(nn.Conv1d(in_channels=self.hidden_dim,
                                      out_channels=self.hidden_dim*2,
                                      kernel_size=5,
                                      stride=2,))
        self.dis_cnn.append(nn.BatchNorm1d(self.hidden_dim*2))
        self.dis_cnn.append(nn.LeakyReLU())
        self.dis_cnn = nn.Sequential(*self.dis_cnn)


        self.dis_rnn = nn.GRU(input_size=self.hidden_dim*2,
                              hidden_size=self.feature_dim,
                              num_layers=self.dis_rnn_num_layers,
                              batch_first=True)

    def forward(self, x):
        # x: B x S x F
        x = x.permute(0, 2, 1) # B x F x S
        x = self.dis_cnn(x)
        x = x.permute(0, 2, 1) # B x S x F
        H, H_t = self.dis_rnn(x)
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

        # Generator Architecture
        self.gen_rnn = nn.GRU(input_size=self.Z_dim,
                             hidden_size=self.gen_rnn_hidden_dim,
                             num_layers=self.gen_rnn_num_layers,
                             batch_first=True)

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


class COTGAN(nn.Module):
    def __init__(self, args):
        super(COTGAN, self).__init__()

        #self.scaling_coef = args["scaling_coef"]
        self.sinkhorn_eps = args["sinkhorn_eps"]
        self.sinkhorn_l = args["sinkhorn_l"]
        self.reg_lam = args["reg_lam"]

        self.generator = SinusGenerator(args=args)
        self.discriminator_h = SinusDiscriminator(args=args)
        self.discriminator_m = SinusDiscriminator(args=args)


    def __discriminator_loss(self, real_data, real_data_p, z1, z2):
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

    def forward(self, z1, z2=None, x1=None, x2=None, obj="inference"):
        if obj == "generator":
            return self.__generator_loss(x1, x2, z1, z2)
        elif obj == "discriminator":
            return self.__discriminator_loss(x1, x2, z1, z2)
        elif obj == "inference":
            X_hat = self.generator(z1)
            return X_hat.cpu().detach()
        else:
            raise ValueError("Invalid obj description. Must be in (generator, discriminator, inference)")


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


