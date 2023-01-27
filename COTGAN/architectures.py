import torch.nn as nn

from cost_utils import *

class SinusDiscriminator(nn.Module):
    def __init__(self, args):
        super(SinusDiscriminator, self).__init__()
        # Basic parameters
        self.device = args["device"]

        self.hidden_dim = args["hidden_dim"]
        #self.dis_rnn_hidden_dim = args["dis_rnn_hidden_dim"]
        # self.dis_rnn_num_layers = args["dis_rnn_num_layers"]
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
                                      stride=1,))
        self.dis_cnn.append(nn.BatchNorm1d(self.hidden_dim))
        self.dis_cnn.append(nn.ReLU())
        self.dis_cnn.append(nn.Conv1d(in_channels=self.hidden_dim,
                                      out_channels=self.hidden_dim*2,
                                      kernel_size=5,
                                      stride=1,))
        self.dis_cnn.append(nn.BatchNorm1d(self.hidden_dim*2))
        self.dis_cnn.append(nn.ReLU())
        self.dis_cnn = nn.Sequential(*self.dis_cnn)


        self.dis_rnn = nn.GRU(input_size=self.hidden_dim*2,
                              hidden_size=self.feature_dim,
                              num_layers=1,
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
        self.gen_rnn_hidden_dim = args["gen_rnn_hidden_dim"]
        self.gen_rnn_num_layers = args["gen_rnn_num_layers"]
        self.feature_dim = args["feature_dim"]
        self.max_seq_len = args["max_seq_len"]

        # Generator Architecture
        self.gen_rnn = nn.GRU(input_size=self.Z_dim,
                             hidden_size=self.gen_rnn_hidden_dim,
                             num_layers=self.gen_rnn_num_layers,
                             batch_first=True)

        self.gen_FC = nn.Sequential(
            nn.Linear(self.gen_rnn_hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.feature_dim),
            nn.Sigmoid()
        )

    def forward(self, z):
        # (B x S x Z)
        H, H_t = self.gen_rnn(z)
        # B x F
        H = H.squeeze(-1)
        # B x F
        logits = self.gen_FC(H)

        return logits


class COTGAN(nn.Module):
    def __init__(self, args):
        super(COTGAN, self).__init__()
        # TODO (Check that everything is on the correct device: cpu or cuda)

        self.scaling_coef = args["scaling_coef"]
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
        "gen_rnn_hidden_dim": 50,
        "gen_rnn_num_layers": 2,
        "dis_rnn_hidden_dim": 10,
        "dis_rnn_num_layers": 2,
        "Z_dim": z.size(-1),
        "max_seq_len": x.size(1),
        "hidden_dim": 3,
        "feature_dim": x.size(-1),
        "device": "cpu"
    }

    gen = SinusGenerator(args)
    dis = SinusDiscriminator(args)
    print(gen(z).size())
    print(dis(x).size())
    model = COTGAN(args)
    print("Generator loss:", model(x, x, z, z, "generator"))
    print("Discriminator loss:", model(x, x, z, z, "discriminator"))


