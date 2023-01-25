import torch
import torchvision.transforms as transforms
import torch.nn as nn

from cost_utils import *

class sinusDiscriminator(nn.Module):
    def __init__(self, args):
        super(sinusDiscriminator, self).__init__()
        # Basic parameters
        self.device = args["device"]

        self.hidden_dim = args["hidden_dim"]
        self.rnn_hidden_dim = args["rnn_hidden_dim"]
        self.feature_dim = args["feature_dim"]
        self.num_layers = args["num_layers"]
        self.max_seq_len = args["max_seq_len"]

        # Discriminator Architecture
        self.dis_cnn = nn.Sequential(
            nn.Conv1d(  in_channels=self.feature_dim,
                        out_channels=self.hidden_dim,
                        kernel_size=5,
                        stride=1,),
            nn.ReLU(),
            nn.Conv1d(  in_channels=self.hidden_dim,
                        out_channels=self.hidden_dim,
                        kernel_size=5,
                        stride=1,),
            nn.ReLU(),
        )

        self.dis_rnn = nn.GRU(  input_size=self.hidden_dim,
                                hidden_size=self.feature_dim,
                                num_layers=self.num_layers,
                                batch_first=True)


    def forward(self, x):
        # x: B x S x F
        x = x.permute(0, 2, 1) # B x F x S
        x = self.dis_cnn(x)
        x = x.permute(0, 2, 1) # B x S x F
        H, H_t = self.dis_rnn(x)
        logits = torch.sigmoid(H)

        return logits

class simpleGenerator(nn.Module):
    def __init__(self,input_size,  args):
        super(simpleGenerator, self).__init__()
        # Basic parameters
        self.device = args["device"]
        self.Z_dim = args["Z_dim"]
        self.hidden_dim = args["hidden_dim"]
        self.rnn_hidden_dim = args["rnn_hidden_dim"]
        self.feature_dim = args["feature_dim"]
        self.num_layers = args["num_layers"]
        self.max_seq_len = args["max_seq_len"]

        # Generator Architecture
        self.gen_rnn = nn.GRU(input_size=input_size,
                             hidden_size=self.rnn_hidden_dim,
                             num_layers=self.num_layers,
                             batch_first=True)

        self.gen_FC = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.feature_dim)
        )

    def forward(self, x):
        # (B x S x Z)
        print(x.shape)
        H, H_t = self.gen_rnn(x)
        print("H.shape", H.shape)
        # B x F
        H = H.squeeze(-1)
        # B x F
        logits = self.gen_FC(H)

        return logits


class COTGAN(nn.Module):
    def __init__(self, args):
        super(COTGAN, self).__init__()
        # Basic parameters
        self.device = args["device"]
        self.Z_dim = args["Z_dim"]
        self.hidden_dim = args["hidden_dim"]
        self.rnn_hidden_dim = args["rnn_hidden_dim"]
        self.feature_dim = args["feature_dim"]
        self.num_layers = args["num_layers"]
        self.max_seq_len = args["max_seq_len"]

        self.generator = simpleGenerator(input_size=self.Z_dim, args=args)
        self.discriminator_h = sinusDiscriminator(args=args)
        self.discriminator_m = sinusDiscriminator(args=args)

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

        # TODO(add 'scaling_coef, 'sinkhorn_eps', 'sinkhorn_l' to arg dictionary)
        scaling_coef = 1
        sinkhorn_eps = 0.1
        sinkhorn_l = 5

        mixed_sinkhorn_loss = compute_mixed_sinkhorn_loss(real_data,   fake_data,   m_real,   m_fake,   h_fake, 10, 0.1,
                                                          real_data_p, fake_data_p, m_real_p, h_real_p, h_fake_p)

        '''
        def compute_mixed_sinkhorn_loss(f_real,   f_fake,   m_real,   m_fake,   h_fake, sinkhorn_eps, sinkhorn_l,
                                        f_real_p, f_fake_p, m_real_p, h_real_p, h_fake_p, scale=False):
            
            :param x and x'(f_real, f_real_p): real data of shape [batch size, time steps, features]
            :param y and y'(f_fake, f_fake_p): fake data of shape [batch size, time steps, features]
            :param h and h'(h_real, h_fake): h(y) of shape [batch size, time steps, J]
            :param m and m'(m_real and m_fake): M(x) of shape [batch size, time steps, J]
            :param scaling_coef: a scaling coefficient
            :param sinkhorn_eps: Sinkhorn parameter - epsilon
            :param sinkhorn_l: Sinkhorn parameter - the number of iterations
            :return: final Sinkhorn loss(and actual number of sinkhorn iterations for monitoring the training process)
        '''
        pm = scale_invariante_martingale_regularization(m_real, reg_lam=0.4) # TODO(add 'reg_lam' to arg dictionary)
        scaling_coef = 1 # TODO(add 'scaling_coef' to arg dictionary)

        return -mixed_sinkhorn_loss + pm



    def __generator_loss(self, z1, z2):
        return z1

    def forward(self, x1, x2, z1, z2, obj):
        if obj == "generator":
            loss = self.__generator_loss(z1, z2)
            return loss
        elif obj == "discriminator":
            loss = self.__discriminator_loss(x1, x2, z1, z2)
            return loss
        elif obj == "inference":
            X_hat = self.generator(z1, z2)
            return X_hat.cpu().detach()
        else:
            raise ValueError("Invalid obj description. Must be in (generator, discriminator, inference)")


if __name__ == "__main__":

    x = torch.randn(16, 100, 5)
    z = torch.randn(16, 100, 100)

    args = {
        "device": "cpu",
        "hidden_dim": 64,
        "rnn_hidden_dim": 64,
        "feature_dim": x.size(-1),
        "num_layers": 1,
        "max_seq_len": x.size(1),
        "Z_dim": z.size(-1)
    }

    input_size = 100
    gen = simpleGenerator(input_size, args)
    dis = sinusDiscriminator(args)
    print(gen(z).size())
    print(dis(x).size())
