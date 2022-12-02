from scipy.stats import wasserstein_distance
import torch
import numpy as np
from utils import minmaxscaler, create_sin3


def compare_sin3_generation(fake_data, alpha, noise):
    sins1_fake = fake_data[:, :, 0]
    sins2_fake = fake_data[:, :, 1]
    sins3_real = fake_data[:, :, 2]

    sins3_fake = []
    for sin1, sin2 in zip(sins1_fake, sins2_fake):
        sins3_fake.append(create_sin3(sin1, sin2, alpha, noise))
    sins3_fake = torch.tensor(sins3_fake)
    sin3_fake_norm = minmaxscaler().fit_transform(sins3_fake)
    mse_error = ((sin3_fake_norm - sins3_real) ** 2).mean()
    return mse_error


from geomloss import SamplesLoss


def sinkhorn_distance(x, y, blur=0.01):
    """
    Sinkhorn distance between two samples x and y. Is an approximation of the Wasserstein distance.
    :param x: torch tensor: samples from the first distribution
    :param y: torch tensor: samples from the second distribution
    :return:  wasserstein distance between x and y

    Permisson is granted in LICENSE.txt
    """
    # Define a Sinkhorn (~Wasserstein) loss between sampled measures
    sinkhorn = SamplesLoss(loss="sinkhorn", p=2, blur=0.05, scaling=0.95)
    return sinkhorn(x, y).detach()


def MMD(x, y):
    """
    Maximum Mean Discrepancy between two samples x and y.
    :param x: torch tensor: samples from the first distribution
    :param y: torch tensor: samples from the second distribution
    :return: MMD between x and y

    Permisson is granted in LICENSE.txt
    """
    # Define a Sinkhorn (~Wasserstein) loss between sampled measures
    MMD = SamplesLoss(loss="gaussian", p=2, blur=0.05)
    return MMD(x, y).detach()


import torch


def sw_approx(mu: torch.Tensor, nu: torch.Tensor):
    """
    Central Limite Theorem approximation of the Sliced Wasserstein distance

    .. math::
        \widehat{\mathbf{S W}}_{2}^{2}\left(\mu_{d}, \nu_{d}\right)=\mathbf{W}_{2}^{2}\left\{\mathrm{~N}\left(0, \mathrm{~m}_{2}\left(\bar{\mu}_{d}\right)\right), \mathrm{N}\left(0, \mathrm{~m}_{2}\left(\bar{\nu}_{d}\right)\right)\right\}+(1 / d)\left\|\mathbf{m}_{\mu_{d}}-\mathbf{m}_{\nu_{d}}\right\|^{2}


    Parameters
    ----------
    mu : ndarray, shape (n_samples_a, dim)
        samples in the source domain
    nu : ndarray, shape (n_samples_b, dim)
        samples in the target domain

    Returns
    -------
    cost: float
        Sliced Wasserstein Cost
    """

    def m_2(X):
        return torch.mean(torch.pow(X, 2), dim=0)

    m_mu = torch.mean(mu, dim=0)
    m_nu = torch.mean(nu, dim=0)
    ### First lets compute d:=W2{N(0, m2(µd_bar)), N(0, m2(νd_bar))}
    # Centered version of mu and nu
    mu_bar = mu - m_mu
    nu_bar = nu - m_nu
    # Compute Wasserstein beetween two centered gaussians
    W = torch.pow(torch.sqrt(m_2(mu_bar)) - torch.sqrt(m_2(nu_bar)), 2)

    ## Compute the mean residuals
    d = mu.size(1)
    res = (1 / d) * torch.pow(m_mu - m_nu, 2)

    ## Approximation of the Sliced Wasserstein
    return torch.norm(W + res, p=2)

class EarlyStopping:
    def __init__(self, tolerance=5, min_delta=0.0):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, validation_loss):
        if (validation_loss - train_loss) > self.min_delta:
            self.counter += 1
            if self.counter >= self.tolerance:
                self.early_stop = True

class PredictionScoreModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, max_seq_len):
        super(PredictionScoreModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        self.model = torch.nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
            )
        self.linear = torch.nn.Linear(hidden_dim, output_dim)
        self.activation = torch.nn.Sigmoid()

    def forward(self, x, T):
        # add padding
        H_packed = torch.nn.utils.rnn.pack_padded_sequence(
            input=x,
            lengths=T,
            batch_first=True,
            enforce_sorted=False
        )

        # 128 x 100 x 10
        H_o, _ = self.model(H_packed)

        # Pad RNN output back to sequence length
        H_o, T = torch.nn.utils.rnn.pad_packed_sequence(
            sequence=H_o,
            batch_first=True,
            padding_value=0.0,
            total_length=self.max_seq_len
        )

        x = self.linear(H_o)
        return self.activation(x)

from tqdm import trange
def prediction_score(train, val, test, epochs=100, device="cpu", neptune_logger=None):
    """Prediction score. A LSTM model trained on Synthetic data and tested on real data.
    Args:
        train: training data of shape (n_samples, seq_len, n_features)
        val: validation data
        test: test data

    """

    # train the LSTM model
    model = PredictionScoreModel(input_dim=train.shape[2], hidden_dim=20, num_layers=2, output_dim=train.shape[2], max_seq_len=train.size(1))
    train_dataloader = torch.utils.data.DataLoader(train, batch_size=32, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=32, shuffle=True)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    train_loss = []
    val_loss = []
    early_stopping = EarlyStopping(tolerance=5, min_delta=1e-3)
    logger = trange(epochs, desc=f"Epoch: 0, Train loss: 0, Val loss: 0")
    for epoch in logger:
        for X_mb in train_dataloader:
            X_mb = X_mb.to(device)
            T = torch.tensor([train.shape[1]] * X_mb.size(0))
            optimizer.zero_grad()
            y_pred = model(X_mb, T)
            loss = criterion(y_pred[:, :-1, :], X_mb[:, 1:, :])
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

        for X_mb in val_dataloader:
            X_mb = X_mb.to(device)
            T = torch.tensor([train.shape[1]] * X_mb.size(0))
            y_pred = model(X_mb, T)
            loss = criterion(y_pred[:, :-1, :], X_mb[:, 1:, :])
            val_loss.append(loss.item())

        logger.set_description(f"Epoch: {epoch}, Train loss: {np.mean(train_loss):.4f}, Val loss: {np.mean(val_loss):.4f}")
        if neptune_logger is not None:
            neptune_logger["PredictionScore/train_loss"].log(np.mean(train_loss))
            neptune_logger["PredictionScore/val_loss"].log(np.mean(val_loss))

        early_stopping(np.mean(train_loss), np.mean(val_loss))
        if early_stopping.early_stop:
            break

    # test the LSTM model
    test_dataloader = torch.utils.data.DataLoader(test, batch_size=32, shuffle=True)
    test_loss = []
    for X_mb in test_dataloader:
        X_mb = X_mb.to(device)
        T = torch.tensor([train.shape[1]] * X_mb.size(0))
        y_pred = model(X_mb, T)
        loss = criterion(y_pred[:, :-1, :], X_mb[:, 1:, :])
        test_loss.append(loss.item())
    return np.mean(test_loss)