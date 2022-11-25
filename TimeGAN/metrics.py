from scipy.stats import wasserstein_distance
import torch
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
def sinkhorn_distance(x,y, blur=0.01):
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

def MMD(x,y):
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


def sw_approx(mu: torch.Tensor, nu: torch.Tensor) -> float:
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



