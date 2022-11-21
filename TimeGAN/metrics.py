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
def sinkhorn_distance(x,y):
    """
    Sinkhorn distance between two samples x and y. Is an approximation of the Wasserstein distance.
    :param x: torch tensor: samples from the first distribution
    :param y: torch tensor: samples from the second distribution
    :return:  wasserstein distance between x and y

    Permisson is granted in LICENSE.txt
    """
    # Define a Sinkhorn (~Wasserstein) loss between sampled measures
    sinkhorn = SamplesLoss(loss="sinkhorn", p=2, blur=0.05)
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


