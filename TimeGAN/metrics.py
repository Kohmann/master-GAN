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
    print(f"MSE Error: {mse_error:.5f}")

