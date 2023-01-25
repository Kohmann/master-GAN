import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

from metrics import sw_approx

def cotgan_trainer(model, dataset, params, val_dataset=None, neptune_logger=None, continue_training=False):

    batch_size = params["batch_size"]
    n_epochs = params["n_epochs"]
    learning_rate = params["l_rate"]
    device = params["device"]
    model_name = params["model_name"]



    for epoch in range(n_epochs):


    return 0
