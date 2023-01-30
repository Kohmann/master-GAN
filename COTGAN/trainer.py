import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

from utils import DatasetSinus, log_visualizations
import neptune.new as neptune

from architectures import COTGAN
from metrics import sw_approx

def cotgan_trainer(model, dataset, params, val_dataset=None, neptune_logger=None, continue_training=False):

    batch_size = params["batch_size"]
    n_epochs = params["n_epochs"]
    learning_rate = params["l_rate"]
    device = params["device"]
    model_name = params["model_name"]
    max_seq_len = params["max_seq_len"]
    Z_dim = params["Z_dim"]

    # Prepare datasets
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size*2,
        shuffle=True
    )
    # TODO - put data into GPU entirely
    #dataloader.train_data.to(torch.device(device))  # put data into GPU entirely
    #dataloader.train_labels.to(torch.device(device))

    if val_dataset is not None:
        dataloader_val = torch.utils.data.DataLoader(
            dataset=val_dataset,
            batch_size=batch_size*2,
            shuffle=True
        )

    # Optimizers
    beta1, beta2 = (params["beta1"], params["beta2"])
    disc_h_opt = torch.optim.Adam(model.discriminator_h.parameters(), lr=learning_rate, betas=(beta1, beta2))
    disc_m_opt = torch.optim.Adam(model.discriminator_m.parameters(), lr=learning_rate, betas=(beta1, beta2))
    gen_opt = torch.optim.Adam(model.generator.parameters(), lr=learning_rate, betas=(beta1, beta2))
    model.to(device)

    x_sw = torch.concat([x for x, _ in dataloader]).detach_().cpu()
    n_samples = len(x_sw)
    fixed_Z_mb = torch.randn(n_samples, max_seq_len, Z_dim, device=device)

    logger = trange(n_epochs, desc=f"Epoch: 0, G_loss: 0, D_loss: 0")
    for epoch in logger:
        for X, _ in dataloader:
            X_mb, X_mb_p = torch.split(X, batch_size, dim=0)
            X_mb = X_mb.to(device)
            X_mb_p = X_mb_p.to(device)

            Z_mb = torch.randn(batch_size, max_seq_len, Z_dim, device=device)
            Z_mb_p = torch.randn(batch_size, max_seq_len, Z_dim, device=device)

            # Train discriminator
            D_loss = model(Z_mb, Z_mb_p, X_mb, X_mb_p, obj="discriminator")
            # Update discriminators
            disc_h_opt.zero_grad()
            disc_m_opt.zero_grad()
            D_loss.backward()
            disc_h_opt.step()
            disc_m_opt.step()

            # Train generator
            gen_opt.zero_grad()
            G_loss = model(Z_mb, Z_mb_p, X_mb, X_mb_p, obj="generator")
            G_loss.backward()
            gen_opt.step()

        G_loss = G_loss.detach().cpu()
        D_loss = D_loss.detach().cpu()
        logger.set_description(
            f"Epoch: {epoch}, G: {G_loss:.4f}, D: {-D_loss:.4f}"
        )
        if neptune_logger is not None:
            neptune_logger["train/Generator"].log(G_loss)
            neptune_logger["train/Discriminator"].log(-D_loss)
            neptune_logger["train/martingale_regularization"].log(-(G_loss-D_loss))
            if (epoch + 1)  > 0: # (epoch + 1) % 10 == 0:
                with torch.no_grad():
                    # generate synthetic data and plot it
                    X_hat = model(z1=fixed_Z_mb, obj="inference")
                    x_axis = np.arange(X_hat.size(dim=1))
                    fig, axs = plt.subplots(3, 3, figsize=(14, 10))

                    for x in range(3):
                        for y in range(3):
                            axs[x, y].plot(x_axis, X_hat[x * 3 + y].cpu().numpy())
                            axs[x, y].set_ylim([0, 1])
                            axs[x, y].set_yticklabels([])

                    fig.suptitle(f"Generation: {epoch}", fontsize=14)
                    # fig.savefig('./images/data_at_epoch_{:04d}.png'.format(epoch))
                    # neptune_logger["generated_image"].upload(fig)
                    neptune_logger["generated_image"].log(fig)
                    neptune_logger["SW"].log(sw_approx(x_sw, X_hat))
                    plt.close(fig)

    # save model
    torch.save(model.state_dict(), f"./models/{model_name}")

def cotgan_generator(model, params, eval=False):
    """The inference procedure for TimeGAN
    Args:
        - model (torch.nn.module): The model that generates synthetic data
        - T (List[int]): The time to be generated on
        - args (dict): The model/training configurations
    Returns:
        - generated_data (np.ndarray): The synthetic data generated by the model
    """
    # Load model for inference
    model_name = params["model_name"]
    device = params["device"]
    Z_dim = params["Z_dim"]
    max_seq_len = params["max_seq_len"]
    trainset_size = params["testset_size"]
    filepath = "./models/"
    if not eval:
        model.load_state_dict(torch.load(filepath + model_name, map_location=device))

    print("\nGenerating Data...", end="")
    # Initialize model to evaluation mode and run without gradients
    model.to(device)
    model.eval()
    with torch.no_grad():
        # Generate fake data
        Z = torch.rand((trainset_size, max_seq_len, Z_dim), device=device)
        generated_data = model(Z, obj="inference")
    print("Done")
    return generated_data.cpu().numpy()

def load_dataset_and_train(params):
    seed = params["seed"]
    alpha = params["alpha"]
    trainset_size = params["trainset_size"]
    testset_size = params["testset_size"]
    max_seq_len = params["max_seq_len"]
    device = params["device"]
    noise = 0
    np.random.seed(seed)
    torch.manual_seed(seed)


    trainset = DatasetSinus(num=trainset_size, seq_len=max_seq_len, alpha=alpha, noise=noise, device=device)
    testset = DatasetSinus(num=testset_size, seq_len=max_seq_len, alpha=alpha, noise=noise, device="cpu")

    print("Num real samples:", len(testset))

    # Start logger
    run = neptune.init_run(
        project="kohmann/COTGAN",
        name="cotgan",
        tags=["tuning"],
        description="",
        source_files=["architectures.py"],
        capture_hardware_metrics=True,
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI3YjFjNGY5MS1kOWU1LTRmZjgtOTNiYS0yOGI2NDdjZGYzNWUifQ==",
    )

    run["parameters"] = params
    run["dataset"] = {"alpha": alpha, "noise": noise,
                      "s1_freq": trainset.s1_freq, "s1_phase": trainset.s1_phase,
                      "s2_freq": trainset.s2_freq, "s2_phase": trainset.s2_phase}

    # Initialize model and train
    model = COTGAN(params)
    cotgan_trainer(model, trainset, params, val_dataset=testset, neptune_logger=run, continue_training=False)


    ### Perform tests on the trained model ###

    # Generate random synthetic data
    gen_z = cotgan_generator(model, params)

    log_visualizations(testset, gen_z, run)  # log pca, tsne, umap, mode_collapse
    run["model_checkpoint"].upload("./models/" + params["model_name"])

    from metrics import compare_sin3_generation, sw_approx
    np.random.seed(seed + 1)
    torch.manual_seed(seed + 1)
    testset2 = DatasetSinus(num=params["testset_size"], seq_len=params["max_seq_len"], alpha=alpha, noise=noise,
                            device="cpu")
    fake_data = cotgan_generator(model, params)

    mse_error = compare_sin3_generation(fake_data, alpha, noise)
    print(f"MSE Error: {mse_error:.5f}")
    x = torch.tensor(fake_data)
    y = testset[:][0]
    y_2 = testset2[:][0]

    sw_baseline = sw_approx(y, y_2)
    sw = sw_approx(y, x)

    run["numeric_results/num_test_samples"] = len(testset)
    run["numeric_results/sin3_generation_MSE_loss"] = mse_error
    run["numeric_results/SW"] = sw#.item()
    run["numeric_results/SW_baseline"] = sw_baseline#.item()
    run.stop()


import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='cotgan')

    # Dataset params
    parser.add_argument('--dataset', type=str, default='sinus', choices=['sinus'])
    parser.add_argument('--max_seq_len', type=int, default=25)
    parser.add_argument('--feature_dim', type=int, default=3)
    parser.add_argument('--alpha', type=float, default=0.7) # exponential decay
    parser.add_argument('--trainset_size', type=int, default=32*2*24)
    parser.add_argument('--testset_size', type=int, default=32*2*12)

    # Hyperparameters
    parser.add_argument('--model_name', type=str, default='model_cotgan.pt')
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--l_rate', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.9)
    # Model architecture
    parser.add_argument('--gen_rnn_num_layers', type=int, default=2)
    parser.add_argument('--gen_rnn_hidden_dim', type=int, default=64)
    parser.add_argument('--hidden_dim', type=int,         default=64*2)
    parser.add_argument('--num_hidden_layers', type=int,  default=3)
    parser.add_argument('--Z_dim', type=int, default=100)
    # Loss params
    parser.add_argument('--scaling_coef', type=float, default=1.0)
    parser.add_argument('--sinkhorn_eps', type=float, default=0.8)
    parser.add_argument('--sinkhorn_l', type=int, default=100)
    parser.add_argument('--reg_lam', type=float, default=0.01)
    # Other
    parser.add_argument('--device', type=str, default='cpu', choices=['cuda', 'cpu', 'mps'])
    parser.add_argument('--seed', type=int, default=1)



    args = parser.parse_args()
    print(vars(args), "\n\n")
    load_dataset_and_train(vars(args))