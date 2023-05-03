import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
import random

import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

from utils import DatasetSinus, log_visualizations, DatasetSoliton, DatasetTwoCollidingSolitons
import neptune.new as neptune

from architectures import COTGAN
from metrics import sw_approx, mae_height_diff, two_sample_kolmogorov_smirnov, compare_sin3_generation, \
    energy_conservation, mass_conservation, momentum_conservation

# FLAGS
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
#setorch.use_deterministic_algorithms(True)


def log_generation(X_hat, epoch, params, x_sw, neptune_logger=None):
    n_samples = X_hat.size(0)
    max_seq_len = X_hat.size(1)
    fig, axs = plt.subplots(3, 3, figsize=(14, 10))
    for x in range(3):
        for y in range(3):
            if "soliton" in params["dataset"]:
                axs[x, y].plot(X_hat[x * 3 + y].cpu().T)
            else:
                axs[x, y].plot(X_hat[x * 3 + y].cpu())
            axs[x, y].set_ylim([0, 1])
            # axs[x, y].set_yticklabels([])

    fig.suptitle(f"Generation: {epoch}", fontsize=14)
    neptune_logger["generated_image"].log(fig)
    plt.close(fig)

    neptune_logger["SW"].log(sw_approx(x_sw.view(n_samples * max_seq_len, -1).cpu(),
                                       X_hat.view(n_samples * max_seq_len, -1).cpu()))

    if params["dataset"] == "sinus":
        # log sin3 error
        if (epoch+1) % 10 == 0:
            neptune_logger["sin3_error"].log(compare_sin3_generation(X_hat.detach().cpu().numpy(), params["alpha"], params["noise"]))


    if "soliton" in params["dataset"]:

        fake = X_hat.clone().detach()

        if "twosolitons" == params["dataset"]:
            d1, d2 = .3, .5
            _, t_res, x_res = fake.shape
            s1_max = int(x_res * d1)
            s2_max = int(x_res * d2)
            c_fake_s1 = fake[:, 0, s1_max].cpu()
            c_fake_s2 = fake[:, 0, s2_max].cpu()

            c_real_s1 = x_sw[:, 0, s1_max].cpu()
            c_real_s2 = x_sw[:, 0, s2_max].cpu()
            p_value_s1 = two_sample_kolmogorov_smirnov(c_real_s1, c_fake_s1)
            p_value_s2 = two_sample_kolmogorov_smirnov(c_real_s2, c_fake_s2)
            p_value = (p_value_s1 + p_value_s2) / 2
            neptune_logger["c_mode_collapse"].log(p_value if p_value > 0.0001 else 0.0)
        else:
            c_fake = fake[:, 0, :].max(dim=1)[0].cpu()
            c_real = x_sw[:, 0, :].max(dim=1)[0].cpu()
            p_value = two_sample_kolmogorov_smirnov(c_real, c_fake)
            neptune_logger["c_mode_collapse"].log(p_value if p_value > 0.0001 else 0.0)
            neptune_logger["height_diff_mae"].log(mae_height_diff(fake))

        # Energy conservation
        dx = params["P"] / params["dx"] if "twosolitons" == params["dataset"] else params["P"] / params["spatial_len"]

        H_error = energy_conservation(fake, dx=dx, eta=params["eta"], gamma=params["gamma"]).mean().item()
        H_mass_error     = mass_conservation(fake, dx=dx).mean().item()
        H_momentum_error = momentum_conservation(fake, dx=dx).mean().item()
        neptune_logger["H_mean_error"].log(H_error)
        neptune_logger["H_mass_error"].log(H_mass_error)
        neptune_logger["H_momentum_error"].log(H_momentum_error)

def cotgan_trainer(model, dataset, params, neptune_logger=None):

    batch_size = params["batch_size"]
    n_epochs = params["n_epochs"]
    learning_rate = params["l_rate"]
    learning_rate_g = params["l_rate_g"]
    model_name = params["model_name"]
    max_seq_len = params["max_seq_len"]
    Z_dim = params["Z_dim"]
    device = params["device"]
    use_opt_scheduler = params["use_opt_scheduler"]
    scheduler_rule = params["scheduler_rule"]# "stepwise" or "linear"
    Z_distribution = params["Z_distribution"]

    use_convservation_loss = params["use_convservation_loss"]
    if use_convservation_loss:
        model.toggle_compute_conservation_loss(toggle=False)

    conservation_loss_init = params["conservation_loss_init"]
    conservation_loss_toggle_epoch = int(n_epochs * conservation_loss_init)



    # Prepare datasets
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size*2,
        shuffle=True,
    )

    # Optimizers
    beta1, beta2 = (params["beta1"], params["beta2"])
    disc_h_opt = torch.optim.Adam(model.discriminator_h.parameters(), lr=learning_rate, betas=(beta1, beta2))
    disc_m_opt = torch.optim.Adam(model.discriminator_m.parameters(), lr=learning_rate, betas=(beta1, beta2))
    gen_opt = torch.optim.Adam(model.generator.parameters(), lr=learning_rate_g, betas=(beta1, beta2))
    # Schedulers (Optional)


    if use_opt_scheduler:
        if scheduler_rule == "linear":
            disc_h_scheduler = torch.optim.lr_scheduler.LinearLR(disc_h_opt, start_factor=1, end_factor=0.5, total_iters=n_epochs)
            disc_m_scheduler = torch.optim.lr_scheduler.LinearLR(disc_m_opt, start_factor=1, end_factor=0.5, total_iters=n_epochs)
            gen_scheduler =    torch.optim.lr_scheduler.LinearLR(disc_m_opt, start_factor=1, end_factor=0.5, total_iters=n_epochs)

        elif scheduler_rule == "stepwise":
            step_size = n_epochs // 3
            disc_h_scheduler = torch.optim.lr_scheduler.StepLR(disc_h_opt, step_size=step_size, gamma=0.8)
            disc_m_scheduler = torch.optim.lr_scheduler.StepLR(disc_m_opt, step_size=step_size, gamma=0.8)
            gen_scheduler    = torch.optim.lr_scheduler.StepLR(gen_opt,    step_size=step_size, gamma=0.8)
        else:
            raise ValueError(f"Unknown scheduler rule. Got {scheduler_rule}")

    model.to(device)
    x_sw = dataset[:].detach().cpu()

    if Z_distribution == "uniform":
        fixed_Z_mb = 2 * torch.rand(len(x_sw), max_seq_len, Z_dim, device=device) - 1.0  # Uniform in [-1, 1]
    else: # normal
        fixed_Z_mb = torch.randn(len(x_sw), max_seq_len, Z_dim, device=device)

    logger = trange(n_epochs, desc=f"Epoch: 0, G_loss: 0, D_loss: 0")
    G_loss = 0
    D_loss = 0
    for epoch in logger:
        for X in dataloader:
            if X.size(0) != batch_size*2: continue
            X = X.to(device)
            Z = torch.randn(X.size(0), max_seq_len, Z_dim, device=device) if Z_distribution == "normal" else 2 * torch.rand(X.size(0), max_seq_len, Z_dim, device=device) - 1.0
            # Train discriminator
            #print(f"X shape {X.size()}, Z shape {Z.size()}")
            D_loss = model(Z, X, obj="discriminator")
            # Update discriminators
            disc_h_opt.zero_grad()
            disc_m_opt.zero_grad()
            D_loss.backward()
            disc_h_opt.step()
            disc_m_opt.step()

            # Train generator
            gen_opt.zero_grad()
            G_loss = model(Z, X, obj="generator")
            G_loss.backward()
            gen_opt.step()

            # Scheduler update
            if use_opt_scheduler:
                disc_h_scheduler.step()
                disc_m_scheduler.step()
                gen_scheduler.step()

        if use_convservation_loss and (epoch % conservation_loss_toggle_epoch == 0):
            model.toggle_compute_conservation_loss(toggle=True)

        #G_loss = G_loss.detach()#.cpu()
        #D_loss = D_loss.detach()#.cpu()
        logger.set_description(
            f"Epoch: {epoch}, G: {G_loss:.4f}, D: {-D_loss:.4f}"
        )
        if neptune_logger is not None:
            neptune_logger["train/Generator"].log(G_loss)
            neptune_logger["train/Discriminator"].log(-D_loss)
            neptune_logger["train/martingale_regularization"].log((D_loss + G_loss) / params["reg_lam"])

            if (epoch + 1)  > 0: # (epoch + 1) % 10 == 0: #
                with torch.no_grad():
                    # generate synthetic data and plot it
                    X_hat = model(Z=fixed_Z_mb, X=None, obj="inference")
                    log_generation(X_hat=X_hat, epoch=epoch,x_sw=x_sw, neptune_logger=neptune_logger, params=params)

    # save model
    torch.save(model.state_dict(), f"./models/{model_name}")

def create_dataset(dataset, n_samples, p, device="cpu"):
    print(f"dataset: {dataset}")
    if "sinus" in dataset:
        return DatasetSinus(num=n_samples, seq_len=p["max_seq_len"],
                            alpha=p["alpha"], noise=p["noise"], device=device)
    elif "soliton" in dataset:
        if "soliton" == dataset:
            t_max = 6 if "t_max" not in p.keys() else p["t_max"]
            t_range = [0, t_max]
            c_range = [0.5, 2] if "c_range" not in p.keys() else p["c_range"]
            return DatasetSoliton(n_samples=n_samples, spatial_len=p["spatial_len"], P=p["P"],
                                  t_steps=p["t_steps"], t_range=t_range,
                                  c_range=c_range, device=device, difficulty=p["difficulty"])
        elif "twosolitons" == dataset:
            print("Loading dataset: Two Solitons")
            file_dir = "datasets/"
            return DatasetTwoCollidingSolitons(file_dir=file_dir, dx=p["dx"], dt=p["dt"],num_samples=n_samples)
    else:
        raise NotImplementedError
def load_dataset_and_train(params):
    seed = params["seed"]
    device = params["device"]
    random.seed(seed)  # python
    np.random.seed(seed) # numpy
    torch.manual_seed(seed) # pytorch


    trainset = create_dataset(dataset=params["dataset"],n_samples=params["trainset_size"], p=params, device=device)
    testset = create_dataset(dataset=params["dataset"], n_samples=params["testset_size"],  p=params)

    print("Num real train samples:", len(testset))
    print("Num real test  samples:", len(testset))

    # Start logger
    run = neptune.init_run(
        project="kohmann/COTGAN",
        name=params["model"],
        tags=["tuning"],
        description="",
        source_files=["architectures.py", "trainer.py"],
        capture_hardware_metrics=True,
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI3YjFjNGY5MS1kOWU1LTRmZjgtOTNiYS0yOGI2NDdjZGYzNWUifQ==",
    )

    run["parameters"] = params
    run["dataset"] = trainset.get_params()

    # Initialize model and train
    if params["model"] == "cotgan":
        model = COTGAN(params)
        cotgan_trainer(model, trainset, params, neptune_logger=run)
    elif params["model"] == "timegan":
        raise NotImplementedError
    else:
        raise NotImplementedError

    # Evaluate
    evaluate_model(model, testset, run, params)

def evaluate_model(model, testset, run, params):
    n_samples = params["testset_size"]
    fake_data = model.generate(n_samples).cpu().detach()

    log_visualizations(testset, fake_data, run)  # log pca, tsne, umap, mode_collapse
    run["model_checkpoint"].upload("./models/" + params["model_name"])

    testset2 = create_dataset(dataset=params["dataset"], n_samples=n_samples, p=params)
    #fake_data = model.generate(n_samples).cpu()

    if "sinus" in params["dataset"]:
        mse_error = compare_sin3_generation(fake_data.numpy(), 0.7, 0)
        print("ALPHA AND NOISE ARE HARD CODED IN THE METRIC FUNCTION to be 0.7 and 0.")
        run["numeric_results/sin3_generation_MSE_loss"] = mse_error

    if "soliton" in params["dataset"]:

        if "twosolitons" == params["dataset"]:
            d1, d2 = .3, .5
            _, t_res, x_res = fake_data.shape
            s1_max = int(x_res * d1)
            s2_max = int(x_res * d2)
            c_fake_s1 = fake_data[:, 0, s1_max].cpu()
            c_fake_s2 = fake_data[:, 0, s2_max].cpu()

            c_real_s1 = testset[:][:, 0, s1_max].cpu()
            c_real_s2 = testset[:][:, 0, s2_max].cpu()
            p_value_s1 = two_sample_kolmogorov_smirnov(c_real_s1, c_fake_s1)
            p_value_s2 = two_sample_kolmogorov_smirnov(c_real_s2, c_fake_s2)
            p_value = (p_value_s1 + p_value_s2) / 2
            run["numeric_results/c_mode_collapse"] = p_value if p_value > 0.0001 else 0.0

            k1_est = np.sqrt(c_fake_s1 / 2)
            k2_est = np.sqrt(c_fake_s2 / 2)

            fig = plt.figure(figsize=(7, 5))
            plt.hist(k1_est, bins=100, density=True, label="k1", alpha=0.5)
            plt.hist(k2_est, bins=100, density=True, label="k2", alpha=0.5)
            plt.legend()
            plt.xlim(0.2, 0.7)  # TODO plt.xlim for c_distribution is always (0.5, 2), make this dynamic
            run["c_fake_distribution"].upload(fig)
            plt.close(fig)


        if "soliton" == params["dataset"]:
            c_fake = fake_data[:, 0, :].max(dim=1)[0]
            c_real = testset[:][:, 0, :].max(dim=1)[0]
            p_value = two_sample_kolmogorov_smirnov(c_real, c_fake)
            run["numeric_results/c_mode_collapse"] = p_value if p_value > 0.0001 else 0.0
            run["numeric_results/height_diff_mae"] = mae_height_diff(fake_data)

            fig = plt.figure(figsize=(7, 5))
            plt.hist(2. * c_fake, bins=100, density=True)
            plt.xlim(0.5, 2) # TODO plt.xlim for c_distribution is always (0.5, 2), make this dynamic
            run["c_fake_distribution"].upload(fig)
            plt.close(fig)

    n, max_seq_len, _ = fake_data.shape
    x = fake_data.view(n_samples * max_seq_len, -1)
    print("testset shape: ", testset[:].shape, "fake_data shape: ", fake_data.shape, "fake_data reshaped (B*T,D): ", x.shape)
    y = testset[:].view(n_samples * max_seq_len, -1)
    y_2 = testset2[:].view(n_samples * max_seq_len, -1)

    sw_baseline = sw_approx(y, y_2)
    sw = sw_approx(y, x)

    run["numeric_results/num_test_samples"] = len(testset)
    run["numeric_results/SW"] = sw  # .item()
    run["numeric_results/SW_baseline"] = sw_baseline  # .item()
    run.stop()


import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='cotgan')

    parser.add_argument('--model', type=str, default='cotgan', choices=['cotgan', 'timegan'])
    parser.add_argument('--model_name', type=str, default='model_cotgan.pt')
    # Dataset params
    parser.add_argument('--dataset',      type=str,   default='soliton', choices=['sinus', 'soliton', 'twosolitons'])
    # For sinus
    parser.add_argument('--max_seq_len',  type=int,   default=25)
    parser.add_argument('--feature_dim',  type=int,   default=3)
    parser.add_argument('--alpha',        type=float, default=0.7)
    parser.add_argument('--noise',        type=float, default=0.0)
    # For soliton
    parser.add_argument('--P',            type=int,   default=50)
    parser.add_argument('--spatial_len',  type=int,   default=120) # M
    parser.add_argument('--t_steps',      type=int,   default=120)  # N
    parser.add_argument('--eta',          type=float, default=6.0)
    parser.add_argument('--gamma',        type=float, default=1.0)
    parser.add_argument('--difficulty',   type=str,   default='easy', choices=['easy', 'medium'])
    parser.add_argument('--t_max',        type=float, default=10.0) # Upper time limit

    # for twosolitons
    parser.add_argument('--dx',           type=int,   default=120)
    parser.add_argument('--dt',           type=int,   default=30)
    parser.add_argument('--use_convservation_loss', type=str,   default="True", choices=["True", "False"])
    parser.add_argument('--conservation_loss_init', type=float, default=0.0)
    parser.add_argument('--conservation_weight',    type=float, default=0.01)

    # Dataset sizes
    parser.add_argument('--trainset_size',type=int,   default=32*2)
    parser.add_argument('--testset_size', type=int,   default=32*2)

    # Hyperparameters
    parser.add_argument('--n_epochs',   type=int,   default=10)
    parser.add_argument('--l_rate',     type=float, default=0.1)
    parser.add_argument('--l_rate_g',   type=float, default=0.1)
    parser.add_argument('--batch_size', type=int,   default=32)
    parser.add_argument('--optimizer',  type=str,   default='Adam')
    parser.add_argument('--beta1',      type=float, default=0.5)
    parser.add_argument('--beta2',      type=float, default=0.9)
    parser.add_argument('--use_opt_scheduler', type=str, default="True", choices=["True", "False"])
    parser.add_argument('--scheduler_rule',    type=str, default="stepwise", choices=["stepwise", "linear"])

    # COTGAN architecture params
    parser.add_argument('--rnn_type',           type=str, default='GRU', choices=['GRU', 'LSTM'])
    parser.add_argument('--gen_rnn_num_layers', type=int, default=2)
    parser.add_argument('--gen_rnn_hidden_dim', type=int, default=64)
    parser.add_argument('--dis_rnn_num_layers', type=int, default=2)
    parser.add_argument('--dis_rnn_hidden_dim', type=int, default=64)
    parser.add_argument('--J_dim',              type=int, default=32)
    parser.add_argument('--hidden_dim',         type=int, default=64)
    parser.add_argument('--num_hidden_layers',  type=int, default=2)
    parser.add_argument('--Z_dim',              type=int, default=10)
    parser.add_argument('--use_bn',             type=str, default="False", choices=["True", "False"])
    parser.add_argument('--Z_distribution',     type=str, default='uniform', choices=['uniform', 'normal'])


    # Loss params
    parser.add_argument('--scaling_coef',     type=float, default=1.0)
    parser.add_argument('--sinkhorn_eps',     type=float, default=0.8)
    parser.add_argument('--sinkhorn_l',       type=int,   default=100)
    parser.add_argument('--reg_lam',          type=float, default=0.01)
    # Other
    parser.add_argument('--device', type=str, default='cpu', choices=['cuda', 'cpu', 'mps'])
    parser.add_argument('--seed',   type=int, default=1)

    args = parser.parse_args()
    args = vars(args)

    args["device_name"] = "cpu"
    if args["device"] == "cuda":
        print("torch.cuda.get_device_name():", torch.cuda.get_device_name(0))
        args["device_name"] = torch.cuda.get_device_name(0)

    args["use_convservation_loss"] = args["use_convservation_loss"] == "True"
    args["use_bn"] = args["use_bn"] == "True"
    args["use_opt_scheduler"] = args["use_opt_scheduler"] == "True"
    if args["dataset"] == "soliton":
        args["max_seq_len"] = args["t_steps"]
        args["feature_dim"] = args["spatial_len"]
        args["dx"] = args["P"] / args["spatial_len"]
        # TODO (Fix this issue properly)
        if args["difficulty"] == "medium":
            args["dataset"] = "medium_soliton"

    if args["dataset"] == "twosolitons":
        args["max_seq_len"] = args["dt"]
        args["feature_dim"] = args["dx"]
        args["P"] = 50
        args["spatial_len"] = args["dx"]
        args["t_steps"] = args["dt"]

    load_dataset_and_train(args)