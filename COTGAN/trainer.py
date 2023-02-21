import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
import random

from utils import DatasetSinus, log_visualizations, DatasetSoliton
import neptune.new as neptune

from architectures import COTGAN
from metrics import sw_approx, mae_height_diff, two_sample_kolmogorov_smirnov

def cotgan_trainer(model, dataset, params, val_dataset=None, neptune_logger=None, continue_training=False):

    batch_size = params["batch_size"]
    n_epochs = params["n_epochs"]
    learning_rate = params["l_rate"]
    learning_rate_g = params["l_rate_g"]
    model_name = params["model_name"]
    max_seq_len = params["max_seq_len"]
    Z_dim = params["Z_dim"]
    device = params["device"]
    use_opt_scheduler = params["use_opt_scheduler"]

    # Prepare datasets
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size*2,
        shuffle=True
    )

    # Optimizers
    beta1, beta2 = (params["beta1"], params["beta2"])
    disc_h_opt = torch.optim.Adam(model.discriminator_h.parameters(), lr=learning_rate, betas=(beta1, beta2))
    disc_m_opt = torch.optim.Adam(model.discriminator_m.parameters(), lr=learning_rate, betas=(beta1, beta2))
    gen_opt = torch.optim.Adam(model.generator.parameters(), lr=learning_rate_g, betas=(beta1, beta2))
    # Schedulers (Optional)
    print("Using use_opt_scheduler:", use_opt_scheduler)
    if use_opt_scheduler:
        step_size = n_epochs // 3
        disc_h_scheduler = torch.optim.lr_scheduler.StepLR(disc_h_opt, step_size=step_size, gamma=0.8)
        disc_m_scheduler = torch.optim.lr_scheduler.StepLR(disc_m_opt, step_size=step_size, gamma=0.8)
        gen_scheduler    = torch.optim.lr_scheduler.StepLR(gen_opt,    step_size=step_size, gamma=0.8)

    model.to(device)
    x_sw = dataset[:].detach().cpu()

    n_samples = len(x_sw)
    fixed_Z_mb = torch.randn(n_samples, max_seq_len, Z_dim, device=device)

    logger = trange(n_epochs, desc=f"Epoch: 0, G_loss: 0, D_loss: 0")
    G_loss = 0
    D_loss = 0
    for epoch in logger:
        for X in dataloader:
            X_mb, X_mb_p = torch.split(X, batch_size, dim=0)
            X_mb = X_mb.to(device)
            X_mb_p = X_mb_p.to(device)

            Z_mb =   torch.randn(batch_size, max_seq_len, Z_dim, device=device)
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

            # Scheduler update
            if use_opt_scheduler:
                disc_h_scheduler.step()
                disc_m_scheduler.step()
                gen_scheduler.step()

        G_loss = G_loss.detach().cpu()
        D_loss = D_loss.detach().cpu()
        logger.set_description(
            f"Epoch: {epoch}, G: {G_loss:.4f}, D: {-D_loss:.4f}"
        )
        if neptune_logger is not None:
            neptune_logger["train/Generator"].log(G_loss)
            neptune_logger["train/Discriminator"].log(-D_loss)
            neptune_logger["train/martingale_regularization"].log((G_loss-D_loss))

            if (epoch + 1)  > 0: # (epoch + 1) % 10 == 0: #
                with torch.no_grad():
                    # generate synthetic data and plot it
                    X_hat = model(z1=fixed_Z_mb, obj="inference")
                    #x_axis = np.arange(max_seq_len)
                    fig, axs = plt.subplots(3, 3, figsize=(14, 10))

                    for x in range(3):
                        for y in range(3):
                            if params["dataset"] == "soliton":
                                axs[x, y].plot(X_hat[x * 3 + y].cpu().T)
                            else:
                                axs[x, y].plot(X_hat[x * 3 + y].cpu())
                            axs[x, y].set_ylim([0, 1])
                            #axs[x, y].set_yticklabels([])

                    fig.suptitle(f"Generation: {epoch}", fontsize=14)
                    neptune_logger["generated_image"].log(fig)
                    plt.close(fig)

                    neptune_logger["SW"].log(sw_approx(x_sw.view(n_samples * max_seq_len, -1),
                                                       X_hat.view(n_samples * max_seq_len, -1)))

                    if "soliton" in params["dataset"]:
                        fake = torch.tensor(X_hat).detach()
                        c_fake = fake[:, 0, :].max(dim=1)[0].cpu()
                        c_real = x_sw[:, 0, :].max(dim=1)[0].cpu()
                        p_value = two_sample_kolmogorov_smirnov(c_real, c_fake)
                        neptune_logger["c_mode_collapse"].log(p_value if p_value > 0.0001 else 0.0)
                        if params["difficulty"] == "medium":
                            neptune_logger["height_diff_mae"].log(mae_height_diff(fake))

    # save model
    torch.save(model.state_dict(), f"./models/{model_name}")

def cotgan_trainer_general(model, dataset, params, val_dataset=None, neptune_logger=None, continue_training=False):

    batch_size = params["batch_size"]
    n_epochs = params["n_epochs"]
    learning_rate = params["l_rate"]
    learning_rate_g = params["l_rate_g"]
    model_name = params["model_name"]
    max_seq_len = params["max_seq_len"]
    Z_dim = params["Z_dim"]
    device = params["device"]
    use_opt_scheduler = params["use_opt_scheduler"]

    # Prepare datasets
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size*2,
        shuffle=True
    )

    # Optimizers
    beta1, beta2 = (params["beta1"], params["beta2"])
    disc_h_opt = torch.optim.Adam(model.discriminator_h.parameters(), lr=learning_rate, betas=(beta1, beta2))
    disc_m_opt = torch.optim.Adam(model.discriminator_m.parameters(), lr=learning_rate, betas=(beta1, beta2))
    gen_opt = torch.optim.Adam(model.generator.parameters(), lr=learning_rate_g, betas=(beta1, beta2))
    # Schedulers (Optional)
    print("Using use_opt_scheduler:", use_opt_scheduler)
    if use_opt_scheduler:
        step_size = n_epochs // 3
        disc_h_scheduler = torch.optim.lr_scheduler.StepLR(disc_h_opt, step_size=step_size, gamma=0.8)
        disc_m_scheduler = torch.optim.lr_scheduler.StepLR(disc_m_opt, step_size=step_size, gamma=0.8)
        gen_scheduler    = torch.optim.lr_scheduler.StepLR(gen_opt,    step_size=step_size, gamma=0.8)

    model.to(device)
    x_sw = dataset[:].detach().cpu()

    n_samples = len(x_sw)
    fixed_Z_mb = torch.randn(n_samples, max_seq_len, Z_dim, device=device)

    logger = trange(n_epochs, desc=f"Epoch: 0, G_loss: 0, D_loss: 0")
    G_loss = 0
    D_loss = 0
    for epoch in logger:
        for X in dataloader:
            X = X.to(device)
            Z = torch.randn(batch_size*2, max_seq_len, Z_dim, device=device)

            # Train discriminator
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

        G_loss = G_loss.detach().cpu()
        D_loss = D_loss.detach().cpu()
        logger.set_description(
            f"Epoch: {epoch}, G: {G_loss:.4f}, D: {-D_loss:.4f}"
        )
        if neptune_logger is not None:
            neptune_logger["train/Generator"].log(G_loss)
            neptune_logger["train/Discriminator"].log(-D_loss)
            neptune_logger["train/martingale_regularization"].log((G_loss-D_loss))

            if (epoch + 1)  > 0: # (epoch + 1) % 10 == 0: #
                with torch.no_grad():
                    # generate synthetic data and plot it
                    X_hat = model(z1=fixed_Z_mb, obj="inference")
                    #x_axis = np.arange(max_seq_len)
                    fig, axs = plt.subplots(3, 3, figsize=(14, 10))

                    for x in range(3):
                        for y in range(3):
                            if params["dataset"] == "soliton":
                                axs[x, y].plot(X_hat[x * 3 + y].cpu().T)
                            else:
                                axs[x, y].plot(X_hat[x * 3 + y].cpu())
                            axs[x, y].set_ylim([0, 1])
                            #axs[x, y].set_yticklabels([])

                    fig.suptitle(f"Generation: {epoch}", fontsize=14)
                    neptune_logger["generated_image"].log(fig)
                    plt.close(fig)

                    neptune_logger["SW"].log(sw_approx(x_sw.view(n_samples * max_seq_len, -1),
                                                       X_hat.view(n_samples * max_seq_len, -1)))

                    if "soliton" in params["dataset"]:
                        fake = torch.tensor(X_hat).detach()
                        c_fake = fake[:, 0, :].max(dim=1)[0].cpu()
                        c_real = x_sw[:, 0, :].max(dim=1)[0].cpu()
                        p_value = two_sample_kolmogorov_smirnov(c_real, c_fake)
                        neptune_logger["c_mode_collapse"].log(p_value if p_value > 0.0001 else 0.0)
                        if params["difficulty"] == "medium":
                            neptune_logger["height_diff_mae"].log(mae_height_diff(fake))

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
        Z = torch.randn((trainset_size, max_seq_len, Z_dim), device=device)
        generated_data = model(Z, obj="inference")
    print("Done")
    return generated_data.cpu().numpy()

def create_dataset(dataset, n_samples, p, device="cpu"):
    print(f"dataset: {dataset}")
    if "sinus" in dataset:
        return DatasetSinus(num=n_samples, seq_len=p["max_seq_len"],
                            alpha=p["alpha"], noise=p["noise"], device=device)
    elif "soliton" in dataset:
        t_range = [0, 6] if not p["t_range"] else p["t_range"]
        c_range = [0.5, 2] if not p["c_range"] else p["c_range"]
        return DatasetSoliton(n_samples=n_samples, spatial_len=p["spatial_len"], P=p["P"],
                              t_steps=p["t_steps"], t_range=t_range,
                              c_range=c_range, device=device, difficulty=p["difficulty"])
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
        name="cotgan",
        tags=["tuning"],
        description="",
        source_files=["architectures.py"],
        capture_hardware_metrics=True,
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI3YjFjNGY5MS1kOWU1LTRmZjgtOTNiYS0yOGI2NDdjZGYzNWUifQ==",
    )

    run["parameters"] = params
    run["dataset"] = trainset.get_params()

    # Initialize model and train
    model = COTGAN(params)
    cotgan_trainer(model, trainset, params, val_dataset=testset, neptune_logger=run, continue_training=False)

    ### Perform tests on the trained model ###

    # Generate random synthetic data
    gen_z = cotgan_generator(model, params)

    log_visualizations(testset, gen_z, run)  # log pca, tsne, umap, mode_collapse
    run["model_checkpoint"].upload("./models/" + params["model_name"])

    from metrics import compare_sin3_generation, sw_approx

    testset2 = create_dataset(dataset=params["dataset"], n_samples=params["testset_size"], p=params)

    # testset2 = DatasetSinus(num=params["testset_size"], seq_len=params["max_seq_len"], alpha=alpha, noise=noise, device="cpu")
    fake_data = cotgan_generator(model, params)

    if "sines" in params["dataset"]:
        mse_error = compare_sin3_generation(fake_data, 0.7, 0)
        print("ALPHA AND NOISE ARE HARD CODED IN THE METRIC FUNCTION to be 0.7 and 0.")
        run["numeric_results/sin3_generation_MSE_loss"] = mse_error

    if "soliton" in params["dataset"]:
        fake = torch.tensor(fake_data)
        c_fake = fake[:, 0, :].max(dim=1)[0]
        c_real = testset[:][:, 0, :].max(dim=1)[0]
        run["numeric_results/c_mode_collapse"] = two_sample_kolmogorov_smirnov(c_real, c_fake)
        if params["difficulty"] == "easy":
            run["numeric_results/height_diff_mae"] = mae_height_diff(fake)
        #run["numeric_results/height_diff_mae"] = mae_height_diff(fake)
        fig = plt.figure(figsize=(7, 5))
        plt.hist(2. * c_fake, bins=100, density=True)
        plt.xlim(0.5, 2)
        run["c_fake_distribution"].upload(fig)
        plt.close(fig)

    n_samples = params["testset_size"]
    max_seq_len = fake_data.shape[1]
    x = torch.tensor(fake_data).view(n_samples * max_seq_len, -1)
    y = testset[:].view(n_samples * max_seq_len, -1)
    y_2 = testset2[:].view(n_samples * max_seq_len, -1)

    sw_baseline = sw_approx(y, y_2)
    sw = sw_approx(y, x)

    run["numeric_results/num_test_samples"] = len(testset)
    run["numeric_results/SW"] = sw  # .item()
    run["numeric_results/SW_baseline"] = sw_baseline  # .item()
    run.stop()


## TIMEGAN

def embedding_trainer(model, dataloader, e_opt, r_opt, n_epochs, neptune_logger=None):
    logger = trange(n_epochs, desc=f"Epoch: 0, Loss: 0")
    for epoch in logger:
        for X_mb in dataloader:
            model.zero_grad()

            _, E_loss0, E_loss_T0 = model(X=X_mb, Z=None, obj="autoencoder")
            loss = np.sqrt(E_loss_T0.item())

            E_loss0.backward()
            e_opt.step()
            r_opt.step()

        logger.set_description(f"Epoch: {epoch}, Loss: {loss:.4f}")
        if epoch % 5 == 0 and neptune_logger is not None:
            neptune_logger["train/Embedding"].log(loss)
def supervisor_trainer(model, dataloader, s_opt, g_opt, n_epochs, neptune_logger=None):
    logger = trange(n_epochs, desc=f"Epoch: 0, Loss: 0")
    for epoch in logger:
        for X_mb in dataloader:
            model.zero_grad()

            S_loss = model(X=X_mb, Z=None, obj="supervisor")
            loss = np.sqrt(S_loss.item())

            S_loss.backward()
            s_opt.step()

        logger.set_description(f"Epoch: {epoch}, Loss: {loss:.4f}")
        if epoch % 5 == 0 and neptune_logger is not None:
            neptune_logger["train/Supervisor"].log(loss)
            # writer.add_scalar("Loss/Supervisor", loss, epoch)
def joint_trainer(model, dataloader, e_opt, r_opt, s_opt, g_opt, d_opt, n_epochs, batch_size, max_seq_len, Z_dim,
                  dis_thresh, neptune_logger=None):
    x_sw = torch.concat([x for x in dataloader])
    n_samples = len(x_sw)
    fixed_Z_mb = torch.rand((n_samples, max_seq_len, Z_dim))
    logger = trange(n_epochs, desc=f"Epoch: 0, E_loss: 0, G_loss: 0, D_loss: 0")
    best_sw = 1000.0
    model_name = neptune_logger["parameters/model_name"].fetch() if neptune_logger is not None else "empty"

    for epoch in logger:
        for X_mb in dataloader:
            for _ in range(2):
                Z_mb = torch.rand(X_mb.size(0), max_seq_len, Z_dim)
                model.zero_grad()
                # Generator
                G_loss = model(X=X_mb, Z=Z_mb, obj="generator")
                G_loss.backward()
                G_loss = np.sqrt(G_loss.item())

                g_opt.step()
                s_opt.step()

                # Embedding
                model.zero_grad()
                E_loss, _, E_loss_T0 = model(X=X_mb, Z=Z_mb, obj="autoencoder")
                E_loss.backward()
                E_loss = np.sqrt(E_loss.item())

                e_opt.step()
                r_opt.step()

            Z_mb = torch.rand((batch_size, max_seq_len, Z_dim))
            model.zero_grad()
            D_loss = model(X=X_mb, Z=Z_mb, obj="discriminator")
            if D_loss > dis_thresh:  # don't let the discriminator get too good
                D_loss.backward()
                d_opt.step()
            D_loss = D_loss.item()

        logger.set_description(
            f"Epoch: {epoch}, E: {E_loss:.4f}, G: {G_loss:.4f}, D: {D_loss:.4f}"
        )

        if neptune_logger is not None:
            neptune_logger["train/Joint/Embedding"].log(E_loss)
            neptune_logger["train/Joint/Generator"].log(G_loss)
            neptune_logger["train/Joint/Discriminator"].log(D_loss)

            if (epoch + 1) % 10 == 0:
                with torch.no_grad():
                    # generate synthetic data and plot it
                    X_hat = model(X=None, Z=fixed_Z_mb, obj="inference")

                    x_axis = np.arange(max_seq_len)
                    fig, axs = plt.subplots(3, 3, figsize=(14, 10))

                    for x in range(3):
                        for y in range(3):
                            axs[x, y].plot(x_axis, X_hat[x * 3 + y].T.cpu().numpy())
                            axs[x, y].set_ylim([0, 1])
                            axs[x, y].set_yticklabels([])

                    fig.suptitle(f"Generation: {epoch}", fontsize=14)
                    # fig.savefig('./images/data_at_epoch_{:04d}.png'.format(epoch))
                    # neptune_logger["generated_image"].upload(fig)
                    neptune_logger["generated_image"].log(fig)
                    sw = sw_approx(x_sw, X_hat)
                    neptune_logger["SW"].log(sw)
                    if sw < best_sw:
                        m_name = model_name[:-3] + "_checkpoint_best_sw.pt"
                        torch.save(model.state_dict(), m_name)
                        neptune_logger["model_checkpoint_best_sw"].upload(m_name)

                    plt.close(fig)
                    # writer.add_figure('Generated data', fig, epoch)
def timegan_trainer(model, dataset, params, neptune_logger=None, continue_training=False):
    """The training procedure for TimeGAN
    Args:
        - model (torch.nn.module): The model that generates synthetic data
        - data (numpy.ndarray): The data for training the model
        - time (numpy.ndarray): The time for the model to be conditioned on
        - args (dict): The model/training configurations
    Returns:
        - generated_data (np.ndarray): The synthetic data generated by the model
    """
    batch_size = params["batch_size"]
    device = params["device"]
    learning_rate = params["l_rate"]
    n_epochs = params["n_epochs"]
    max_seq_len = params["max_seq_len"]
    dis_thresh = params["dis_thresh"]
    model_name = params["model_name"]
    ae_lr = params["l_rate_ae"]
    Z_dim = params["Z_dim"]

    # Initialize TimeGAN dataset and dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True
    )
    if continue_training:
        model.load_state_dict(torch.load(model_name))
        print("Continuing training from previous checkpoint")
    model.to(device)

    # Initialize Optimizers
    e_opt = torch.optim.Adam(model.embedder.parameters(), lr=ae_lr)
    r_opt = torch.optim.Adam(model.recovery.parameters(), lr=ae_lr)
    s_opt = torch.optim.Adam(model.supervisor.parameters(), lr=learning_rate)
    g_opt = torch.optim.Adam(model.generator.parameters(), lr=learning_rate)
    d_opt = torch.optim.Adam(model.discriminator.parameters(), lr=learning_rate)

    if not continue_training:
        print("\nStart Embedding Network Training")
        embedding_trainer(
            model=model,
            dataloader=dataloader,
            e_opt=e_opt,
            r_opt=r_opt,
            n_epochs=500 if n_epochs > 500 else n_epochs,
            neptune_logger=neptune_logger
        )

        print("\nStart Training with Supervised Loss Only")
        supervisor_trainer(
            model=model,
            dataloader=dataloader,
            s_opt=s_opt,
            g_opt=g_opt,
            n_epochs=500 if n_epochs > 500 else n_epochs,
            neptune_logger=neptune_logger
        )

    print("\nStart Joint Training")
    joint_trainer(
        model=model,
        dataloader=dataloader,
        e_opt=e_opt,
        r_opt=r_opt,
        s_opt=s_opt,
        g_opt=g_opt,
        d_opt=d_opt,
        n_epochs=n_epochs,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        Z_dim=Z_dim,
        dis_thresh=dis_thresh,
        neptune_logger=neptune_logger
    )
    # Save model, args, and hyperparameters
    torch.save(model.state_dict(), model_name)
    print(f"Training Complete and {model_name} saved")


import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='cotgan')

    # Dataset params
    parser.add_argument('--dataset',      type=str,   default='sinus', choices=['sinus', 'soliton'])
    # For sinus
    parser.add_argument('--max_seq_len',  type=int,   default=25)
    parser.add_argument('--feature_dim',  type=int,   default=3)
    parser.add_argument('--alpha',        type=float, default=0.7)
    parser.add_argument('--noise',        type=float, default=0.0)
    # For soliton
    parser.add_argument('--P',            type=int,   default=20)
    parser.add_argument('--spatial_len',  type=int,   default=50)
    parser.add_argument('--t_steps',      type=int,   default=5)
    parser.add_argument('--difficulty',   type=str,   default='easy', choices=['easy', 'medium'])
    #parser.add_argument('--t_range',      type=float, default=1.0) # Hard coded
    #parser.add_argument('--c_range',      type=float, default=1.0) # Hard coded

    # Dataset sizes
    parser.add_argument('--trainset_size',type=int,   default=32*2*24)
    parser.add_argument('--testset_size', type=int,   default=32*2*24)

    # Hyperparameters
    parser.add_argument('--model_name', type=str,   default='model_cotgan.pt')
    parser.add_argument('--n_epochs',   type=int,   default=1)
    parser.add_argument('--l_rate',     type=float, default=0.001)
    parser.add_argument('--l_rate_g',   type=float, default=0.001)
    parser.add_argument('--batch_size', type=int,   default=32)
    parser.add_argument('--optimizer',  type=str,   default='Adam')
    parser.add_argument('--beta1',      type=float, default=0.5)
    parser.add_argument('--beta2',      type=float, default=0.9)
    parser.add_argument('--use_opt_scheduler', type=bool, default=False)

    # Model architecture
    parser.add_argument('--rnn_type',           type=str, default='GRU', choices=['GRU', 'LSTM'])
    parser.add_argument('--gen_rnn_num_layers', type=int, default=2)
    parser.add_argument('--gen_rnn_hidden_dim', type=int, default=64)
    parser.add_argument('--dis_rnn_num_layers', type=int, default=2)
    parser.add_argument('--dis_rnn_hidden_dim', type=int, default=64)
    parser.add_argument('--J_dim',              type=int, default=10)
    parser.add_argument('--hidden_dim',         type=int, default=64*2)
    parser.add_argument('--num_hidden_layers',  type=int, default=3)
    parser.add_argument('--Z_dim',              type=int, default=100)
    parser.add_argument('--use_bn',             type=bool,default=False)
    # Loss params
    parser.add_argument('--scaling_coef',     type=float, default=1.0)
    parser.add_argument('--sinkhorn_eps',     type=float, default=0.8)
    parser.add_argument('--sinkhorn_l',       type=int,   default=100)
    parser.add_argument('--reg_lam',          type=float, default=0.01)
    # Other
    parser.add_argument('--device', type=str, default='cpu', choices=['cuda', 'cpu'])
    parser.add_argument('--seed',   type=int, default=1)



    args = parser.parse_args()
    args = vars(args)
    if args["dataset"] == "soliton":
        args["max_seq_len"] = args["t_steps"]
        args["feature_dim"] = args["spatial_len"]
        # TODO (Fix this issue properly)
        if args["difficulty"] == "medium":
            args["dataset"] = "medium_soliton"

    load_dataset_and_train(args)