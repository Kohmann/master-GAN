import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

from metrics import sw_approx


def embedding_trainer(model, dataloader, e_opt, r_opt, n_epochs, neptune_logger=None):
    logger = trange(n_epochs, desc=f"Epoch: 0, Loss: 0")
    for epoch in logger:
        for X_mb, T_mb in dataloader:
            model.zero_grad()

            _, E_loss0, E_loss_T0 = model(X=X_mb, T=T_mb, Z=None, obj="autoencoder")
            loss = np.sqrt(E_loss_T0.item())

            E_loss0.backward()
            e_opt.step()
            r_opt.step()

        logger.set_description(f"Epoch: {epoch}, Loss: {loss:.4f}")
        if epoch % 5 == 0:
            neptune_logger["train/Embedding"].log(loss)


def supervisor_trainer(model, dataloader, s_opt, g_opt, n_epochs, neptune_logger=None):
    logger = trange(n_epochs, desc=f"Epoch: 0, Loss: 0")
    for epoch in logger:
        for X_mb, T_mb in dataloader:
            model.zero_grad()

            S_loss = model(X=X_mb, T=T_mb, Z=None, obj="supervisor")
            loss = np.sqrt(S_loss.item())

            S_loss.backward()
            s_opt.step()

        logger.set_description(f"Epoch: {epoch}, Loss: {loss:.4f}")
        if epoch % 5 == 0:
            neptune_logger["train/Supervisor"].log(loss)
            # writer.add_scalar("Loss/Supervisor", loss, epoch)


def joint_trainer(model, dataloader, e_opt, r_opt, s_opt, g_opt, d_opt, n_epochs, batch_size, max_seq_len, Z_dim,
                  dis_thresh, neptune_logger=None):
    x_sw = torch.concat([x for x, _ in dataloader])
    n_samples = len(x_sw)
    fixed_Z_mb = torch.rand((n_samples, max_seq_len, Z_dim))
    logger = trange(n_epochs, desc=f"Epoch: 0, E_loss: 0, G_loss: 0, D_loss: 0")
    best_sw = 1000.0
    model_name = neptune_logger["parameters/model_name"].fetch()

    for epoch in logger:
        for X_mb, T_mb in dataloader:
            for _ in range(2):
                #
                Z_mb = torch.rand(X_mb.size(0), max_seq_len, Z_dim)
                model.zero_grad()
                # Generator
                G_loss = model(X=X_mb, T=T_mb, Z=Z_mb, obj="generator")
                G_loss.backward()
                G_loss = np.sqrt(G_loss.item())

                g_opt.step()
                s_opt.step()

                # Embedding
                model.zero_grad()
                E_loss, _, E_loss_T0 = model(X=X_mb, T=T_mb, Z=Z_mb, obj="autoencoder")
                E_loss.backward()
                E_loss = np.sqrt(E_loss.item())

                e_opt.step()
                r_opt.step()

            Z_mb = torch.rand((batch_size, max_seq_len, Z_dim))
            model.zero_grad()
            D_loss = model(X=X_mb, T=T_mb, Z=Z_mb, obj="discriminator")
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
                    X_hat = model(X=None, Z=fixed_Z_mb, T=[max_seq_len for _ in range(n_samples)], obj="inference")

                    x_axis = np.arange(max_seq_len)
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
            n_epochs= 500 if n_epochs > 500 else n_epochs,
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
        Z_dim=100,
        dis_thresh=dis_thresh,
        neptune_logger=neptune_logger
    )
    # Save model, args, and hyperparameters
    torch.save(model.state_dict(), model_name)
    print(f"Training Complete and {model_name} saved")


def rtsgan_autoencoder_trainer(model, dataloader, val_dataset, e_opt, d_opt, n_epochs, neptune_logger=None):

    #if True:
    #    model.load_ae()
    #    return 0

    n_epochs = 350 if n_epochs > 350 else n_epochs
    logger = trange(n_epochs, desc=f"Epoch: 0, Loss: 0")
    loss = 0
    for epoch in logger:
        for X_mb, _ in dataloader:
            model.zero_grad()

            reconstruction_loss = model(X=X_mb, Z=None, obj="autoencoder")
            loss = reconstruction_loss.item()
            reconstruction_loss.backward()
            e_opt.step()
            d_opt.step()

        logger.set_description(f"Epoch: {epoch}, Loss: {loss:.4f}")
        
        if neptune_logger is not None:
            neptune_logger["train/Autoencoder"].log(loss)
            if val_dataset is not None:
                with torch.no_grad():
                    reconstruction_loss = model(X=val_dataset[:][0], Z=None, obj="autoencoder")
                neptune_logger["train/Autoencoder_val"].log(reconstruction_loss.item())

#    inputs, _ = next(iter(dataloader))
#    "rtsgan_encoder" +str(inputs[1])+ ".pt"
#    torch.save(model.encoder.state_dict(), "rtsgan_encoder" +str(inputs.size()[1])+ ".pt")
#    torch.save(model.decoder.state_dict(), "rtsgan_decoder" +str(inputs.size()[1])+ ".pt")
#    print("Saved autoencoder")

def rtsgan_gan_trainer(model, dataloader, gen_opt, disc_opt, n_epochs, d_steps, device, Z_dim, neptune_logger=None):
    # Wasserstein training

    x_sw = torch.concat([x for x, _ in dataloader])
    n_samples = len(x_sw)
    logger = trange(n_epochs, desc=f"Epoch: 0, G_loss: 0, D_loss: 0")
    fixed_Z_mb = torch.randn(n_samples, Z_dim, device=device)

    step = 0
    for epoch in logger:
        for X_mb, _ in dataloader:
            step += 1
            model.zero_grad()
            Z = torch.randn(X_mb.shape[0], Z_dim, device=device)
            D_loss = model(X=X_mb, Z=Z, obj="discriminator")
            D_loss.backward()
            disc_opt.step()

            if d_steps % step  == 0:
                model.zero_grad()
                Z = torch.randn(X_mb.shape[0], Z_dim, device=device)
                G_loss = model(X=None, Z=Z, obj="generator")
                G_loss.backward()
                gen_opt.step()
                step = 0

        logger.set_description(
            f"Epoch: {epoch}, G: {-G_loss:.4f}, D: {-D_loss:.4f}"
        )

        if neptune_logger is not None:
            # neptune_logger["train/Joint/Embedding"].log(E_loss)
            neptune_logger["train/Joint/Generator"].log(-G_loss)
            neptune_logger["train/Joint/Discriminator"].log(-D_loss)
            if (epoch + 1) % 10 == 0:
                with torch.no_grad():
                    # generate synthetic data and plot it
                    X_hat = model(X=None, Z=fixed_Z_mb, obj="inference")
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


def rtsgan_trainer(model, dataset, params, val_dataset=None, neptune_logger=None, continue_training=False):
    """The training procedure for RTS-GAN
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
    l_rate_g = params["l_rate_g"]
    learning_rate_ae = params["l_rate_ae"]
    n_epochs = params["n_epochs"]
    model_name = params["model_name"]
    d_steps = params["d_steps"]

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
    encoder_opt = torch.optim.Adam(model.encoder.parameters(), lr=learning_rate_ae)  # betas=(0.9, 0.999) by default
    decoder_opt = torch.optim.Adam(model.decoder.parameters(), lr=learning_rate_ae)

    # RMSprop is used in the original paper
    if params["optimizer"] == "Adam":
        gen_opt = torch.optim.Adam(model.generator.parameters(), lr=l_rate_g, betas=(0, 0.999))
        disc_opt = torch.optim.Adam(model.discriminator.parameters(), lr=learning_rate, betas=(0, 0.999))
        print("Using optimizer: Adam")
    else:
        print("Using optimizer: RMSprop")
        gen_opt = torch.optim.RMSprop(model.generator.parameters(), lr=l_rate_g)
        disc_opt = torch.optim.RMSprop(model.discriminator.parameters(), lr=learning_rate)

    if not continue_training:
        print("\nStart Embedding Network Training")
        rtsgan_autoencoder_trainer(
            model=model,
            dataloader=dataloader,
            val_dataset=val_dataset,
            e_opt=encoder_opt,
            d_opt=decoder_opt,
            n_epochs=n_epochs,
            neptune_logger=neptune_logger
        )

    print("\nStart GAN Training")
    rtsgan_gan_trainer(
        model=model,
        dataloader=dataloader,
        gen_opt=gen_opt,
        disc_opt=disc_opt,
        n_epochs=n_epochs,
        d_steps=d_steps,
        device=device,
        Z_dim=params["Z_dim"],
        neptune_logger=neptune_logger
    )

    # Save model, args, and hyperparameters
    torch.save(model.state_dict(), model_name)
    print(f"Training Complete and {model_name} saved")


def rtsgan_generator(model, params, eval=False):
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
    trainset_size = params["testset_size"]
    if not eval:
        model.load_state_dict(torch.load(model_name, map_location=device))

    print("\nGenerating Data...", end="")
    # Initialize model to evaluation mode and run without gradients
    model.to(device)
    model.eval()
    with torch.no_grad():
        # Generate fake data
        Z = torch.rand((trainset_size, Z_dim), device=device)
        generated_data = model(X=None, Z=Z, obj="inference")
    print("Done")
    return generated_data.cpu().numpy()


def timegan_generator(model, T, model_path, device, max_seq_len, Z_dim):
    """The inference procedure for TimeGAN
    Args:
        - model (torch.nn.module): The model that generates synthetic data
        - T (List[int]): The time to be generated on
        - args (dict): The model/training configurations
    Returns:
        - generated_data (np.ndarray): The synthetic data generated by the model
    """
    # Load model for inference
    if not eval:
        model.load_state_dict(torch.load(model_path, map_location=device))

    print("\nGenerating Data...", end="")
    # Initialize model to evaluation mode and run without gradients
    model.to(device)
    model.eval()
    with torch.no_grad():
        # Generate fake data
        Z = torch.rand((len(T), max_seq_len, Z_dim))

        generated_data = model(X=None, T=T, Z=Z, obj="inference")
    print("Done")
    return generated_data.numpy()


def timegan_generate_data(model, T, max_seq_len, Z_dim):
    """The inference procedure for TimeGAN
    Args:
        - model (torch.nn.module): The model that generates synthetic data
        - T (List[int]): The time to be generated on
        - args (dict): The model/training configurations
    Returns:
        - generated_data (np.ndarray): The synthetic data generated by the model
    """
    model.eval()
    with torch.no_grad():
        Z = torch.rand((len(T), max_seq_len, Z_dim))
        generated_data = model(X=None, T=T, Z=Z, obj="inference")
    return generated_data.numpy()


# RGAN
def rgan_trainer(model, dataset, batch_size, device, learning_rate, n_epochs, max_seq_len, dis_thresh,
                 continue_training=False, neptune_logger=None, model_name="model.pt"):
    """Traniner for RGAN"""

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
    g_opt = torch.optim.Adam(model.generator.parameters(), lr=learning_rate)
    d_opt = torch.optim.Adam(model.discriminator.parameters(), lr=learning_rate)
    Z_dim = 100

    print("\nStart Training")
    x_sw = torch.concat([x for x, _ in dataloader])
    n_samples = len(x_sw)
    fixed_Z_mb = torch.randn(n_samples,max_seq_len, Z_dim, device=device)

    #fixed_Z_mb = torch.rand((9, max_seq_len, Z_dim), device=device)
    logger = trange(n_epochs, desc=f"Epoch: 0, G_loss: 0, D_loss: 0")
    for epoch in logger:
        for X_mb, T_mb in dataloader:

            Z_mb = torch.rand(X_mb.size(0), max_seq_len, Z_dim, device=device)
            X_mb = X_mb.to(device)

            # Discriminator
            model.zero_grad()
            D_loss, _ = model(X=X_mb, T=T_mb, Z=Z_mb, gamma=100)
            if D_loss > dis_thresh:  # don't let the discriminator get too good
                D_loss.backward()
                d_opt.step()
            # D_loss = D_loss.item()

            # Generator
            model.zero_grad()
            _, G_loss = model(X=X_mb, Z=Z_mb, T=T_mb, gamma=100)
            G_loss.backward()
            g_opt.step()
            # G_loss = G_loss.item()

        logger.set_description(
            f"Epoch: {epoch}, G: {G_loss.item():.4f}, D: {D_loss.item():.4f}"
        )

        if neptune_logger is not None:
            neptune_logger["train/Joint/Generator"].log(G_loss)
            neptune_logger["train/Joint/Discriminator"].log(D_loss)
            if (epoch + 1) % 10 == 0:
                with torch.no_grad():
                    # generate synthetic data and plot it
                    X_hat = model.generate(Z=fixed_Z_mb, T=[max_seq_len for _ in range(n_samples)])
                    x_axis = np.arange(X_hat.size(dim=1))
                    fig, axs = plt.subplots(3, 3, figsize=(14, 10))

                    for x in range(3):
                        for y in range(3):
                            axs[x, y].plot(x_axis, X_hat[x * 3 + y].cpu().numpy())
                            axs[x, y].set_ylim([0, 1])
                            axs[x, y].set_yticklabels([])

                    fig.suptitle(f"Generation: {epoch}", fontsize=14)
                    neptune_logger["generated_image"].log(fig)
                    neptune_logger["SW"].log(sw_approx(x_sw, X_hat.cpu()))
                    plt.close(fig)

    # Save model, args, and hyperparameters
    torch.save(model.state_dict(), model_name)
    print(f"Training Complete and {model_name} saved")


def rgan_generator(model, T, model_path, device, max_seq_len, Z_dim):
    """The inference procedure for TimeGAN
    Args:
        - model (torch.nn.module): The model that generates synthetic data
        - T (List[int]): The time to be generated on
        - args (dict): The model/training configurations
    Returns:
        - generated_data (np.ndarray): The synthetic data generated by the model
    """
    # Load model for inference

    model.load_state_dict(torch.load(model_path, map_location=device))

    print("\nGenerating Data...", end="")
    # Initialize model to evaluation mode and run without gradients
    model.to(device)
    model.eval()
    with torch.no_grad():
        # Generate fake data
        Z = torch.rand((len(T), max_seq_len, Z_dim), device=device)

        generated_data = model.generate(Z=Z, T=T.cpu())
    print("Done")
    return generated_data.cpu().numpy()

def rgan_generate_data(model, T, max_seq_len, Z_dim):

    model.eval()
    with torch.no_grad():
        Z = torch.rand((len(T), max_seq_len, Z_dim))
        generated_data = model.generate(Z=Z, T=T.cpu())
    return generated_data.numpy()