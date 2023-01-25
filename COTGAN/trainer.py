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
    max_seq_len = params["max_seq_len"]
    Z_dim = params["Z_dim"]

    # Prepare datasets
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size*2,
        shuffle=True
    )
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

    x_sw = torch.concat([x for x, _ in dataloader])
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
            disc_h_opt.zero_grad()
            disc_m_opt.zero_grad()
            D_loss = model(Z_mb, Z_mb_p, X_mb, X_mb_p, obj="discriminator")
            D_loss.backward()
            disc_h_opt.step()
            disc_m_opt.step()

            # Train generator
            gen_opt.zero_grad()
            G_loss = model(Z_mb, Z_mb_p, X_mb, X_mb_p, obj="generator")
            G_loss.backward()
            gen_opt.step()

        logger.set_description(
            f"Epoch: {epoch}, G: {-G_loss:.4f}, D: {-D_loss:.4f}"
        )
        if neptune_logger is not None:
            neptune_logger["train/Generator"].log(-G_loss)
            neptune_logger["train/Discriminator"].log(-D_loss)
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
    torch.save(model.state_dict(), f"./models/{model_name}.pt")
