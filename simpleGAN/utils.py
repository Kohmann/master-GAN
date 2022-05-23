from torch.utils.data import DataLoader


def get_data(train_ds, valid_ds, bs):
    # returns dataloaders for the training and validation sets.
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(valid_ds, batch_size=bs * 2),
    )