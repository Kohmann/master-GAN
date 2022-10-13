import numpy as np
import torch
import matplotlib.pyplot as plt


class minmaxscaler:
    def __init__(self):
        self.matrix_min = None
        self.matrix_max = None
        self.denom = None

    def fit(self, X):
        self.matrix_min = X.min(axis=1, keepdims=True).values
        self.matrix_max = X.max(axis=1, keepdims=True).values
        self.denom = self.matrix_max - self.matrix_min

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def transform(self, X):
        return (X - self.matrix_min) / self.denom

    def inverse_transform(self, X):
        return X * self.denom + self.matrix_min


def create_sin3(sin1, sin2, temporal=False):
    sin1, sin2 = np.array(sin1), np.array(sin2)
    e = 0.7  # temporal information weight
    seq_len = len(sin1)
    importance = np.array([e ** i for i in range(seq_len)])

    if temporal:
        sin3 = []
        for i in range(1, seq_len + 1):
            sin3.append(((importance[:i][::-1] * sin1[:i] + importance[:i][::-1] * sin2[:i]) / 2).sum())
        return sin3
    else:
        return [(s1 + s2) / 2 for s1, s2 in zip(sin1, sin2)]


def sine_data_generation(no, seq_len, dim=3, temporal=False):
    """Sine data generation.

    Args:
      - no: the number of samples
      - seq_len: sequence length of the time-series
      - dim: feature dimensions
      - temporal: whether to add temporal information

    Returns:
      - data: generated data
    """
    # Initialize the output
    data = list()
    # Generate sine data
    for i in range(no):
        # Initialize each time-series
        temp = []
        # For each feature

        # Randomly drawn frequency and phase
        freq1 = np.random.uniform(0.05, 0.15)
        phase1 = np.random.uniform(-np.pi / 2, 0)
        sin1 = np.sin(np.arange(seq_len) * freq1 + phase1).tolist()

        freq2 = np.random.uniform(0.3, 0.4)
        phase2 = np.random.uniform(0, np.pi / 2)
        sin2 = np.sin(np.arange(seq_len) * freq2 + phase2).tolist()

        sin3 = create_sin3(sin1, sin2, temporal)
        temp.extend([sin1, sin2, sin3])

        # Align row/column
        temp = torch.tensor(temp).transpose(0, 1)
        # Normalize to [0,1]
        # temp = (temp + 1) * 0.5
        # Stack the generated data
        data.append(temp)

    return torch.stack(data)


class TimeGANDatasetSinus(torch.utils.data.Dataset):
    """TimeGAN Dataset for sampling data with their respective time
    Args:
        - data (numpy.ndarray): the padded dataset to be fitted (D x S x F)
        - time (numpy.ndarray): the length of each data (D)
    Parameters:
        - x (torch.FloatTensor): the real value features of the data
        - t (torch.LongTensor): the temporal feature of the data
    """

    def __init__(self, num, seq_len, features, temporal=False):
        # sanity check
        self.X_raw = sine_data_generation(num, seq_len, features, temporal=temporal)
        self.X_scaler = minmaxscaler()
        self.X = self.X_scaler.fit_transform(self.X_raw)
        # self.X = torch.tensor(sine_data_generation(num, seq_len, features, temporal=temporal),
        #                      dtype=torch.float32).clone().detach_()
        self.T = [x.size(0) for x in self.X]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx].float(), self.T[idx]

    def collate_fn(self, batch):
        """Minibatch sampling
        """
        # Pad sequences to max length
        X_mb = [X for X in batch[0]]

        # The actual length of each data
        T_mb = [T for T in batch[1]]

        return X_mb, T_mb


#######################################################################
# Visualization

# Commented out IPython magic to ensure Python compatibility.
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


def visualization(ori_data, generated_data, analysis):
    """Using PCA or tSNE for generated and original data visualization.

    Args:
      - ori_data: original data
      - generated_data: generated synthetic data
      - analysis: tsne or pca
    """
    # Analysis sample size (for faster computation)
    anal_sample_no = min([1000, len(ori_data)])
    idx = np.random.permutation(len(ori_data))[:anal_sample_no]

    # Data preprocessing
    ori_data = np.asarray(ori_data)
    generated_data = np.asarray(generated_data)

    ori_data = ori_data[idx]
    generated_data = generated_data[idx]

    no, seq_len, dim = ori_data.shape

    for i in range(anal_sample_no):
        if (i == 0):
            prep_data = np.reshape(np.mean(ori_data[0, :, :], 1), [1, seq_len])
            prep_data_hat = np.reshape(np.mean(generated_data[0, :, :], 1), [1, seq_len])
        else:
            prep_data = np.concatenate((prep_data,
                                        np.reshape(np.mean(ori_data[i, :, :], 1), [1, seq_len])))
            prep_data_hat = np.concatenate((prep_data_hat,
                                            np.reshape(np.mean(generated_data[i, :, :], 1), [1, seq_len])))

    # Visualization parameter
    colors = ["tab:blue" for i in range(anal_sample_no)] + ["tab:orange" for i in range(anal_sample_no)]

    if analysis == 'pca':
        # PCA Analysis
        pca = PCA(n_components=2)
        pca.fit(prep_data)
        pca_results = pca.transform(prep_data)
        pca_hat_results = pca.transform(prep_data_hat)

        # Plotting
        f, ax = plt.subplots(1)
        plt.scatter(pca_results[:, 0], pca_results[:, 1],
                    c=colors[:anal_sample_no], alpha=0.2, label="Original")
        plt.scatter(pca_hat_results[:, 0], pca_hat_results[:, 1],
                    c=colors[anal_sample_no:], alpha=0.2, label="Synthetic")

        ax.legend()
        plt.title('PCA plot')
        plt.xlabel('x-pca')
        plt.ylabel('y_pca')
        # plt.show()
        return f

    elif analysis == 'tsne':

        # Do t-SNE Analysis together
        prep_data_final = np.concatenate((prep_data, prep_data_hat), axis=0)

        # TSNE anlaysis
        tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300)
        tsne_results = tsne.fit_transform(prep_data_final)

        # Plotting
        f, ax = plt.subplots(1)

        plt.scatter(tsne_results[:anal_sample_no, 0], tsne_results[:anal_sample_no, 1],
                    c=colors[:anal_sample_no], alpha=0.2, label="Original")
        plt.scatter(tsne_results[anal_sample_no:, 0], tsne_results[anal_sample_no:, 1],
                    c=colors[anal_sample_no:], alpha=0.2, label="Synthetic")

        ax.legend()

        plt.title('t-SNE plot')
        plt.xlabel('x-tsne')
        plt.ylabel('y_tsne')
        # plt.show()
        return f


def modeCollapseEvaluator(ori_data, generated_data):
    # Analysis sample size (for faster computation)
    anal_sample_no = min([1000, len(ori_data)])
    idx = np.random.permutation(len(ori_data))[:anal_sample_no]

    # Data preprocessing
    ori_data = np.asarray(ori_data)
    generated_data = np.asarray(generated_data)

    ori_data = ori_data[idx]
    generated_data = generated_data[idx]

    no, seq_len, dim = ori_data.shape

    for i in range(anal_sample_no):
        if (i == 0):
            prep_data = np.reshape(np.mean(ori_data[0, :, :], 1), [1, seq_len])
            prep_data_hat = np.reshape(np.mean(generated_data[0, :, :], 1), [1, seq_len])
        else:
            prep_data = np.concatenate((prep_data,
                                        np.reshape(np.mean(ori_data[i, :, :], 1), [1, seq_len])))
            prep_data_hat = np.concatenate((prep_data_hat,
                                            np.reshape(np.mean(generated_data[i, :, :], 1), [1, seq_len])))

    # PCA Analysis
    pca = PCA(n_components=2)
    pca.fit(prep_data)
    pca_results = pca.transform(prep_data)
    pca_hat_results = pca.transform(prep_data_hat)

    real_std = pca_results.std(axis=0)
    fake_std = pca_hat_results.std(axis=0)
    print("Real std: ", real_std)
    print("Fake std: ", fake_std)
    if np.all(real_std / fake_std) < 2:
        return False
    else:
        return True
