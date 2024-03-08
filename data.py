import random
from math import floor

import numpy as np

import torch

from scipy.io import loadmat

from sklearn.impute import SimpleImputer


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_uci_data(
    data_dir,
    dataset,
    seed,
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    verbose=False,
):
    if dataset == "airline":
        return load_airline_data(data_dir, seed)

    set_seed(seed)

    if dataset == "airquality":
        from datasets.uci import AirQuality
        X, y = AirQuality(dtype=torch.float32).tensors
    elif dataset == "sgemmgpu":
        from datasets.uci import SGEMMGPU
        X, y = SGEMMGPU(dtype=torch.float32).tensors
    elif dataset == "gassensors":
        from datasets.uci import GasSensors
        X, y = GasSensors(dtype=torch.float32).tensors
    else:
        data = torch.Tensor(loadmat(data_dir + dataset + ".mat")["data"])

        X = data[:, :-1]
        y = data[:, -1]

    good_dimensions = X.var(dim=-2) > 1.0e-10
    if int(good_dimensions.sum()) < X.size(1):
        if verbose:
            print(
                "Removed %d dimensions with no variance"
                % (X.size(1) - int(good_dimensions.sum()))
            )
        X = X[:, good_dimensions]

    if dataset in ["keggundirected", "slice"]:
        X = torch.Tensor(
            SimpleImputer(missing_values=np.nan).fit_transform(X.data.numpy())
        )

    X = X - X.min(0)[0]
    X = 2.0 * (X / X.max(0)[0]) - 1.0
    y -= y.mean()
    y /= y.std()

    shuffled_indices = torch.randperm(X.size(0))
    X = X[shuffled_indices, :]
    y = y[shuffled_indices]

    train_n = int(floor(0.8 * X.size(0)))
    valid_n = 0

    train_x = X[:train_n, :].contiguous().to(device)
    train_y = y[:train_n].contiguous().to(device)

    valid_x = X[train_n: train_n + valid_n, :].contiguous().to(device)
    valid_y = y[train_n: train_n + valid_n].contiguous().to(device)

    test_x = X[train_n + valid_n:, :].contiguous().to(device)
    test_y = y[train_n + valid_n:].contiguous().to(device)

    if verbose:
        print("Loaded data with input dimension of {}".format(test_x.size(-1)))

    return train_x, train_y, test_x, test_y, valid_x, valid_y, None
