"""Synthetic dataset for Gaussian process regression."""

from typing import Dict, Optional

import gpytorch
import torch
from torch.utils.data import TensorDataset


class SyntheticDataset(TensorDataset):
    """Creates a synthetic dataset from a latent function defined as a linear combination of kernel functions centered at random datapoints.

    Parameters
    ----------
    X
        Training and test inputs with shape ``(n_train + n_test, input_dim)``.
    kernel
        Kernel defining the latent process.
    num_kernel_fns
        Number of kernel functions to linearly combine.
    lengthscale
        Lengthscale of the kernel.
    noise
        Observation noise.
    dtype
        Data type. Defaults to data type of :attr:`X`.
    device
        Device. Defaults to device of :attr:`X`.
    """

    def __init__(
        self,
        X: torch.Tensor,
        kernel: gpytorch.kernels.Kernel,
        num_kernel_fns: int = 10,
        noise: float = 1e-2,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        # Set dtype and device
        if dtype is None:
            dtype = X.dtype
        if device is None:
            device = X.device

        with torch.no_grad():
            kernel.to(device)

            # Sample locations of kernel functions
            X_min = torch.min(X)
            X_max = torch.max(X)

            X_kernel_fns = (
                torch.rand(num_kernel_fns, X.shape[1], device=device, dtype=dtype)
                * (X_max - X_min)
                + X_min
            )

            # Sample representer weights
            representer_weights = torch.randn(
                num_kernel_fns, dtype=dtype, device=device
            )
            representer_weights = representer_weights.div(
                torch.linalg.vector_norm(representer_weights)
            )

            # Create latent fnction
            self.latent_fn = lambda x: kernel(x, X_kernel_fns) @ representer_weights

            # Generate data from latent function
            y = self.latent_fn(X) + torch.sqrt(
                torch.as_tensor(noise, dtype=dtype, device=device)
            ) * torch.randn(X.shape[0], dtype=dtype, device=device)

            super().__init__(X, y)

    @classmethod
    def from_size_and_dim(
        cls,
        n_train_plus_n_test: int,
        input_dim: int,
        kernel: gpytorch.kernels.Kernel,
        num_kernel_fns: int = 10,
        noise: float = 1e-2,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        """Create a synthetic dataset in the unit cube [0, 1)^d for given dimension and number of datapoints.

        Parameters
        ----------
        n_train_plus_n_test
            Number of training and test datapoints.
        input_dim
            Input dimension.
        kernel
            Kernel defining the latent process.
        num_kernel_fns
            Number of kernel functions to linearly combine.
        lengthscale
            Lengthscale of the kernel.
        noise
            Observation noise.
        dtype
            Data type. Defaults to data type of :attr:`X`.
        device
            Device. Defaults to device of :attr:`X`.
        """
        return cls(
            X=torch.rand(n_train_plus_n_test, input_dim, dtype=dtype, device=device),
            kernel=kernel,
            num_kernel_fns=num_kernel_fns,
            noise=noise,
            dtype=dtype,
            device=device,
        )
