"""Base class for UCI datasets."""
import abc
import os
import pathlib
from typing import Optional, Tuple

import torch
from torch.utils.data import TensorDataset


class UCIDataset(TensorDataset, abc.ABC):
    """Dataset from the UCI repository.

    Parameters
    ----------
    dir
        Directory where data is retrieved from or saved to. If ``None``, data is not
        saved to file.
    overwrite
        Whether to overwrite any potentially saved data on disk.
    dtype
        Data type.
    device
        Device.
    """

    def __init__(
        self,
        dir: Optional[str] = "datasets/raw_data/uci",
        overwrite: bool = False,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> None:

        self.dtype = dtype
        self.device = device

        file = dir + "/" + self.__class__.__name__.lower() + ".pt"

        if os.path.isfile(file) and not overwrite:
            # Load data from file
            X, y = torch.load(file, map_location=self.device)[:]
        else:
            # Download data from the web
            raw_data = self._download()
            X, y = self._preprocess(raw_data)

            # Save to disk if data does not yet exist on disk
            if dir is not None:
                pathlib.Path(dir).mkdir(parents=True, exist_ok=True)
                torch.save(TensorDataset(X, y), file)

        super().__init__(
            X.to(dtype=self.dtype, device=self.device),
            y.to(dtype=self.dtype, device=self.device),
        )

    @abc.abstractmethod
    def _download(self) -> torch.Tensor:
        """Download data from the UCI repository."""
        raise NotImplementedError

    @abc.abstractmethod
    def _preprocess(
        self,
        raw_data: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Preprocess and normalize data."""
        raise NotImplementedError
