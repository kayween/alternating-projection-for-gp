"""Protein dataset from the UCI machine learning repository."""

from typing import Optional, Tuple

import pandas as pd
import torch

from ._uci_dataset import UCIDataset


class Protein(UCIDataset):
    """Protein dataset (45,730 × 9).

    This is a data set of Physicochemical Properties of Protein Tertiary Structure. The
    data set is taken from CASP 5-9. There are 45730 decoys and size varying from 0 to
    21 armstrong.

    Source: https://archive.ics.uci.edu/ml/datasets/Physicochemical+Properties+of+Protein+Tertiary+Structure
    """

    URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00265/"

    def __init__(
        self,
        dir: Optional[str] = "datasets/raw_data/uci/protein",
        overwrite: bool = False,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__(dir, overwrite, dtype=dtype, device=device)

    def _download(self) -> torch.Tensor:

        # Read data
        df = pd.read_csv(Protein.URL + "CASP.csv")

        return torch.as_tensor(df.values, dtype=self.dtype, device=self.device)

    def _preprocess(
        self,
        raw_data: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # Preprocess
        X = raw_data[:, 1::]
        y = raw_data[:, 0]

        # Transform outputs
        y = torch.log(y + 1)
        y = y - torch.mean(y, dim=0)

        # Normalize features
        X = (X - torch.mean(X, dim=0)) / torch.std(X, dim=0)

        return X, y
