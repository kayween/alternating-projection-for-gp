"""Bike sharing dataset from the UCI machine learning repository."""


from io import BytesIO
from typing import Optional, Tuple
from zipfile import ZipFile

import pandas as pd
import requests
import torch

from ._uci_dataset import UCIDataset


class BikeSharing(UCIDataset):
    """Bike sharing dataset (17,379 × 16). [1]_

    This dataset contains the hourly (and daily) count of rental bikes between years
    2011 and 2012 of the Capital bikeshare system with the corresponding weather and
    seasonal information.

    Source: https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset

    References
    ----------
    .. [1] Fanaee-T, Hadi, and Gama, Joao, "Event labeling combining ensemble detectors
           and background knowledge", Progress in Artificial Intelligence (2013): pp.
           1-15, Springer Berlin Heidelberg, doi:10.1007/s13748-013-0040-3.
    """

    URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00275/"

    def __init__(
        self,
        dir: Optional[str] = "datasets/raw_data/uci/bike",
        overwrite: bool = False,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__(dir, overwrite, dtype=dtype, device=device)

    def _download(self) -> torch.Tensor:
        # Download and unzip archive
        r = requests.get(BikeSharing.URL + "Bike-Sharing-Dataset.zip")
        files = ZipFile(BytesIO(r.content))

        # Read data for the hourly count
        df = pd.read_csv(files.open("hour.csv"))

        # Convert dates to numeric
        df["dteday"] = pd.to_datetime(df["dteday"]).astype(int)

        return torch.as_tensor(df.values, dtype=self.dtype, device=self.device)

    def _preprocess(
        self,
        raw_data: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Preprocess
        X = raw_data[:, 0:-1]
        y = raw_data[:, -1]

        # Transform outputs
        y = torch.log(y)
        y = y - torch.mean(y, dim=0)

        # Normalize features
        X = (X - torch.mean(X, dim=0)) / torch.std(X, dim=0)

        return X, y
