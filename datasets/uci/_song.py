"""Song dataset from the UCI machine learning repository."""

from io import BytesIO
from typing import Optional, Tuple
from zipfile import ZipFile

import pandas as pd
import requests
import torch

from ._uci_dataset import UCIDataset


class Song(UCIDataset):
    """Song dataset (515,345 × 90).

    Prediction of the release year of a song from audio features. Songs are mostly western, commercial tracks ranging from 1922 to 2011, with a peak in the year 2000s.

    Source: https://archive.ics.uci.edu/ml/datasets/yearpredictionmsd)
    """

    URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00203/"

    def __init__(
        self,
        dir: Optional[str] = "datasets/raw_data/uci/song",
        overwrite: bool = False,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__(dir, overwrite, dtype=dtype, device=device)

    def _download(self) -> torch.Tensor:

        # Download and unzip archive
        r = requests.get(Song.URL + "YearPredictionMSD.txt.zip")
        files = ZipFile(BytesIO(r.content))

        # Read data for the hourly count
        df = pd.read_csv(files.open("YearPredictionMSD.txt"), header=None)

        return torch.as_tensor(df.values, dtype=self.dtype, device=self.device)

    def _preprocess(
        self,
        raw_data: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # Preprocess
        X = raw_data[:, 1::]
        y = raw_data[:, 0]

        # Transform outputs
        y = y - torch.mean(y, dim=0)

        # Normalize features
        X = (X - torch.mean(X, dim=0)) / torch.std(X, dim=0)

        return X, y
