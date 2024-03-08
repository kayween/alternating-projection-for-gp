"""Gas sensor dataset from the UCI machine learning repository."""


from io import BytesIO
from typing import Optional, Tuple
from zipfile import ZipFile

import numpy as np
import pandas as pd
import requests
import torch

from ._uci_dataset import UCIDataset


class GasSensors(UCIDataset):
    """Gas sensor array under dynamic gas mixtures (4,178,504 × 17).

    Recordings of 16 chemical sensors exposed to a dynamic gas mixture of Methane and Ethylene in air at varying concentrations. Measurements were constructed by the continuous acquisition of the 16-sensor array signals for a duration of about 12 hours without interruption. The regression task is to estimate the concentration of Methane (in ppm) given the time and results of the sensor recordings.

    Source: https://archive-beta.ics.uci.edu/dataset/322/gas+sensor+array+under+dynamic+gas+mixtures
    """

    URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00322/"

    def __init__(
        self,
        dir: Optional[str] = "datasets/raw_data/uci/gas",
        overwrite: bool = False,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__(dir, overwrite, dtype=dtype, device=device)

    def _download(self) -> torch.Tensor:
        # Download and unzip archive
        r = requests.get(GasSensors.URL + "data.zip")
        files = ZipFile(BytesIO(r.content))

        # Read data for the gas mixture measurements
        df = pd.read_csv(
            files.open("ethylene_methane.txt"),
            header=None,
            names=["Time (seconds)", "Methane conc (ppm)", "Ethylene conc (ppm)"]
            + [f"Sensor {i}" for i in range(16)],
            skiprows=1,
            delim_whitespace=True,
        )

        return torch.as_tensor(df.values, dtype=self.dtype, device=self.device)

    def _preprocess(
        self,
        raw_data: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Preprocess
        X = raw_data[:, 3:]
        # X = torch.hstack((X[:, 0].reshape(-1, 1), X))
        y = raw_data[:, 1]

        # Transform outputs
        y = y - torch.mean(y, dim=0)

        # Normalize features
        X = (X - torch.mean(X, dim=0)) / torch.std(X, dim=0)

        return X, y
