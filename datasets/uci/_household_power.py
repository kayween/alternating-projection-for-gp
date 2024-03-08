"""Household power dataset from the UCI machine learning repository."""


from io import BytesIO
from typing import Optional, Tuple
from zipfile import ZipFile

import numpy as np
import pandas as pd
import requests
import torch

from ._uci_dataset import UCIDataset


class HouseholdPower(UCIDataset):
    """Individual household electric power consumption (2,049,280 × 7).

    Measurements of electric power consumption in one household with a one-minute sampling rate over a period of almost 4 years. Different electrical quantities and some sub-metering values are available.

    Source: https://archive-beta.ics.uci.edu/dataset/235/individual+household+electric+power+consumption
    """

    URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00235/"

    def __init__(
        self,
        dir: Optional[str] = "datasets/raw_data/uci/power",
        overwrite: bool = False,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__(dir, overwrite, dtype=dtype, device=device)

    def _download(self) -> torch.Tensor:

        # Download and unzip archive
        r = requests.get(HouseholdPower.URL + "household_power_consumption.zip")
        files = ZipFile(BytesIO(r.content))

        # Read data for the hourly count
        df = pd.read_csv(
            files.open("household_power_consumption.txt"),
            sep=";",
            header=0,
            parse_dates=True,
            low_memory=False,
        )

        # Convert dates to numeric
        df["date_time"] = df["Date"].str.cat(df["Time"], sep=" ")
        df.drop(["Date", "Time"], inplace=True, axis=1)
        df["date_time"] = pd.to_datetime(df["date_time"], dayfirst=True).astype(int)

        # Drop NaNs
        df.replace("?", np.nan, inplace=True)
        df.dropna(inplace=True)

        # Numeric dtypes
        df = df.astype("float")

        return torch.as_tensor(df.values, dtype=self.dtype, device=self.device)

    def _preprocess(
        self,
        raw_data: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # Preprocess
        X = raw_data[:, 1:]
        y = raw_data[:, 0]

        # Transform outputs
        y = y - torch.mean(y, dim=0)

        # Normalize features
        X = (X - torch.mean(X, dim=0)) / torch.std(X, dim=0)

        return X, y
