"""Protein dataset from the UCI machine learning repository."""

from typing import Optional, Tuple

import pandas as pd
import torch

from ._uci_dataset import UCIDataset


class Parkinsons(UCIDataset):
    """Parkinsons Telemonitoring dataset (5,875 × 21). [1]_

    This dataset is composed of a range of biomedical voice measurements from 42 people
    with early-stage Parkinson's disease recruited to a six-month trial of a
    telemonitoring device for remote symptom progression monitoring. The recordings were
    automatically captured in the patient's homes. The original study used a range of
    linear and nonlinear regression methods to predict the clinician's Parkinson's
    disease symptom score on the UPDRS scale.

    Source: https://archive.ics.uci.edu/ml/datasets/parkinsons+telemonitoring

    References
    ----------
    .. [1] A Tsanas, MA Little, PE McSharry, LO Ramig (2009) 'Accurate telemonitoring of
           Parkinson's disease progression by non-invasive speech tests', IEEE
           Transactions on Biomedical Engineering
    """

    URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/telemonitoring/"

    def __init__(
        self,
        dir: Optional[str] = "datasets/raw_data/uci/parkinsons",
        overwrite: bool = False,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__(dir, overwrite, dtype=dtype, device=device)

    def _download(self) -> torch.Tensor:

        # Read data
        df = pd.read_csv(Parkinsons.URL + "parkinsons_updrs.data")
        df.drop(["motor_UPDRS"], axis=1)

        # Move column to predict
        column_to_move = df.pop("total_UPDRS")
        df.insert(0, "total_UPDRS", column_to_move)

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
