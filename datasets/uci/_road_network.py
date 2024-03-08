"""Road network dataset from the UCI machine learning repository."""


from typing import Optional, Tuple

import pandas as pd
import torch

from ._uci_dataset import UCIDataset


class RoadNetwork(UCIDataset):
    """3D Road Network (North Jutland, Denmark) (434,874 × 3).

    Dataset of longitude, latitude and altitude values of a road network in
    North Jutland, Denmark (covering a region of 185x135 km2). Elevation values where
    extracted from a publicly available massive Laser Scan Point Cloud for Denmark. This
    3D road network was eventually used for benchmarking various fuel and CO2 estimation
    algorithms.

    Source: https://archive.ics.uci.edu/ml/datasets/3D+Road+Network+(North+Jutland,+Denmark)
    """

    URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00246/"

    def __init__(
        self,
        dir: Optional[str] = "datasets/raw_data/uci/3droad",
        overwrite: bool = False,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__(dir, overwrite, dtype=dtype, device=device)

    def _download(self) -> torch.Tensor:

        # Read data
        df = pd.read_csv(
            RoadNetwork.URL + "3D_spatial_network.txt",
            header=None,
            names=["OSM_ID", "longitude", "latitude", "altitude"],
        )

        return torch.as_tensor(df.values, dtype=self.dtype, device=self.device)

    def _preprocess(
        self,
        raw_data: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # Preprocess
        X = raw_data[:, 1:-1]
        y = raw_data[:, -1]

        # Transform outputs
        y = y - torch.mean(y, dim=0)

        # Normalize features
        X = (X - torch.mean(X, dim=0)) / torch.std(X, dim=0)

        return X, y
