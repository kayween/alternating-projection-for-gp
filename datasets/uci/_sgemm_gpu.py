"""Gas sensor dataset from the UCI machine learning repository."""


from io import BytesIO
from typing import Optional, Tuple
from zipfile import ZipFile

import numpy as np
import pandas as pd
import requests
import torch

from ._uci_dataset import UCIDataset


class SGEMMGPU(UCIDataset):
    URL = "https://archive.ics.uci.edu/static/public/440/sgemm+gpu+kernel+performance.zip"

    def __init__(
        self,
        dir: Optional[str] = "datasets/raw_data/uci/sgemmgpu",
        overwrite: bool = False,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__(dir, overwrite, dtype=dtype, device=device)

    def _download(self) -> torch.Tensor:
        # Download and unzip archive
        r = requests.get(SGEMMGPU.URL)
        files = ZipFile(BytesIO(r.content))

        df = pd.read_csv(files.open("sgemm_product.csv"))

        df['Runtime'] = df[['Run{:d} (ms)'.format(i) for i in (1, 2, 3, 4)]].mean(axis=1)
        df.drop(columns=['Run{:d} (ms)'.format(i) for i in (1, 2, 3, 4)], axis=1, inplace=True)

        return torch.as_tensor(df.values, dtype=self.dtype, device=self.device)

    def _preprocess(
        self,
        raw_data: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        X = raw_data[:, :-1]
        y = raw_data[:, -1]

        y = np.log(y)

        return X, y
