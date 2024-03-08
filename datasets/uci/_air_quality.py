"""Gas sensor dataset from the UCI machine learning repository."""


from io import BytesIO
from typing import Optional, Tuple
from zipfile import ZipFile

import numpy as np
import pandas as pd
import requests
import torch

from ._uci_dataset import UCIDataset


class AirQuality(UCIDataset):
    URL = "https://archive.ics.uci.edu/static/public/501/beijing+multi+site+air+quality+data.zip"

    def __init__(
        self,
        dir: Optional[str] = "datasets/raw_data/uci/airquality",
        overwrite: bool = False,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__(dir, overwrite, dtype=dtype, device=device)

    def _download(self) -> torch.Tensor:
        # Download and unzip archive
        r = requests.get(AirQuality.URL)
        files = ZipFile(BytesIO(r.content))

        data_zip_file = ZipFile(BytesIO(files.read('PRSA2017_Data_20130301-20170228.zip')))

        lst_frames = []
        for name in data_zip_file.namelist():
            if name[-4:] == ".csv":
                lst_frames.append(
                    pd.read_csv(data_zip_file.open(name))
                )
        assert len(lst_frames) == 12
        df = pd.concat(lst_frames)

        # drop missing data
        df.dropna(inplace=True)

        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        df['station'] = le.fit_transform(df['station'])
        df['wd'] = le.fit_transform(df['wd'])

        # drop index column
        df.drop(["No"], axis=1, inplace=True)

        # drop PM 10, as it may be highly co-related with PM 2.5
        df.drop(["PM10"], axis=1, inplace=True)

        # drop year and day, as it may be not informative with the prediction
        df.drop(["year"], axis=1, inplace=True)
        df.drop(["day"], axis=1, inplace=True)

        X = df.drop(["PM2.5"], axis=1)
        y = df['PM2.5']

        return torch.cat(
            [
                torch.as_tensor(X.values, dtype=self.dtype, device=self.device),
                torch.as_tensor(y.values, dtype=self.dtype, device=self.device).unsqueeze(-1),
            ], dim=-1
        )

    def _preprocess(
        self,
        raw_data: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        X = raw_data[:, :-1]
        y = raw_data[:, -1]

        return X, y
