"""KEGG undirected dataset from the UCI machine learning repository."""

from typing import Optional, Tuple

import pandas as pd
import torch

from ._uci_dataset import UCIDataset


class KEGGUndir(UCIDataset):
    """KEGG Metabolic pathways (Undirected) dataset (63,608 × 26).

    KEGG Metabolic pathways modelled as a graph. A variety of network features were
    computed using Cytoscape. [1]_

    Source: https://archive.ics.uci.edu/ml/datasets/KEGG+Metabolic+Reaction+Network+(Undirected)

    References
    ----------
    .. [1] Shannon,P., Markiel,A., Ozier,O., Baliga,N.S., Wang,J.T.,Ramage,D., Amin,N.,
           Schwikowski,B. and Ideker,T. (2003) Cytoscape: a software environment for
           integrated models of biomolecular interaction networks. Genome Res., 13
    """

    URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00221/"

    def __init__(
        self,
        dir: Optional[str] = "datasets/raw_data/uci/keggu",
        overwrite: bool = False,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__(dir, overwrite, dtype=dtype, device=device)

    def _download(self) -> torch.Tensor:
        # Read data
        df = pd.read_csv(
            KEGGUndir.URL + "Reaction%20Network%20(Undirected).data",
            index_col=0,
            header=None,
        )
        df.drop(df[df[4] == "?"].index, inplace=True)
        df[4] = df[4].astype(float)
        df.drop(df[df[21] > 1].index, inplace=True)
        df.drop(columns=[10], inplace=True)

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
