"""Datasets from the UCI machine learning repository."""

from ._bike_sharing import BikeSharing
from ._gas_sensors import GasSensors
from ._household_power import HouseholdPower
from ._kegg_undir import KEGGUndir
from ._parkinsons import Parkinsons
from ._protein import Protein
from ._road_network import RoadNetwork
from ._song import Song
from ._uci_dataset import UCIDataset
from ._air_quality import AirQuality
from ._sgemm_gpu import SGEMMGPU

# Public classes and functions. Order is reflected in documentation.
__all__ = [
    "UCIDataset",
    "BikeSharing",
    "KEGGUndir",
    "Parkinsons",
    "Protein",
    "RoadNetwork",
    "Song",
    "HouseholdPower",
    "GasSensors",
    "AirQuality",
    "SGEMMGPU",
]

# Set correct module paths. Corrects links and module paths in documentation.
UCIDataset.__module__ = "datasets.uci"
BikeSharing.__module__ = "datasets.uci"
KEGGUndir.__module__ = "datasets.uci"
Parkinsons.__module__ = "datasets.uci"
Protein.__module__ = "datasets.uci"
RoadNetwork.__module__ = "datasets.uci"
Song.__module__ = "datasets.uci"
HouseholdPower.__module__ = "datasets.uci"
GasSensors.__module__ = "datasets.uci"
AirQuality.__module__ = "datasets.uci"
