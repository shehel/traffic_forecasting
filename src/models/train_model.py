
from torch import nn
from src.models.DataTransform import DataTransform


class Model:
    """
    Model class composed of a network architecture and the
    associated data transformation class
    Args:
        network: Network architecture to train on
        data_transform: DataTransform object that contains the
                relevant pre/post processing methods
    """

    def __init__(self,
                 network: nn.module,
                 data_transform: DataTransform,
                 dataset: T4CDataset
                 ) -> None:
        self.network = network
        self.data_transform = data_transform
        self.dataset = dataset
