import torch 
from src.data.dataset import T4CDataset

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
                 network: torch.nn.Module,
                 dataset: T4CDataset
                 ) -> None:
        self.network = network
        self.dataset = dataset
