import torch 
import copy
from src.data.dataset import T4CDataset

import pdb
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
                 dataset: T4CDataset,
                 valset: bool,
                 valset_limit: int,
                 val_filter: str = "**/validation/*8ch.h5"
                 ) -> None:
        self.network = network
        self.t_dataset = dataset

        # If there's validation data, duplicate dataset
        # and change file filter and reload dataset files
        if valset == True:
            self.v_dataset = copy.deepcopy(dataset)
            self.v_dataset.file_filter = val_filter
            self.v_dataset._load_dataset()
            if valset_limit:
                self.v_dataset.limit = valset_limit
        # load dataset after to prevent populating
        # valset with training data
        self.t_dataset._load_dataset()
