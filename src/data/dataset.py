import random
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Optional
from typing import Tuple
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset

from src.competition.competition_constants import MAX_TEST_SLOT_INDEX
from src.competition.prepare_test_data.prepare_test_data import prepare_test
from src.common.h5_util import load_h5_file

import pdb

import tensorly as tl
from pytorch_wavelets import DWTForward, DWTInverse

from clearml import Task

perm = [[0,1,2,3,4,5,6,7],
        [2,3,4,5,6,7,0,1],
        [4,5,6,7,0,1,2,3],
        [6,7,0,1,2,3,4,5]
        ]
class T4CDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        file_filter: str = None,
            static_filter: str = None,
        limit: Optional[int] = None,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        use_npy: bool = False,
        sampling_height: int = 1,
        sampling_width: int = 1,
        dim_start: int = 0,
        dim_step: int = 1,
        output_start: int = 0,
        output_step: int = 1,
        reduced: bool = False,
        factors_task_id: str = None,
        single_channel: int = None,
        time_step: int = None,
        perm: bool = False
    ):
        """torch dataset from training data.
        Parameters
        ----------
        root_dir
            data root folder, by convention should be `data/raw`, see `data/README.md`. All `**/training/*8ch.h5` will be added to the dataset.
        file_filter: str
            filter files under `root_dir`, defaults to `"**/training/*ch8.h5`
        limit
            truncate dataset size
        transform
            transform applied to both the input and label
        """
        self.root_dir = root_dir
        self.limit = limit
        self.files = []
        self.static_dict = {}
        self.file_filter = file_filter
        self.static_filter = static_filter
        self.use_npy = use_npy
        if self.file_filter is None:
            self.file_filter = "**/training/*8ch.h5"
            if self.use_npy:
                self.file_filter = "**/training_npy/*.npy"
        if self.static_filter is None:
            self.static_filter = "**/*_static.h5"
        self.transform = transform
        self.len = 0
        self.sampling_height = sampling_height
        self.sampling_width = sampling_width
        self.dim_start = dim_start
        self.dim_step = dim_step
        self.output_start = output_start
        self.output_step = output_step
        self.reduced = reduced
        self.single_channel = single_channel
        self.time_step = time_step
        self.file_list = None
        self.perm = perm
        if self.reduced:
            preprocess_task = Task.get_task(task_id=factors_task_id)
            with open(preprocess_task.artifacts['trainset factors'].get_local_copy(), 'rb') as file:
                self.factors = pickle.load(file)


    def _load_dataset(self):
        self.file_list = list(Path(self.root_dir).rglob(self.file_filter))
        static_list = list(Path(self.root_dir).rglob(self.static_filter))

        for city in static_list:
            self.static_dict[city.parts[-2]] = load_h5_file(city)
        self.file_list.sort()
        for file in self.file_list:
            self.files.append(load_h5_file(file))
        self.len = len(self.files) * MAX_TEST_SLOT_INDEX

    def _load_h5_file(self, fn, sl: Optional[slice]):
        if self.use_npy:
            return np.load(fn)
        else:
            return load_h5_file(fn, sl=sl)

    def __len__(self):
        if self.limit is not None:
            return min(self.len, self.limit)
        else:
            return self.len


    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        if idx > self.__len__():
            raise IndexError("Index out of bounds")

        file_idx = idx // MAX_TEST_SLOT_INDEX
        start_hour = idx % MAX_TEST_SLOT_INDEX


        #two_hours = self._load_h5_file(self.files[file_idx], sl=slice(start_hour, start_hour + 12 * 2 + 1))
        two_hours = self.files[file_idx][start_hour:start_hour+24]

        two_hours = two_hours[:,::self.sampling_height,::self.sampling_width,self.dim_start::self.dim_step]

        if self.perm:
            dir_sel = random.randint(0,3)
            two_hours = two_hours[:,:,:,perm[dir_sel]]
        #input_data, output_data = prepare_test(two_hours)
        dynamic_input, output_data = two_hours[:12], two_hours[[12, 13, 14, 17, 20, 23]]

        # get static channels
        static_ch = self.static_dict[self.file_list[file_idx].parts[-3]]

        output_data = output_data[:,:,:,self.output_start::self.output_step]

        if self.single_channel is not None:
            output_data = output_data[:,:,:,self.single_channel:self.single_channel+1]

        if self.time_step is not None:
            output_data = output_data[self.time_step:self.time_step+1,:,:,:]

        # if self.transform is not None:
        #     dynamic_input = self.transform.pre_transform(dynamic_input)
        #     output_data = self.transform.pre_transform(output_data)
            # static_ch = self.transform.pre_transform(static_ch, stack_time=False)
        #input_data = torch.cat((input_data, static_ch), dim=0)
        if self.reduced:
            dynamic_input = dynamic_input.numpy()
            reduc = tl.tenalg.multi_mode_dot(dynamic_input, self.factors, transpose=True)
            dynamic_input = torch.from_numpy(reduc).float()

        return dynamic_input, static_ch, output_data

    def _to_torch(self, data):
        data = torch.from_numpy(data)
        data = data.to(dtype=torch.float)
        return data
def train_collate_fn(batch):
    dynamic_input_batch, static_input_batch, target_batch = zip(*batch)
    dynamic_input_batch = np.stack(dynamic_input_batch, axis=0)
    static_input_batch = np.stack(static_input_batch, axis=0)
    target_batch = np.stack(target_batch, axis=0)
    dynamic_input_batch = np.moveaxis(dynamic_input_batch, source=4, destination=2)
    dynamic_input_batch = torch.from_numpy(dynamic_input_batch).float()
    static_input_batch = torch.from_numpy(static_input_batch)
    target_batch = np.moveaxis(target_batch, source=4, destination=2)
    target_batch = torch.from_numpy(target_batch).float()
    pdb.set_trace()
    return dynamic_input_batch, static_input_batch, target_batch
