from pathlib import Path
from typing import Any
from typing import Callable
from typing import Optional
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from src.competition.competition_constants import MAX_TEST_SLOT_INDEX
from src.competition.prepare_test_data.prepare_test_data import prepare_test
from src.common.h5_util import load_h5_file

import pdb
import tensorly as tl

from clearml import Task
class T4CDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        file_filter: str = None,
        limit: Optional[int] = None,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        use_npy: bool = False,
        sampling_height: int = 1,
        sampling_width: int = 1,
        reduced: bool = False
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
        self.file_filter = file_filter
        self.use_npy = use_npy
        if self.file_filter is None:
            self.file_filter = "**/training/*8ch.h5"
            if self.use_npy:
                self.file_filter = "**/training_npy/*.npy"
        self.transform = transform
        self._load_dataset()
        self.sampling_height = sampling_height
        self.sampling_width = sampling_width
        self.reduced = reduced
        if self.reduced:
            preprocess_task = Task.get_task(task_id='ea686462a02146d3b17b01c71b299161')
            self.factors = preprocess_task.artifacts['trainset_factors'].get_local_copy()

    def _load_dataset(self):
        self.files = list(Path(self.root_dir).rglob(self.file_filter))

    def _load_h5_file(self, fn, sl: Optional[slice]):
        if self.use_npy:
            return np.load(fn)
        else:
            return load_h5_file(fn, sl=sl)

    def __len__(self):
        size_240_slots_a_day = len(self.files) * MAX_TEST_SLOT_INDEX
        #if self.limit is not None:
        #    return min(size_240_slots_a_day, self.limit)
        return size_240_slots_a_day

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        if idx > self.__len__():
            raise IndexError("Index out of bounds")

        file_idx = idx // MAX_TEST_SLOT_INDEX
        start_hour = idx % MAX_TEST_SLOT_INDEX

        two_hours = self._load_h5_file(self.files[file_idx], sl=slice(start_hour, start_hour + 12 * 2 + 1))
        two_hours = two_hours[:,::self.sampling_height,::self.sampling_width,:]

        input_data, output_data = prepare_test(two_hours)
        pdb.set_trace()

        input_data = self._to_torch(input_data)
        output_data = self._to_torch(output_data)

        if self.transform is not None:
            input_data = self.transform.pre_transform(input_data)
            output_data = self.transform.pre_transform(output_data)


        if self.reduced:
            input_data = input_data.numpy()
            reduc = tl.tenalg.multi_mode_dot(input_data, self.factors, transpose=True)
            input_data = torch.from_numpy(reduc).float()



        return input_data, output_data

    def _to_torch(self, data):
        data = torch.from_numpy(data)
        data = data.to(dtype=torch.float)
        return data
