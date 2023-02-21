import datetime
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

# import einops rearrange
from einops import rearrange

perm = [[0,1,2,3,4,5,6,7],
        [2,3,4,5,6,7,0,1],
        [4,5,6,7,0,1,2,3],
        [6,7,0,1,2,3,4,5]
        ]
filters = list(range(3, 50, 2))
def extract_feats(pixel, dynamic_input, static_mask):
    
    point_x, point_y = pixel

    feats = np.zeros(
        (dynamic_input.shape[0], dynamic_input.shape[-1], len(filters)+1),
        dtype=np.float32,
    )
    accum = np.zeros(
        (dynamic_input.shape[0], dynamic_input.shape[-1]), dtype=np.float32
    )

    static_accum = 0
    mask_sum = 0
    for filter_idx, filter in enumerate(filters):
        offset = int(np.floor(filter / 2))
        feat = dynamic_input[
            :,
            point_x - offset : point_x + offset + 1,
            point_y - offset : point_y + offset + 1,
            :
        ].copy()
        
        sum_feat = np.zeros(
            (dynamic_input.shape[0], dynamic_input.shape[-1]), dtype=np.float32
        )
        # TODO this approach doesn't seem right
        mask_feat = static_mask[
            point_x - offset : point_x + offset + 1,
            point_y - offset : point_y + offset + 1,
        ]
        
        mask_sum = np.sum(mask_feat)
        mask_sum = mask_sum - static_accum
        static_accum = static_accum + mask_sum

        # reducefeat = reduce(feat, 'f h w c -> f c', 'sum')/mask_feat
        if mask_sum != 0:
            #print (f'Accum is {accum[0,0]}')
            feat = np.moveaxis(feat, -1 ,1)
            feat_ch = feat.reshape(6, 8, -1).copy()
            sum_feat = np.sum(feat_ch, axis=2)
            #print (f'Total for filter {filter} is {sum_feat[0,0]}')
            sum_feat = sum_feat - accum
            #print (f'Total after subtracting accum is {sum_feat[0,0]}')
            accum = accum + sum_feat
            #sum_feat = sum_feat / mask_sum
            #print (f'Total after dividing mask_sum {mask_sum} is {sum_feat[0,0]}')
        feats[:,:,filter_idx] = sum_feat
        # if np.count_nonzero(np.isnan(feats)) > 0:
        #    pdb.set_trace()
    # feats = feats.reshape(feats, 'k f c -> (k f c)')
    
    feats[:,:,filter_idx+1] = dynamic_input[:, point_x, point_y, :]
    return feats

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
        # static_input = np.stack(list(self.static_dict.values()), axis=0)
        # static_input_mean = static_input.mean(axis=(0, 2, 3))[:, None, None]
        # static_input_std = static_input.std(axis=(0, 2, 3))[:, None, None]
        # for city in self.static_dict:
        #     self.static_dict[city] = (self.static_dict[city] - static_input_mean) / static_input_std
        self.file_list.sort()
        # for file in self.file_list:
        #     self.files.append(load_h5_file(file))
        self.len = len(self.file_list) * MAX_TEST_SLOT_INDEX

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

        start_h = start_hour // 12
        start_m = (start_hour % 12)*5
        # convert start_hour to list called hm of hour and minutes format where start_hour 0 is [0,0] and start_hourt 1 is [0,5] in 5 minute increments
        #hm = [0, start_hour * 5]

        file_name = self.file_list[file_idx].parts[-1]
        # extract date from from file_name containing date and place and convert to list of [year, month, day]
        date = [int(x) for x in file_name.split('_')[0].split('-')]

        # get day of the week from date given as list of [year, month, day]
        day = datetime.date(date[0], date[1], date[2]).weekday() + 1
        
        # set month to 0 and day to day of the week
        # so its consistent with corresponding data in test set
        date[1] = 0
        date[2] = day
        #date = [start_h, start_m, 0] + date
        if self.time_step == 'random':
            # pick random integer between 0 and 6
            self.time_step = random.randint(0, 5)
        date = [self.time_step]#[day*start_m]

        two_hours = self._load_h5_file(self.file_list[file_idx], sl=slice(start_hour, start_hour + 12 * 2 + 1))
        # convert 
        #two_hours = self.files[file_idx][start_hour:start_hour+24]
         
        # get static channels
        static_ch = self.static_dict[self.file_list[file_idx].parts[-3]][0]


        dynamic_input, output_data = two_hours[:6], two_hours[[6,7,8,9,10,11]]
        points = []

        for i in range(100):
            random_int_x = random.randint(128, 300)
            random_int_y = random.randint(128, 250)

            # while random_int_x and random_int_y position in static_ch are less than 5, recalculate random_int_x and random_int_y
            while static_ch[random_int_x, random_int_y] < 20:
                random_int_x = random.randint(128, 300)
                random_int_y = random.randint(128, 250)

            feats = extract_feats((random_int_x, random_int_y), dynamic_input, static_ch)
            points.append(feats)
        
        # stack points into numpy array
        points = np.stack(points, axis=0)
        inp = points[:,:,1::2,:24]
        outp = points[:,0,1::2,24:]

        inp = rearrange(inp, 's t c x->  s (t c x)')
        outp = rearrange(outp, 's t c ->  s (t c)')
        
        #dynamic_input = np.concatenate((np.tile(date, (100,1)), dynamic_input), axis=1)

        # concatenate date to dynamic input using numpy
        return inp, outp

    def _to_torch(self, data):
        data = torch.from_numpy(data)
        data = data.to(dtype=torch.float)
        return data

def train_collate_fn(batch):
    dynamic_input_batch, target_batch= zip(*batch)
    # stack dynamic_input_batch so that the shape is (batch size*100, -1)
    dynamic_input_batch = np.stack(dynamic_input_batch, axis=0)
    target_batch = np.stack(target_batch, axis=0)

    # rearrange dynamic_input_batch so that the shape is (batch size*100, -1)
    dynamic_input_batch = rearrange(dynamic_input_batch, 's t c ->  (s t) c')
    target_batch = rearrange(target_batch, 's t c ->  (s t) c')



    #dynamic_input_batch = np.moveaxis(dynamic_input_batch, source=4, destination=2)
    dynamic_input_batch = torch.from_numpy(dynamic_input_batch).float()
    #target_batch = np.moveaxis(target_batch, source=4, destination=2)
    target_batch = torch.from_numpy(target_batch).float()

    return dynamic_input_batch, target_batch

