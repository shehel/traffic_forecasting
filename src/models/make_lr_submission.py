import argparse
import binascii
import logging
import os
import random
import zipfile
import h5py
from sklearn.linear_model import LinearRegression
import pickle
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
from pathlib import Path
from typing import Optional
import pdb

import copy
import numpy as np
import torch
import torch.nn.functional as F  # noqa
import torch.optim as optim
import tqdm
from ignite.contrib.handlers import TensorboardLogger
from ignite.contrib.handlers.tensorboard_logger import GradsHistHandler
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.engine import create_supervised_evaluator
from ignite.engine import create_supervised_trainer
from ignite.engine import Engine
from ignite.engine import Events
from ignite.handlers import Checkpoint
from ignite.handlers import DiskSaver
from ignite.handlers import global_step_from_engine
from ignite.metrics import Loss
from ignite.metrics import RunningAverage
from ignite.utils import convert_tensor
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler

from src.data.dataset import T4CDataset, train_collate_fn


import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import mean_squared_error

from pytorch_wavelets import DWTForward, DWTInverse  # (or import DWT, IDWT)
import matplotlib.animation as animation

from hydra import initialize, initialize_config_module, initialize_config_dir, compose
from hydra.utils import instantiate
from omegaconf import OmegaConf

from clearml import Dataset, Task

perm = np.array(
    [
        [0, 1, 2, 3, 4, 5, 6, 7],
        [2, 3, 4, 5, 6, 7, 0, 1],
        [4, 5, 6, 7, 0, 1, 2, 3],
        [6, 7, 0, 1, 2, 3, 4, 5],
    ]
)
EXTENDED_CHALLENGE_CITIES = ["NEWYORK", "VIENNA"]


def reset_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class NaiveAverage(torch.nn.Module):  # noqa
    def __init__(self):
        """Returns prediction consisting of repeating last frame."""
        super(NaiveAverage, self).__init__()

    def forward(self, x: torch.Tensor, *args, **kwargs):
        x = torch.mean(x, dim=1)
        x = torch.unsqueeze(x, dim=1)
        x_out = torch.repeat_interleave(x, repeats=6, dim=1)
        x_in = torch.repeat_interleave(x, repeats=12, dim=1)
        return x_in, x_out


average_model = NaiveAverage()


def write_data_to_h5(
    data: np.ndarray,
    filename: str,
    compression="gzip",
    compression_level=9,
    dtype="uint8",
):
    with h5py.File(filename, "w", libver="latest") as f:
        f.create_dataset(
            "array",
            shape=data.shape,
            data=data,
            chunks=(1, *data.shape[1:]),
            dtype=dtype,
            compression=compression,
            compression_opts=compression_level,
        )


def prepare_data(batch, dynamic_channels, out_channels, transform_p, switch=None):
    in_h = transform_p["in_h"]
    in_w = transform_p["in_w"]
    is_static = transform_p["static"]
    pad_tuple = tuple(transform_p["pad_tuple"])
    if is_static == True:
        dynamic_channels = dynamic_channels - 9
    dynamic, static, target = batch
    # target = dynamic[:, 11:12, switch[0][0]:switch[0][0]+1, :, :] - target
    # dynamic = (dynamic - dynamic_input_mean) / dynamic_input_std

    # post_d = dynamic.clone()
    # post_d[:, :10, 0, :, :] = dynamic[:, 1:11, 0, :, :]

    # mean_in, mean_out = average_model.forward(dynamic)
    # target = target - mean_out
    # dynamic = dynamic - mean_in
    dynamic = dynamic.reshape(-1, dynamic_channels, in_h, in_w)
    if switch is not None:
        dynamic = dynamic[:, switch.flatten(), :, :]

    # target = target.reshape(-1, out_channels, in_h, in_w)
    # target = F.pad(target, pad=pad_tuple)
    if is_static:
        input_batch = torch.cat([dynamic, static], dim=1)
    else:
        input_batch = dynamic

    input_batch = F.pad(input_batch, pad=pad_tuple)

    # pred = t_model(input_batch.to("cuda"))
    # height, width = pred.shape[-2], pred.shape[-1]
    # left, right, top, bottom = pad_tuple
    # right = width - right
    # bottom = height - bottom
    # pred = pred[:, :, top:bottom, left:right]

    # pred = pred.reshape(-1, 1, 1, 495, 436)
    # pdb.set_trace()
    # post_d[:,-1,0, :,:] = pred[:,0,0,:,:]

    # dynamic = post_d.reshape(-1, dynamic_channels, in_h, in_w)
    # if is_static:
    #    input_batch = torch.cat([dynamic, static], dim=1)
    # else:
    #    input_batch = dynamic
    # input_batch = F.pad(input_batch, pad=pad_tuple)

    # target = (target - dynamic_input_mean) / dynamic_input_std

    return input_batch, target


class ValidDataset(Dataset):
    def __init__(self, args: argparse.Namespace, valid_data: np.ndarray):
        self.args = args
        self.valid_data = valid_data

    def __getitem__(self, index):
        dynamic_input = self.valid_data[index, :, :, :, :]
        return dynamic_input

    def __len__(self):
        return self.valid_data.shape[0]


def valid_collate_fn(batch):
    dynamic_input_batch = batch
    dynamic_input_batch = np.stack(dynamic_input_batch, axis=0)
    dynamic_input_batch = np.moveaxis(dynamic_input_batch, source=4, destination=2)
    #dynamic_input_batch = torch.from_numpy(dynamic_input_batch).float()
    return dynamic_input_batch


"""
Provides evaluation information for a given model and dataset.
Information include
- MSE
- Single pixel time series
- Single sample true and predicted traffic maps
"""


def main():
    reset_seeds(123)
    task = Task.init(project_name="t4c_eval", task_name="LR submission")
    logger = task.get_logger()
    args = {
        "task_id": "84b6f5b666354ae2a01cd4933466ce72",
        "data_dir": "/home/shehel/ml/NeurIPS2021-traffic4cast/data/raw/",
        "submission_name": "lr7d",
        "batch_size": 1,
        "num_workers": 2,
        "pixel": (108, 69),  # (221, 192),
        "loader": "val",
        "num_channels": 4,
        "max_idx": 240,
        "use_static": False,
        "avg": False,
    }

    task.connect(args)
    print("Arguments: {}".format(args))

    bs =1
    is_perm = False


    g = torch.Generator()
    g.manual_seed(123)

    pkl_filename = "models/lr_model.pkl"
    with open(pkl_filename, 'rb') as file:
        model = pickle.load(file)

    static_input_dict = dict()
    for city in EXTENDED_CHALLENGE_CITIES:
        with h5py.File(args["data_dir"] + f"{city}/{city}_static.h5", "r") as file_in:
            static_input = file_in.get("array")
            static_input = np.array(static_input, dtype=np.float32)
            static_input = torch.from_numpy(static_input).float()
            static_input = F.pad(static_input, pad=(6, 6, 1, 0))
            static_input = static_input.cuda()
            static_input_dict[city] = static_input

    with zipfile.ZipFile(f"./submission/{args['submission_name']}.zip", "w") as z:
        for city in EXTENDED_CHALLENGE_CITIES:
            with h5py.File(
                args["data_dir"] + f"{city}/{city}_test_spatiotemporal.h5", "r"
            ) as file_in:
                valid_data = file_in.get("array")
                valid_data = np.array(valid_data)
            valid_dataset = ValidDataset(args=args, valid_data=valid_data)
            valid_loader = DataLoader(
                dataset=valid_dataset,
                batch_size=bs,
                shuffle=False,
                num_workers=1,
                collate_fn=valid_collate_fn,
                pin_memory=True,
                drop_last=False,
            )
            mean_outs = []
            valid_predictions = np.zeros((100, 96, 495, 436))
            with torch.no_grad():
                for idx, dynamic_input_batch in tqdm.tqdm(enumerate(valid_loader)):
                    batch_size = dynamic_input_batch.shape[0]

                    # if args.input_normalization:
                    #     dynamic_input_batch = (
                    #         dynamic_input_batch - dynamic_input_mean
                    #    ) / dynamic_input_std
                    if args["avg"]:
                        mean_in, mean_out = average_model(dynamic_input_batch)
                        mean_outs.append(mean_out.cpu().numpy())
                        dynamic_input_batch = dynamic_input_batch - mean_in

                    dynamic_input_batch = dynamic_input_batch.reshape(-1, 96, 495, 436)
                    #dynamic_input_batch = F.pad(dynamic_input_batch, pad=(6, 6, 1, 0))


                    input_batch = dynamic_input_batch
                    for h_id in range(495):
                        for w_id in range(436):
                            if input_batch[0, :, h_id, w_id].sum() == 0:
                                continue
                            else:
                                valid_predictions[idx, :, h_id, w_id] = model.predict(input_batch[:, :, h_id, w_id])

            valid_predictions = (
                valid_predictions
                .clip(min=0.0, max=255.0)
                .astype(np.float32)
            )



                #valid_predictions = valid_predictions[:, :, 1:, 6:-6]
            valid_predictions = valid_predictions.reshape(-1, 12, 8, 495, 436)
            valid_predictions = valid_predictions[:,[0, 1, 2, 5, 8, 11], :, :, :]

            if args["avg"]:
                mean_outs = np.concatenate(mean_outs, axis=0).astype(np.float32)
                valid_predictions = valid_predictions + mean_outs
            valid_predictions = np.moveaxis(valid_predictions, source=2, destination=4)
            np.save(
                f"./submission/extended_model_v{args['submission_name']}_{city}_predictions.npy",
                valid_predictions,
            )
            write_data_to_h5(
                data=valid_predictions,
                filename=f"./submission/{args['submission_name']}_{city}_predictions.h5",
                compression_level=6,
            )
            z.write(
                f"./submission/{args['submission_name']}_{city}_predictions.h5",
                arcname=f"{city}/{city}_test_spatiotemporal.h5",
            )


if __name__ == "__main__":
    main()
