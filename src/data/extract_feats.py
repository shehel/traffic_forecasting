# -*- coding: utf-8 -*-
import logging
from pathlib import Path
from typing import Optional
from typing import Union
import glob
import random
import os
import shutil

from clearml import Dataset
from clearml import Task

import hydra

# from omegaconf import DictConfig, OmegaConf


import h5py
import numba
import numpy as np
from einops import rearrange, reduce

import pdb
from tqdm import tqdm
from numba import jit, prange, njit

import h5py


def load_h5_file(
    file_path: Union[str, Path], sl: Optional[slice] = None, to_torch: bool = False
) -> np.ndarray:
    """Given a file path to an h5 file assumed to house a tensor, load that
    tensor into memory and return a pointer.

    Parameters
    ----------
    file_path: str
        h5 file to load
    sl: Optional[slice]
        slice to load (data is written in chunks for faster access to rows).
    """
    # load
    with h5py.File(
        str(file_path) if isinstance(file_path, Path) else file_path, "r"
    ) as fr:
        data = fr.get("array")
        if sl is not None:
            data = np.array(data[sl], np.float32)
        else:
            data = np.array(data, np.float32)
        return data


MAX_TEST_SLOT_INDEX = 240  # since a test goes over 2 hours, the latest possibility is 10p.m. However, `22*12 > 256 = 2^8` and so does not fit into uint8. Therefore, we (somewhat arbitrarily) chose to start the last test slot at 8-10p.m.


rng = np.random.default_rng()


@njit(parallel=True)
def extract_feats(dynamic_input, output_data, static_mask, filters, sample_idx):
    Xs = np.zeros((len(sample_idx), len(filters)*12*8), np.float32)
    ys = np.zeros((len(sample_idx), 6*8), np.float32)

    for idx in prange(sample_idx.shape[0]):
        point_x, point_y = sample_idx[idx]
        feats = np.zeros(
            (len(filters), dynamic_input.shape[0], dynamic_input.shape[1]),
            dtype=np.float32,
        )
        accum = np.zeros(
            (dynamic_input.shape[0], dynamic_input.shape[1]), dtype=np.float32
        )
        static_accum = 0
        mask_sum = 0
        for filter_idx, filter in enumerate(filters):
            offset = int(np.floor(filter / 2))
            feat = dynamic_input[
                :,
                :,
                point_x - offset : point_x + offset + 1,
                point_y - offset : point_y + offset + 1,
            ].copy()

            sum_feat = np.zeros(
                (dynamic_input.shape[0], dynamic_input.shape[1]), dtype=np.float32
            )
            # TODO this approach doesn't seem right
            mask_feat = static_mask[
                point_x - offset : point_x + offset + 1,
                point_y - offset : point_y + offset + 1,
            ]
            mask_sum = np.sum(mask_feat)
            mask_sum = mask_sum - static_accum
            static_accum = static_accum + mask_sum

            if mask_sum != 0:
                feat_ch = feat.reshape(12, 8, -1).copy()
                sum_feat = np.sum(feat_ch, axis=2)

                sum_feat = sum_feat - accum
                accum = accum + sum_feat
                sum_feat = sum_feat / mask_sum
            feats[filter_idx] = sum_feat

        feats = feats.flatten()
        output_feats = output_data[:, point_x, point_y, :].flatten()
        Xs[idx] = feats
        ys[idx] = output_feats

    return Xs, ys


def write_data_to_h5(
    data_X: np.ndarray,
    data_y: np.ndarray,
    filename: Union[str, Path],
    compression="gzip",
    compression_level=9,
    verbose=False,
):
    """write data in gzipped h5 format.

    Parameters
    ----------
    data
    filename
    compression
    compression_level
    verbose
    """
    with h5py.File(
        filename if isinstance(filename, str) else str(filename), "w", libver="latest"
    ) as f:
        # if data.dtype != dtype:
        # logging.warning(f"Found data with {data.dtype}, expected {dtype}.")
        if verbose:
            print(f"writing {filename} ...")
        f.create_dataset(
            # `chunks=(1, *data.shape[1:])`: optimize for row access!
            "X",
            shape=data_X.shape,
            data=data_X,
            chunks=(1, *data_X.shape[1:]),
            dtype=data_X.dtype,
            compression=compression,
            compression_opts=compression_level,
        )
        f.create_dataset(
            # `chunks=(1, *data.shape[1:])`: optimize for row access!
            "y",
            shape=data_y.shape,
            data=data_y,
            chunks=(1, *data_y.shape[1:]),
            dtype=data_y.dtype,
            compression=compression,
            compression_opts=compression_level,
        )

        if verbose:
            print(f"... done writing {filename}")


# TODO change logger to common.util
# @hydra.main(config_path="../../config/data", config_name="feat_7days")
def main() -> None:
    """Extracts features based on specified filters and stores as
    .npy files.
    """
    task = Task.init(project_name="t4c", task_name="extract_features")
    logger = logging.getLogger(__name__)
    args = {
        "random_seed": 123,
        "name": "feat_7days_v5_3",
        "ds_name": "7days",
        "input_path": "/home/shehel/ml/NeurIPS2021-traffic4cast/data/raw",
        "cities": ["ANTWERP","BANGKOK","BARCELONA", "BERLIN", "CHICAGO", "ISTANBUL", "MELBOURNE", "MOSCOW"],
        "count": 7,
        "filters": [3, 7, 21, 51, 71, 91],
        "static_filter": None,
        "slices_per_day": 40,
        "pixels_per_slice": 128,

    }
    task.connect(args)
    print ('Arguments: {}'.format(args))

    random.seed(args["random_seed"])

    # uses the name of the yaml file aos dataset folder name
    try:
        Dataset.get(dataset_project="t4c", dataset_name=args["name"])
        logger.info("Dataset exists. Skipping to next step.")
    except:

        root_dir = Dataset.get(
            dataset_project="t4c", dataset_name=args["ds_name"]
        ).get_local_copy()
        # root_dir = "../../../data/raw/debug/"
        logger.info("Loading dataset")
        input_path = Path(root_dir)
        output_path = Path("data/processed/" + args["name"])
        if os.path.exists(output_path):
            logger.info("Folder exists. Fatal error, exiting!")
            return

        if args["static_filter"] is None:
            static_filter = "**/*_static.h5"
        else:
            static_filter = args["static_filter"]

        filters = args["filters"]
        static_dict = {}
        static_list = list(input_path.rglob(static_filter))

        for city in static_list:
            static_dict[city.parts[-2]] = load_h5_file(city)

        # f = h5py.File("7day.hdf5", "a")
        for city in tqdm(args["cities"], desc="Cities"):

            # grp = f.create_group(city)

            # Get static channel corresponding to the city
            static_ch = static_dict[city]
            for ds in tqdm(["training", "validation"], desc="type"):
                if not os.path.exists(output_path / city / ds):
                    os.makedirs(output_path / city / ds)

                logger.info("Sampling %s for %s" % (city, ds))


                static_mask = np.pad(static_ch[0], ((100, 100), (100, 100)), 'constant', constant_values=0)
                road_pixels = np.transpose((static_mask>0).nonzero())
                static_mask = np.where(static_mask > 0, 1, 0)
                # get a Nx2 matrix giving the indices of non-zero pixels


                for file in tqdm(glob.glob(str(input_path / city / ds / "*")), "file"):
                    # sample non-zero pixel indices
                    sample_idx = road_pixels[np.random.randint(static_mask.shape[0], size = args["pixels_per_slice"]), :]

                    X = np.zeros((len(sample_idx)*args["slices_per_day"], len(filters)*12*8), np.float32)
                    y = np.zeros((len(sample_idx)*args["slices_per_day"], 6*8), np.float32)

                    # sample slots at certain interval defined by dividing max slot by required number of slices
                    for idx, start_hour in tqdm(enumerate(range(0, MAX_TEST_SLOT_INDEX, int(MAX_TEST_SLOT_INDEX/args["slices_per_day"]))), desc="slot"):
                        two_hours = load_h5_file(
                            file, sl=slice(start_hour, start_hour + 12 * 2 + 1)
                        )
                        dynamic_input, output_data = (
                            two_hours[:12],
                            two_hours[[12, 13, 14, 17, 20, 23]],
                        )

                        dynamic_input = np.pad(dynamic_input, ((0,0),(100, 100), (100, 100),(0,0)), 'constant', constant_values=0)
                        output_data = np.pad(output_data, ((0,0),(100, 100), (100, 100),(0,0)), 'constant', constant_values=0)
                        # Normalize by removing
                        # mean of inputs from input and output
                        mean = np.mean(dynamic_input, 0)
                        output_data = output_data - mean
                        dynamic_input = dynamic_input - mean

                        dynamic_input = np.moveaxis(dynamic_input, -1, 1)

                        Xs, ys = extract_feats(
                            dynamic_input,
                            output_data,
                            static_mask,
                            np.array(filters),
                            sample_idx
                        )
                        # if ds == "training":
                        X[idx*Xs.shape[0]:idx*Xs.shape[0]+Xs.shape[0]] = Xs
                        y[idx*Xs.shape[0]:idx*Xs.shape[0]+Xs.shape[0]] = ys
                    f_name = file.split("/")[-1]
                    # X = np.concatenate(X, axis=0)
                    # y = np.concatenate(y, axis=0)
                    write_data_to_h5(
                        X,
                        y,
                        str(output_path / city / ds / f_name),
                    )

if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
