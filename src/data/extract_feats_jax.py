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

import pdb
from tqdm import tqdm
from numba import jit, prange, njit

import h5py


from multiprocessing import Pool, RawArray

try:
    import joblib
except ImportError:
    from sklearn.externals import joblib

import warnings
warnings.filterwarnings('ignore')
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

var_dict = {}
def init_worker(sh_Xs, sh_ys, dynamic_input, output_data, filters, X_shape, y_shape):
    # Using a dictionary is not strictly necessary. You can also
    # use global variables.
    var_dict["dynamic_input"] = dynamic_input
    var_dict["output_data"] = output_data
    var_dict["filters"] = filters
    var_dict["sh_Xs"] = sh_Xs
    var_dict["sh_ys"] = sh_ys
    var_dict["X_shape"] = X_shape
    var_dict["y_shape"] = y_shape


#@njit()

def feat_extract_p(i):

    point_x, point_y, s_idx = i

    X_np = np.frombuffer(var_dict["sh_Xs"]).reshape(var_dict["X_shape"])
    y_np = np.frombuffer(var_dict["sh_ys"]).reshape(var_dict["y_shape"])

    feats = np.zeros(
        (
            len(var_dict["filters"]),
            var_dict["dynamic_input"].shape[0],
            var_dict["dynamic_input"].shape[1],
        ),
        dtype=np.float64,
    )

    for filter_idx, filter in enumerate(var_dict["filters"]):
        offset = int(np.floor(filter / 2))
        feat = var_dict["dynamic_input"][
            :,
            :,
            point_x - offset : point_x + offset + 1,
            point_y - offset : point_y + offset + 1,
        ].copy()

        # Zero out regions of previous filter
        if filter_idx > 0:
            offset_prev = int(np.floor(var_dict["filters"][filter_idx - 1] / 2))
            feat[
                :,
                :,
                point_x - offset_prev : point_x + offset_prev + 1,
                point_y - offset_prev : point_y + offset_prev + 1,
            ] = 0


        feat = feat.reshape(12, 8, -1)
        sum_feat = feat.sum(axis=-1)
        # get the total number of non-zero values along time and direction dim
        sum_mask = (feat != 0).sum(axis=-1)

        # mean taking into account only the non-zero values
        feats[filter_idx] = sum_feat/sum_mask

        output_feats = var_dict["output_data"][:, point_x, point_y, :].flatten()
        # if np.count_nonzero(np.isnan(feats)) > 0:
        #    pdb.set_trace()
    # feats = feats.reshape(feats, 'k f c -> (k f c)

    np.copyto(X_np[s_idx, :], feats.flatten())
    np.copyto(y_np[s_idx, :], output_feats)





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
    task = Task.init(project_name="t4c_lr", task_name="extract_features mp")
    logger = logging.getLogger(__name__)
    args = {
        "random_seed": 123,
        "name": "feat_7days_v23_mp",
        "desc": 'baseline',
        "ds_name": "7days",
        "input_path": "/home/shehel/ml/NeurIPS2021-traffic4cast/data/raw",
        "cities": [
            "ANTWERP",
            "BANGKOK",
            "BARCELONA",
            "BERLIN",
            "CHICAGO",
            "ISTANBUL",
            "MELBOURNE",
            "MOSCOW",
        ],
        "count": 7,
        "filters": [0, 5, 7, 21, 51, 71],
        "static_filter": None,
        "slices_per_day": 40,
        "pixels_per_slice": 128,
        "road_thresh": 0,
        "processes": 8

    }
    task.connect(args)
    print("Arguments: {}".format(args))

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

                static_mask = np.pad(
                    static_ch[0],
                    ((100, 100), (100, 100)),
                    "constant",
                    constant_values=0,
                )
                road_pixels = np.transpose((static_mask > args["road_thresh"]).nonzero())
                #static_mask = np.where(static_mask > 0, 1, 0)
                # get a Nx2 matrix giving the indices of non-zero pixels


                for file in tqdm(glob.glob(str(input_path / city / ds / "*")), "file"):
                    X = np.zeros(
                        (
                            args["pixels_per_slice"] * args["slices_per_day"],
                            len(filters) * 12 * 8,
                        ),
                        np.float32,
                    )
                    y = np.zeros(
                        (args["pixels_per_slice"] * args["slices_per_day"], 6 * 8),
                        np.float32,
                    )

                    # sample slots at certain interval defined by dividing max slot by required number of slices
                    for idx, start_hour in tqdm(
                        enumerate(
                            range(
                                0,
                                MAX_TEST_SLOT_INDEX,
                                int(MAX_TEST_SLOT_INDEX / args["slices_per_day"]),
                            )
                        ),
                        desc="slot",
                    ):
                        # sample non-zero pixel indices
                        np.random.shuffle(road_pixels)
                        sample_idx = road_pixels[:args["pixels_per_slice"]]
                        sample_idx = np.append(sample_idx, np.arange(sample_idx.shape[0]).reshape(-1, 1), axis=-1)

                        two_hours = load_h5_file(
                            file, sl=slice(start_hour, start_hour + 12 * 2 + 1)
                        )
                        dynamic_input, output_data = (
                            two_hours[:12],
                            two_hours[[12, 13, 14, 17, 20, 23]],
                        )

                        # Normalize by removing
                        # mean of inputs from input and output
                        # mean = np.mean(dynamic_input, 0)
                        # output_data = output_data - mean
                        # dynamic_input = dynamic_input - mean

                        dynamic_input = np.pad(
                                dynamic_input,
                                ((0, 0), (100, 100), (100, 100), (0, 0)),
                                "constant",
                                constant_values=0,
                            )
                        output_data = np.pad(
                                output_data,
                                ((0, 0), (100, 100), (100, 100), (0, 0)),
                                "constant",
                                constant_values=0,
                            )
                        dynamic_input = np.moveaxis(dynamic_input, -1, 1)


                        Xs = np.zeros((len(sample_idx), len(filters) * 12 * 8), np.float32)
                        ys = np.zeros((len(sample_idx), 6 * 8), np.float32)

                        X_shape = Xs.shape
                        y_shape = ys.shape
                        # sh_pred = RawArray('d', pred.shape[0], pred.shape[1], pred.shape[2], pred.shape[3])
                        sh_Xs = RawArray(
                            "d", Xs.shape[0] * Xs.shape[1]
                        )
                        sh_ys = RawArray(
                            "d", ys.shape[0] * ys.shape[1]
                        )

                        sh_Xs_np = np.frombuffer(sh_Xs).reshape(Xs.shape)
                        sh_ys_np = np.frombuffer(sh_ys).reshape(ys.shape)
                        np.copyto(sh_Xs_np, Xs)
                        np.copyto(sh_ys_np, ys)

                        with Pool(
                                processes=args["processes"],
                            initializer=init_worker,
                            initargs=(sh_Xs, sh_ys, dynamic_input, output_data, filters, X_shape, y_shape),
                        ) as pool:
                            pool.map(feat_extract_p, sample_idx)

                        # if ds == "training":
                        X[idx * Xs.shape[0] : idx * Xs.shape[0] + Xs.shape[0]] = sh_Xs_np
                        y[idx * Xs.shape[0] : idx * Xs.shape[0] + Xs.shape[0]] = sh_ys_np


                    X = np.nan_to_num(X, posinf=0, neginf=0, nan=0)
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
