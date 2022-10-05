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


# # @jit(parallel=True)
# def extract_feats(dynamic_input, output_data, static_mask, cfg, filters, static_ch):
#     # Xs = []
#     # ys = []
#     Xs = np.zeros((128, 128, 576), np.float32)
#     ys = np.zeros((128, 128, 48), np.float32)
#     # Xs = np.zeros((100, 576), np.float32)
#     # ys = np.zeros((100, 48), np.float32)

#     for idx, point_x in tqdm(
#         enumerate(prange(cfg.patch_start, cfg.patch_start + cfg.patch_step))
#     ):
#         cand_mask = (
#             static_mask[
#                 cfg.patch_start : cfg.patch_start + cfg.patch_step,
#                 cfg.patch_start : cfg.patch_start + cfg.patch_step,
#             ]
#             == 1
#         )
#         # cand_mask = cand_mask[0::4, 0::4]
#         # pdb.set_trace()
#         for idy, point_y in enumerate(
#             prange(cfg.patch_start, cfg.patch_start + cfg.patch_step)
#         ):
#             # find a pixel with road
#             # for sample in range(100):
#             #     point_x = rng.integers(cfg.patch_start, cfg.patch_start + cfg.patch_step)
#             #     point_y = rng.integers(cfg.patch_start, cfg.patch_start + cfg.patch_step)
#             #     while static_ch[0][point_x, point_y] < 5:
#             #         point_x = rng.integers(cfg.patch_start, cfg.patch_start + cfg.patch_step)
#             #         point_y = rng.integers(cfg.patch_start, cfg.patch_start + cfg.patch_step)

#             # if static_mask[point_x, point_y] == 0:
#             #    continue
#             feats = np.zeros(
#                 (len(filters), dynamic_input.shape[0], dynamic_input.shape[-1])
#
# )
#             accum = np.zeros((dynamic_input.shape[0], dynamic_input.shape[-1]))
#             static_accum = 0
#             for filter_idx, filter in enumerate(filters):
#                 offset = int(np.floor(filter / 2))
#                 feat = dynamic_input[
#                     :,
#                     point_x - offset : point_x + offset + 1,
#                     point_y - offset : point_y + offset + 1,
#                     :,
#                 ]
#                 # TODO this approach doesn't seem right
#                 mask_feat = static_mask[
#                     point_x - offset : point_x + offset + 1,
#                     point_y - offset : point_y + offset + 1,
#                 ]
#                 mask_feat = np.sum(mask_feat)
#                 mask_feat = mask_feat - static_accum
#                 static_accum = static_accum + mask_feat

#                 # reducefeat = reduce(feat, 'f h w c -> f c', 'sum')/mask_feat
#                 feat = np.sum(feat, axis=(1, 2))
#                 if mask_feat == 0:
#                     feat[:] = 0
#                 else:
#                     feat = feat - accum
#                     accum = accum + feat
#                     feat = feat / mask_feat
#                 feats[filter_idx] = feat
#                 # if np.count_nonzero(np.isnan(feats)) > 0:
#                 #    pdb.set_trace()
#             # feats = feats.reshape(feats, 'k f c -> (k f c)')
#             feats = feats.flatten()
#             output_feats = output_data[:, point_x, point_y, :].flatten()
#             Xs[idx, idy] = feats
#             ys[idx, idy] = output_feats

#         # Xs[sample] = feats
#         # ys[sample] = output_feats

#         # jXs.append(feats)
#         # ys.append(output_feats)
#     # pdb.set_trace()
#     Xs = Xs[cand_mask]
#     ys = ys[cand_mask]
#     return Xs, ys
# return Xs, ys
@njit(parallel=True)
def extract_feats(dynamic_input, output_data, static_mask, filters):
    # Xs = []
    # ys = []
    Xs = np.zeros((128, 128, 576), np.float32)
    ys = np.zeros((128, 128, 48), np.float32)
    # Xs = np.zeros((100, 576), np.float32)
    # ys = np.zeros((100, 48), np.float32)
    patch_start = 128
    patch_step = 128
    for point_x in prange(patch_start, patch_start + patch_step):
        # cand_mask = cand_mask[0::4, 0::4]
        # pdb.set_trace()
        idx = patch_start - point_x
        for point_y in prange(patch_start, patch_start + patch_step):
            idy = patch_start - point_y
            # find a pixel with road
            # for sample in range(100):
            #     point_x = rng.integers(cfg.patch_start, cfg.patch_start + cfg.patch_step)
            #     point_y = rng.integers(cfg.patch_start, cfg.patch_start + cfg.patch_step)
            #     while static_ch[0][point_x, point_y] < 5:
            #         point_x = rng.integers(cfg.patch_start, cfg.patch_start + cfg.patch_step)
            #         point_y = rng.integers(cfg.patch_start, cfg.patch_start + cfg.patch_step)

            # if static_mask[point_x, point_y] == 0:
            #    continue
            feats = np.zeros(
                (len(filters), dynamic_input.shape[0], dynamic_input.shape[-1]),
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
                    :,
                    point_x - offset : point_x + offset + 1,
                    point_y - offset : point_y + offset + 1,
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
                    feat_ch = feat.reshape(12, 8, -1).copy()
                    sum_feat = np.sum(feat_ch, axis=2)

                    sum_feat = sum_feat - accum
                    accum = accum + sum_feat
                    sum_feat = sum_feat / mask_sum
                feats[filter_idx] = sum_feat
                # if np.count_nonzero(np.isnan(feats)) > 0:
                #    pdb.set_trace()
            # feats = feats.reshape(feats, 'k f c -> (k f c)')
            feats = feats.flatten()
            output_feats = output_data[:, point_x, point_y, :].flatten()
            Xs[idx, idy] = feats
            ys[idx, idy] = output_feats

        # Xs[sample] = feats
        # ys[sample] = output_feats

        # jXs.append(feats)
        # ys.append(output_feats)
    # pdb.set_trace()
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
    cfg = {
        "random_seed": 123,
        "name": "feat_7days_v3",
        "ds_name": "88 days single pixel 329 92",
        "input_path": "/home/shehel/ml/NeurIPS2021-traffic4cast/data/raw",
        "cities": ["BERLIN"],
        "count": 7,
        "filters": [3, 7, 21, 51, 71, 91],
        "static_filter": None,
        "patch_start": 128,
        "patch_step": 128,
    }

    random.seed(cfg["random_seed"])

    # uses the name of the yaml file aos dataset folder name
    try:
        Dataset.get(dataset_project="t4c", dataset_name=cfg["name"])
        logger.info("Dataset exists. Skipping to next step.")
    except:

        root_dir = Dataset.get(
            dataset_project="t4c", dataset_name=cfg["ds_name"]
        ).get_local_copy()
        # root_dir = "../../../data/raw/debug/"
        logger.info("Loading dataset")
        input_path = Path(root_dir)
        output_path = Path("data/processed/" + cfg["name"])
        if os.path.exists(output_path):
            logger.info("Folder exists. Fatal error, exiting!")
            return

        if cfg["static_filter"] is None:
            static_filter = "**/*_static.h5"
        else:
            static_filter = cfg["static_filter"]

        filters = cfg["filters"]
        static_dict = {}
        static_list = list(input_path.rglob(static_filter))

        for city in static_list:
            static_dict[city.parts[-2]] = load_h5_file(city)

        # f = h5py.File("7day.hdf5", "a")
        for city in tqdm(cfg["cities"], desc="Cities"):

            # grp = f.create_group(city)

            # Get static channel corresponding to the city
            static_ch = static_dict[city]
            for ds in tqdm(["training", "validation"], desc="type"):
                if not os.path.exists(output_path / city / ds):
                    os.makedirs(output_path / city / ds)

                logger.info("Sampling %s for %s" % (city, ds))
                # sub_grp = grp.create_group(ds)
                # test_y = []
                # X_test = []

                static_mask = np.where(static_ch[0] > 0, 1, 0)
                cand_mask = (
                    static_mask[
                        cfg["patch_start"] : cfg["patch_start"] + cfg["patch_step"],
                        cfg["patch_start"] : cfg["patch_start"] + cfg["patch_step"],
                    ]
                    == 1
                )
                for file in tqdm(glob.glob(str(input_path / city / ds / "*")), "file"):
                    X = []
                    y = []
                    for start_hour in tqdm(range(0, MAX_TEST_SLOT_INDEX), desc="slot"):
                        two_hours = load_h5_file(
                            file, sl=slice(start_hour, start_hour + 12 * 2 + 1)
                        )
                        dynamic_input, output_data = (
                            two_hours[:12],
                            two_hours[[12, 13, 14, 17, 20, 23]],
                        )

                        # Normalize by removing
                        # mean of inputs from input and output
                        mean = np.mean(dynamic_input, 0)
                        output_data = output_data - mean
                        dynamic_input = dynamic_input - mean

                        Xs, ys = extract_feats(
                            dynamic_input,
                            output_data,
                            static_mask,
                            np.array(filters),
                        )
                        # if ds == "training":
                        #
                        Xs = Xs[cand_mask]
                        print (Xs.shape[0]*40)
                        pdb.set_trace()
                        ys = ys[cand_mask]

                        X.append(Xs)
                        y.append(ys)
                    f_name = file.split("/")[-1]
                    X = np.concatenate(X, axis=0)
                    y = np.concatenate(y, axis=0)
                    write_data_to_h5(
                        X,
                        y,
                        str(output_path / city / ds / f_name),
                    )

                    # if len(X) == 0:
                    #     continue

                    # pdb.set_trace()
                    # else:
                    # test_X.append(Xs)
                    # test_y.append(ys)

                # sub_grp.create_dataset(
                #     "X",
                #     shape=X.shape,
                #     data=X,
                #     chunks=(1, *X.shape[1:]),
                #     dtype=X.dtype,
                #     compression="gzip",
                #     compression_opts=9,
                # )
                # sub_grp.create_dataset(
                #     "y",
                #     shape=y.shape,
                #     data=y,
                #     chunks=(1, *y.shape[1:]),
                #     dtype=y.dtype,
                #     compression="gzip",
                #     compression_opts=9,
                # )

                # train_y = np.concatenate(train_y, axis=0)
                # test_y = np.concatenate(test_y, axis=0)
        # pdb.set_trace()

        # loop through the cities and select the required number of samples
        # for train set and val set. Note that cities appearing in both train and
        # val set is not supported (no guarantees on duplicates)

        # there should be enough samples to create the subset without duplication
        # assert (cfg.train_set.count) <= len(files)
        # random.shuffle(files)
        # for file in files[:cfg.train_set.count]:
        # shutil.copy(file, output_path/city/'training'/Path(file).name)

        # res_file = city+"_map_high_res.h5"
        # static_file = city+"_static.h5"
        # shutil.copy(input_path/city/static_file, output_path/city/static_file)
        # s

        # val set

        # handling the case when there's a single
        # validation city.
        # TODO handle 1 or more cities well. yaml doesn't parse
        # single element list correctly.
        # registering clearml dataset
        # dataset = Dataset.create(dataset_name=cfg.name, dataset_project="t4c")

        # dataset.add_files(path=output_path)
        # dataset.upload()
        # dataset.finalize()


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
