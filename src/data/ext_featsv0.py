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
from omegaconf import DictConfig, OmegaConf

import h5py
import numba
import numpy as np
from einops import rearrange, reduce

import pdb
from tqdm import tqdm
from numba import jit


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


# @jit()
def extract_feats(dynamic_input, output_data, static_mask, cfg, filters, static_ch):
    # Xs = []
    # ys = []
    # Xs = np.empty((128, 128, 576), np.float64)
    # ys = np.empty((128, 128, 48), np.float64)
    Xs = np.zeros((100, 576), np.float32)
    ys = np.zeros((100, 48), np.float32)

    # for idx, point_x in enumerate(
    #     range(cfg.patch_start, cfg.patch_start + cfg.patch_step, 16)
    # ):
    #     for idy, point_y in enumerate(
    #         range(cfg.patch_start, cfg.patch_start + cfg.patch_step, 16)
    # ):
    # find a pixel with road
    for sample in range(100):
        point_x = rng.integers(cfg.patch_start, cfg.patch_start + cfg.patch_step)
        point_y = rng.integers(cfg.patch_start, cfg.patch_start + cfg.patch_step)
        while static_ch[0][point_x, point_y] < 5:
            point_x = rng.integers(cfg.patch_start, cfg.patch_start + cfg.patch_step)
            point_y = rng.integers(cfg.patch_start, cfg.patch_start + cfg.patch_step)

        # if static_mask[point_x, point_y] == 0:
        #    continue
        feats = np.zeros(
            (len(filters), dynamic_input.shape[0], dynamic_input.shape[-1])
        )
        accum = np.zeros((dynamic_input.shape[0], dynamic_input.shape[-1]))
        static_accum = 0
        for idx, filter in enumerate(filters):
            offset = int(np.floor(filter / 2))
            feat = dynamic_input[
                :,
                point_x - offset : point_x + offset + 1,
                point_y - offset : point_y + offset + 1,
                :,
            ]
            # TODO this approach doesn't seem right
            mask_feat = static_mask[
                point_x - offset : point_x + offset + 1,
                point_y - offset : point_y + offset + 1,
            ]
            mask_feat = np.sum(mask_feat)
            mask_feat = mask_feat - static_accum
            static_accum = static_accum + mask_feat

            # reducefeat = reduce(feat, 'f h w c -> f c', 'sum')/mask_feat
            feat = np.sum(feat, axis=(1, 2)) / mask_feat
            feat = feat - accum
            accum = accum + feat
            feats[idx] = feat
            if np.count_nonzero(np.isnan(feats)) > 0:
                pdb.set_trace()
        # feats = feats.reshape(feats, 'k f c -> (k f c)')
        feats = feats.flatten()
        output_feats = output_data[:, point_x, point_y, :].flatten()
        # Xs[idx, idy] = feats
        # ys[idx, idy] = output_feats
        Xs[sample] = feats
        ys[sample] = output_feats

        # jXs.append(feats)
        # ys.append(output_feats)
    # return Xs.reshape(-1, 576), ys.reshape(-1, 48)
    return Xs, ys


# TODO change logger to common.util
@hydra.main(config_path="../../config/data", config_name="feat_7days")
def main(cfg: DictConfig) -> None:
    """Extracts features based on specified filters and stores as
    .npy files.
    """
    task = Task.init(project_name="t4c", task_name="extract_features")
    logger = logging.getLogger(__name__)

    random.seed(cfg.random_seed)

    # uses the name of the yaml file aos dataset folder name
    try:
        Dataset.get(dataset_project="t4c", dataset_name=cfg.name)
        logger.info("Dataset exists. Skipping to next step.")
    except:

        root_dir = Dataset.get(
            dataset_project="t4c", dataset_name=cfg.ds_name
        ).get_local_copy()
        logger.info("Loading dataset")
        input_path = Path(root_dir)
        output_path = Path("data/processed/" + cfg.name)
        if os.path.exists(output_path):
            logger.info("Folder exists. Fatal error, exiting!")
            return

        if cfg.static_filter is None:
            static_filter = "**/*_static.h5"
        else:
            static_filter = cfg.static_filter

        filters = cfg.filters
        static_dict = {}
        static_list = list(input_path.rglob(static_filter))
        for city in static_list:
            static_dict[city.parts[-2]] = load_h5_file(city)

        train_X = []
        train_y = []
        test_X = []
        test_y = []
        for city in tqdm(cfg.cities, desc="Cities"):

            logger.info("Sampling %s for training", city)

            # Get static channel corresponding to the city
            static_ch = static_dict[city]
            for ds in tqdm(["training", "validation"], desc="type"):
                for file in tqdm(glob.glob(str(input_path / city / ds / "*")), "file"):
                    for start_hour in tqdm(
                        range(0, MAX_TEST_SLOT_INDEX, 16), desc="slot"
                    ):

                        two_hours = load_h5_file(
                            file, sl=slice(start_hour, start_hour + 12 * 2 + 1)
                        )
                        dynamic_input, output_data = (
                            two_hours[:12],
                            two_hours[[12, 13, 14, 17, 20, 23]],
                        )

                        mean = np.mean(dynamic_input, 0)
                        output_data = output_data - mean
                        dynamic_input = dynamic_input - mean

                        static_mask = np.where(static_ch[0] > 0, 1, 0)
                        Xs, ys = extract_feats(
                            dynamic_input,
                            output_data,
                            static_mask,
                            cfg,
                            filters,
                            static_ch,
                        )
                        if ds == "training":
                            train_X.append(Xs)
                            train_y.append(ys)
                        else:
                            test_X.append(Xs)
                            test_y.append(ys)
        train_X = np.concatenate(train_X, axis=0)
        test_X = np.concatenate(test_X, axis=0)
        train_y = np.concatenate(train_y, axis=0)
        test_y = np.concatenate(test_y, axis=0)
        pdb.set_trace()

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
