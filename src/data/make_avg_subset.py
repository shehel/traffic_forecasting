#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
from pathlib import Path
import glob
import random
import os
import shutil

from clearml import Dataset
from clearml import Task

import hydra
from omegaconf import DictConfig, OmegaConf

from src.common.h5_util import load_h5_file, write_data_to_h5
import numpy as np
import bottleneck as bn
import pdb
import h5py
# TODO change logger to common.util
def main() -> None:
    """ Runs data processing scripts to turn raw data from input path into
        a subset ready for quick training saved in output path.
    """
    task = Task.init(project_name='t4c', task_name='Moving avg subset')
    logger = logging.getLogger(__name__)

    args = {
        'task_id': '53ec3f01cb1e4385bf0e441551af0452',
        'name': '5avg7days',
    }

    task.connect(args)
    print ('Arguments: {}'.format(args))

    # get OmegaConf file from ClearML and parse
    train_task = Task.get_task(task_id=args['task_id'])
    cfg = train_task.get_configuration_object("OmegaConf")
    cfg = OmegaConf.create(cfg)
    print (cfg)

    random.seed(cfg.random_seed)
    # uses the name of the yaml file aos dataset folder name
    try:
        print (abssdf)
        Dataset.get(dataset_project="t4c", dataset_name=args['name'])
        logger.info('Dataset exists. Skipping dataset creation.')
    except:

        root_dir = Dataset.get(dataset_project="t4c", dataset_name=cfg.name).get_local_copy()
        logger.info('Loading dataset')
        input_path = Path(root_dir)
        output_path = Path("data/raw/"+args['name'])
        if os.path.exists(output_path):
            logger.info('Folder exists. Fatal error, exiting!')
            # return

        avg_term = 5

        # loop through the cities and select the required number of samples
        # for train set and val set. Note that cities appearing in both train and
        # val set is not supported (no guarantees on duplicates)

        # for city in cfg.train_set.cities:
        #     files = []
        #     logger.info('Opening %s files for training set processing', city)

        #     if not os.path.exists(output_path/city/'training'):
        #         os.makedirs(output_path/city/'training')
        #     for file in glob.glob(str(input_path/city/'training'/'*')):
        #         files.append(file)

        #     for file in files:
        #         day_data = load_h5_file(file)

        #         day_data = bn.move_mean(day_data, avg_term, axis=0).astype(np.uint8)
        #         day_data = day_data[(avg_term-1):,:,:,:]
        #         write_data_to_h5(day_data,output_path/city/'training'/Path(file).name)
        #     res_file = city+"_map_high_res.h5"
        #     static_file = city+"_static.h5"
        #     shutil.copy(input_path/city/static_file, output_path/city/static_file)
        #     shutil.copy(input_path/city/res_file, output_path/city/res_file)

        # val set

        # handling the case when there's a single
        # validation city.
        # TODO handle 1 or more cities well. yaml doesn't parse
        # single element list correctly.
        if type(cfg.val_set.cities) == str:
            cfg.val_set.cities = [cfg.val_set.cities]
        for city in cfg.val_set.cities:

            files = []
            logger.info('Opening %s files for validation set processing', city)

            if not os.path.exists(output_path/city/'validation'):
                os.makedirs(output_path/city/'validation')
            for file in glob.glob(str(input_path/city/'validation'/'*')):
                day_data = load_h5_file(file)
                day_data = bn.move_mean(day_data, avg_term, axis=0).astype(np.uint8)
                day_data = day_data[(avg_term-1):, :, :, :]
                write_data_to_h5(day_data, output_path/city/'validation'/Path(file).name)


            res_file = city+"_map_high_res.h5"
            static_file = city+"_static.h5"

            shutil.copy(input_path/city/static_file, output_path/city/static_file)
            shutil.copy(input_path/city/res_file, output_path/city/res_file)

        # registering clearml dataset
        dataset = Dataset.create(dataset_name=args['name'], dataset_project="t4c")

        dataset.add_files(path=output_path)
        dataset.upload()
        dataset.finalize()


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
