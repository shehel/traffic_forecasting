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

# TODO change logger to common.util
@hydra.main(config_path="../../config/data", config_name="21d")
def main(cfg: DictConfig) -> None:
    """ Runs data processing scripts to turn raw data from input path into
        a subset ready for quick training saved in output path.
    """
    task = Task.init(project_name='t4c', task_name='subset_creation')
    logger = logging.getLogger(__name__)

    random.seed(cfg.random_seed)

    # uses the name of the yaml file aos dataset folder name
    try:
        Dataset.get(dataset_project="t4c", dataset_name=cfg.name)
        logger.info('Dataset exists. Skipping to next step.')
    except:

        logger.info('Creating dataset')
        input_path = Path(cfg.input_path)
        output_path = Path("data/raw/"+cfg.name)
        if os.path.exists(output_path):
            logger.info('Folder exists. Fatal error, exiting!')
            return

        # loop through the cities and select the required number of samples
        # for train set and val set. Note that cities appearing in both train and
        # val set is not supported (no guarantees on duplicates)
        for city in cfg.train_set.cities:
            files = []
            logger.info('Sampling %s for training', city)

            if not os.path.exists(output_path/city/'training'):
                os.makedirs(output_path/city/'training')
            for file in glob.glob(str(input_path/city/'training'/'*')):
                files.append(file)

            # there should be enough samples to create the subset without duplication
            assert (cfg.train_set.count) <= len(files)
            random.shuffle(files)
            for file in files[:cfg.train_set.count]:
                shutil.copy(file, output_path/city/'training'/Path(file).name)

            res_file = city+"_map_high_res.h5"
            static_file = city+"_static.h5"
            shutil.copy(input_path/city/static_file, output_path/city/static_file)
            shutil.copy(input_path/city/res_file, output_path/city/res_file)

        # val set

        # handling the case when there's a single
        # validation city.
        # TODO handle 1 or more cities well. yaml doesn't parse
        # single element list correctly.
        if type(cfg.val_set.cities) == str:
            cfg.val_set.cities = [cfg.val_set.cities]

        for city in cfg.val_set.cities:

            files = []
            logger.info('Sampling %s for validation', city)

            if not os.path.exists(output_path/city/'validation'):
                os.makedirs(output_path/city/'validation')
            for file in glob.glob(str(input_path/city/'training'/'*')):
                files.append(file)

            # there should be enough samples to create the subset without duplication
            assert (cfg.val_set.count) <= len(files)
            random.shuffle(files)
            for file in files[:cfg.val_set.count]:
                shutil.copy(file, output_path/city/'validation'/Path(file).name)

            res_file = city+"_map_high_res.h5"
            static_file = city+"_static.h5"

            shutil.copy(input_path/city/static_file, output_path/city/static_file)
            shutil.copy(input_path/city/res_file, output_path/city/res_file)

        # registering clearml dataset
        dataset = Dataset.create(dataset_name=cfg.name, dataset_project="t4c")

        dataset.add_files(path=output_path)
        dataset.upload()
        dataset.finalize()


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
