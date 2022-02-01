# -*- coding: utf-8 -*-
from box import Box
import logging
from pathlib import Path
import argparse
import glob
import random
import os
import shutil

from clearml import Dataset
from clearml import Task


def main():
    """ Runs data processing scripts to turn raw data from input path into
        a subset ready for quick training saved in output path.
    """

    task = Task.init(project_name='t4c', task_name='subset_creation')


    parser = argparse.ArgumentParser(description="T4C data subset creation")
    parser.add_argument('-c', '--config',type=str, help='location of config file')

    logger = logging.getLogger(__name__)

    args = parser.parse_args()
    print (args)

    conf = Box.from_yaml(filename=args.config)
    task.connect_configuration(configuration=args.config, name="yaml file")
    print (conf.val_set.cities)

    # random.seed(args.seed)

    input_path = Path(conf.input_path)
    output_path = Path(conf.output_path)
    
    # uses the name of the yaml file as dataset folder name
    ds_name = Path(args.config).stem
    output_path = output_path/ds_name
    
    if os.path.exists(output_path):
        logger.info('Folder exists already. Exiting!')
        return

    # registering clearml dataset
    dataset = Dataset.create(dataset_name=ds_name, dataset_project="subset_creation")

    # loop through the cities and select the required number of samples
    # for train set and val set. Note that cities appearing in both train and
    # val set is not supported (no guarantees on duplicates)
    for city in conf.train_set.cities:
        files = []
        logger.info('Sampling %s for training', city)

        if not os.path.exists(output_path/city/'training'):
            os.makedirs(output_path/city/'training')
        for file in glob.glob(str(input_path/city/'training'/'*')):
            files.append(file)

        # there should be enough samples to create the subset without duplication
        assert (conf.train_set.count) <= len(files)
        random.shuffle(files)
        for file in files[:conf.train_set.count]:
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
    if type(conf.val_set.cities) == str:
        conf.val_set.cities = [conf.val_set.cities]

    for city in conf.val_set.cities:

        files = []
        logger.info('Sampling %s for validation', city)

        if not os.path.exists(output_path/city/'validation'):
            os.makedirs(output_path/city/'validation')
        for file in glob.glob(str(input_path/city/'training'/'*')):
            files.append(file)

        # there should be enough samples to create the subset without duplication
        assert (conf.val_set.count) <= len(files)
        random.shuffle(files)
        for file in files[:conf.val_set.count]:
            shutil.copy(file, output_path/city/'validation'/Path(file).name)

        res_file = city+"_map_high_res.h5"
        static_file = city+"_static.h5"

        shutil.copy(input_path/city/static_file, output_path/city/static_file)
        shutil.copy(input_path/city/res_file, output_path/city/res_file)

    dataset.add_files(path=output_path)
    dataset.upload()
    dataset.finalize()

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
