# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
import argparse
import glob

from clearml import Task

def main():
    """ Runs data processing scripts to turn raw data from input path into
        a subset ready for quick training saved in output path.
    """

    #task = Task.init(project_name='t4c', task_name='subset_creation')

    parser = argparse.ArgumentParser(description="T4C data subset creation")
    parser.add_argument('-i', '--input_path',type=str, required=True, help='path to the raw full dataset')
    parser.add_argument('-o', '--output_path',type=str, required=True, help='path to where the subset will be stored', default=Path('../data/.raw/'))
    parser.add_argument('-t', '--train_count',type=int, help='Count of training samples from each map', default=10)
    parser.add_argument('-v', '--val_count',type=int, help='Count of training samples from each map', default=10)
    parser.add_argument('-s', '--seed',type=int, help='random seed (default:1)', default=1)

    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    args = parser.parse_args()
    print (args)
    cities = ['ANTWERP', 'BANGKOK', 'BARCELONA', 'BERLIN', 'CHICAGO', 'ISTANBUL', 'MELBOURNE', 'MOSCOW', 'NEWYORK', 'VIENNA']

    files = []
    for city in cities:
        for file in glob.glob(args.input_path + '/'+ city + '/training/*'):
            files.append(file)
        print (len(files))
        break







if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
