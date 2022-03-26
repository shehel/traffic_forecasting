import argparse
import binascii
import logging
import os
import random
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
from pathlib import Path
from typing import Optional
import pdb

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
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler

from src.data.dataset import T4CDataset



import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error

from pytorch_wavelets import DWTForward, DWTInverse # (or import DWT, IDWT)


from hydra import initialize, initialize_config_module, initialize_config_dir, compose
from hydra.utils import instantiate
from omegaconf import OmegaConf

from clearml import Dataset, Task
perm = np.array([[0,1,2,3,4,5,6,7],
        [2,3,4,5,6,7,0,1],
        [4,5,6,7,0,1,2,3],
        [6,7,0,1,2,3,4,5]
        ])

def reset_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def plot_dims(task, true_series, pred_series, dim=8):

    x = list(range(true_series.shape[0]))
    # For Sine Function
    plt.plot(x, true_series[:,0])
    task.logger.report_matplotlib_figure(title="VolNW", series="VolNW", figure=plt)
    plt.plot(x, pred_series[:,0])
    task.logger.report_matplotlib_figure(title="VolNW", series="VolNW", figure=plt)
    plt.show()
    plt.close()
    #
    # For Cosine Function
    plt.plot(x, true_series[:,1])
    task.logger.report_matplotlib_figure(title="SpeedNW", series="SpeedNW", figure=plt)
    plt.plot(x, pred_series[:,1])
    task.logger.report_matplotlib_figure(title="SpeedNW", series="SpeedNW", figure=plt)
    plt.show()
    plt.close()
    plt.plot(x, true_series[:,2])
    task.logger.report_matplotlib_figure(title="VolumeNE", series="VolumeNE", figure=plt)
    plt.plot(x, pred_series[:,2])
    task.logger.report_matplotlib_figure(title="VolumeNE", series="VolumeNE", figure=plt)
    plt.show()
    plt.close()
    # For Cosine Function
    plt.plot(x, true_series[:,3])
    task.logger.report_matplotlib_figure(title="SpeedNE", series="SpeedNE", figure=plt)
    plt.plot(x, pred_series[:,3])
    task.logger.report_matplotlib_figure(title="SpeedNE", series="SpeedNE", figure=plt)
    plt.show()
    plt.close()
    if dim == 8:
        plt.plot(x, true_series[:,4])
        task.logger.report_matplotlib_figure(title="VolumeSE", series="VolumeSE", figure=plt)
        plt.plot(x, pred_series[:,4])
        task.logger.report_matplotlib_figure(title="VolumeSE", series="VolumeSE", figure=plt)
        plt.show()
        plt.close()

        # For Cosine Function
        plt.plot(x, true_series[:,5])
        task.logger.report_matplotlib_figure(title="SpeedSE", series="SpeedSE", figure=plt)
        plt.plot(x, pred_series[:,5])
        task.logger.report_matplotlib_figure(title="SpeedSE", series="SpeedSE", figure=plt)
        plt.show()
        plt.close()

        plt.plot(x, true_series[:,6])
        task.logger.report_matplotlib_figure(title="VolumeSW", series="VolumeSW", figure=plt)
        plt.plot(x, pred_series[:,6])
        task.logger.report_matplotlib_figure(title="VolumeSW", series="VolumeSW", figure=plt)
        plt.show()
        plt.close()

        plt.plot(x, true_series[:,7])
        task.logger.report_matplotlib_figure(title="SpeedSW", series="SpeedSW", figure=plt)
        plt.plot(x, pred_series[:,7])
        task.logger.report_matplotlib_figure(title="SpeedSW", series="SpeedSW", figure=plt)
        plt.show()
        plt.close()


def unstack_on_time(data: torch.Tensor, batch_dim:bool = False, num_channels=4):
        """
        `(k, 12 * 8, 495, 436) -> (k, 12, 495, 436, 8)`
        """
        _, _, height, width = data.shape
        if not batch_dim:
            # `(12, 495, 436, 8) -> (1, 12, 495, 436, 8)`
            data = torch.unsqueeze(data, 0)

        num_time_steps = int(data.shape[1] / num_channels)
        # (k, 12 * 8, 495, 436) -> (k, 12, 8, 495, 436)
        data = torch.reshape(data, (data.shape[0],
                                    num_time_steps,
                                    num_channels,
                                    height,
                                    width))

        # (k, 12, 8, 495, 436) -> (k, 12, 495, 436, 8)
        data = torch.moveaxis(data, 2, 4)

        if not batch_dim:
            # `(1, 12, 495, 436, 8) -> (12, 495, 436, 8)`
            data = torch.squeeze(data, 0)
        return data

"""
Produces dimension-MSE error plots for the following
case:
- avg train tensor decomposition and avg train tensor reconstruction
- avg train tensor decomposition and random train tensor reconstruction
- avg train tensor decomposition and avg val tensor reconstruction
- avg train tensor decomposition and random val tensor reconstruction
- avg val tensor decomposition and avg val tensor reconstruction
- avg val tensor decomposition and random val tensor reconstruction
"""
def main():
    reset_seeds(123)
    task = Task.init(project_name="t4c_eval", task_name="Model Evaluation")
    logger = task.get_logger()
    args = {
        'task_id': '406605c977064682a4753d40ea5636ae',#'845d6fef6e4e439ab02574d54a4888a3',
        'batch_size': 1,
        'num_workers': 0,
        'pixel': (108, 69),
        'loader': 'val',
        'num_channels': 8
    }

    task.connect(args)
    print ('Arguments: {}'.format(args))

    # get OmegaConf file from ClearML and parse
    train_task = Task.get_task(task_id=args['task_id'])
    cfg = train_task.get_configuration_object("OmegaConf")
    cfg = OmegaConf.create(cfg)
    print (cfg)
    # instantiate model
    try:
        root_dir = Dataset.get(dataset_project="t4c", dataset_name=cfg.model.dataset.root_dir).get_local_copy()
    except:
        print("Could not find dataset in clearml server. Exiting!")

    model = instantiate(cfg.model, dataset={"root_dir":root_dir})
    #model_path = train_task.artifacts['model_checkpoint'].get_local_copy()
    model_path = "/home/shehel/waste/1dir2.5708.pt"
    network = model.network
    network = network.to('cuda')
    model_state_dict = torch.load(model_path)
    #model_state_dict = torch.load(model_path+'/'+os.listdir(model_path)[0])#,map_location=torch.device('cpu'))
    network.load_state_dict(model_state_dict['train_model'])
    network.eval()

    max_idx = 240
    bs = args['batch_size']
    d = args['num_channels']
    #dataloader_config = configs[model_str].get("dataloader_config", {})
    if args['loader'] == 'val':
        loader = DataLoader(model.v_dataset, batch_size=bs, num_workers=args['num_workers'], shuffle=False)
    else:
        loader = DataLoader(model.t_dataset, batch_size=bs, num_workers=args['num_workers'], shuffle=False)
    print ('Dataloader first few files: {}'.format(loader.dataset.files[:10]))
    trues = np.zeros((max_idx, d))
    preds = np.zeros((max_idx, d))


    mse=[]
    msenz=[]
    mse1=[]
    mse2=[]

    pixel_x, pixel_y = args['pixel']
    try:
        mode = cfg.model.dataset.transform.mode
        wave = cfg.model.dataset.transform.wave
        xfm = DWTForward(J=1, mode=mode, wave=wave)  # Accepts all wave types available to PyWavelets
        ifm = DWTInverse(mode=mode, wave=wave)
        is_waveTransform = True
    except:
        is_waveTransform = False

    print ('Wavelet Transform: {}'.format(is_waveTransform))

    #pixel_x, pixel_y = 101,132#108, 69
    pixel_x, pixel_y = 108, 69
    t = 0

    for idx, i in (enumerate(loader)):
        for directions in range(4):
            switch = perm[directions]
            for c in range(1,12): switch = np.vstack([switch, perm[directions]+(8*c)])
            inp = i[0][:,switch.flatten(),:,:]
            outp = i[1][:,directions::4, :,:]
            batch_prediction = network(inp.to('cuda'))
            batch_prediction = batch_prediction.cpu().detach()#.numpy()

            pred = model.t_dataset.transform.post_transform(batch_prediction)
            true = model.t_dataset.transform.post_transform(outp)

        # pred1 = pred[:,:,:,:,::2]
        # true1 = true[:,:,:,:,::2]
        # pred2 = pred[:,:,:,:,1::2]
        # true2 = true[:,:,:,:,1::2]
            if is_waveTransform:
                _,_,rh,rw = pred.shape
                Yl = pred[:, :24,:,:]
                Yh = [pred[:, 24:,:,:].reshape((bs, 24, 3, rh, rw))]
                Yh[0][:,:,:,:,:] = 0
                pred = ifm((Yl, Yh))

                Yl = true[:, :24,:,:]
                Yh = [true[:, 24:,:,:].reshape((bs, 24, 3, rh, rw))]
                true = ifm((Yl, Yh))

            try:
                mse.append(mean_squared_error(pred.flatten(), true.flatten()))
            except:
                pdb.set_trace()
        print (mse)
        # mse1.append(mean_squared_error(pred1.flatten(), true1.flatten()))
        # mse2.append(mean_squared_error(pred2.flatten(), true2.flatten()))

        # if idx>=max_idx/bs:
        #     continue
        # else:
        #     if is_waveTransform:
        #         true = unstack_on_time(true[:,:,:-1,:], d)
        #         pred = unstack_on_time(pred[:,:,:-1,:], d)

        #     p_pred = (pred[:,t, pixel_x, pixel_y, :].numpy())
        #     p_true = (true[:,t, pixel_x, pixel_y, :].numpy())
        #     trues[idx*bs:idx*bs+bs] = p_true
        #     preds[idx*bs:idx*bs+bs] = p_pred

        #msenz.append(mse_func(pred.flatten(), true.flatten(), nonzero))
        #trues.extend(p_true)
        #preds.extend(p_pred)
        #if idx==240:
        #break
    print (mse)
    print("Overall MSE: {}".format(sum(mse)/len(mse)))
    # print("MSE vol: {}".format(sum(mse1)/len(mse1)))
    # print("MSE speed: {}".format(sum(mse2)/len(mse2)))
    # plot_dims(task, trues, preds, d)
if __name__ == "__main__":
    main()
