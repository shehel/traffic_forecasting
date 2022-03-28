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



import matplotlib.animation as animation
from PIL import Image
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
def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img.convert('RGB')

def get_ani(mat):
    fig, ax = plt.subplots(figsize=(8, 8))
    imgs = []
    for img in mat:
        img = ax.imshow(img, animated=True, vmax=np.median(img.flatten())+50)
        imgs.append([img])
    ani = animation.ArtistAnimation(fig, imgs, interval=1000, blit=True, repeat_delay=3000)
    return ani.to_html5_video()

def plot_tmaps(true, pred, viz_dir, logger):
    for dir in viz_dir:
        fig = plt.figure(figsize=(50, 35))

        # setting values to rows and column variables
        rows = 2
        columns = pred.shape[0]
        for t_step in range(pred.shape[0]):

            # reading images


            # Adds a subplot at the 1st position
            fig.add_subplot(rows, columns, t_step+1)

            # showing image
            _ = plt.imshow(pred[t_step,:,:,dir])
            plt.axis('off')

        plt.title("pred")

        for t_step in range(true.shape[0]):

            # Adds a subplot at the 1st position
            fig.add_subplot(rows, columns, t_step+pred.shape[0]+1)
            # showing image
            _ = plt.imshow(true[t_step,:,:,dir])
            plt.axis('off')

        plt.title("true")
        plt.close()

        logger.current_logger().report_image("viz", "images", iteration=dir, image=fig2img(fig))

        logger.current_logger().report_media(
                "viz", "true frames", iteration=dir, stream=get_ani(true[:,:,:,dir]), file_extension='html')

        logger.current_logger().report_media(
                "viz", "pred frames", iteration=dir, stream=get_ani(pred[:,:,:,dir]), file_extension='html')


def plot_dims(logger, true_series, pred_series, dim=8):

    x = list(range(true_series.shape[0]))

    for i in range(0, true_series.shape[-1]):
        logger.current_logger().report_scatter2d(
        str(i),
        "true",
        iteration=0,
        scatter=np.dstack((x, true_series[:,i])).squeeze(),
        xaxis="t",
        yaxis="count",
        mode='lines+markers'
    )
        logger.current_logger().report_scatter2d(
            str(i),
            "pred",
            iteration=0,
            scatter=np.dstack((x, pred_series[:,i])).squeeze(),
            xaxis="t",
            yaxis="count",
            mode='lines+markers'
        )




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
        'task_id': '9be6fe52a8c44efe8052bfd4e24f2351',
        'batch_size': 1,
        'num_workers': 0,
        'pixel': (108, 69),
        'loader': 'val',
        'num_channels': 4,
        'viz_dir': [0,1,2,3],
        'viz_idx': 0
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
    model_path = "/data/best1dir.pt"
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

    pred_comb = np.zeros((bs, 6, 495, 436, d))
    true_comb = np.zeros((bs, 6, 495, 436, d))
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
            pred_comb[:, :, :, :,directions:directions+1] = pred.numpy()
            true_comb[:, :, :, :,directions:directions+1] = true.numpy()


        # pred1 = pred[:,:,:,:,::2]
        # true1 = true[:,:,:,:,::2]
        # pred2 = pred[:,:,:,:,1::2]
        # true2 = true[:,:,:,:,1::2]
            if is_waveTransform:
                _,_,rh,rw = pred.shape
                Yl = pred[:, :24,:,:]
                Yh = [pred[:, 24:,:,:].reshape((bs, 24, 3, rh, rw))]
                #Yh[0][:,:,:,:,:] = 0
                pred = ifm((Yl, Yh))

                Yl = true[:, :24,:,:]
                Yh = [true[:, 24:,:,:].reshape((bs, 24, 3, rh, rw))]
                true = ifm((Yl, Yh))

        try:
            mse.append(mean_squared_error(pred_comb.flatten(), true_comb.flatten()))
        except:
            print ("Failed in mse calc!")

        if idx == args['viz_idx']:
            plot_tmaps(true_comb[0], pred_comb[0], args['viz_dir'], logger)
        # mse1.append(mean_squared_error(pred1.flatten(), true1.flatten()))
        # mse2.append(mean_squared_error(pred2.flatten(), true2.flatten()))

        if idx>=max_idx/bs:
            continue
        else:
        #     if is_waveTransform:
        #         true = unstack_on_time(true[:,:,:-1,:], d)
        #         pred = unstack_on_time(pred[:,:,:-1,:], d)

             p_pred = (pred_comb[:,t, pixel_x, pixel_y, :])
             p_true = (true_comb[:,t, pixel_x, pixel_y, :])
             trues[idx*bs:idx*bs+bs] = p_true
             preds[idx*bs:idx*bs+bs] = p_pred

        #msenz.append(mse_func(pred.flatten(), true.flatten(), nonzero))
        #trues.extend(p_true)
        #preds.extend(p_pred)
        #if idx==240:
        #break
    print (mse)
    print("Overall MSE: {}".format(sum(mse)/len(mse)))
    # print("MSE vol: {}".format(sum(mse1)/len(mse1)))
    # print("MSE speed: {}".format(sum(mse2)/len(mse2)))
    plot_dims(logger, trues, preds, d)
if __name__ == "__main__":
    main()
