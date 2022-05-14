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
from ignite.utils import convert_tensor
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler

from src.data.dataset import T4CDataset, train_collate_fn

import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import mean_squared_error

from pytorch_wavelets import DWTForward, DWTInverse # (or import DWT, IDWT)
import matplotlib.animation as animation

from hydra import initialize, initialize_config_module, initialize_config_dir, compose
from hydra.utils import instantiate
from omegaconf import OmegaConf

from clearml import Dataset, Task
perm = np.array([[0,1,2,3,4,5,6,7],
        [2,3,4,5,6,7,0,1],
        [4,5,6,7,0,1,2,3],
        [6,7,0,1,2,3,4,5]
        ])
# dynamic_input_mean = np.load('data/processed/dynamic_input_mean.npy')
# dynamic_input_std = np.load('data/processed/dynamic_input_std.npy')

# dynamic_input_mean = torch.from_numpy(dynamic_input_mean)[None, None, :, None, None].float()
# dynamic_input_std = torch.from_numpy(dynamic_input_std)[None, None, :, None, None].float()


def reset_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


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
        img = ax.imshow(img, animated=True)
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

class NaiveAverage(torch.nn.Module):  # noqa
    def __init__(self):
        """Returns prediction consisting of repeating last frame."""
        super(NaiveAverage, self).__init__()

    def forward(self, x: torch.Tensor, *args, **kwargs):
        x = torch.mean(x, dim=1)
        x = torch.unsqueeze(x, dim=1)
        x = torch.repeat_interleave(x, repeats=6, dim=1)
        return x

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


def unstack_on_time(data: torch.Tensor, batch_dim:bool = False, num_channels=4, crop= None):
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
        if crop is not None:
            left, right, top, bottom = crop
            right = width - right
            bottom = height - bottom
            data = data[:, :,:, top:bottom, left:right]

        #data = (data+dynamic_input_mean)*dynamic_input_std
        # (k, 12, 8, 495, 436) -> (k, 12, 495, 436, 8)
        data = torch.moveaxis(data, 2, 4)

        if not batch_dim:
            # `(1, 12, 495, 436, 8) -> (12, 495, 436, 8)`
            data = torch.squeeze(data, 0)
        return data

def prepare_data(batch, dynamic_channels, out_channels, transform_p, switch=None):
        in_h = transform_p['in_h']
        in_w = transform_p['in_w']
        is_static = transform_p['static']
        pad_tuple = tuple(transform_p['pad_tuple'])
        if is_static == True:
            dynamic_channels = dynamic_channels - 9
        dynamic, static, target  = batch
        target = dynamic[:, 11:12, switch[0][0]:switch[0][0]+1, :, :] - target
        #dynamic = (dynamic - dynamic_input_mean) / dynamic_input_std
        dynamic = dynamic.reshape(-1, dynamic_channels, in_h, in_w)
        if switch is not None:
            dynamic = dynamic[:,switch.flatten(),:,:]

        #target = target.reshape(-1, out_channels, in_h, in_w)
        #target = F.pad(target, pad=pad_tuple)
        if is_static:
            input_batch = torch.cat([dynamic, static], dim=1)
        else:
            input_batch = dynamic

        input_batch = F.pad(input_batch, pad=pad_tuple)

        #target = (target - dynamic_input_mean) / dynamic_input_std

        return input_batch, target

"""
Provides evaluation information for a given model and dataset.
Information include
- MSE
- Single pixel time series
- Single sample true and predicted traffic maps
"""
def main():
    reset_seeds(123)
    task = Task.init(project_name="t4c_eval", task_name="Eval avg")
    logger = task.get_logger()
    args = {
        'task_id': 'fb40f8c008b04d5cbe10655a6ffe151e',
        'batch_size': 4,
        'num_workers': 2,
        'pixel': (108, 69),#(221, 192),
        'loader': 'val',
        'num_channels': 8,
        'viz_dir': [0,1,2,3],
        'viz_idx': 0,
        'time_step': 0, #time step to plot
        'max_idx': 240
    }

    task.connect(args)
    print ('Arguments: {}'.format(args))

    # get OmegaConf file from ClearML and parse
    train_task = Task.get_task(task_id=args['task_id'])
    cfg = train_task.get_configuration_object("OmegaConf")
    cfg = OmegaConf.create(cfg)
    print (cfg)
    cfg.model.dataset.root_dir = "7days"
    # instantiate model
    try:
        root_dir = Dataset.get(dataset_project="t4c", dataset_name=cfg.model.dataset.root_dir).get_local_copy()
    except:
        print("Could not find dataset in clearml server. Exiting!")
    
    if cfg.model.dataset.perm == True:
        is_perm = True
        cfg.model.dataset.perm = False
        cfg.model.dataset.single_channel = None
    else:
        is_perm = False
    model = instantiate(cfg.model, dataset={"root_dir":root_dir})
    model_path = train_task.artifacts['model_checkpoint'].get_local_copy()
    network = model.network
    network = network.to('cuda')
    #model_path = "/data/t5chx.pt"
    #model_state_dict = torch.load(model_path)
    model_state_dict = torch.load(model_path+'/'+os.listdir(model_path)[0])#,map_location=torch.device('cpu'))
    network.load_state_dict(model_state_dict['train_model'])
    network.eval()

    max_idx = args['max_idx']
    bs = args['batch_size']
    d = args['num_channels']
    #dataloader_config = configs[model_str].get("dataloader_config", {})

    g = torch.Generator()
    g.manual_seed(123)

    if args['loader'] == 'val':
        loader = DataLoader(model.v_dataset, batch_size=bs, num_workers=args['num_workers'], worker_init_fn=seed_worker, generator=g, shuffle=False, collate_fn=train_collate_fn)
    else:
        loader = DataLoader(model.t_dataset, batch_size=bs, num_workers=args['num_workers'], worker_init_fn=seed_worker, generator=g, shuffle=False, collate_fn=train_collate_fn)
    print ('Dataloader first few files: {}'.format(loader.dataset.file_list[:10]))

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
    pixel_x, pixel_y = args['pixel']
    t = args['time_step']

    m = NaiveAverage()
    #hack_map = [1, 2, 3, 0]
    if is_perm == False:
        for idx, i in (enumerate(loader)):
            #inp, true = prepare_data(i, cfg.model.network.in_channels,
            #                          cfg.model.network.out_channels, cfg.train.transform)
            #pred = network(inp.to("cuda"))
            #pred = pred.cpu().detach()#.numpy()

            #pred = unstack_on_time(pred, d, num_channels=d,
            #                           crop = tuple(cfg.train.transform.pad_tuple))
            #true = (true+dynamic_input_mean)*dynamic_input_std
            pred = m.forward(i[0])
            true = i[2]
            pred = torch.moveaxis(pred, 2, 4)
            true = torch.moveaxis(true, 2, 4)

            pred1 = pred[:,:,:,:,0::2]
            true1 = true[:,:,:,:,0::2]
            pred2 = pred[:,:,:,:,1::2]
            true2 = true[:,:,:,:,1::2]
            if is_waveTransform:
                _,_,rh,rw = pred.shape
                Yl = pred[:, :24,:,:]
                Yh = [pred[:, 24:,:,:].reshape((bs, 24, 3, rh, rw))]
                #Yh[0][:,:,:,:,:] = 0
                pred = ifm((Yl, Yh))

                Yl = true[:, :24,:,:]
                Yh = [true[:, 24:,:,:].reshape((bs, 24, 3, rh, rw))]
                true = ifm((Yl, Yh))


            pred = np.clip(pred, 0, 255)
            #print (mean_squared_error(pred.flatten(), true.flatten()))
            mse.append(mean_squared_error(pred.flatten(), true.flatten()))
            mse1.append(mean_squared_error(pred1.flatten(), true1.flatten()))
            mse2.append(mean_squared_error(pred2.flatten(), true2.flatten()))

            if idx>=max_idx/bs:
                continue
            else:
                if is_waveTransform:
                    true = unstack_on_time(true[:,:,:-1,:], d)
                    pred = unstack_on_time(pred[:,:,:-1,:], d)

                p_pred = (pred[:,t, pixel_x, pixel_y, :].numpy())
                p_true = (true[:,t, pixel_x, pixel_y, :].numpy())
                trues[idx*bs:idx*bs+bs] = p_true
                preds[idx*bs:idx*bs+bs] = p_pred
            if len(args['viz_dir']) != 0:
                # with open('data/interim/'+str(cfg.model.dataset.time_step)+"true_5avg.npy", 'wb') as f: np.save(f, true[0])
                # with open('data/interim/'+str(cfg.model.dataset.time_step)+"pred_5avg.npy", 'wb') as f: np.save(f, pred[0])
                # return
                if idx==args['viz_idx']:
                    plot_tmaps(true[0].numpy(), pred[0].numpy(), args['viz_dir'], logger)
            #msenz.append(mse_func(pred.flatten(), true.flatten(), nonzero))
            #trues.extend(p_true)
            #preds.extend(p_pred)
            #if idx==240:
            #break
    else:
        if cfg.model.dataset.time_step is None:
            print("Using default timesteps 6")
            time_steps = 6
        else:
            time_steps = cfg.model.dataset.time_step
        h = 495
        w = 436
        pred_comb = np.zeros((bs, time_steps, h, w, d))
        true_comb = np.zeros((bs, time_steps, h, w, d))
        for idx, i in (enumerate(loader)):

            for directions in range(4):
                switch = perm[directions]
                for c in range(1,12): switch = np.vstack([switch, perm[directions]+(8*c)])
                inp, true = prepare_data(i, cfg.model.network.in_channels,
                                      cfg.model.network.out_channels, cfg.train.transform, switch)
                true = torch.moveaxis(true, 2, 4)


                pred = network(inp.to('cuda'))
                pred = pred.cpu().detach()#.numpy()

                # Remove padding and unstack time channels from predictions
                pred = unstack_on_time(pred, d, num_channels=1,
                                       crop = tuple(cfg.train.transform.pad_tuple))

                #print (F.mse_loss(pred[:,:,:,:,0], true[:,:,:,:,directions]))
                pred_comb[:, :, :, :,directions] = pred[:,:,:,:,0]#pred.numpy()
                true_comb[:, :, :, :,directions] = true[:,:,:,:,directions]#outp.numpy()

                if is_waveTransform:
                    _,_,rh,rw = pred.shape
                    Yl = pred[:, :24,:,:]
                    Yh = [pred[:, 24:,:,:].reshape((bs, 24, 3, rh, rw))]
                    #Yh[0][:,:,:,:,:] = 0
                    pred = ifm((Yl, Yh))

                    Yl = true[:, :24,:,:]
                    Yh = [true[:, 24:,:,:].reshape((bs, 24, 3, rh, rw))]
                    true = ifm((Yl, Yh))

            chk = i[0][:,11:12, 0::2, :,:]
            chk = torch.moveaxis(chk, 2, 4)
            pred_comb = chk.numpy() + pred_comb
            true_comb = chk.numpy() + true_comb

            pred_comb = np.clip(pred_comb, 0, 255)
            try:
                # timestep slice
                #pred1 = pred_comb[:,0,:,:,1]
                #true1 = true_comb[:,0,:,:,1]
                #pred2 = pred_comb[:,5,:,:,1]
                #true2 = true_comb[:,5,:,:,1]

                #print(mean_squared_error(pred_comb.flatten(), true_comb.flatten()))

                #pdb.set_trace()
                mse.append(mean_squared_error(pred_comb.flatten(), (true_comb).flatten()))
                #print (mse)
                #mse2.append(mean_squared_error(pred2.flatten(), true2.flatten()))

            except:
                print ("Failed in mse calc!")

            if idx == args['viz_idx']:
                #plot_tmaps(true_comb[0], pred_comb[0], args['viz_dir'], logger)
                continue
            # mse1.append(mean_squared_error(pred1.flatten(), true1.flatten()))
            # mse2.append(mean_squared_error(pred2.flatten(), true2.flatten()))

            if idx>=max_idx/bs:
                break
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
    print("MSE vol: {}".format(sum(mse1)/len(mse1)))
    print("MSE speed: {}".format(sum(mse2)/len(mse2)))
    plot_dims(logger, trues, preds, d)
if __name__ == "__main__":
    main()
