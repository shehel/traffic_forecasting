import random
import os
import pdb
from tqdm import tqdm
import numpy as np
import sys
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from clearml import Task, StorageManager, Dataset, Logger
from src.data.dataset import T4CDataset
from src.data.transform import UNetTransform

import tensorly as tl
from tensorly.decomposition import tucker
from sklearn.metrics import mean_squared_error

def reset_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_recon_error(tensor, factors):
    reduc = tl.tenalg.multi_mode_dot(tensor, factors, transpose=True)
    recon = tl.tenalg.multi_mode_dot(reduc, factors, transpose=False)
    return mean_squared_error(tensor.flatten(), recon.flatten())

def plot(x, y, title, labelx="dimensions", labely="MSE"):
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(labelx)
    plt.ylabel(labely)
    plt.draw()
    plt.show()

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
    task = Task.init(project_name="t4c_tensor_decomp", task_name="Reconstruction error")
    logger = task.get_logger()
    args = {
        'trainset_dir': 'third',
        'valset_dir': 'third',
        'train_filter': 'training/',
        'val_filter': 'validation/',
        'train_city':'**/',
        'val_city': '**/',
        'batch_size': 8,
        'num_workers': 4 ,
        'iter': 0,
        'sampling_height': 1,
        'sampling_width': 1,
        'labels': 0, # decompose inputs or outputs of the model
        'dim': 0,
        'step_size': 32,
    }

    task.connect(args)
    print ('Arguments: {}'.format(args))

    try:
        trainset_dir = Dataset.get(dataset_project="t4c", dataset_name=args['trainset_dir']).get_local_copy()
        valset_dir = Dataset.get(dataset_project="t4c", dataset_name=args['valset_dir']).get_local_copy()
        print ("Dataset found.")
    except:
        print("Unable to load dataset. Exiting!")
        sys.exit()

    # TODO Transform can be done better in config
    transform = UNetTransform(stack_time=True, pre_batch_dim=False, post_batch_dim=True, crop_pad=[6,6,1,0], num_channels=8)
    train_dataset = T4CDataset(root_dir=trainset_dir, file_filter=args['train_city']+args['train_filter']+'*8ch.h5',
                            transform=transform, sampling_height=args['sampling_height'], sampling_width=args['sampling_width'])
    val_dataset = T4CDataset(root_dir=valset_dir, file_filter=args['val_city']+args['val_filter']+'*8ch.h5',
                            transform=transform, sampling_height=args['sampling_height'], sampling_width=args['sampling_width'])
    #train_dataset = T4CDataset(root_dir=trainset_dir, file_filter='ISTANBUL/training/*8ch.h5')
    #val_dataset = T4CDataset(root_dir=valset_dir, file_filter='**/validation/*8ch.h5')

    val_loader = DataLoader(val_dataset, batch_size=args['batch_size'], num_workers=args['num_workers'], shuffle=False)
    train_loader = DataLoader(train_dataset, batch_size=args['batch_size'], num_workers=args['num_workers'])

    dims = iter(train_loader).next()[args['labels']][0].shape
    dims_v = iter(val_loader).next()[args['labels']][0].shape

    print ("Image dimension: %sx%sx%s" % (dims[0], dims[1], dims[2]))

    train_avg = np.zeros(dims)
    val_avg = np.zeros(dims_v)
    for idx, sample in enumerate(train_loader):
        b_avg = torch.mean(sample[args['labels']], 0).numpy()
        train_avg += b_avg
        if idx == args['iter']:
            break
    t_sample = sample[args['labels']][0].numpy()

    train_avg = train_avg/(idx+1)

    for idx, sample in enumerate(val_loader):
        b_avg = torch.mean(sample[args['labels']], 0).numpy()
        val_avg += b_avg
        if idx == args['iter']:
            break
    v_sample = sample[args['labels']][-1].numpy()

    val_avg = val_avg/(idx+1)

    errors_tavg_tavg = []
    errors_tavg_train = []
    errors_tavg_vavg = []
    errors_tavg_val = []
    errors_vavg_vavg = []
    errors_vavg_val = []
    pts = list(range(args['step_size'], dims[args['dim']]+args['step_size'],args['step_size']))
    for x in tqdm(pts):
        if args['dim']==0:
            tten = tucker(train_avg, rank=[x,dims[1],dims[2]])
            tten_val = tucker(val_avg, rank=[x,dims[1],dims[2]])
        elif args['dim']==1:
            tten = tucker(train_avg, rank=[dims[0],x,dims[2]])
            tten_val = tucker(val_avg, rank=[dims[0],x,dims[2]])
        else:
            tten = tucker(train_avg, rank=[dims[0],dims[1],x])
            tten_val = tucker(val_avg, rank=[dims[0],dims[1],x])

        #recon = tl.tucker_to_tensor(tten)
        factors = tten[1]
        factors_val = tten_val[1]

        # errors_tavg_tavg.append(get_recon_error(train_avg, train_avg, factors))
        # errors_tavg_train.append(get_recon_error(train_avg, t_sample, factors))
        # errors_tavg_vavg.append(get_recon_error(train_avg, val_avg, factors))
        # errors_tavg_val.append(get_recon_error(train_avg, v_sample, factors))
        # errors_vavg_vavg.append(get_recon_error(val_avg, val_avg, factors_val))
        # errors_vavg_val.append(get_recon_error(val_avg, v_sample, factors_val))
        #pdb.set_trace()
        logger.report_scalar("train avg", "train avg", iteration=x,value=get_recon_error(train_avg, factors))
        logger.report_scalar("train avg", "train sample", iteration=x,value=get_recon_error(t_sample, factors))
        logger.report_scalar("train avg", "val avg", iteration=x, value=get_recon_error(val_avg, factors))
        logger.report_scalar("train avg", "val sample", iteration=x, value=get_recon_error(v_sample, factors))
        logger.report_scalar("val avg", "val avg", iteration=x, value=get_recon_error(val_avg, factors_val))
        logger.report_scalar("val avg", "val sample", iteration=x, value=get_recon_error(v_sample, factors_val))
    # figure, axis = plt.subplots(3, 2)


    # # Make dimension-MSE plots
    # axis[0, 0].plot(pts, errors_tavg_tavg)
    # axis[0, 0].set_title("train avg - train avg")

    # axis[0, 1].plot(pts, errors_tavg_train)
    # axis[0, 1].set_title("train avg - train sample")

    # axis[1, 0].plot(pts, errors_tavg_vavg)
    # axis[1, 0].set_title("train avg - val avg")

    # axis[1, 1].plot(pts, errors_tavg_val)
    # axis[1, 1].set_title("train avg - val sample")

    # axis[2, 0].plot(pts, errors_vavg_vavg)
    # axis[2, 0].set_title("val avg - val avg")

    # axis[2, 1].plot(pts, errors_vavg_val)
    # axis[2, 1].set_title("val avg - val sample")

    # figure.text(0.5, 0.01, "dimension", va='center')
    # figure.text(0.01, 0.5, "MSE", va='center', rotation='vertical')


    # plot(pts, errors_tavg_tavg, "train avg - train avg")

    # plot(pts, errors_tavg_train, "train avg - train sample")

    # plot(pts, errors_tavg_vavg, "train avg - val avg")

    # plot(pts, errors_tavg_val, "train avg - val sample")

    # plot(pts, errors_vavg_vavg, "val avg - val avg")

    # plot(pts, errors_vavg_val, "val avg - val sample")


if __name__ == "__main__":
    main()
