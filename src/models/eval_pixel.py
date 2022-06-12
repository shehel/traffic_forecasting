#!/usr/bin/env python3
import numpy as np
import h5py
import matplotlib.pyplot as plt
import glob
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

from src.data.dataset import T4CDataset
from torch.utils.data import DataLoader

from clearml import Dataset, Task
import itertools
from multiprocessing import Pool, RawArray

from tqdm import tqdm
import pdb
try:
    import joblib
except ImportError:
    from sklearn.externals import joblib


def train_collate_fn(batch):
    dynamic_input_batch, static_input_batch, target_batch = zip(*batch)
    dynamic_input_batch = np.stack(dynamic_input_batch, axis=0)
    static_input_batch = np.stack(static_input_batch, axis=0)
    target_batch = np.stack(target_batch, axis=0)
    dynamic_input_batch = np.moveaxis(dynamic_input_batch, source=4, destination=2)
    target_batch = np.moveaxis(target_batch, source=4, destination=2)

    return dynamic_input_batch, static_input_batch, target_batch

var_dict = {}

def init_worker(sh_pred, dynamic_input, static_mask, filters, pred_shape, reg):
    # Using a dictionary is not strictly necessary. You can also
    # use global variables.
    var_dict['dynamic_input'] = dynamic_input
    var_dict['static_mask'] = static_mask
    var_dict['filters'] = filters
    var_dict['sh_pred'] = sh_pred
    var_dict['shape'] = pred_shape
    var_dict['reg'] = reg

def feat_extract_p(i):
    point_x, point_y = i
    X_np = np.frombuffer(var_dict['sh_pred']).reshape(var_dict['shape'])
    if var_dict['static_mask'][point_x, point_y] == 0:
        return

    feats = np.zeros(
                (len(var_dict['filters']), var_dict['dynamic_input'].shape[0], var_dict['dynamic_input'].shape[1]),
                dtype=np.float64,
            )
    accum = np.zeros(
        (var_dict['dynamic_input'].shape[0], var_dict['dynamic_input'].shape[1]), dtype=np.float64
    )
    static_accum = 0
    mask_sum = 0
    for filter_idx, filter in enumerate(var_dict['filters']):
        offset = int(np.floor(filter / 2))
        feat = var_dict['dynamic_input'][
            :,
            :,
            point_x - offset : point_x + offset + 1,
            point_y - offset : point_y + offset + 1,
        ].copy()

        sum_feat = np.zeros(
            (var_dict['dynamic_input'].shape[0], var_dict['dynamic_input'].shape[1]), dtype=np.float64
        )
        # TODO this approach doesn't seem right
        mask_feat = var_dict['static_mask'][
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
    # feats = feats.reshape(feats, 'k f c -> (k f c)

    pr = var_dict["reg"].predict(feats.reshape(1, -1))
    np.copyto(X_np[:,:,point_x, point_y], pr.reshape(6, 8))


def main():
    task = Task.init(project_name="t4c_eval/lr", task_name="eval pixel lr model")
    args = {
        "ds_name": "7days",
        "filters": [0],
        "processes": 8,
        "model": "ba23830a9a5a42fbad24d04d8853bf09",
    }
    task.connect(args)
    print ('Arguments: {}'.format(args))

    root_dir = Dataset.get(dataset_project="t4c", dataset_name=args["ds_name"]).get_local_copy()
    train_task = Task.get_task(task_id=args['model'])
    model_path = train_task.artifacts['model'].get_local_copy()

    reg = joblib.load(model_path)


    ds = T4CDataset(root_dir = root_dir, file_filter="**/validation/*8ch.h5")
    bs = 1
    ds._load_dataset()
    loader = DataLoader(ds, batch_size=bs, collate_fn=train_collate_fn)

    a = range(695)
    b = range(636)

    paramlist = list(itertools.product(a,b))

    filters = np.array(args["filters"])
    mses=[]
    for idx, sample in tqdm((enumerate(loader))):
        dynamic_input, static_ch, output_data = sample[0][0], sample[1][0], sample[2][0]
        pred = np.zeros((6, 8, 695, 636))
        static_mask = np.pad(static_ch[0], ((100, 100), (100, 100)), 'constant', constant_values=0)
        #road_pixels = np.transpose((static_mask>0).nonzero())
        static_mask = np.where(static_mask > 0, 1, 0)

        #output_data = np.pad(output_data, ((0,0),(0,0),(100, 100), (100, 100)), 'constant', constant_values=0)

        mean = np.mean(dynamic_input, 0)
        output_data = output_data - mean
        dynamic_input = dynamic_input - mean
        dynamic_input = np.pad(dynamic_input, ((0,0),(0,0), (100, 100), (100, 100)), 'constant', constant_values=0)

        X_shape = pred.shape
        #sh_pred = RawArray('d', pred.shape[0], pred.shape[1], pred.shape[2], pred.shape[3])
        sh_pred = RawArray('d', pred.shape[0]*pred.shape[1]*pred.shape[2]*pred.shape[3])
        sh_pred_np = np.frombuffer(sh_pred).reshape(pred.shape)
        np.copyto(sh_pred_np, pred)

        with Pool(processes=args["processes"], initializer=init_worker, initargs=(sh_pred, dynamic_input, static_mask, filters, pred.shape, reg)) as pool:
            pool.map(feat_extract_p, paramlist)



        mses.append(mean_squared_error(sh_pred_np[:,:,100:-100,100:-100].flatten(), output_data.flatten()))
        print (mses)
    print ("Overall MSE: ", sum(mses)/len(mses))
if __name__ == "__main__":
    main()
