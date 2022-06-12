import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
import glob
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

from clearml import Dataset
from clearml import Task

try:
    import joblib
except ImportError:
    from sklearn.externals import joblib

def main():
    task = Task.init(project_name="t4c_lr", task_name="train LR model")
    args = {
        "ds_name": "7days",
        "train_path": "./data/processed/feat_7days_v5_2/**/training/*.h5",
        "val_path": "./data/processed/feat_7days_v5_2/**/validation/*.h5",
    }
    task.connect(args)
    print ('Arguments: {}'.format(args))

    ds_X = []
    ds_y = []
    for file in (glob.glob(args["train_path"])):
        f = h5py.File(file, 'r')
        ds_X.append(f['X'][:])
        ds_y.append(f['y'][:])

    X = np.concatenate(ds_X, axis=0)
    y = np.concatenate(ds_y, axis=0)
    reg = LinearRegression().fit(X, y)

    joblib.dump(reg, 'models/LRmodel.pkl', compress=True)
    task.upload_artifact(
    'model',
    artifact_object=os.path.join(
        'models',
        'LRmodel.pkl'
    )
    )

    ds = []
    for file in (glob.glob(args["val_path"])):
        f = h5py.File(file, 'r')
        preds = reg.predict(f['X'][:])
        ds.append(mean_squared_error(f['y'][:], preds))
    print ("MSE: ", sum(ds)/len(ds))

if __name__ == "__main__":
    main()
