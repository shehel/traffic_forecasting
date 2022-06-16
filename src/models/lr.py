import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
import glob
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor

from clearml import Dataset
from clearml import Task

try:
    import joblib
except ImportError:
    from sklearn.externals import joblib

import pdb
from sklearn.ensemble import RandomForestRegressor


def main():
    task = Task.init(project_name="t4c_lr", task_name="train rf model mean on non-zero")
    args = {
        "ds_name": "7days",
        "train_path": "./data/processed/feat_7days_v8/**/training/*.h5",
        "val_path": "./data/processed/feat_7days_v8/**/validation/*.h5",
    }
    name = args["train_path"].split("/")[-4]
    task.connect(args)
    print("Arguments: {}".format(args))

    ds_X = []
    ds_y = []
    for file in glob.glob(args["train_path"]):
        f = h5py.File(file, "r")
        ds_X.append(f["X"][:])
        ds_y.append(f["y"][:])

    X = np.concatenate(ds_X, axis=0)
    y = np.concatenate(ds_y, axis=0)
    # reg = LinearRegression().fit(X, y)
    # reg = MLPRegressor(
    #     hidden_layer_sizes=(256), activation="relu", random_state=1, max_iter=2000
    # )
    # .fit(X, y)
    reg = RandomForestRegressor(max_depth=2, random_state=0).fit(X, y)
    joblib.dump(reg, "models/" + name + "MLPmodel.pkl", compress=True)
    task.upload_artifact(
        "model", artifact_object=os.path.join("models", name + "MLPmodel.pkl")
    )

    ds = []
    for file in glob.glob(args["val_path"]):
        f = h5py.File(file, "r")
        preds = reg.predict(f["X"][:])
        ds.append(mean_squared_error(f["y"][:], preds))
    print("MSE: ", sum(ds) / len(ds))


if __name__ == "__main__":
    main()
