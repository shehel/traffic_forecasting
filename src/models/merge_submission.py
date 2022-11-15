#!/usr/bin/env python3
import zipfile

import numpy as np
from tqdm import trange
import h5py


import pdb

EXTENDED_CHALLENGE_CITIES = ["NEWYORK", "VIENNA"]


def write_data_to_h5(
    data: np.ndarray,
    filename: str,
    compression="gzip",
    compression_level=9,
    dtype="uint8",
):
    with h5py.File(filename, "w", libver="latest") as f:
        f.create_dataset(
            "array",
            shape=data.shape,
            data=data,
            chunks=(1, *data.shape[1:]),
            dtype=dtype,
            compression=compression,
            compression_opts=compression_level,
        )


def main():
    with zipfile.ZipFile("./submission/extended_submission.zip", "w") as z:
        for city in EXTENDED_CHALLENGE_CITIES:
            city_predictions = np.zeros((100, 6, 495, 436, 8))
            city_predictions[:, :, :, :, 0::2] = np.load(
                f"./submission/extended_model_vd17dvol_{city}_predictions.npy"
            )

            city_predictions[:, :, :, :, 1::2] = np.load(
                f"./submission/extended_model_vd17dsp_{city}_predictions.npy"
            )
            # city_predictions /= 7.0
            city_predictions = np.clip(city_predictions, a_min=0.0, a_max=255.0)
            write_data_to_h5(
                data=city_predictions,
                filename=f"./submission/extended_submission_{city}_predictions.h5",
                compression_level=6,
            )
            z.write(
                f"./submission/extended_submission_{city}_predictions.h5",
                arcname=f"{city}/{city}_test_spatiotemporal.h5",
            )


if __name__ == "__main__":
    main()
