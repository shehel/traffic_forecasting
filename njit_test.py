from numba import jit, njit, prange
import numpy as np
from tqdm import tqdm 

import pdb
dynamic_input = np.random.rand(12,8,495,436)
output_data = np.random.rand(6, 495, 436, 8)
static_mask = np.ones((495,436))
cfg = {'random_seed': 123, 'name': 'feat_7days', 'ds_name': '7days', 'input_path': '/home/shehel/ml/NeurIPS2021-traffic4cast/data/raw', 'cities': ['MOSCOW'], 'count': 7, 'filters': [3, 7, 21, 51, 71, 91], 'static_filter': None, 'patch_start': 128, 'patch_step': 128}
filters = np.array([3, 7, 21, 51, 71, 91])
static_ch = np.random.rand(9, 495, 436)

@njit(parallel=True)
def extract_feats(dynamic_input, output_data, static_mask,filters, static_ch):
    # Xs = []
    # ys = []
    Xs = np.zeros((128, 128, 576), np.float64)
    ys = np.zeros((128, 128, 48), np.float64)
    # Xs = np.zeros((100, 576), np.float32)
    # ys = np.zeros((100, 48), np.float32)
    patch_start = 128
    patch_step = 128
    for point_x in (
        (prange(patch_start, patch_start + patch_step))
    ):
        cand_mask = (
            static_mask[
                patch_start : patch_start + patch_step,
                patch_start : patch_start + patch_step,
            ]
            == 1
        )
        #cand_mask = cand_mask[0::4, 0::4]
        # pdb.set_trace()
        idx = patch_start-point_x
        for point_y in (
            prange(patch_start, patch_start + patch_step)
        ):
            idy = patch_start-point_y
            # find a pixel with road
            # for sample in range(100):
            #     point_x = rng.integers(cfg.patch_start, cfg.patch_start + cfg.patch_step)
            #     point_y = rng.integers(cfg.patch_start, cfg.patch_start + cfg.patch_step)
            #     while static_ch[0][point_x, point_y] < 5:
            #         point_x = rng.integers(cfg.patch_start, cfg.patch_start + cfg.patch_step)
            #         point_y = rng.integers(cfg.patch_start, cfg.patch_start + cfg.patch_step)

            # if static_mask[point_x, point_y] == 0:
            #    continue
            feats = np.zeros(
                (len(filters), dynamic_input.shape[0], dynamic_input.shape[-1]), dtype=np.float64
            )
            accum = np.zeros((dynamic_input.shape[0], dynamic_input.shape[-1]), dtype=np.float64)
            static_accum = 0
            mask_sum = 0
            for filter_idx, filter in enumerate(filters):
                offset = int(np.floor(filter / 2))
                feat = dynamic_input[
                        :,:,
                    point_x - offset : point_x + offset + 1,
                    point_y - offset : point_y + offset + 1,
                ].copy()
                
                sum_feat = np.zeros((dynamic_input.shape[0], dynamic_input.shape[-1]), dtype=np.float64)
                # TODO this approach doesn't seem right
                mask_feat = static_mask[
                    point_x - offset : point_x + offset + 1,
                    point_y - offset : point_y + offset + 1,
                ]
                mask_sum = np.sum(mask_feat)
                mask_sum = mask_sum - static_accum
                static_accum = static_accum + mask_sum
                
                # reducefeat = reduce(feat, 'f h w c -> f c', 'sum')/mask_feat
                if mask_sum != 0:
                    feat_ch = feat.reshape(12,8,-1).copy()
                    sum_feat = np.sum(feat_ch, axis=2)
                    
                    sum_feat = sum_feat - accum
                    accum = accum + sum_feat
                    sum_feat = sum_feat / mask_feat
                feats[filter_idx] = sum_feat
                # if np.count_nonzero(np.isnan(feats)) > 0:
                #    pdb.set_trace()
            # feats = feats.reshape(feats, 'k f c -> (k f c)')
            feats = feats.flatten()
            output_feats = output_data[:, point_x, point_y, :].flatten()
            Xs[idx, idy] = feats
            ys[idx, idy] = output_feats

        # Xs[sample] = feats
        # ys[sample] = output_feats

        # jXs.append(feats)
        # ys.append(output_feats)
    # pdb.set_trace()
    #Xs = Xs[cand_mask]
    #ys = ys[cand_mask]
    return Xs, ys 



extract_feats(dynamic_input, output_data, static_mask, filters, static_ch)
