#  Copyright 2021 Institute of Advanced Research in Artificial Intelligence (IARAI) GmbH.
#  IARAI licenses this file to You under the Apache License, Version 2.0
#  (the "License"); you may not use this file except in compliance with
#  the License. You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import argparse
import binascii
import logging
import os
import sys
import random
import tqdm
from pathlib import Path
from typing import Optional
import pdb

import numpy as np
import torch
import torch.nn.functional as F  # noqa
import torch.optim as optim
from ignite.contrib.handlers import TensorboardLogger
from ignite.contrib.handlers.tensorboard_logger import GradsHistHandler, GradsScalarHandler, WeightsScalarHandler
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

from ignite.contrib.engines.common import save_best_model_by_val_score


from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler

from src.models.checkpointing import load_torch_model_from_checkpoint
from src.models.checkpointing import save_torch_model_to_checkpoint
from src.data.dataset import T4CDataset, train_collate_fn

import pdb

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

from src.models.model import Model
from src.models.ar import AR
from einops import rearrange

from src.common.utils import t4c_apply_basic_logging_config

from clearml import Task
from clearml import Dataset

import ignite.distributed as idist

def reset_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def run_model(
        rank: int,
        task: Task,
        root_dir: str,
        cfg: DictConfig,
    ):  # noqa

    t4c_apply_basic_logging_config()
    logging.info(
        idist.get_rank(),
        ": ",
        "backend=",
        idist.backend(),
        "- world size",
        idist.get_world_size(),
    )

    device = idist.device()
    logging.info(f"Using device {device}")
    # Fetch dataset if it exists in clearml server, otherwise
    # assume config contains path and try local dir

    #
    model = instantiate(cfg.model, dataset={"root_dir":root_dir})
    # pytorch_total_params = sum(p.numel() for p in model.network.parameters() if p.requires_grad)
    # print (pytorch_total_params)
    # r
    #eturn
    assert len(model.t_dataset) > 0

    # TODO Model restricted to unet.
    if cfg.train.resume_checkpoint is not None:
        logging.info("Reload checkpoint %s", cfg.train.resume_checkpoint)
        #load_torch_model_from_checkpoint(checkpoint=cfg.train.resume_checkpoint, model=model.network)
        to_load = {"train_model": model.network}
        checkpoint = torch.load(cfg.train.resume_checkpoint)
        #Checkpoint.load_objects(to_load, checkpoint, map_location=torch.device('cuda'))
        model.network.load_state_dict(checkpoint['train_model'].module.state_dict())
        # TODO
        #optimizer.load_state_dict(checkpoint['optimizer'].state_dict())

            # https://stackoverflow.com/questions/59249563/runtimeerror-module-must-have-its-parameters-and-buffers-on-device-cuda1-devi


    model.network = idist.auto_model(model.network)

    optimizer = idist.auto_optim(optim.Adam(model.network.parameters(), lr = cfg.train.optimizer.lr))
    logging.info("Model and optimizer instantiated.")

    logging.info("dataset has size %s", len(model.t_dataset))
    if cfg.model.valset == True:
        logging.info("val dataset has size %s", len(model.v_dataset))

    # Train / Dev / Test set splits
    logging.info("train/dev split")
    full_dataset_size = len(model.t_dataset)

    if cfg.model.valset == True:
        full_dataset_size_val = len(model.v_dataset)
        effective_val_dataset_size = full_dataset_size_val

    effective_dataset_size = full_dataset_size

    if model.t_dataset.limit is not None:
        effective_dataset_size = min(full_dataset_size, model.t_dataset.limit)

    try:
        if model.v_dataset.limit is not None:
            effective_val_dataset_size = min(full_dataset_size_val, model.v_dataset.limit)
    except:
        pass

    indices = list(range(full_dataset_size))

    if cfg.model.valset == True:
        val_indices = list(range(full_dataset_size_val))
        np.random.shuffle(val_indices)

    np.random.shuffle(indices)

    assert np.isclose(cfg.train.train_fraction + cfg.train.val_fraction, 1.0)
    num_train_items = max(int(np.floor(cfg.train.train_fraction * effective_dataset_size)),
                          cfg.train.dataloader.batch_size)

    if cfg.model.valset == True:
        num_val_items = max(int(np.floor(effective_val_dataset_size)), cfg.train.dataloader.batch_size)
        train_indices, dev_indices = indices[:num_train_items], val_indices[:num_val_items]
    else:
        num_val_items = max(int(np.floor(cfg.train.val_fraction * effective_dataset_size)), cfg.train.dataloader.batch_size)
        train_indices, dev_indices = indices[:num_train_items], indices[num_train_items : num_train_items + num_val_items]
    logging.info(
        "Taking %s from dataset of length %s, splitting into %s train items, %s train eval items and %s val items",
        effective_dataset_size,
        full_dataset_size,
        num_train_items,
        min([cfg.train.dataloader.train_eval, num_train_items]),
        num_val_items,
    )

    train_sampler = SubsetRandomSampler(train_indices)
    dev_sampler = SubsetRandomSampler(dev_indices)

    train_loader = idist.auto_dataloader(model.t_dataset, batch_size=cfg.train.dataloader.batch_size,
                                         num_workers=cfg.train.dataloader.num_workers,
                              sampler=train_sampler, collate_fn=train_collate_fn, pin_memory=False, drop_last=True)


    # Needed as train evaluator uses the entire training set for
    # train loss which is prohibitively long. If the number of samplers
    # specified for traing evaluation is more than number of samples in
    # training set, default to train loader for loss calculation
    if cfg.train.dataloader.train_eval and cfg.train.dataloader.train_eval < num_train_items:
        train_eval_sampler = SubsetRandomSampler(indices[:cfg.train.dataloader.train_eval])
        train_eval_loader = idist.auto_dataloader(model.t_dataset, batch_size=cfg.train.dataloader.batch_size,
                                                  num_workers=cfg.train.dataloader.num_workers,
                                                  sampler=train_eval_sampler, collate_fn=train_collate_fn, pin_memory=False)
    else:
        train_eval_loader = train_loader

    if cfg.model.valset:
        val_loader = idist.auto_dataloader(model.v_dataset, batch_size=cfg.train.dataloader.batch_size,
                                           num_workers=cfg.train.dataloader.num_workers, sampler=dev_sampler, collate_fn=train_collate_fn, pin_memory=False)
    else:
        val_loader = idist.auto_dataloader(model.t_dataset, batch_size=cfg.train.dataloader.batch_size,
                                           num_workers=cfg.train.dataloader.num_workers, sampler=dev_sampler, collate_fn =train_collate_fn, pin_memory=False)

        # Loss
    loss = F.mse_loss

    logging.info("Going to run train_model.")
    # logging.info(system_status())

    checkpoints_dir = os.path.join(os.path.curdir, "checkpoints")
    
    t_model = AR(48, 48)

    # load model to predict t+1
    train_task = Task.get_task(task_id="0008d142e02545b2b898206cb0be9cba")
    model_path = train_task.artifacts['model_checkpoint'].get_local_copy()#"/data/t5chx.pt"
    #model_state_dict = torch.load(model_path)
    model_state_dict = torch.load(model_path+'/'+os.listdir(model_path)[-1])#,map_location=torch.device('cpu'))
    t_model.load_state_dict(model_state_dict['train_model'], strict=False)
    t_model = t_model.to('cuda')
    t_model.eval()

    
    train_ignite(device, loss, optimizer, train_loader, train_eval_loader, val_loader, model.network, checkpoints_dir, cfg, t_model)
    logging.info("End training of train_model %s on %s for %s epochs", model.network, device, cfg.train.epochs)

    # Upload checkpoint folder con
    task.upload_artifact(name='model_checkpoint', artifact_object=checkpoints_dir)

def train_ignite(device, loss, optimizer, train_loader, train_eval_loader, val_loader,
                 train_model, checkpoints_dir, cfg, t_model):
    # Validator
    is_static = cfg.train.transform.static
    if is_static:
        dynamic_channels = cfg.model.network.in_channels - 9
    else:
        dynamic_channels = cfg.model.network.in_channels

    out_channels = cfg.model.network.out_channels

    in_h = cfg.train.transform.in_h
    in_w = cfg.train.transform.in_w
    epochs = cfg.train.epochs
    amp_mode = cfg.train.amp_mode
    scaler = cfg.train.scaler
    pad_tuple = tuple(cfg.train.transform.pad_tuple)

    # dynamic_input_mean = np.load('data/processed/dynamic_input_mean.npy')
    # dynamic_input_std = np.load('data/processed/dynamic_input_std.npy')

    # dynamic_input_mean = torch.from_numpy(dynamic_input_mean)[None, None, :, None, None].float().cuda()
    # dynamic_input_std = torch.from_numpy(dynamic_input_std)[None, None, :, None, None].float().cuda()

    def prepare_batch_fn(batch, device, non_blocking):
        dynamic, static, target, dates  = batch
        dynamic = convert_tensor(dynamic, device, non_blocking)
        target = convert_tensor(target, device, non_blocking)
        dates = convert_tensor(dates, device, non_blocking)
        #dynamic = (dynamic - dynamic_input_mean) / dynamic_input_std
        pred = t_model.forward(rearrange(dynamic, 'b t c h w -> (b h w) (t c)'))
        # rearrange pred back to b t c h w where h = 128 w = 128 c = 8 and t = 6
        pred = rearrange(pred, '(b h w) (t c) -> b t c h w', h=128, w=128, c=8, t=6)
        target = target - pred
        #pdb.set_trace()
        #target = dynamic[:, 11:12, 0:1, :, :] - target
        dynamic = dynamic.reshape(-1, dynamic_channels, in_h, in_w)
        target = target.reshape(-1, out_channels, in_h, in_w)
        target = F.pad(target, pad=pad_tuple)
        static = convert_tensor(static, device, non_blocking)
        if is_static:
            input_batch = torch.cat([dynamic, static], dim=1)
        else:
            input_batch = dynamic
        input_batch = F.pad(input_batch, pad=pad_tuple)

        return [input_batch,dates], target

    validation_evaluator = create_supervised_evaluator(train_model, metrics={"val_loss": Loss(loss), "neg_val_loss": Loss(loss)*-1}, device=device, amp_mode=amp_mode,
                                                       prepare_batch=prepare_batch_fn)
    train_evaluator = create_supervised_evaluator(train_model, metrics={"loss": Loss(loss)}, device=device, amp_mode=amp_mode,
                                                  prepare_batch=prepare_batch_fn)

    # Trainer
    trainer = create_supervised_trainer(train_model, optimizer, loss, device=device,
                                        amp_mode=amp_mode, scaler=scaler,
                                        prepare_batch = prepare_batch_fn)
    run_id = binascii.hexlify(os.urandom(15)).decode("utf-8")
    artifacts_path = os.path.join(os.path.curdir, f"artifacts/{run_id}")
    logs_path = os.path.join(artifacts_path, "tensorboard")
    RunningAverage(output_transform=lambda x: x).attach(trainer, name="loss")
    pbar = ProgressBar(persist=True, bar_format="{desc}[{n_fmt}/{total_fmt}] {percentage:3.0f}%|{bar}{postfix} [{elapsed}<{remaining}]{rate_fmt}")
    pbar.attach(trainer, metric_names="all")

    @trainer.on(Events.EPOCH_STARTED)  # noqa
    def log_epoch_start(engine: Engine):
        logging.info(f"Started epoch {engine.state.epoch}")
        # logging.info(system_status()

    @trainer.on(Events.EPOCH_COMPLETED)  # noqa
    #@trainer.on(Events.EPOCH_STARTED)  # noqa
    def log_epoch_summary(engine: Engine):
        # Training
        train_evaluator.run(train_eval_loader)
        metrics = train_evaluator.state.metrics
        train_avg_loss = metrics["loss"]

        # Validation
        validation_evaluator.run(val_loader)
        metrics = validation_evaluator.state.metrics
        val_avg_loss = metrics["val_loss"]

        msg = f"Epoch summary for epoch {engine.state.epoch}: loss: {train_avg_loss:.4f}, val_loss: {val_avg_loss:.4f}\n"
        pbar.log_message(msg)
        logging.info(msg)
        # logging.info(system_status())

    tb_logger = TensorboardLogger(log_dir=logs_path)
    #tb_logger.attach(trainer, log_handler=GradsScalarHandler(train_model), event_name=Events.ITERATION_COMPLETED(every=200))
    #tb_logger.attach(trainer, log_handler=WeightsScalarHandler(train_model), event_name=Events.ITERATION_COMPLETED(every=500))
    tb_logger.attach_output_handler(
       train_evaluator, event_name=Events.EPOCH_COMPLETED, tag="train", metric_names=["loss"], global_step_transform=global_step_from_engine(trainer)
    )
    tb_logger.attach_output_handler(
        validation_evaluator,
        event_name=Events.EPOCH_COMPLETED,
        tag="validation",
        metric_names=["val_loss"],
        global_step_transform=global_step_from_engine(trainer),
    )
    to_save = {"train_model": train_model, "optimizer": optimizer}
    checkpoint_handler = Checkpoint(to_save, DiskSaver(checkpoints_dir, create_dir=True, require_empty=False), n_saved=2)
    checkpoint_handler = save_best_model_by_val_score(checkpoints_dir, validation_evaluator, to_save,
                                                      metric_name="neg_val_loss", n_saved=1, trainer=trainer)
    validation_evaluator.add_event_handler(Events.COMPLETED, checkpoint_handler)
    # Run Training
    logging.info("Start training of train_model %s on %s for %s epochs", train_model, device, epochs)
    logging.info(f"tensorboard --logdir={artifacts_path}")

    # To get better trace of Nan events
    #torch.autograd.set_detect_anomaly(True)

    trainer.run(train_loader, max_epochs=epochs)

    pbar.close()

@hydra.main(config_path=str("../../config"), config_name="default")
def main(cfg: DictConfig):
    spawn_kwargs = dict()
    spawn_kwargs["nproc_per_node"] = cfg.train.n_process
    reset_seeds(cfg.train.random_seed)
    #sd = Dataset.get(dataset_project="t4c", dataset_name="default").get_mutable_local_copy("data/raw")
    task = Task.init(project_name='t4c', task_name='train_model')

    try:
        root_dir = Dataset.get(dataset_project="t4c", dataset_name=cfg.model.dataset.root_dir).get_local_copy()
    except:
        root_dir = cfg.model.dataset.root_dir
        logging.info(f"Could not find dataset in clearml server. Using {root_dir} as path.")

    with idist.Parallel(backend=cfg.train.parallel_backend, **spawn_kwargs) as parallel:
        parallel.run(run_model, task, root_dir, cfg)



    # for competition in competitions:
    #     additional_args = {}
    #     if geometric:
    #         processed_dir = str(Path(data_raw_path).parent)
    #         additional_args = {
    #             "gt": GraphTransformer(processed_dir=processed_dir, raw_dir=data_raw_path, batch_size=1),
    #             "processed_dir": processed_dir,
    #         }
    #     submission = package_submission(
    #         data_raw_path=data_raw_path,
    #         competition=competition,
    #         model=model,
    #         model_str=model_str,
    #         device=device,
    #         h5_compression_params={"compression_level": None},
    #         submission_output_dir=Path(args.submission_output_dir if args.submission_output_dir is not None else "."),
    #         # batch mode for submission
    #         batch_size=1 if geometric else args.batch_size,
    #         num_tests_per_file=args.num_tests_per_file,
    #         **additional_args,
    #     )
    #     ground_truth_dir = args.ground_truth_dir
    #     if ground_truth_dir is not None:
    #         ground_truth_dir = Path(ground_truth_dir)
    #         scorecomp.score_participant(
    #             ground_truth_archive=str(ground_truth_dir / f"ground_truth_{competition}.zip"),
    #             input_archive=str(submission),
    #             batch_size=args.batch_size_scoring,
    #         )
    #     else:
    #         scorecomp.verify_submission(input_archive=submission, competition=competition, batch_size=args.batch_size_scoring)


if __name__ == "__main__":
    main()
