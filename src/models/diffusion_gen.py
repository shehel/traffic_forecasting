
import torch
from video_diffusion_pytorch import Unet3D, GaussianDiffusion, Trainer
from clearml import Task

task = Task.init(project_name='t4c_gen', task_name='vid_train_diffusion')
model = Unet3D(
    dim = 64,
    dim_mults = (1, 2, 4, 8)
).cuda()

diffusion = GaussianDiffusion(
    model,
    image_size = 64,
    num_frames = 6,
    timesteps = 1000,   # number of steps
    loss_type = 'l1'    # L1 or L2
).cuda()

trainer = Trainer(
    diffusion,
    '../NeurIPS2021-traffic4cast/data/raw',
    train_batch_size = 2,
    train_lr = 2e-4,
    train_num_steps = 700000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = False,
    save_and_sample_every = 500# turn on mixed precision
)

trainer.train()
