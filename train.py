import os
import pytorch_lightning as pl
import torch
import random

from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.strategies import DDPStrategy,FSDPStrategy
from pytorch_lightning.loggers import WandbLogger

import argparse
import numpy as np

import json

def set_seed(seed: int = 0):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


if __name__ == '__main__':

    set_seed(114514)

    parser = argparse.ArgumentParser(description='Train CODEC model using PyTorch Lightning')
    parser.add_argument('--num-nodes', type=int, default=1)
    parser.add_argument('--num-gpus', type=int, default=1)

    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-workers', type=int, default=8)

    parser.add_argument('--project-name', type=str,default="Xcodec2")
    parser.add_argument('--run-name', type=str, default=None)
    parser.add_argument('--seed', type=int, default=49)
    parser.add_argument("--dataset-config", type=str, default = "config/dataset/test_audio.json")
    parser.add_argument("--trainer-config", type=str, default = "config/trainer/muq_25hz.json")
    parser.add_argument("--precision", type=str, default = "32")
    parser.add_argument("--save-dir", type=str, default = "exp/")
    parser.add_argument("--ckpt-path", type=str, default = None)



    args = parser.parse_args()


    with open(args.dataset_config) as f:
        dataset_config: dict = json.load(f)

    with open(args.trainer_config) as f:
        trainer_config: dict = json.load(f)

    from xcodec2.dataset import create_dataloader_from_config

    train_dl = create_dataloader_from_config(
        dataset_config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sample_rate=trainer_config["sample_rate"],
        sample_size=trainer_config["sample_size"],
        audio_channels=trainer_config.get("audio_channels", 1)
    )


    lr_monitor = LearningRateMonitor(logging_interval='step')



    from datetime import datetime,timedelta
    timestamp = (datetime.now() + timedelta(hours=8)).strftime("%m%d_%H%M")
    save_dir = os.path.join(args.save_dir, f"{timestamp}")
    checkpoint_dir = os.path.join(save_dir, "checkpoints")

    os.makedirs(save_dir, exist_ok=True)

    from xcodec2.model import create_model_from_config
    from xcodec2.trainer import create_training_wrapper_from_config
    model = create_model_from_config(trainer_config["model"])
    training_wrapper = create_training_wrapper_from_config(trainer_config, model)

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir, 
        save_top_k=-1, 
        save_last=True,
        every_n_train_steps=20000
    )

    from xcodec2.trainer import DemoCallback
    demo_callback = DemoCallback(
        demo_every=trainer_config['training']["demo_every"], 
        sample_rate=trainer_config["sample_rate"],
    )


    callbacks = [checkpoint_callback, lr_monitor,demo_callback]

    wandb_logger = WandbLogger(
        project=args.project_name,
        name=args.run_name,
    )    

    # xcodec2-muq-25

    # log_dir_name = os.path.basename(os.path.normpath(cfg.log_dir))

    ckpt_path = None
    if args.ckpt_path is not None:
        last_ckpt = args.ckpt_path
        ckpt_path = last_ckpt
        print(f"Resuming from checkpoint: {ckpt_path}")
    else:
        print("No checkpoint found, starting training from scratch.")
    

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=args.num_gpus,
        num_nodes=args.num_nodes,
        strategy=DDPStrategy(find_unused_parameters=True),
        callbacks=callbacks,
        logger=wandb_logger,
        log_every_n_steps=1,
        precision=args.precision,
        default_root_dir=save_dir,


    )
    torch.backends.cudnn.benchmark = True  
    # training_wrapper.strict_loading = False
    # LightningModule.strict_loading = True

    # trainer.fit(training_wrapper, datamodule=datamodule,ckpt_path=ckpt_path )
    trainer.fit(training_wrapper, train_dl,ckpt_path=ckpt_path )

    print(f'Training ends, best score: {checkpoint_callback.best_model_score}, ckpt path: {checkpoint_callback.best_model_path}')

