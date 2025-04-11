# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import logging
import os
import shutil
import sys
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from datetime import datetime
from glob import glob
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from ignite.engine import Events
from ignite.metrics import Loss

from monai.handlers import CheckpointSaver, LrScheduleHandler, MeanDice, ValidationHandler
from monai.handlers.checkpoint_saver import Checkpoint
from monai.inferers import SimpleInferer, SlidingWindowInferer
from dice import DeepSupervisionLoss, DiceCELoss
from monai.utils import set_determinism
from torch.nn.parallel import DistributedDataParallel

from create_dataset import get_dataloader, get_datalist_from_prepared_paths
from create_network import get_network, get_kernels_strides

from evaluator import DynUNetEvaluator
from custom_handlers import print_avg_epoch_loss, log_validation_metrics, print_validation_metrics, print_loss, \
    collect_epoch_metrics
from transforms import determine_normalization_param_from_crop, get_channel_mapping
from trainer import DynUNetTrainer

from utils import setup_root_logger, save_stdout_in_logfile, print_args, create_or_load_training_prior, \
    output_transform_dice, output_transform_loss, get_tensorboard_writer, \
    create_and_sync_csv_logfile_with_log_dict, get_class_weights
from args import add_training_args

from config import debug, multi_gpu_flag


def train(args):
    local_rank = args.local_rank
    if multi_gpu_flag:
        dist.init_process_group(backend="nccl", init_method="env://")
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cuda")

    model_dir = os.path.join(args.models_root_dir, args.expr_name, f"fold{args.fold}")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_stdout_in_logfile(os.path.join(model_dir, "logs", f"train_{timestamp}.log"), multi_gpu_flag, args.local_rank)
    csv_logfile_path = os.path.join(model_dir, "logs", "train_log.csv")

    if args.local_rank == 0:
        print(f"Training on multiple GPUs: {multi_gpu_flag}")
        print_args(args)

    # load hyperparameters
    fold = args.fold
    resume_latest_checkpoint = args.resume_latest_checkpoint
    interval = args.interval
    learning_rate = args.learning_rate
    max_epochs = args.max_epochs
    amp_flag = args.amp
    lr_decay_flag = args.lr_decay
    patch_size = args.patch_size
    sw_batch_size = args.sw_batch_size
    batch_dice = args.batch_dice
    window_mode = args.window_mode
    eval_overlap = args.eval_overlap
    determinism_flag = args.determinism_flag
    determinism_seed = args.determinism_seed
    checkpoint = args.checkpoint
    modalities_path = args.modalities_path
    label_names_path = args.label_names_path
    splits_path = args.splits_path
    use_prior = args.use_prior

    for p in [modalities_path, label_names_path, splits_path,]:
        if p is not None and not os.path.exists(p):
            raise FileNotFoundError(f"Provided path {p} does not exist.")
    if checkpoint is not None and not os.path.exists(os.path.join(model_dir, checkpoint)):
        raise FileNotFoundError(f"Provided checkpoint {checkpoint} does not exist in {model_dir}.")

    if determinism_flag:
        set_determinism(seed=determinism_seed)
        if local_rank == 0:
            print("Using deterministic training.")

    # load input modality names
    modalities = json.load(open(modalities_path, "r"))

    args.modalities = modalities

    # where to save/load the prior
    prior_path = os.path.join(model_dir, "prior.nii.gz") if use_prior else None

    # where to store the training arguments (or load them if training is resumed)
    train_args_out_path = os.path.join(model_dir, "train_args_out.json")

    # load the train_args from the provided json file
    train_args_out_previous = None
    if checkpoint or resume_latest_checkpoint:
        if os.path.exists(train_args_out_path):
            with open(train_args_out_path, "r") as f:
                train_args_out_previous = json.load(f)

    datalist_train = get_datalist_from_prepared_paths("train", modalities, splits_path=splits_path, fold=fold, prior_path=prior_path)
    datalist_validation = get_datalist_from_prepared_paths("validation", modalities, splits_path=splits_path, fold=fold, prior_path=prior_path)

    # calculate kernel sizes and strides
    kernels, strides = get_kernels_strides(args.patch_size, args.spacing)

    # determine label mapping to consecutive integer labels
    channel_mapping, num_channels_out = get_channel_mapping(datalist_train,
                                                            key='label',
                                                            label_names_path=label_names_path)

    # add to args for saving in json file later on
    args.num_channels_out = num_channels_out
    args.channel_mapping = channel_mapping

    # parameters used by transforms
    transform_args = {
        "patch_size": args.patch_size,
        "strides": strides,
        "deep_supr_num": args.deep_supr_num,
        "pos_sample_num": args.pos_sample_num,
        "neg_sample_num": args.neg_sample_num,
        "use_prior": args.use_prior,
        "channel_mapping": channel_mapping,
    }

    # calculate prior input
    if use_prior:
        create_or_load_training_prior(prior_path, datalist_train, model_dir, channel_mapping)

    # parameters used by dataloaders
    dataloader_args = {
        "train_num_workers": args.train_num_workers,
        "val_num_workers": args.val_num_workers,
    }

    # define the cache_dir
    task_folder_name = os.path.basename(os.getcwd())
    cache_dir = os.path.join(Path.home(), ".cache", "MONAI", task_folder_name, args.expr_name, str(local_rank))

    # delete the cache dir if it exists
    if os.path.exists(cache_dir):
        print(f"Deleting cache dir {cache_dir}...")
        shutil.rmtree(cache_dir, ignore_errors=True)


    if not train_args_out_previous:
        # prep loader will perform pre-processing transforms including cropping and cache the dataset
        prep_loader = get_dataloader(datalist_train,
                                     transform_args,
                                     multi_gpu_flag,
                                     mode="prep",
                                     batch_size=args.data_loader_args["batch_size"],
                                     cache_dir=cache_dir,
                                     **dataloader_args)

        # based on cached preprocessed dataset, additional transform parameters such as "use_nonzero" can be calculated
        print("Crop images and determine normalization parameters...")
        transform_args["use_nonzero"] = determine_normalization_param_from_crop(prep_loader, key='image_0000', multi_gpu=multi_gpu_flag)
    else:
        print("Resume training: Loading use_nonzero transform parameter from previous training run...")
        transform_args["use_nonzero"] = train_args_out_previous["use_nonzero"]

    # subsequent data_loaders can continue from cached dataset
    val_loader = get_dataloader(datalist_validation,
                                transform_args,
                                multi_gpu_flag,
                                mode="validation",
                                batch_size=2,
                                cache_dir=cache_dir,
                                **dataloader_args)

    train_loader = get_dataloader(datalist_train,
                                  transform_args,
                                  multi_gpu_flag,
                                  mode="train",
                                  batch_size=args.data_loader_args["batch_size"],
                                  cache_dir=cache_dir,
                                  **dataloader_args)

    # produce the network (checkpoint is loaded later if provided)
    net = get_network(n_classes=num_channels_out,
                      n_in_channels=len(modalities),
                      kernels=kernels,
                      strides=strides,
                      deep_supr_num=args.deep_supr_num,
                      prior_path=prior_path,
                      )
    net = net.to(device)

    if multi_gpu_flag:
        net = DistributedDataParallel(module=net, device_ids=[device])

    optimizer = torch.optim.SGD(
        net.parameters(),
        lr=learning_rate,
        momentum=0.99,
        weight_decay=3e-5,
        nesterov=True,
    )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: (1 - epoch / max_epochs) ** 0.9)

    # define the loss function that uses the DiceCELoss separately for each channel (wrapper of the DiceCELoss)
    loss_each_head = DiceCELoss(include_background=False,
                                to_onehot_y=True,
                                softmax=True,
                                batch=batch_dice)

    print(f"Loss function: {loss_each_head}")

    loss = DeepSupervisionLoss(loss_each_head=loss_each_head,
                               num_deep_supervision_heads=args.deep_supr_num,)

    # produce evaluator
    evaluator = DynUNetEvaluator(
        device=device,
        val_data_loader=val_loader,
        network=net,
        inferer=SlidingWindowInferer(
            roi_size=patch_size,
            sw_batch_size=sw_batch_size,
            overlap=eval_overlap,
            mode=window_mode,
        ),
        decollate=False,
        postprocessing=None,
        key_val_metric={
            "val_mean_dice": MeanDice(
                include_background=False,
                output_transform=output_transform_dice,
            )
        },
        additional_metrics={
            "val_loss": Loss(loss_fn=loss,
                             output_transform=output_transform_loss)
        },
        val_handlers=None,
        amp=amp_flag,
    )

    # add a tensorboard logger
    if local_rank == 0:
        tb_writer = get_tensorboard_writer(f'commands/logs/runs/experiment_{timestamp}')
    else:
        tb_writer = None

    # produce trainer
    num_minibatches_per_epoch = 250 if not debug else 4
    if multi_gpu_flag:
        epoch_len = num_minibatches_per_epoch // dist.get_world_size()
    else:
        epoch_len = num_minibatches_per_epoch
    trainer = DynUNetTrainer(
        device=device,
        max_epochs=max_epochs,
        train_data_loader=train_loader,
        network=net,
        optimizer=optimizer,
        loss_function=loss,
        inferer=SimpleInferer(),
        postprocessing=None,
        key_train_metric=None,
        train_handlers=None,
        amp=amp_flag,
        epoch_length=epoch_len,
    )

    trainer.state.csv_logfile_dict = {"epoch": [],
                                      "iter": [],
                                      "train_loss": [],
                                      "val_loss": [],
                                      "val_mean_dice": [],
                                      "lr": []}
    trainer.state_dict_user_keys.append("csv_logfile_dict")

    # add evaluator handlers
    evaluator.add_event_handler(Events.EPOCH_COMPLETED(once=1), log_validation_metrics, csv_logfile_path, tb_writer, trainer, local_rank)
    evaluator.add_event_handler(Events.EPOCH_COMPLETED(every=interval), log_validation_metrics, csv_logfile_path, tb_writer, trainer, local_rank)

    # print the validation metrics
    evaluator.add_event_handler(Events.EPOCH_COMPLETED(every=1), print_validation_metrics, local_rank)

    # add train handlers

    # add learning rate scheduler
    if lr_decay_flag:
        lrScheduleHandler = LrScheduleHandler(lr_scheduler=scheduler, print_lr=True)

    # create directory to store the model
    os.makedirs(model_dir, exist_ok=True)
    # add evaluator handlers
    checkpoint_dict = {"net": net, "optimizer": optimizer, "scheduler": scheduler, "trainer": trainer}

    # checkpoint saver (for multi_gpu, the handler makes sure internally that only one process saves the checkpoint)
    checkpointSaver = CheckpointSaver(save_dir=model_dir, save_dict=checkpoint_dict, save_key_metric=True)
    checkpointSaver.attach(evaluator)
    # add another checkpoint saver to save the current model after each 5 epochs
    checkpointSaver2 = CheckpointSaver(save_dir=model_dir, save_dict=checkpoint_dict, save_interval=5, n_saved=1)
    checkpointSaver2.attach(trainer)

    # print losses for each iteration only during first epoch
    trainer.add_event_handler(Events.ITERATION_COMPLETED, print_loss, local_rank)

    # collect the losses for each epoch
    trainer.add_event_handler(Events.ITERATION_COMPLETED, collect_epoch_metrics, local_rank)

    # print average loss for each epoch
    trainer.add_event_handler(Events.EPOCH_COMPLETED, print_avg_epoch_loss, csv_logfile_path, tb_writer, scheduler, local_rank)

    # add this handler after the print_avg_epoch_loss handler to make sure the lr is logged before it is updated
    if lr_decay_flag:
        lrScheduleHandler.attach(trainer)

    ValidationHandler(validator=evaluator, interval=interval, epoch_level=True, exec_at_start=True).attach(trainer)

    if local_rank > 0:
        evaluator.logger.setLevel(logging.WARNING)
        trainer.logger.setLevel(logging.WARNING)

    # store the training arguments in a json file for use during inference
    with open(train_args_out_path, "w") as f:
        save_dict = vars(args)
        save_dict.update(transform_args)
        del save_dict["class_weights"]  # class weights are not JSON serializable and not needed for inference
        json.dump(save_dict, f, indent=2)

    # checkpoint handling
    if checkpoint:
        checkpoint_path = os.path.join(model_dir, checkpoint)
        if os.path.exists(checkpoint_path):
            Checkpoint.load_objects(to_load=checkpoint_dict, checkpoint=checkpoint_path)
            if local_rank == 0:
                print("Resuming from provided checkpoint: ", checkpoint_path)
        else:
            raise Exception(f"Provided checkpoint {checkpoint_path} not found.")
    elif resume_latest_checkpoint:
        checkpoints = glob(os.path.join(model_dir, "checkpoint_*=*.pt"))
        if len(checkpoints) == 0:
            if local_rank == 0:
                print(f"No checkpoints found in {model_dir}. Start training from beginning...")
        else:
            checkpoints.sort(key=lambda x: os.path.getmtime(x))  # sort by modification time
            checkpoint_latest = checkpoints[-1]  # pick the latest checkpoint
            if local_rank == 0:
                print("Resuming from latest checkpoint: ", checkpoint_latest)
            Checkpoint.load_objects(to_load=checkpoint_dict, checkpoint=checkpoint_latest)

            # if max_epochs is provided as argument, it should overwrite the value stored in the trainer
            trainer.state.max_epochs = max_epochs
            # same for learning rate (note, this is the initial learning rate, not the learning rate at the resumed
            # epoch number)
            trainer.optimizer.param_groups[0]['initial_lr'] = learning_rate
            scheduler.base_lrs = [learning_rate for i in scheduler.base_lrs]

    else:
        if local_rank == 0:
            print("No checkpoints provided. Start training from beginning...")

    if not trainer.state.max_epochs == trainer.state.epoch:
        if local_rank == 0:
            create_and_sync_csv_logfile_with_log_dict(csv_logfile_path, trainer.state.csv_logfile_dict)
        trainer.run()
    else:
        if local_rank == 0:
            print("Training from checkpoint already completed.")
    if multi_gpu_flag:
        dist.destroy_process_group()


if __name__ == "__main__":
    print("Training script started.")
    setup_root_logger()

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "-task_dir",
        "--task_dir",
        type=str,
        help="Path to the task directory, that contains the data, models, results and config directories",
        required=True,
    )

    parser.add_argument(
        "-args_file",
        "--args_file",
        type=str,
        help="Path to the json file containing the training arguments",
        required=True,
    )

    args = parser.parse_args()

    # change the current working directory to the task directory
    os.chdir(args.task_dir)

    args = add_training_args(args)

    train(args)

    if args.local_rank == 0:
        # after successul training, copy the model dir to the models directory
        src_dir = os.path.join(args.models_root_dir, args.expr_name, f"fold{args.fold}")
        dest = os.path.join("models", os.path.relpath(src_dir, args.models_root_dir))

        try:
            if os.path.exists(dest):
                print(f"FileExistsError: Model directory {dest} already exists. Please replace it manually before running inference.")
                raise FileExistsError
            else:
                print(f"Copying model directory {src_dir} to {dest}...")
                shutil.copytree(src_dir, dest)
        except FileExistsError:
            sys.exit(1)

