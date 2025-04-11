import csv

import numpy as np

from monai.handlers import StatsHandler
from utils import append_to_dict


def print_validation_metrics(engine, local_rank):
    if local_rank == 0:
        print("validation metrics: ", engine.state.metrics)

def log_validation_metrics(engine, csv_logfile_path, tensorboard_writer, trainer, local_rank):
    if local_rank == 0:
        for key, value in engine.state.metrics.items():
            tensorboard_writer.add_scalar(key, value, trainer.state.epoch)
        # write to logfile and logdict
        row = {"epoch": trainer.state.epoch,
               "iter": trainer.state.iteration,
               "val_loss": engine.state.metrics["val_loss"],
               "val_mean_dice": engine.state.metrics["val_mean_dice"], }
        trainer.state.csv_logfile_dict = append_to_dict(trainer.state.csv_logfile_dict, row)
        with open(csv_logfile_path, "a") as f:
            csv_writer = csv.DictWriter(f, fieldnames=trainer.state.csv_logfile_dict.keys())
            csv_writer.writerow(row)


def collect_epoch_metrics(engine, local_rank):
    if local_rank == 0:
        if not hasattr(engine.state, "losses"):
            engine.state.losses = []
        # clear the losses list at the beginning of each epoch
        if (engine.state.iteration-1) % engine.state.epoch_length == 0:
            engine.state.losses = []
        engine.state.losses.append(engine.state.output['loss'].item())


def print_loss(engine, local_rank):
    if local_rank == 0:
        if engine.state.epoch == 1:
            StatsHandler(
                name="StatsHandler",
                iteration_log=True,
                epoch_log=False,
                tag_name="train_loss",
                output_transform=lambda x: x['loss'].item()
            ).iteration_completed(engine)

def print_avg_epoch_loss(engine, csv_logfile_path, tensorboard_writer, scheduler, local_rank):
    if local_rank == 0:
        avg_train_loss = np.mean(engine.state.losses)
        print("average train epoch loss: ", avg_train_loss)
        tensorboard_writer.add_scalar('train_loss', avg_train_loss, engine.state.epoch)

        # write to logfile and logdict
        row = {"epoch": engine.state.epoch,
               "iter": engine.state.iteration,
               "train_loss": avg_train_loss,
               "lr": scheduler._last_lr[0]}
        engine.state.csv_logfile_dict = append_to_dict(engine.state.csv_logfile_dict, row)
        with open(csv_logfile_path, "a") as f:
            csv_writer = csv.DictWriter(f, fieldnames=engine.state.csv_logfile_dict.keys())
            csv_writer.writerow(row)