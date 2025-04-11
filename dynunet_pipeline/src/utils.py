import csv
import os
import subprocess
import sys
import logging
import nibabel as nib
import numpy as np
import torch
import torch.distributed as dist

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import multi_gpu_flag
from monai.transforms import AsDiscrete


def setup_root_logger():
    logger = logging.root
    logger.propagate = False
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(fmt="%(asctime)s - %(levelname)s - %(message)s", datefmt=None)
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def save_stdout_in_logfile(logfile, multi_gpu_flag, local_rank):
    if multi_gpu_flag:
        if local_rank == 0:  # only one process should delete the file
            if os.path.isfile(logfile):
                os.remove(logfile)
            if not os.path.exists(os.path.dirname(logfile)):
                os.makedirs(os.path.dirname(logfile))
        dist.barrier()  # wait until all processes arrive here before creating the logfile in the next step
    else:
        if os.path.isfile(logfile):
            os.remove(logfile)
        if not os.path.exists(os.path.dirname(logfile)):
            os.makedirs(os.path.dirname(logfile))

    print(f"Save stdout in {logfile}...", flush=True)
    print(f"Save stderr in {logfile}...", file=sys.stderr, flush=True)
    
    tee_stdout = subprocess.Popen(f"(tee -a {logfile})", stdin=subprocess.PIPE, shell=True)
    tee_stderr = subprocess.Popen(f"(tee -a {logfile}) >&2", stdin=subprocess.PIPE, shell=True)

    # Cause tee's stdin to get a copy of our stdin/stdout (as well as that
    # of any child processes we spawn)
    os.dup2(tee_stdout.stdin.fileno(), sys.stdout.fileno())
    os.dup2(tee_stderr.stdin.fileno(), sys.stderr.fileno())

    # # The flush flag is needed to guarantee these lines are written before
    # # the two spawned /bin/ls processes emit any output
    # print("\nstdout", flush=True)
    # print("stderr", file=sys.stderr, flush=True)

    # # These child processes' stdin/stdout are
    # os.spawnve("P_WAIT", "/bin/ls", ["/bin/ls"], {})
    # os.execve("/bin/ls", ["/bin/ls"], os.environ)
    
        
def run_on_rank0_then_broadcast_object_list(fun, *args, **kwargs):
    """
    This is a decorator for a function that returns a list.
    It will make sure that the function only runs on process with rank 0 or when distributed processing is not enabled.
    It will then make the list available on processes with other rank.

    The function torch.distributed.broadcast_object_list cannot be used directly, because it requires the length of the list
    to be known on all processes prior to broadcasting.
    This is circumvented here by broadcasting the length of the list first, and then the list itself using torch.distributed.broadcast_object_list
    """
    def wrapper(*args, **kwargs):
        if not multi_gpu_flag or dist.get_rank() == 0:  # do the heavy lifting only on rank 0
            list_out = fun(*args, **kwargs)

            list_out_len = torch.tensor(len(list_out), device="cuda")  # get length of list on rank 0 process
        else:
            list_out_len = torch.tensor(0, device="cuda")  # initialize placeholder with same name on other processes

        if multi_gpu_flag:
            dist.broadcast(list_out_len, src=0)  # broadcast length of list to all processes

            if dist.get_rank() > 0:
                list_out = [0] * list_out_len  # initialize output list with correct length on other processes

        if multi_gpu_flag:
            dist.broadcast_object_list(list_out, src=0)  # broadcast list to all processes

        return list_out  # return output list on all processes

    return wrapper


def print_args(args):
    print("Arguments:", flush=True)
    for i, arg in enumerate(vars(args)):
        print(f"\t {i}. {arg} : ", getattr(args, arg), flush=True)


def load_label_support(label_support_path, device="cuda"):
    """
    This function loads the label support from the label_support_path. The label support is a tensor that contains the
    label support for each label.
    :param label_support_path: path to the label support
    :param device: device to load the label support to
    :return: label_support: an array of shape (num_labels, *data_shape) that contains the label support for each label
    """
    if label_support_path.endswith(".npz"):
        print(f"Loading compressed label support from {label_support_path}")
        label_support = torch.from_numpy(np.load(label_support_path)["label_support"]).to(device)
    else:
        label_support = torch.load(label_support_path, map_location=device)

    return label_support


def calculate_or_load_label_support(datalist_train, model_dir, channel_mapping):
    save_label_support_path = os.path.join(model_dir, "label_merging", "label_support.pt.npz")
    if not os.path.exists(os.path.dirname(save_label_support_path)):
        print("Calculate label support...")
        label_paths = [d['label'] for d in datalist_train]
        os.makedirs(os.path.dirname(save_label_support_path), exist_ok=True)
        label_support = get_label_support(label_paths=label_paths,
                                          label_to_channel_map=channel_mapping,
                                          save_path=save_label_support_path,
                                          compress=True)
    else:
        print("Load label support...")
        label_support = load_label_support(save_label_support_path)

    return label_support


def get_label_support(label_paths, label_to_channel_map, save_path=None, compress=True):  # equation 1
    """
    This function calculates the label support for each label in found in the label_paths data arrays, i.e. it counts
    the number of times each label appears in the label_paths data arrays at each voxel.
    This requires that labels were co-registered to the same space.
    :param label_paths: list of paths to the label files
    :param label_to_channel_map: a dictionary that maps each label to a consecutive channel number
    :param save_path: path to save the label support
    :
    :return: label_support: an array of shape (num_labels, *data_shape) that contains the label support for each label
    """

    for i, parc_path in enumerate(tqdm(label_paths)):
        parc_nii = nib.load(parc_path)
        parc_data = torch.tensor(parc_nii.get_fdata().astype(int), device="cuda")

        # check if all label files have the same affine
        if i == 0:
            reference_affine = parc_nii.affine
        else:
            assert np.allclose(parc_nii.affine, reference_affine, rtol=0.01), ("All label files must have the same "
                                                                               "affine to create label support.")

        if i == 0:
            nb_labels = len(label_to_channel_map)
            label_support = torch.zeros((nb_labels,) + parc_data.shape, device="cuda").float()

        for lab in torch.unique(parc_data):
            to_add_to_channel = parc_data == lab

            lab = lab.to('cpu').item()
            channel = label_to_channel_map[lab]
            label_support[channel] += to_add_to_channel

    if save_path:
        # # convert to int16 to save space (max value allowed: 32767)
        # assert (torch.max(label_support) < 32767)
        # label_support = label_support.to(torch.int16)

        if not compress:
            print(f"Uncompressed saving of label support to {save_path}")
            torch.save(label_support, save_path)
        else:
            # append ".npz" to the save_path
            if not save_path.endswith(".npz"):
                save_path = save_path + ".npz"
            print(f"Compressed saving of label support to {save_path}")

            np.savez_compressed(save_path, label_support=label_support.cpu().numpy())

    return label_support


def get_training_prior(label_support, channel_to_label_mapping=None):
    """
    This function calculates the training prior from the label support. The training prior is a volume that can be
    passed as an additional input channel to the CNN to help with patch-based training.
    At each voxel the training prior is the label that has the highest label support when normalized label-wise
    to sum to 1 across the volume. This makes sure small labels are not ignored in the training prior.
    :param label_support: an array of shape (num_labels, *data_shape) that contains the label support for each label
    :param channel_to_label_mapping: a dictionary that maps each channel to the original label
    :return: training_prior: an array of shape (*data_shape) that contains the training prior
    """
    label_support = label_support.float()  # convert to float for normalization
    for ch in range(label_support.shape[0]):
        channel_sum = torch.sum(label_support[ch])  # normalize the label support to sum to 1 label-wise across the volume
        if channel_sum > 0:  # avoid division by zero
            label_support[ch] = label_support[ch] / channel_sum

    # get the label with the highest normalized label support at each voxel
    training_prior = torch.argmax(label_support, dim=0).type(torch.int)

    # map each channel to the original label
    training_prior_mapped = torch.zeros(training_prior.shape, dtype=torch.int)
    for channel, label in channel_to_label_mapping.items():
        training_prior_mapped[training_prior == channel] = label

    return training_prior_mapped


@run_on_rank0_then_broadcast_object_list
def create_or_load_training_prior(prior_path, datalist_train, model_dir, channel_mapping):
    if os.path.isfile(prior_path):  # if prior_path is provided, leave it as is
        print(f"Use existing prior at {prior_path}")
    else:  # if prior is not provided at prior path, create it
        print("Creating prior...")

        label_support = calculate_or_load_label_support(datalist_train, model_dir, channel_mapping)

        # get the channel to label mapping
        channel_to_label_mapping = {v: k for k, v in channel_mapping.items()}

        # pick the correct label mapping
        final_label_mapping = channel_to_label_mapping
        prior = get_training_prior(label_support, final_label_mapping)
        prior = prior.numpy()
        reference_affine = nib.load(datalist_train[0]['label']).affine

        print(f"Saving prior to {prior_path}")
        nib.save(nib.Nifti1Image(prior, reference_affine), prior_path)

    # return this just to make sure this function is compatible with the decorator
    # run_on_rank0_then_broadcast_object_list
    return [1]


def output_transform_dice(output):
    preds = output['pred'][0]
    labels = output['label'][0]

    # discretize the predictions for dice calculation
    if labels.shape[1] == 1:  # the label is ordinal (only one channel)
        preds = torch.argmax(preds, dim=1, keepdim=True)
    else:
        # assume here that the label is binary-encoded  (but this will fail if the label is one-hot encoded)
        assert torch.max(labels) <= 1, f"Labels should be binary-encoded, but found max label value {torch.max(labels)}"
        preds = AsDiscrete(threshold=0.5)(preds)


    predictions_decollated = [preds[i] for i in range(len(preds))]
    targets_decollated = [labels[i] for i in range(len(labels))]

    return {"y_pred": predictions_decollated, "y": targets_decollated}


def output_transform_loss(output):
    preds = output['pred']
    labels = output['label']

    predictions_decollated = [preds[i] for i in range(len(preds))]
    targets_decollated = [labels[i] for i in range(len(labels))]

    criterion_kwargs = {}

    return {"y_pred": predictions_decollated,
            "y": targets_decollated,
            "criterion_kwargs": criterion_kwargs}


def append_to_dict(d, row):
    for key in d:
        if key in row:
            d[key].append(row[key])
        else:
            d[key].append(None)
    return d


def get_tensorboard_writer(summary_dir):
    layout = {
        "metrics": {
            "all": ["Multiline", ["train_loss", "val_loss", "val_mean_dice"]],
            "losses": ["Multiline", ["train_loss", "val_loss"]],
            "dice": ["Multiline", ["val_mean_dice"]],
        },
    }
    writer = SummaryWriter(summary_dir)
    writer.add_custom_scalars(layout)

    return writer


def create_and_sync_csv_logfile_with_log_dict(csv_logfile_path, log_dict):
    with open(csv_logfile_path, "w") as f:
        csv_writer = csv.DictWriter(f, fieldnames=log_dict.keys())
        csv_writer.writeheader()
        if log_dict["epoch"]:
            for i in range(len(log_dict["epoch"])):
                row = {k: log_dict[k][i] for k in log_dict.keys()}
                csv_writer.writerow(row)

