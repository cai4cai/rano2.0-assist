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
import os
import shutil
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from datetime import datetime
from pathlib import Path

import torch
import torch.distributed as dist
from monai.inferers import SlidingWindowInferer
from torch.nn.parallel import DistributedDataParallel

from create_dataset import get_dataloader, get_datalist_from_prepared_paths
from create_network import get_network, get_kernels_strides
from inferrer import DynUNetInferrer

from utils import setup_root_logger, save_stdout_in_logfile, print_args
from args import add_inference_args
import argparse

from config import debug, multi_gpu_flag

def inference(args):
    local_rank = args.local_rank
    if multi_gpu_flag:
        dist.init_process_group(backend="nccl", init_method="env://")
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cuda")

    infer_out_dir = os.path.join(args.out_dir, args.expr_name, f"fold{args.fold}")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_stdout_in_logfile(os.path.join(infer_out_dir, "logs", f"infer_{timestamp}.log"), multi_gpu_flag, args.local_rank)
    if args.local_rank == 0:
        print(f"Inference on multiple GPUs: {multi_gpu_flag}")
        print(f"debug: {debug}")
        print_args(args)

    # load hyper parameters
    checkpoint = args.checkpoint
    model_dir = args.model_dir
    sw_batch_size = args.sw_batch_size
    window_mode = args.window_mode
    eval_overlap = args.eval_overlap
    amp = args.amp
    tta_val = args.tta_val
    local_rank = args.local_rank
    num_channels_out = args.num_channels_out
    channel_mapping = args.channel_mapping
    use_prior = args.use_prior
    inference_files_path = args.inference_files_path

    # load input modality names
    modalities = args.modalities

    prior_path = os.path.join(model_dir, "prior.nii.gz") if use_prior else None

    # define output directories
    preproc_out_dir = os.path.join(infer_out_dir, "preprocessed")
    os.makedirs(infer_out_dir, exist_ok=True)

    # get datalist
    datalist_testing = get_datalist_from_prepared_paths("test",
                                                        modalities,
                                                        infer_out_dir=infer_out_dir,
                                                        inference_files_path=inference_files_path,
                                                        prior_path=prior_path, )

    # calculate kernel sizes and strides
    kernels, strides = get_kernels_strides(args.patch_size, args.spacing)

    # parameters used by transforms
    channel_mapping = {float(k): v for k, v in channel_mapping.items()} if channel_mapping else None

    # registration template path needs to be set depending on whether the input images are brain extracted or not
    if args.input_is_bet:
        registration_template_path = args.registration_template_bet_path
    else:
        registration_template_path = args.registration_template_path

    # brain extraction needs to be deactivated if the input images are brain extracted
    if args.input_is_bet:
        bet = False
    else:
        bet = args.bet

    transform_args = {
        "patch_size": args.patch_size,
        "use_nonzero": args.use_nonzero,
        "reg": args.reg,
        "registration_template_path": registration_template_path,
        "preproc_out_dir": preproc_out_dir,
        "bet": bet,
        "use_prior": use_prior,
        "channel_mapping": channel_mapping,
    }

    # parameters used by dataloaders
    dataloader_args = {
        "val_num_workers": args.val_num_workers,
    }

    # define the cache_dir
    task_folder_name = os.path.basename(os.getcwd())
    cache_dir = os.path.join(Path.home(), ".cache", "MONAI", task_folder_name, args.expr_name, str(local_rank))

    # delete the cache dir if it exists
    if os.path.exists(cache_dir):
        print(f"Deleting cache dir {cache_dir}...")
        shutil.rmtree(cache_dir, ignore_errors=True)

    # get dataloader
    test_loader = get_dataloader(datalist_testing,
                                 transform_args,
                                 multi_gpu_flag,
                                 mode="test",
                                 batch_size=1,
                                 cache_dir=cache_dir,
                                 **dataloader_args,
                                 )

    net = get_network(
        n_classes=num_channels_out,
        n_in_channels=len(modalities),
        kernels=kernels,
        strides=strides,
        deep_supr_num=args.deep_supr_num,
        prior_path=prior_path,
        pretrain_path=model_dir,
        checkpoint=checkpoint)
    net = net.to(device)

    if multi_gpu_flag:
        net = DistributedDataParallel(module=net, device_ids=[device], find_unused_parameters=True)

    net.eval()

    inferrer = DynUNetInferrer(
        device=device,
        val_data_loader=test_loader,
        network=net,
        output_dir=infer_out_dir,
        num_classes=num_channels_out,
        inferer=SlidingWindowInferer(
            roi_size=args.patch_size,
            sw_batch_size=sw_batch_size,
            overlap=eval_overlap,
            mode=window_mode,
            device="cpu",
            progress=True,
        ),
        amp=amp,
        tta_val=tta_val,
        channel_mapping=channel_mapping,
    )

    inferrer.run()


if __name__ == "__main__":
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

    parser.add_argument(
        "-inference_files_path",
        "--inference_files_path",
        type=str,
        default=argparse.SUPPRESS,
        help="Path to the JSON file that defines the input files for inference and the save path for the predictions",
    )

    parser.add_argument(
        "-out_dir",
        "--out_dir",
        type=str,
        default=argparse.SUPPRESS,
        help="Path to the output directory",
    )

    parser.add_argument(
        "-registration_template_path",
        "--registration_template_path",
        type=str,
        default=argparse.SUPPRESS,
        help="Path to the registration template",
    )

    # in the following, use default=argparse.SUPPRESS to not set a default value, so that the default value from the
    # infer_args.json/args.py is used instead if the argument is not specified in the command line
    parser.add_argument('-reg', '--reg', dest='reg', action='store_true', default=argparse.SUPPRESS,
                        help="whether to perform brain extraction during preprocessing.")
    parser.add_argument('-no-reg', '--no-reg', dest='reg', action='store_false', default=argparse.SUPPRESS)

    parser.add_argument('-bet', '--bet', dest='bet', action='store_true', default=argparse.SUPPRESS,
                        help="whether to perform  brain extraction during preprocessing.")
    parser.add_argument('-no-bet', '--no-bet', dest='bet', action='store_false', default=argparse.SUPPRESS)

    parser.add_argument('-input_is_bet', '--input_is_bet', dest='input_is_bet', action='store_true',
                        default=argparse.SUPPRESS,
                        help="use if the input image has already been skull-stripped. In this case, "
                             "the registration is performed with respect to a skull-stripped MNI template, and "
                             "further brain extraction is deactivated (priority over the -bet argument).")
    parser.add_argument('-no-input_is_bet', '--no-input_is_bet', dest='input_is_bet', action='store_false',
                        default=argparse.SUPPRESS)

    args = parser.parse_args()

    # change the current working directory to the task directory
    os.chdir(args.task_dir)

    args = add_inference_args(args)

    inference(args)
