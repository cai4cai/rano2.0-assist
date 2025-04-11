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

import os
from glob import glob
import json

import torch.distributed as dist
from monai.data import PersistentStagedDataset, DataLoader, load_decathlon_datalist, \
    partition_dataset, PersistentDataset

from transforms import get_task_transforms

from utils import run_on_rank0_then_broadcast_object_list

import config


@run_on_rank0_then_broadcast_object_list
def get_datalist(mode,
                 modalities,
                 fold=0,
                 splits_path=None,
                 test_files_dir=None,
                 infer_out_dir=None,
                 prior_path=None,
                 grtr_dir=None):

    # for testing, get the datalist by checking which files are available in test_files_dir directory
    if mode == "test":
        assert test_files_dir, f"test_files_dir must be provided, but is {test_files_dir}..."
        datalist = [{'image': p.replace("_0000.nii.gz", ".nii.gz")}
                    for p in sorted(glob(os.path.join(test_files_dir, "*")), key=str.lower)
                    if "_0000.nii.gz" in p]
        assert (len(datalist) > 0), f"No cases found in {test_files_dir} to run inference on..."

        # remove cases from datalist that are already in the inference folder
        files_in_infer_dir = os.listdir(infer_out_dir)
        reduced_datalist = []
        for d in datalist:
            if not os.path.basename(d['image']) in files_in_infer_dir:
                reduced_datalist.append(d)
            else:
                print(f"Found {d['image']} in {infer_out_dir}. Remove from datalist...")

        datalist = reduced_datalist
        assert (len(datalist) > 0), (f"No cases left to run inference on...\n"
                                     f"Make sure the test files directory exists and is not empty: {test_files_dir} ...\n" 
                                     f"Make sure that the inference folder is empty: {infer_out_dir} ...")

        # add the ground truth to the data dictionary if available
        if grtr_dir:
            for d in datalist:
                assumed_path = os.path.join(grtr_dir, os.path.basename(d["image"]))
                assert os.path.exists(assumed_path), f"Ground truth file not found: {assumed_path}... Set grtr_dir to None if no ground truth is available..."
                d["grtr"] = assumed_path

    # for prep, train, and validation, get the datalist from the dataset...json file
    else:
        if mode in ["prep", "train"]:
            list_key = f"train_fold{fold}"
        elif mode in ["validation"]:
            list_key = f"validation_fold{fold}"
        else:
            raise Exception(f"mode needs to be 'prep', 'train' or 'validation' but is '{mode}'...")

        datalist = json.load(open(splits_path, "r"))[list_key]

    def expand_paths_for_modalities(data_dict, modality_dict, prior_path):
        """
        Expands the "image" entry in data_dict by "image_0000" for first modality, "image_0001" for second modality etc..
        If prior is used, includes the prior as an additional modality...
        """
        for i, mod in modality_dict.items():
            mod_val_str = "{:04d}".format(int(i))
            data_dict["image_" + mod_val_str] = data_dict["image"].replace(".nii.gz", "_" + mod_val_str + ".nii.gz")

        # add the MNI prior to the data dictionary
        if prior_path:
            i = str(int(i) + 1)
            mod_val_str = "{:04d}".format(int(i))
            data_dict["image_" + mod_val_str] = prior_path

        data_dict.pop("image")
        return data_dict

    datalist = [expand_paths_for_modalities(d, modalities, prior_path) for d in datalist]

    if config.debug:
        max_datalist_lenth = 4
        print(f"Debugging active... Reduce {mode} datalist to {max_datalist_lenth} cases ...", flush=True)
        datalist = datalist[:4]
        for d in datalist:
            print("\t", d)

    return datalist


@run_on_rank0_then_broadcast_object_list
def get_datalist_from_prepared_paths(mode,
                                     modalities,
                                     fold=0,
                                     splits_path=None,
                                     inference_files_path=None,
                                     infer_out_dir=None,

                                     prior_path=None):


    def reorganise_dicts(data_dict, modality_dict, prior_path):
        """
        Converts the "images" entry in data_dict to new keys "image_0000" for first modality, "image_0001" for second modality etc..
        If prior is used, includes the prior as an additional modality...
        """
        for i, mod in modality_dict.items():
            mod_val_str = "{:04d}".format(int(i))
            if mod in data_dict["images"]:
                data_dict["image_" + mod_val_str] = data_dict["images"][mod]
            elif i in data_dict["images"]:
                data_dict["image_" + mod_val_str] = data_dict["images"][i]
            else:
                raise Exception(f"Modality {i} or {mod} not found in data_dict...")

        # add the MNI prior to the data dictionary
        if prior_path:
            i = str(int(i) + 1)
            mod_val_str = "{:04d}".format(int(i))
            data_dict["image_" + mod_val_str] = prior_path

        data_dict.pop("images")
        return data_dict

    if mode == "test":
        assert inference_files_path, f"inference_files_path must be provided, but is {inference_files_path}..."
        assert infer_out_dir, f"infer_out_dir must be provided, but is {infer_out_dir}..."

        datalist = json.load(open(inference_files_path, "r"))

        datalist = [reorganise_dicts(d, modalities, prior_path) for d in datalist]

        # modify the save path by the inference output directory
        for d in datalist:
            d["save_path"] = os.path.join(infer_out_dir, d["save_path"])

        # remove cases from datalist that are already in the inference folder
        if os.path.isdir(infer_out_dir):
            reduced_datalist = []
            for d in datalist:
                save_path = d["save_path"]
                if os.path.isfile(save_path):
                    print(f"Found {d['save_path']}. Remove from datalist...")
                else:
                    reduced_datalist.append(d)

            datalist = reduced_datalist

        assert (len(datalist) > 0), (f"No cases left to run inference on...\n"
                                     f"Make sure that the inference folder is empty: {infer_out_dir} ...")

    else:
        if mode in ["prep", "train"]:
            list_key = f"train"
        elif mode in ["validation"]:
            list_key = f"val"
        else:
            raise Exception(f"mode needs to be 'prep', 'train' or 'validation' but is '{mode}'...")

        datalist = json.load(open(splits_path, "r"))[str(fold)][list_key]
        datalist = [reorganise_dicts(d, modalities, prior_path) for d in datalist]

        if config.debug:
            max_datalist_lenth = 4
            print(f"Debugging active... Reduce {mode} datalist to {max_datalist_lenth} cases ...", flush=True)
            datalist = datalist[:4]
            for d in datalist:
                print("\t", d)

    return datalist


def get_dataloader(
        datalist,
        transform_args,
        multi_gpu=False,
        mode="train",
        batch_size=1,
        train_num_workers=None,
        val_num_workers=None,
        cache_dir="./cache_dir"
):

    modality_keys = sorted([k for k in datalist[0].keys() if "image_" in k], key=str.lower)

    # for multi-GPU, split datalist into multiple lists
    if multi_gpu:
        if mode in ["prep", "train", "test"] or (mode in ["validation"] and dist.get_world_size() <= len(datalist)):
            datalist = partition_dataset(
                data=datalist,
                shuffle=True if mode in ["prep", "train"] else False,
                num_partitions=dist.get_world_size(),
                even_divisible=True if mode in ["prep", "train"] else False,
            )[dist.get_rank()]
        elif mode in ["validation"] and dist.get_world_size() > len(datalist):
            print(f"Warning: Number of GPUs ({dist.get_world_size()}) is larger than the number of validation cases ({len(datalist)})."
                  f"As a solution, use all validation cases on each GPU.")
            datalist = [datalist] * dist.get_world_size()

    if mode == "prep":
        prep_load_tfm = get_task_transforms(mode, modality_keys, **transform_args)
        prep_ds = PersistentStagedDataset(
            new_transform=prep_load_tfm,
            old_transform=None,
            data=datalist,
            cache_dir=cache_dir,
        )

        data_loader = DataLoader(
            prep_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=train_num_workers,
            collate_fn=lambda x: x,
        )

    elif mode == "train":
        prep_load_tfm = get_task_transforms("prep", modality_keys, **transform_args)
        new_tfm = get_task_transforms(mode, modality_keys, **transform_args)

        train_ds = PersistentStagedDataset(
            new_transform=new_tfm,
            old_transform=prep_load_tfm,
            data=datalist,
            cache_dir=cache_dir,
        )
        data_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=train_num_workers,
        )

    elif mode in ["validation", "test"]:
        tfm = get_task_transforms(mode, modality_keys, **transform_args)

        # no caching for testing set
        cache_dir = cache_dir if mode == "validation" else None

        val_ds = PersistentDataset(
            transform=tfm,
            data=datalist,
            cache_dir=cache_dir,
        )

        data_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=val_num_workers, #args.val_num_workers,  # because of the brain-extraction transform, multiprocessing cannot be used, since subprocesses cannot initialize their own CUDA processes. #RuntimeError: Cannot re-initialize CUDA in forked subprocess. To use CUDA with multiprocessing, you must use the 'spawn' start method TODO: try the 'spawn' method: mp.set_start_method("spawn")
        )

    else:
        raise ValueError(f"mode should be train, validation or test.")

    return data_loader
