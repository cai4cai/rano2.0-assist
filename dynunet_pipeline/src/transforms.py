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
import csv
import json
import shutil
import sys

from tqdm import tqdm

import numpy as np
import torch
import torch.distributed as dist
import nibabel as nib

from utils import run_on_rank0_then_broadcast_object_list

from monai.transforms import (
    CastToTyped,
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    LoadImaged,
    NormalizeIntensityd,
    RandFlipd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandScaleIntensityd,
    EnsureTyped,
    CenterSpatialCropd,
    ConcatItemsd,
    DeleteItemsd,
    ANTsAffineRegistrationd,
    BrainExtractiond,
    Identityd,
    MapLabelValued,
    BinarizeLabeld,
    ToTensord,
    SampleForegroundLocationsd,
    RandScaleIntensityFixedMeand,
    RandAdjustContrastd,
    RandSimulateLowResolutiond,
    AppendDownsampledd,
    RandAffined, SpatialPadd
)


def get_deep_supr_label_shapes(deep_supr_num, patch_size, strides):
    assert len(strides) >= deep_supr_num + 2, "The number of strides must be at least deep_supr_num + 2."
    supr_label_shapes = [patch_size]
    for i in range(deep_supr_num):
        last_shape = supr_label_shapes[-1]
        curr_strides = strides[
            i + 1]  # ignore first set of strides, since they apply to downsampling prior to the first level
        downsampled_shape = [int(np.round(last / curr)) for last, curr in zip(last_shape, curr_strides)]
        supr_label_shapes.append(downsampled_shape)
    return supr_label_shapes


def get_task_transforms(mode,
                        modality_keys,
                        patch_size,
                        strides=None,
                        deep_supr_num=None,
                        pos_sample_num=None,
                        neg_sample_num=None,
                        use_nonzero=False,
                        reg=None,
                        registration_template_path=None,
                        preproc_out_dir=None,
                        bet=False,
                        use_prior=False,
                        channel_mapping=None,
                        ):

    label_keys = ["label"]
    all_keys = modality_keys + label_keys

    # prior key
    prior_key = [modality_keys[-1]] if use_prior else []

    # exclude the prior from the intensity transforms
    mod_inty_keys = modality_keys[:-1] if use_prior else modality_keys

    load_image = LoadImaged(keys=all_keys, image_only=True)
    ensure_channel_first = EnsureChannelFirstd(keys=all_keys)
    crop_transform = CropForegroundd(keys=all_keys, source_key=mod_inty_keys[0], start_coord_key=None, end_coord_key=None, allow_smaller=True)

    prep_load_tfm = Compose([load_image, ensure_channel_first, crop_transform], unpack_items=True)

    if mode == "prep":
        return prep_load_tfm

    elif mode in ["train", "validation"]:
        """ -2 Map the labels to consecutive integers or merged labels"""
        if channel_mapping:
            map_labels = MapLabelValued(keys=label_keys+prior_key,
                                        orig_labels=list(channel_mapping.keys()),
                                        target_labels=list(channel_mapping.values()))

        elif not channel_mapping and not merged_label_mapping and not binary_label_mapping:
            map_labels = Identityd(keys=label_keys+prior_key)
        else:
            raise ValueError("Inconsistent label mapping configuration.")

        """ -1. add normalization to list of transforms based on the median of all crop size factors """
        """ Also normalize the prior image, if it is used"""
        norm_transform = NormalizeIntensityd(keys=modality_keys, nonzero=use_nonzero)

        """ nnU-Net first calculates a larger patch-size, then samples the image across the image border (self.need_to_pad), so that the final
        smaller patch size (which is cropped from the center of the larger patch) will still cover the image borders.
        Here, instead, we directly achieve this by specifying the translate_range="cover" of the random affine. Internally,
        the translate_range is chosen such that after augmentation patches can have their center point as close as half the
        patch size from the image border"""

        """0. Oversample the foreground. This samples the foreground at a desired number of locations. Although the sampling is
        random, the transform doesn't inherit from RandomizableTransform, so that the result will automatically be cached for
        training. The sampled locations will be saved in the metadata and can be used by subsequent transforms."""

        # transform that creates a list of foreground locations (doesn't change the image/label data)
        sample_foreground_locations = SampleForegroundLocationsd(label_keys=label_keys, num_samples=10000)

        """## 1. Random affine transformation

        The following affine transformation is defined to output a (300, 300, 50) image patch.
        The patch location is randomly chosen in a range of (-40, 40), (-40, 40), (-2, 2) in x, y, and z axes respectively.
        The translation is relative to the image centre.
        The 3D rotation angle is randomly chosen from (-45, 45) degrees around the z axis, and 5 degrees around x and y axes.
        The random scaling factor is randomly chosen from (1.0 - 0.15, 1.0 + 0.15) along each axis.
        """

        ### Hyperparameters from experiment planning
        rotate_range = (30 / 360 * 2 * np.pi, 30 / 360 * 2 * np.pi, 30 / 360 * 2 * np.pi)
        scale_range = ((-0.3, 0.4), (-0.3, 0.4), (-0.3, 0.4))  # 1 is added to these values, so the scale factor will be (0.7, 1.4)

        rand_affine = RandAffined(
            keys=all_keys,
            mode=(3,)*len(modality_keys) + (1, ),  # 3 means third order spline interpolation,  1 means linear interpolation.
            multilabel=(False,)*len(modality_keys) + (True, ),  # If multilabel is False for the label, then the mode should be 0 (nearest interpolation), otherwise the interpolated label can be in between the original label values
            prob=1.0,
            spatial_size=patch_size,
            rotate_range=rotate_range,
            prob_rotate=0.2,
            translate_range=(0, 0, 0),
            foreground_oversampling_prob=pos_sample_num / (pos_sample_num + neg_sample_num),  # None for random sampling according to translate_range, 1.0 for foreground sampling plus random translation according to translate_range, 0.0 for random within "valid" range
            label_key_for_foreground_oversampling="label",  # Determine which dictionary entry's metadata contains the sampled foreground locations
            prob_translate=1.0,
            scale_range=scale_range,
            prob_scale=0.2,
            padding_mode=("constant",)*len(modality_keys) + ("border", ),
        )

        """2. Gaussian Noise augmentation"""
        rand_gauss_noise = RandGaussianNoised(keys=mod_inty_keys, std=0.1, prob=0.1)

        """3. Gaussian Smoothing augmentation (might need adjustment, because image channels are smoothed independently in nnU-Net"""
        rand_gauss_smooth = RandGaussianSmoothd(keys=mod_inty_keys,
                                                sigma_x=(0.5, 1.0),
                                                sigma_y=(0.5, 1.0),
                                                sigma_z=(0.5, 1.0),
                                                prob=0.2 * 0.5, )  # 0.5 comes from the per_channel_probability

        """4. Intensity scaling transform"""
        scale_intensity = RandScaleIntensityd(keys=mod_inty_keys, factors=[-0.25, 0.25], prob=0.15)

        """5. ContrastAugmentationTransform"""
        shift_intensity = RandScaleIntensityFixedMeand(keys=mod_inty_keys, factors=[-0.25, 0.25], preserve_range=True,
                                                       prob=0.15)

        """6. Simulate Lowres transform"""
        sim_lowres = RandSimulateLowResolutiond(keys=mod_inty_keys, prob=0.25*0.5, zoom_range=(0.5, 1.0))

        """7. Adjust contrast transform with image inversion"""
        adjust_contrast_inverted = RandAdjustContrastd(keys=mod_inty_keys, prob=0.1 * 1.0, gamma=(0.7, 1.5),
                                                       invert_image=True, retain_stats=True)

        """8. Adjust contrast transform """
        adjust_contrast = RandAdjustContrastd(keys=mod_inty_keys, prob=0.3 * 1.0, gamma=(0.7, 1.5), invert_image=False,
                                              retain_stats=True)

        """9. Mirror Transform"""
        mirror_x = RandFlipd(all_keys, spatial_axis=[0], prob=0.5)
        mirror_y = RandFlipd(all_keys, spatial_axis=[1], prob=0.5)
        mirror_z = RandFlipd(all_keys, spatial_axis=[2], prob=0.5)

        """10. Downsampled labels"""
        supr_label_shapes = get_deep_supr_label_shapes(deep_supr_num, patch_size, strides)

        if mode == "train":
            new_transform = Compose([
                map_labels,  # -2
                norm_transform,  # -1
                sample_foreground_locations,  # 0
                ToTensord(keys=all_keys, device=torch.device("cuda")),  # moving data to GPU to run RandAffine spline interpolation with cupy backend rather than scipy. This requires 0 train_num_workers in the dataloader because cupy doesn't support multiprocessing

                rand_affine,  # 1
                # SpatialPadd(keys=all_keys, spatial_size=patch_size),  # pad to patch size
                # CenterSpatialCropd(keys=all_keys, roi_size=patch_size),

                rand_gauss_noise,  # 2
                rand_gauss_smooth,  # 3
                scale_intensity,  # 4
                shift_intensity,  # 5
                sim_lowres,  # 6
                adjust_contrast_inverted,  # 7
                adjust_contrast,  # 8
                mirror_x, mirror_y, mirror_z,  # 9
                AppendDownsampledd(label_keys, downsampled_shapes=supr_label_shapes),
                CastToTyped(keys=modality_keys, dtype=np.float32),
                EnsureTyped(keys=modality_keys),
                ConcatItemsd(keys=modality_keys, name="image", dim=0),
                DeleteItemsd(keys=modality_keys),
            ], unpack_items=True)

            return new_transform

        elif mode == "validation":
            transform = Compose([
                load_image,
                ensure_channel_first,
                crop_transform,
                map_labels,  # -2
                norm_transform,  # -1
                ToTensord(keys=all_keys, device=torch.device("cuda")), # moving data to GPU

                SpatialPadd(keys=all_keys, spatial_size=patch_size),
                CenterSpatialCropd(keys=all_keys, roi_size=patch_size),  # to make sliding-window-inference much faster for validation (but restrict it to a central patch

                AppendDownsampledd(label_keys, downsampled_shapes=supr_label_shapes),
                CastToTyped(keys=modality_keys, dtype=(np.float32,)*len(modality_keys)),
                EnsureTyped(keys=modality_keys),
                ConcatItemsd(keys=modality_keys, name="image", dim=0),
                DeleteItemsd(keys=modality_keys),
            ], unpack_items=True)

            return transform

    elif mode == "test":
        affine_reg = ANTsAffineRegistrationd(keys=mod_inty_keys,  # register only the intensity image, the prior is already registered to the template
                                             moving_img_key=modality_keys[0],
                                             save_path_key="save_path",
                                             template_path=registration_template_path)
        identity = Identityd(all_keys, allow_missing_keys=True)
        brain_extraction = BrainExtractiond(keys=mod_inty_keys, save_path_key="save_path")
        load_image = LoadImaged(keys=modality_keys, image_only=True)
        ensure_channel_first = EnsureChannelFirstd(keys=modality_keys)
        crop_transform = CropForegroundd(keys=modality_keys, source_key=mod_inty_keys[0], start_coord_key=None, end_coord_key=None)
        norm_transform = NormalizeIntensityd(keys=modality_keys, nonzero=use_nonzero)

        # map the training prior to the labels the model was trained with
        if use_prior and channel_mapping:
            map_labels = MapLabelValued(keys=prior_key,
                                        orig_labels=list(channel_mapping.keys()),
                                        target_labels=list(channel_mapping.values()))

        else:
            map_labels = identity

        transform = Compose([
            affine_reg if reg else identity,
            brain_extraction if bet else identity,
            load_image,
            ensure_channel_first,
            crop_transform,
            map_labels,
            norm_transform,
            #CenterSpatialCropd(keys=modality_keys, roi_size=patch_size), # to make sliding-window-inference much faster for validation (but restrict it to a central patch
            CastToTyped(keys=modality_keys, dtype=(np.float32,) * len(modality_keys)),
            EnsureTyped(keys=modality_keys),
            ConcatItemsd(keys=modality_keys, name="image", dim=0),
            DeleteItemsd(keys=modality_keys),
        ], unpack_items=True)

        return transform


def determine_normalization_param_from_crop(prep_data_loader, key, multi_gpu):
    '''
    Helper function to determine the "use_nonzero" parameter of the NormalizeIntensity transform. It loads the whole
    dataset via the provided dataloader, whose preprocessing transform includes the CropForeground transform. If the
    median volume reduction achieved by the cropping is more than 25%, use_nonzero will be returned as True, otherwise
    false.
    :param prep_data_loader: The dataloader, must be configured to go through the complete dataset once
    :param key: The key of the dictionaries that points to the cropped MetaTensor
    :param multi_gpu: Whether training data is split across multiple GPUs
    :return: bool, that indicates if zeros should be included in normalization or not.
    '''

    all_crop_size_factors = []

    def get_crop_size_factor(img):
        # among applied transforms, find the idx of the crop transform
        for idx, tfm in enumerate(img.applied_operations):
            if tfm['class'] == 'CropForeground':
                break
        crop_size_factor = np.prod(img.shape) / np.prod(img.applied_operations[idx]['orig_size'])
        return crop_size_factor

    for batch in tqdm(prep_data_loader, file=sys.stdout):
        for sample in batch:
            all_crop_size_factors.append(get_crop_size_factor(sample[key]))

    # In case of data on multiple GPUs, all cropping results need to be gathered
    if multi_gpu:
        world_size = dist.get_world_size()

        # initialize tensor of expected size
        gathered = [torch.zeros(len(all_crop_size_factors)).cuda() for _ in range(world_size)]

        # Gather all padded lists into a single tensor on each process (requires conversion to torch tensor)
        dist.all_gather(gathered, torch.tensor(all_crop_size_factors, dtype=gathered[0].dtype).cuda())

        # convert back to list
        all_crop_size_factors = torch.stack(gathered).flatten().tolist()

    median_crop_size_factor = np.median(all_crop_size_factors)
    non_zero_normalization = True if median_crop_size_factor < 0.75 else False
    
    if not multi_gpu or dist.get_rank() == 0:
        print(f"Median crop size factor of {len(all_crop_size_factors)} images is {median_crop_size_factor}, "
              f"therefore non_zero_normalization is {non_zero_normalization}")
    return non_zero_normalization


def check_label_mapping(label_mapping, label_names_path):
    # check consistency of label mapping with label names file
    if os.path.isfile(label_names_path):
        print(f"Label names file provided at {label_names_path}. Checking label mapping...")
        named_labels = [int(la) for la in json.load(open(label_names_path, "r"))]
        for la in label_mapping.keys():
            if la not in named_labels:
                raise Exception(f"Label mapping check failed. Label name {la} present in data but not found in {label_names_path}.")
        for la in named_labels:
            if la not in label_mapping:
                print(f"Warning: Label {la} found in {label_names_path} but not present in training data. "
                      f"The model will not predict this label.")
        print("Label mapping check passed.")
    else:
        print(f"No label names file provided at {label_names_path}. Label mapping check skipped...")


@run_on_rank0_then_broadcast_object_list
def get_channel_mapping(datalist_train, key, label_names_path):
    """
    Determine the mapping from the original labels to consecutive integers (channels).
    :param datalist_train: List of dictionaries containing the training data
    :param key: The key of the dictionaries that points to the original label
    :param label_names_path: Path to the json file that contains the label names
    :return: dict, that maps original labels to consecutive integers
    """
    print("Determining label to channel mapping...")
    # get all labels present in the dataset
    all_labels = set()
    label_paths = [la[key] for la in datalist_train]
    for la in tqdm(label_paths, file=sys.stdout):
        curr_unique_labels = set(np.unique(nib.load(la).get_fdata()))
        all_labels = all_labels.union(curr_unique_labels)

    # sort labels
    all_labels = sorted(list(all_labels))

    # check if the labels are already consecutive integers (or floats with .0)
    if all_labels == list(range(len(all_labels))):
        print("Labels are already consecutive integers, no mapping needed.")
        num_channels_out = len(all_labels)
        return None, num_channels_out

    # create mapping to consecutive integers (channels)
    channel_mapping = {label: idx for idx, label in enumerate(all_labels)}
    print("Label mapping determined.")
    check_label_mapping(channel_mapping, label_names_path)

    num_channels_out = len(channel_mapping)
    return channel_mapping, num_channels_out
