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

import torch

from monai.networks.nets import DynUNet


def get_kernels_strides(patch_size, spacing):
    """
    This function is only used for decathlon datasets with the provided patch sizes.
    When refering this method for other tasks, please ensure that the patch size for each spatial dimension should
    be divisible by the product of all strides in the corresponding dimension.
    In addition, the minimal spatial size should have at least one dimension that has twice the size of
    the product of all strides. For patch sizes that cannot find suitable strides, an error will be raised.

    """
    sizes, spacings = patch_size, spacing
    input_size = sizes
    strides, kernels = [], []
    while True:
        spacing_ratio = [sp / min(spacings) for sp in spacings]
        stride = [2 if ratio <= 2 and size >= 8 else 1 for (ratio, size) in zip(spacing_ratio, sizes)]
        kernel = [3 if ratio <= 2 else 1 for ratio in spacing_ratio]
        if all(s == 1 for s in stride):
            break
        for idx, (i, j) in enumerate(zip(sizes, stride)):
            if i % j != 0:
                raise ValueError(
                    f"Patch size is not supported, please try to modify the size {input_size[idx]} in the spatial dimension {idx}. Remember that the"
                    f"patch size for each spatial dimension should be divisible by the product of all strides in the corresponding dimension."
                )
        sizes = [i / j for i, j in zip(sizes, stride)]
        spacings = [i * j for i, j in zip(spacings, stride)]
        kernels.append(kernel)
        strides.append(stride)

    strides.insert(0, len(spacings) * [1])
    kernels.append(len(spacings) * [3])
    return kernels, strides


def get_network(
                n_classes,
                n_in_channels,
                kernels,
                strides,
                deep_supr_num,
                prior_path,
                pretrain_path=None,
                checkpoint=None):

    # increase number of input channels by one if a prior is passed as an additional image
    if prior_path:
        n_in_channels = n_in_channels + 1

    net = DynUNet(
        spatial_dims=3,
        in_channels=n_in_channels,
        out_channels=n_classes,
        kernel_size=kernels,
        strides=strides,
        upsample_kernel_size=strides[1:],
        norm_name=("INSTANCE", {"affine": True}),
        deep_supervision=True,
        deep_supr_num=deep_supr_num,
        trans_bias=True,
        return_list=True,
    )

    if checkpoint is not None:
        pretrain_path = os.path.join(pretrain_path, checkpoint)
        if os.path.exists(pretrain_path):
            net.load_state_dict(torch.load(pretrain_path, weights_only=False)['net'])
            print("Pretrained checkpoint: {} loaded".format(pretrain_path))
        else:
            raise Exception(f"Checkpoint {pretrain_path} not found.")
    return net
