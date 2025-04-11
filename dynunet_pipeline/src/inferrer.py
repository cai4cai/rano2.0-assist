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
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from ignite.engine import Engine
from monai.transforms import SaveImaged, BatchInverseTransform, ANTsApplyTransformd, allow_missing_keys_mode, MapLabelValue
from monai.engines import SupervisedEvaluator
from monai.engines.utils import IterationEvents
from monai.inferers import Inferer
from monai.networks.utils import eval_mode
from monai.transforms import AsDiscrete
from torch.utils.data import DataLoader


class DynUNetInferrer(SupervisedEvaluator):
    """
    This class inherits from SupervisedEvaluator in MONAI, and is used with DynUNet
    on Decathlon datasets. As a customized inferrer, some of the arguments from
    SupervisedEvaluator are not supported. For example, the actual
    post processing method used is hard coded in the `_iteration` function, thus the
    argument `postprocessing` from SupervisedEvaluator is not exist. If you need
    to change the post processing way, please modify the `_iteration` function directly.

    Args:
        device: an object representing the device on which to run.
        val_data_loader: Ignite engine use data_loader to run, must be
            torch.DataLoader.
        network: use the network to run model forward.
        output_dir: the path to save inferred outputs.
        num_classes: the number of classes (output channels) for the task.
        inferer: inference method that execute model forward on input data, like: SlidingWindow, etc.
        amp: whether to enable auto-mixed-precision evaluation, default is False.
        tta_val: whether to do the 8 flips (8 = 2 ** 3, where 3 represents the three dimensions)
            test time augmentation, default is False.

    """

    def __init__(
        self,
        device: torch.device,
        val_data_loader: DataLoader,
        network: torch.nn.Module,
        output_dir: str,
        num_classes: Union[str, int],
        inferer: Optional[Inferer] = None,
        amp: bool = False,
        tta_val: bool = False,
        channel_mapping: Optional[Dict[Union[float, int], int]] = None,
    ) -> None:
        super().__init__(
            device=device,
            val_data_loader=val_data_loader,
            network=network,
            inferer=inferer,
            amp=amp,
        )

        if not isinstance(num_classes, int):
            num_classes = int(num_classes)
        self.output_dir = output_dir
        self.tta_val = tta_val
        self.num_classes = num_classes
        self.channel_mapping = channel_mapping

        if channel_mapping is not None:
            self.map_labels_back = MapLabelValue(orig_labels=list(channel_mapping.values()),
                                                 target_labels=list(channel_mapping.keys()))


    def _iteration(self, engine: Engine, batchdata: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        callback function for the Supervised Evaluation processing logic of 1 iteration in Ignite Engine.
        Return below item in a dictionary:
            - PRED: prediction result of model.

        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.
            batchdata: input data for this iteration, usually can be dictionary or tuple of Tensor data.

        Raises:
            ValueError: When ``batchdata`` is None.

        """
        if batchdata is None:
            raise ValueError("Must provide batch data for current iteration.")
        batch = self.prepare_batch(batchdata, engine.state.device, engine.non_blocking)
        if len(batch) == 2:
            inputs, _ = batch
            args: Tuple = ()
            kwargs: Dict = {}
        else:
            inputs, _, args, kwargs = batch

        def _compute_pred():
            counter = 0
            for i, dims in enumerate([[], [2], [3], [4], (2, 3), (2, 4), (3, 4), (2, 3, 4)]):
                flipped_inputs = torch.flip(inputs, dims=dims)
                pred_flipped_input = torch.flip(self.inferer(flipped_inputs, self.network), dims=dims)
                pred_flipped_input = nn.functional.sigmoid(pred_flipped_input)
                if i == 0:
                    pred = pred_flipped_input
                else:
                    pred += pred_flipped_input
                counter += 1

                if not self.tta_val:
                    break

            pred = pred / counter

            pred = torch.argmax(pred, dim=1, keepdim=True)

            return pred

        # execute forward computation
        print("run sliding window inference...")
        with eval_mode(self.network):
            if self.amp:
                with torch.amp.autocast('cuda'):
                    predictions = _compute_pred()
            else:
                predictions = _compute_pred()

        # map the consecutive integer labels to the original labels in the dataset
        if self.channel_mapping is not None:
            predictions = self.map_labels_back(predictions)

        # here we overwrite the "image_0000" with the predictions, because the inverse transformation with
        # BatchInverseTransform will only work on the keys that were used in the forward transform, which was only
        # the "image_0000" key
        batchdata["image_0000"] = predictions
        mni_registered = True if 'image_0000_meta_dict_affine_trfm_file_path' in batchdata else False

        batch_inverter = BatchInverseTransform(self.data_loader.dataset.transform, self.data_loader)
        with allow_missing_keys_mode(self.data_loader.dataset.transform):
            data_list = batch_inverter(batchdata)  # the batch inverter decollates the batch into a list of dicts

        # save each prediction in the list of dicts
        for data_dict in data_list:
            if not mni_registered:
                output_original_space_segm_path = data_dict["save_path"]
                SaveImaged(
                    keys=["image_0000"],
                    output_dir=os.path.dirname(output_original_space_segm_path),
                    output_name_formatter=lambda x, y: {"subject": os.path.basename(output_original_space_segm_path)},
                    output_postfix="",
                    output_dtype=np.uint16,
                    separate_folder=False,
                    resample=False,
                )(data_dict)

            else:
                mni_template_space_output_file_path = os.path.join(os.path.dirname(data_dict["save_path"]), "preprocessed", "registered", os.path.basename(data_dict["save_path"]))

                SaveImaged(
                    keys=["image_0000"],
                    output_dir=os.path.dirname(mni_template_space_output_file_path),
                    output_name_formatter=lambda x, y: {"subject": os.path.basename(mni_template_space_output_file_path)},
                    output_postfix="",
                    output_dtype=np.uint16,
                    separate_folder=False,
                    resample=False,
                )(data_dict)

                # transform saved segmentation into the space of the original image
                print("transform from MNI template space to original image space")
                # Define transform that uses ANTs to invert the affine registration to the template space
                ants_apply_transform = ANTsApplyTransformd(keys="image_0000")

                output_original_space_segm_path = data_dict["save_path"]
                ants_apply_transform(data_dict,
                                     input_file_path=mni_template_space_output_file_path,
                                     output_file_path=output_original_space_segm_path,
                                     use_inverse_trfm=True)

        engine.fire_event(IterationEvents.FORWARD_COMPLETED)
        return {"pred": predictions}
