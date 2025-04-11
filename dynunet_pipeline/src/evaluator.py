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

from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from ignite.engine import Engine
from ignite.metrics import Metric
from monai.data import decollate_batch
from monai.engines import SupervisedEvaluator
from monai.engines.utils import CommonKeys as Keys
from monai.engines.utils import IterationEvents, default_prepare_batch
from monai.inferers import Inferer
from monai.networks.utils import train_mode
from monai.transforms import AsDiscrete, Transform
from torch.utils.data import DataLoader


class DynUNetEvaluator(SupervisedEvaluator):
    """
    This class inherits from SupervisedEvaluator in MONAI, and is used with DynUNet
    on Decathlon datasets.

    Args:
        device: an object representing the device on which to run.
        val_data_loader: Ignite engine use data_loader to run, must be
            torch.DataLoader.
        network: use the network to run model forward.
        epoch_length: number of iterations for one epoch, default to
            `len(val_data_loader)`.
        non_blocking: if True and this copy is between CPU and GPU, the copy may occur asynchronously
            with respect to the host. For other cases, this argument has no effect.
        prepare_batch: function to parse image and label for current iteration.
        iteration_update: the callable function for every iteration, expect to accept `engine`
            and `batchdata` as input parameters. if not provided, use `self._iteration()` instead.
        inferer: inference method that execute model forward on input data, like: SlidingWindow, etc.
        postprocessing: execute additional transformation for the model output data.
            Typically, several Tensor based transforms composed by `Compose`.
        key_val_metric: compute metric when every iteration completed, and save average value to
            engine.state.metrics when epoch completed. key_val_metric is the main metric to compare and save the
            checkpoint into files.
        additional_metrics: more Ignite metrics that also attach to Ignite Engine.
        val_handlers: every handler is a set of Ignite Event-Handlers, must have `attach` function, like:
            CheckpointHandler, StatsHandler, etc.
        amp: whether to enable auto-mixed-precision evaluation, default is False.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def _iteration(self, engine: Engine, batchdata: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        callback function for the Supervised Evaluation processing logic of 1 iteration in Ignite Engine.
        Return below items in a dictionary:
            - IMAGE: image Tensor data for model input, already moved to device.
            - LABEL: label Tensor data corresponding to the image, already moved to device.
            - PRED: prediction result of model.

        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.
            batchdata: input data for this iteration, usually can be dictionary or tuple of Tensor data.

        Raises:
            ValueError: When ``batchdata`` is None.

        """
        # put all MetaTensors on device
        inputs = batchdata['image'].to(device=engine.state.device, non_blocking=engine.non_blocking)
        targets = [l.to(device=engine.state.device, non_blocking=engine.non_blocking) for l in batchdata['label']]

        # compute in train mode, since we need the deep supervision outputs to calculate the validation loss
        with train_mode(self.network):
            with torch.no_grad():  # but no need to compute gradients
                if self.amp:
                    with torch.amp.autocast('cuda'):
                        predictions = self.inferer(inputs, self.network)
                else:
                    predictions = self.inferer(inputs, self.network)

        engine.state.output = {Keys.IMAGE: inputs,
                               Keys.LABEL: targets,
                               Keys.PRED: predictions,}

        engine.fire_event(IterationEvents.FORWARD_COMPLETED)
        engine.fire_event(IterationEvents.MODEL_COMPLETED)

        return engine.state.output
