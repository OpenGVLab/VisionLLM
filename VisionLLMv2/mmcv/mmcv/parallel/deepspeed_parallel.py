# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, List, Tuple
from deepspeed.runtime.engine import DeepSpeedEngine
from mmcv.parallel import MODULE_WRAPPERS

from .scatter_gather import ScatterInputs, scatter_kwargs


@MODULE_WRAPPERS.register_module()
class MMDeepSpeedEngine(DeepSpeedEngine):
    """A prototype for Deepspeed Enginer.

    MMDeepSpeedEngine is a protytpe to support mmcv and mmengine to use Deepspeed

    - It implement two APIs ``train_step()`` and ``val_step()``.
    """

    def to_kwargs(self, inputs: ScatterInputs, kwargs: ScatterInputs,
                device_id: int) -> Tuple[tuple, tuple]:
        # Use `self.to_kwargs` instead of `self.scatter` in pytorch1.8
        # to move all tensors to device_id
        return scatter_kwargs(inputs, kwargs, [device_id], dim=0)

    def scatter(self, inputs: ScatterInputs, kwargs: ScatterInputs,
                device_ids: List[int]) -> Tuple[tuple, tuple]:
        return scatter_kwargs(inputs, kwargs, device_ids, dim=0)
    
    def train_step(self, *inputs, **kwargs):
        inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
            # Inputs is a nested tuple. Therefore inputs[0][0] is required.
        input_img = inputs[0][0].pop('img')
        kwargs[0].update(inputs[0][0])
        losses = super().forward(input_img, **kwargs[0])
        loss, log_vars = self._parse_losses(losses)
        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(inputs[0][0]['img_metas']))
        return outputs
    
    def forward(self, *inputs: Any, **kwargs: Any):
        # Eval mode will use this method.
        inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
        input_img = kwargs[0].pop('img')
        losses = super().forward(input_img, **kwargs[0])
        return losses