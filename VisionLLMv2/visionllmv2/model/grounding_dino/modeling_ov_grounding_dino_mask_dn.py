# coding=utf-8
# Copyright 2023 IDEA Research and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch Grounding DINO model."""

import copy
import enum
import math
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from detectron2.layers import Conv2d, ShapeSpec, get_norm
import fvcore.nn.weight_init as weight_init
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from transformers.activations import ACT2FN
from transformers.file_utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_scipy_available,
    is_torch_cuda_available,
    is_vision_available,
    replace_return_docstrings,
    requires_backends,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPooling,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, meshgrid, prune_linear_layer
from transformers.utils import is_ninja_available, logging
from transformers.models.auto import AutoBackbone
from .configuration_grounding_dino import GroundingDinoConfig, GroundingDinoTextConfig

# internimage
import warnings
import torch.nn as nn
from collections import OrderedDict
import torch.utils.checkpoint as checkpoint
from timm.models.layers import trunc_normal_, DropPath
from mmcv.runner import _load_checkpoint
from mmcv.cnn import constant_init, trunc_normal_init
from mmdet.utils import get_root_logger
from mmdet.models.builder import BACKBONES
try:
    from ..ops_dcnv3 import modules as opsm
except:
    warnings.warn("please compile DCNv3 operator if taking InternImage as backbone!")


logger = logging.get_logger(__name__)

from transformers.models.deformable_detr.load_custom import load_cuda_kernels

# def load_cuda_kernels():
#     from torch.utils.cpp_extension import load

#     root = Path(__file__).resolve().parent.parent.parent / "kernels" / "grounding_dino"
#     src_files = [
#         root / filename
#         for filename in [
#             "vision.cpp",
#             os.path.join("cpu", "ms_deform_attn_cpu.cpp"),
#             os.path.join("cuda", "ms_deform_attn_cuda.cu"),
#         ]
#     ]

#     load(
#         "MultiScaleDeformableAttention",
#         src_files,
#         with_cuda=True,
#         extra_include_paths=[str(root)],
#         extra_cflags=["-DWITH_CUDA=1"],
#         extra_cuda_cflags=[
#             "-DCUDA_HAS_FP16=1",
#             "-D__CUDA_NO_HALF_OPERATORS__",
#             "-D__CUDA_NO_HALF_CONVERSIONS__",
#             "-D__CUDA_NO_HALF2_OPERATORS__",
#         ],
#     )

#     import MultiScaleDeformableAttention as MSDA

#     return MSDA


# Move this to not compile only when importing, this needs to happen later, like in __init__.
if is_torch_cuda_available() and is_ninja_available():
    logger.info("Loading custom CUDA kernels...")
    try:
        MultiScaleDeformableAttention = load_cuda_kernels()
    except Exception as e:
        logger.warning(f"Could not load the custom kernel for multi-scale deformable attention: {e}")
        MultiScaleDeformableAttention = None
else:
    MultiScaleDeformableAttention = None

if is_vision_available():
    from transformers.image_transforms import center_to_corners_format

if is_scipy_available():
    from scipy.optimize import linear_sum_assignment

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "GroundingDinoConfig"
_CHECKPOINT_FOR_DOC = "EduardoPacheco/grounding-dino-tiny"

GROUNDING_DINO_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "EduardoPacheco/grounding-dino-tiny",
    # See all Grounding DINO models at https://huggingface.co/models?filter=grounding-dino
]


# Copied from transformers.models.deformable_detr.modeling_deformable_detr.MultiScaleDeformableAttentionFunction
class MultiScaleDeformableAttentionFunction(Function):
    @staticmethod
    def forward(
        context,
        value,
        value_spatial_shapes,
        value_level_start_index,
        sampling_locations,
        attention_weights,
        im2col_step,
    ):
        context.im2col_step = im2col_step
        output = MultiScaleDeformableAttention.ms_deform_attn_forward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
            context.im2col_step,
        )
        context.save_for_backward(
            value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights
        )
        return output

    @staticmethod
    @once_differentiable
    def backward(context, grad_output):
        (
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
        ) = context.saved_tensors
        grad_value, grad_sampling_loc, grad_attn_weight = MultiScaleDeformableAttention.ms_deform_attn_backward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
            grad_output,
            context.im2col_step,
        )

        return grad_value, None, None, grad_sampling_loc, grad_attn_weight, None


@dataclass
class GroundingDinoDecoderOutput(ModelOutput):
    """
    Base class for outputs of the GroundingDinoDecoder. This class adds two attributes to
    BaseModelOutputWithCrossAttentions, namely:
    - a stacked tensor of intermediate decoder hidden states (i.e. the output of each decoder layer)
    - a stacked tensor of intermediate reference points.
    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        intermediate_hidden_states (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, num_queries, hidden_size)`):
            Stacked intermediate hidden states (output of each layer of the decoder).
        intermediate_reference_points (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, sequence_length, hidden_size)`):
            Stacked intermediate reference points (reference points of each layer of the decoder).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer
            plus the initial embedding outputs.
        attentions (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of tuples of `torch.FloatTensor` (one for attention for each layer) of shape `(batch_size, num_heads,
            sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the
            weighted average in the self-attention, cross-attention and multi-scale deformable attention heads.
    """

    last_hidden_state: torch.FloatTensor = None
    intermediate_hidden_states: torch.FloatTensor = None
    intermediate_reference_points: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None


@dataclass
class GroundingDinoEncoderOutput(ModelOutput):
    """
    Base class for outputs of the GroundingDinoEncoder. This class extends BaseModelOutput, due to:
    - vision and text last hidden states
    - vision and text intermediate hidden states
    Args:
        last_hidden_state_vision (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the vision encoder.
        last_hidden_state_text (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the text encoder.
        hidden_states_vision (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the vision embeddings + one for the output of each
            layer) of shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the vision encoder at the
            output of each layer plus the initial embedding outputs.
        hidden_states_text (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the text embeddings + one for the output of each layer)
            of shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the text encoder at the output of
            each layer plus the initial embedding outputs.
        attentions (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of tuples of `torch.FloatTensor` (one for attention for each layer) of shape `(batch_size, num_heads,
            sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the
            weighted average in the text-vision attention, vision-text attention, text-enhancer (self-attention) and
            multi-scale deformable attention heads.
    """

    last_hidden_state_vision: torch.FloatTensor = None
    last_hidden_state_text: torch.FloatTensor = None
    hidden_states_vision: Optional[Tuple[torch.FloatTensor]] = None
    hidden_states_text: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None


@dataclass
class GroundingDinoModelOutput(ModelOutput):
    """
    Base class for outputs of the Grounding DINO encoder-decoder model.
    Args:
        init_reference_points (`torch.FloatTensor` of shape  `(batch_size, num_queries, 4)`):
            Initial reference points sent through the Transformer decoder.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the decoder of the model.
        intermediate_hidden_states (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, num_queries, hidden_size)`):
            Stacked intermediate hidden states (output of each layer of the decoder).
        intermediate_reference_points (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, num_queries, 4)`):
            Stacked intermediate reference points (reference points of each layer of the decoder).
        decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, num_queries, hidden_size)`. Hidden-states of the decoder at the output of each layer
            plus the initial embedding outputs.
        decoder_attentions (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of tuples of `torch.FloatTensor` (one for attention for each layer) of shape `(batch_size, num_heads,
            sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the
            weighted average in the self-attention, cross-attention and multi-scale deformable attention heads.
        encoder_last_hidden_state_vision (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_last_hidden_state_text (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_hidden_states_vision (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the vision embeddings + one for the output of each
            layer) of shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the vision encoder at the
            output of each layer plus the initial embedding outputs.
        encoder_hidden_states_text (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the text embeddings + one for the output of each layer)
            of shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the text encoder at the output of
            each layer plus the initial embedding outputs.
        encoder_attentions (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of tuples of `torch.FloatTensor` (one for attention for each layer) of shape `(batch_size, num_heads,
            sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the
            weighted average in the text-vision attention, vision-text attention, text-enhancer (self-attention) and
            multi-scale deformable attention heads. attention softmax, used to compute the weighted average in the
            bi-attention heads.
        enc_outputs_class (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_labels)`, *optional*, returned when `config.two_stage=True`):
            Predicted bounding boxes scores where the top `config.num_queries` scoring bounding boxes are picked as
            region proposals in the first stage. Output of bounding box binary classification (i.e. foreground and
            background).
        enc_outputs_coord_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, 4)`, *optional*, returned when `config.two_stage=True`):
            Logits of predicted bounding boxes coordinates in the first stage.
    """

    init_reference_points: torch.FloatTensor = None
    last_hidden_state: torch.FloatTensor = None
    intermediate_hidden_states: torch.FloatTensor = None
    intermediate_reference_points: torch.FloatTensor = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    encoder_last_hidden_state_vision: Optional[torch.FloatTensor] = None
    encoder_last_hidden_state_text: Optional[torch.FloatTensor] = None
    encoder_hidden_states_vision: Optional[Tuple[torch.FloatTensor]] = None
    encoder_hidden_states_text: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    enc_outputs_class: Optional[torch.FloatTensor] = None
    enc_outputs_coord_logits: Optional[torch.FloatTensor] = None
    mask_features: torch.FloatTensor = None
    output_mask_intermediate: torch.FloatTensor = None
    enc_topk_coords_logits: torch.FloatTensor = None
    enc_topk_class_logits: torch.FloatTensor = None


@dataclass
class GroundingDinoObjectDetectionOutput(ModelOutput):
    """
    Output type of [`GroundingDinoForObjectDetection`].
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` are provided)):
            Total loss as a linear combination of a negative log-likehood (cross-entropy) for class prediction and a
            bounding box loss. The latter is defined as a linear combination of the L1 loss and the generalized
            scale-invariant IoU loss.
        loss_dict (`Dict`, *optional*):
            A dictionary containing the individual losses. Useful for logging.
        logits (`torch.FloatTensor` of shape `(batch_size, num_queries, num_classes + 1)`):
            Classification logits (including no-object) for all queries.
        pred_boxes (`torch.FloatTensor` of shape `(batch_size, num_queries, 4)`):
            Normalized boxes coordinates for all queries, represented as (center_x, center_y, width, height). These
            values are normalized in [0, 1], relative to the size of each individual image in the batch (disregarding
            possible padding). You can use [`~GroundingDinoProcessor.post_process_object_detection`] to retrieve the
            unnormalized bounding boxes.
        auxiliary_outputs (`List[Dict]`, *optional*):
            Optional, only returned when auxilary losses are activated (i.e. `config.auxiliary_loss` is set to `True`)
            and labels are provided. It is a list of dictionaries containing the two above keys (`logits` and
            `pred_boxes`) for each decoder layer.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the decoder of the model.
        decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, num_queries, hidden_size)`. Hidden-states of the decoder at the output of each layer
            plus the initial embedding outputs.
        decoder_attentions (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of tuples of `torch.FloatTensor` (one for attention for each layer) of shape `(batch_size, num_heads,
            sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the
            weighted average in the self-attention, cross-attention and multi-scale deformable attention heads.
        encoder_last_hidden_state_vision (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_last_hidden_state_text (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_hidden_states_vision (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the vision embeddings + one for the output of each
            layer) of shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the vision encoder at the
            output of each layer plus the initial embedding outputs.
        encoder_hidden_states_text (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the text embeddings + one for the output of each layer)
            of shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the text encoder at the output of
            each layer plus the initial embedding outputs.
        encoder_attentions (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of tuples of `torch.FloatTensor` (one for attention for each layer) of shape `(batch_size, num_heads,
            sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the
            weighted average in the text-vision attention, vision-text attention, text-enhancer (self-attention) and
            multi-scale deformable attention heads.
        intermediate_hidden_states (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, num_queries, hidden_size)`):
            Stacked intermediate hidden states (output of each layer of the decoder).
        intermediate_reference_points (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, num_queries, 4)`):
            Stacked intermediate reference points (reference points of each layer of the decoder).
        init_reference_points (`torch.FloatTensor` of shape  `(batch_size, num_queries, 4)`):
            Initial reference points sent through the Transformer decoder.
        enc_outputs_class (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_labels)`, *optional*, returned when `config.two_stage=True`):
            Predicted bounding boxes scores where the top `config.num_queries` scoring bounding boxes are picked as
            region proposals in the first stage. Output of bounding box binary classification (i.e. foreground and
            background).
        enc_outputs_coord_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, 4)`, *optional*, returned when `config.two_stage=True`):
            Logits of predicted bounding boxes coordinates in the first stage.
    """

    loss: Optional[torch.FloatTensor] = None
    loss_dict: Optional[Dict] = None
    logits: torch.FloatTensor = None
    pred_boxes: torch.FloatTensor = None
    pred_masks: torch.FloatTensor = None
    auxiliary_outputs: Optional[List[Dict]] = None
    init_reference_points: Optional[torch.FloatTensor] = None
    last_hidden_state: Optional[torch.FloatTensor] = None
    intermediate_hidden_states: Optional[torch.FloatTensor] = None
    intermediate_reference_points: Optional[torch.FloatTensor] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    encoder_last_hidden_state_vision: Optional[torch.FloatTensor] = None
    encoder_last_hidden_state_text: Optional[torch.FloatTensor] = None
    encoder_hidden_states_vision: Optional[Tuple[torch.FloatTensor]] = None
    encoder_hidden_states_text: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    enc_outputs_class: Optional[torch.FloatTensor] = None
    enc_outputs_coord_logits: Optional[torch.FloatTensor] = None


def _get_clones(module, num_copies):
    return nn.ModuleList([copy.deepcopy(module) for i in range(num_copies)])


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


# Copied from transformers.models.detr.modeling_detr.DetrFrozenBatchNorm2d with Detr->GroundingDino
class GroundingDinoFrozenBatchNorm2d(nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.
    Copy-paste from torchvision.misc.ops with added eps before rqsrt, without which any other models than
    torchvision.models.resnet[18,34,50,101] produce nans.
    """

    def __init__(self, n):
        super().__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        num_batches_tracked_key = prefix + "num_batches_tracked"
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def forward(self, x):
        # move reshapes to the beginning
        # to make it user-friendly
        weight = self.weight.reshape(1, -1, 1, 1)
        bias = self.bias.reshape(1, -1, 1, 1)
        running_var = self.running_var.reshape(1, -1, 1, 1)
        running_mean = self.running_mean.reshape(1, -1, 1, 1)
        epsilon = 1e-5
        scale = weight * (running_var + epsilon).rsqrt()
        bias = bias - running_mean * scale
        return x * scale + bias


# Copied from transformers.models.detr.modeling_detr.replace_batch_norm with Detr->GroundingDino
def replace_batch_norm(model):
    r"""
    Recursively replace all `torch.nn.BatchNorm2d` with `GroundingDinoFrozenBatchNorm2d`.
    Args:
        model (torch.nn.Module):
            input model
    """
    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm2d):
            new_module = GroundingDinoFrozenBatchNorm2d(module.num_features)

            if not module.weight.device == torch.device("meta"):
                new_module.weight.data.copy_(module.weight)
                new_module.bias.data.copy_(module.bias)
                new_module.running_mean.data.copy_(module.running_mean)
                new_module.running_var.data.copy_(module.running_var)

            model._modules[name] = new_module

        if len(list(module.children())) > 0:
            replace_batch_norm(module)


class GroundingDinoConvEncoder(nn.Module):
    """
    Convolutional backbone using the AutoBackbone API.
    nn.BatchNorm2d layers are replaced by GroundingDinoFrozenBatchNorm2d as defined above.
    """

    def __init__(self, config):
        super().__init__()

        self.config = config
        backbone = AutoBackbone.from_config(config.backbone_config)

        # replace batch norm by frozen batch norm
        with torch.no_grad():
            replace_batch_norm(backbone)
        self.model = backbone
        self.intermediate_channel_sizes = self.model.channels

        backbone_model_type = config.backbone_config.model_type
        if "resnet" in backbone_model_type:
            for name, parameter in self.model.named_parameters():
                if "stage.1" not in name and "stage.2" not in name and "stage.3" not in name:
                    parameter.requires_grad_(False)

    def forward(self, pixel_values: torch.Tensor, pixel_mask: torch.Tensor):
        # send pixel_values through the model to get list of feature maps
        features = self.model(pixel_values).feature_maps

        out = []
        for feature_map in features:
            # downsample pixel_mask to match shape of corresponding feature_map
            mask = nn.functional.interpolate(pixel_mask[None].float(), size=feature_map.shape[-2:]).to(torch.bool)[0]
            out.append((feature_map, mask))
        return out


# Copied from transformers.models.detr.modeling_detr.DetrConvModel with Detr->GroundingDino
class GroundingDinoConvModel(nn.Module):
    """
    This module adds 2D position embeddings to all intermediate feature maps of the convolutional encoder.
    """

    def __init__(self, conv_encoder, position_embedding):
        super().__init__()
        self.conv_encoder = conv_encoder
        self.position_embedding = position_embedding

    def forward(self, pixel_values, pixel_mask):
        # send pixel_values and pixel_mask through backbone to get list of (feature_map, pixel_mask) tuples
        out = self.conv_encoder(pixel_values, pixel_mask)
        pos = []
        for feature_map, mask in out:
            # position encoding
            pos.append(self.position_embedding(feature_map, mask).to(feature_map.dtype))

        return out, pos


class GroundingDinoSinePositionEmbedding(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one used by the Attention is all you
    need paper, generalized to work on images.
    """

    def __init__(self, embedding_dim=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, pixel_values, pixel_mask):
        if pixel_mask is None:
            raise ValueError("No pixel mask provided")
        y_embed = pixel_mask.cumsum(1, dtype=torch.float32)
        x_embed = pixel_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.embedding_dim, dtype=torch.float32, device=pixel_values.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / self.embedding_dim)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


# Copied from transformers.models.detr.modeling_detr.DetrLearnedPositionEmbedding
class GroundingDinoLearnedPositionEmbedding(nn.Module):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, embedding_dim=256):
        super().__init__()
        self.row_embeddings = nn.Embedding(50, embedding_dim)
        self.column_embeddings = nn.Embedding(50, embedding_dim)

    def forward(self, pixel_values, pixel_mask=None):
        height, width = pixel_values.shape[-2:]
        width_values = torch.arange(width, device=pixel_values.device)
        height_values = torch.arange(height, device=pixel_values.device)
        x_emb = self.column_embeddings(width_values)
        y_emb = self.row_embeddings(height_values)
        pos = torch.cat([x_emb.unsqueeze(0).repeat(height, 1, 1), y_emb.unsqueeze(1).repeat(1, width, 1)], dim=-1)
        pos = pos.permute(2, 0, 1)
        pos = pos.unsqueeze(0)
        pos = pos.repeat(pixel_values.shape[0], 1, 1, 1)
        return pos


def build_position_encoding(config):
    n_steps = config.d_model // 2
    if config.position_embedding_type == "sine":
        # TODO find a better way of exposing other arguments
        position_embedding = GroundingDinoSinePositionEmbedding(
            n_steps, config.positional_embedding_temperature, normalize=True
        )
    elif config.position_embedding_type == "learned":
        position_embedding = GroundingDinoLearnedPositionEmbedding(n_steps)
    else:
        raise ValueError(f"Not supported {config.position_embedding_type}")

    return position_embedding


# Copied from transformers.models.deformable_detr.modeling_deformable_detr.multi_scale_deformable_attention
def multi_scale_deformable_attention(
    value: Tensor, value_spatial_shapes: Tensor, sampling_locations: Tensor, attention_weights: Tensor
) -> Tensor:
    batch_size, _, num_heads, hidden_dim = value.shape
    _, num_queries, num_heads, num_levels, num_points, _ = sampling_locations.shape
    value_list = value.split([height.item() * width.item() for height, width in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level_id, (height, width) in enumerate(value_spatial_shapes):
        # batch_size, height*width, num_heads, hidden_dim
        # -> batch_size, height*width, num_heads*hidden_dim
        # -> batch_size, num_heads*hidden_dim, height*width
        # -> batch_size*num_heads, hidden_dim, height, width
        value_l_ = (
            value_list[level_id].flatten(2).transpose(1, 2).reshape(batch_size * num_heads, hidden_dim, height, width)
        )
        # batch_size, num_queries, num_heads, num_points, 2
        # -> batch_size, num_heads, num_queries, num_points, 2
        # -> batch_size*num_heads, num_queries, num_points, 2
        sampling_grid_l_ = sampling_grids[:, :, :, level_id].transpose(1, 2).flatten(0, 1)
        # batch_size*num_heads, hidden_dim, num_queries, num_points
        sampling_value_l_ = nn.functional.grid_sample(
            value_l_, sampling_grid_l_, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        sampling_value_list.append(sampling_value_l_)
    # (batch_size, num_queries, num_heads, num_levels, num_points)
    # -> (batch_size, num_heads, num_queries, num_levels, num_points)
    # -> (batch_size, num_heads, 1, num_queries, num_levels*num_points)
    attention_weights = attention_weights.transpose(1, 2).reshape(
        batch_size * num_heads, 1, num_queries, num_levels * num_points
    )
    output = (
        (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights)
        .sum(-1)
        .view(batch_size, num_heads * hidden_dim, num_queries)
    )
    return output.transpose(1, 2).contiguous()


# Copied from transformers.models.deformable_detr.modeling_deformable_detr.DeformableDetrMultiscaleDeformableAttention with DeformableDetr->GroundingDino,Deformable DETR->Grounding DINO
class GroundingDinoMultiscaleDeformableAttention(nn.Module):
    """
    Multiscale deformable attention as proposed in Deformable DETR.
    """

    def __init__(self, config: GroundingDinoConfig, num_heads: int, n_points: int):
        super().__init__()
        if config.d_model % num_heads != 0:
            raise ValueError(
                f"embed_dim (d_model) must be divisible by num_heads, but got {config.d_model} and {num_heads}"
            )
        dim_per_head = config.d_model // num_heads
        # check if dim_per_head is power of 2
        if not ((dim_per_head & (dim_per_head - 1) == 0) and dim_per_head != 0):
            warnings.warn(
                "You'd better set embed_dim (d_model) in GroundingDinoMultiscaleDeformableAttention to make the"
                " dimension of each attention head a power of 2 which is more efficient in the authors' CUDA"
                " implementation."
            )

        self.im2col_step = 64

        self.d_model = config.d_model
        self.n_levels = config.num_feature_levels
        self.n_heads = num_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(config.d_model, num_heads * self.n_levels * n_points * 2)
        self.attention_weights = nn.Linear(config.d_model, num_heads * self.n_levels * n_points)
        self.value_proj = nn.Linear(config.d_model, config.d_model)
        self.output_proj = nn.Linear(config.d_model, config.d_model)

        self.disable_custom_kernels = config.disable_custom_kernels

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.constant_(self.sampling_offsets.weight.data, 0.0)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
            .view(self.n_heads, 1, 1, 2)
            .repeat(1, self.n_levels, self.n_points, 1)
        )
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        nn.init.constant_(self.attention_weights.weight.data, 0.0)
        nn.init.constant_(self.attention_weights.bias.data, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight.data)
        nn.init.constant_(self.value_proj.bias.data, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight.data)
        nn.init.constant_(self.output_proj.bias.data, 0.0)

    def with_pos_embed(self, tensor: torch.Tensor, position_embeddings: Optional[Tensor]):
        return tensor if position_embeddings is None else tensor + position_embeddings

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        position_embeddings: Optional[torch.Tensor] = None,
        reference_points=None,
        spatial_shapes=None,
        level_start_index=None,
        output_attentions: bool = False,
    ):
        # add position embeddings to the hidden states before projecting to queries and keys
        if position_embeddings is not None:
            hidden_states = self.with_pos_embed(hidden_states, position_embeddings)

        batch_size, num_queries, _ = hidden_states.shape
        batch_size, sequence_length, _ = encoder_hidden_states.shape
        if (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() != sequence_length:
            raise ValueError(
                "Make sure to align the spatial shapes with the sequence length of the encoder hidden states"
            )

        value = self.value_proj(encoder_hidden_states)
        if attention_mask is not None:
            # we invert the attention_mask
            value = value.masked_fill(~attention_mask[..., None], float(0))
        value = value.view(batch_size, sequence_length, self.n_heads, self.d_model // self.n_heads)
        if self.sampling_offsets.bias.dtype != hidden_states.dtype:
            self.sampling_offsets.bias.data = self.sampling_offsets.bias.data.to(hidden_states.dtype)
        sampling_offsets = self.sampling_offsets(hidden_states).view(
            batch_size, num_queries, self.n_heads, self.n_levels, self.n_points, 2
        )
        attention_weights = self.attention_weights(hidden_states).view(
            batch_size, num_queries, self.n_heads, self.n_levels * self.n_points
        )
        attention_weights = F.softmax(attention_weights, -1).view(
            batch_size, num_queries, self.n_heads, self.n_levels, self.n_points
        )
        # batch_size, num_queries, n_heads, n_levels, n_points, 2
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
            )
        else:
            raise ValueError(f"Last dim of reference_points must be 2 or 4, but got {reference_points.shape[-1]}")

        if self.disable_custom_kernels:
            # PyTorch implementation
            output = multi_scale_deformable_attention(value, spatial_shapes, sampling_locations, attention_weights)
        else:
            if value.dtype != torch.float32:
                value = value.to(torch.float32)
                attention_weights = attention_weights.to(torch.float32)
            try:
                # custom kernel
                output = MultiScaleDeformableAttentionFunction.apply(
                    value,
                    spatial_shapes,
                    level_start_index,
                    sampling_locations,
                    attention_weights,
                    self.im2col_step,
                )
            except Exception:
                # PyTorch implementation
                output = multi_scale_deformable_attention(value, spatial_shapes, sampling_locations, attention_weights)

        output = output.to(self.output_proj.weight.dtype)
        output = self.output_proj(output)

        return output, attention_weights


class GroundingDinoTextEnhancerLayer(nn.Module):
    """Vanilla Transformer with text embeddings as input"""

    def __init__(self, config):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=config.encoder_attention_heads // 2,
            dropout=config.text_enhancer_dropout,
            batch_first=True,
        )
        # Implementation of Feedforward model
        self.fc1 = nn.Linear(config.d_model, config.encoder_ffn_dim // 2)
        self.fc2 = nn.Linear(config.encoder_ffn_dim // 2, config.d_model)

        self.layer_norm_before = nn.LayerNorm(config.d_model)
        self.layer_norm_after = nn.LayerNorm(config.d_model)

        self.activation = ACT2FN[config.activation_function]
        self.num_heads = config.encoder_attention_heads // 2
        self.dropout = config.text_enhancer_dropout

    def with_pos_embed(self, hidden_state: Tensor, position_embeddings: Optional[Tensor]):
        return hidden_state if position_embeddings is None else hidden_state + position_embeddings

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_masks: Optional[torch.BoolTensor] = None,
        position_embeddings: Optional[torch.FloatTensor] = None,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """Text self-attention to enhance projection of text features generated by
        the text encoder (GroundingDinoTextPrenet) within GroundingDinoEncoderLayer
        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_dim)`):
                Text features generated by the text encoder.
            attention_masks (`torch.BoolTensor`, *optional*):
                Attention mask for text self-attention. False for real tokens and True for padding tokens.
            position_embeddings (`torch.FloatTensor`, *optional*):
                Position embeddings to be added to the hidden states.
        Returns:
            `tuple(torch.FloatTensor)` comprising two elements:
            - **hidden_states** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) --
                Output of the text self-attention layer.
            - **attention_weights** (`torch.FloatTensor` of shape `(batch_size, num_heads, sequence_length,
              sequence_length)`) --
                Attention weights of the text self-attention layer.
        """

        # repeat attn mask
        if attention_masks.dim() == 3 and attention_masks.shape[0] == hidden_states.shape[0]:
            # bs, num_q, num_k
            attention_masks = attention_masks.repeat(self.num_heads, 1, 1)

        queries = keys = self.with_pos_embed(hidden_states, position_embeddings)
        attention_output, attention_weights = self.self_attn(
            query=queries, key=keys, value=hidden_states, attn_mask=attention_masks, average_attn_weights=False
        )
        attention_output = nn.functional.dropout(attention_output, p=self.dropout, training=self.training)
        hidden_states = hidden_states + attention_output
        hidden_states = self.layer_norm_before(hidden_states)

        residual = hidden_states
        hidden_states = self.activation(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = hidden_states + residual
        hidden_states = self.layer_norm_after(hidden_states)

        return hidden_states, attention_weights


class GroundingDinoBiMultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        vision_dim = text_dim = config.d_model
        embed_dim = config.encoder_ffn_dim // 2
        num_heads = config.encoder_attention_heads // 2
        dropout = config.fusion_dropout

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.vision_dim = vision_dim
        self.text_dim = text_dim

        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by `num_heads` (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
            )
        self.scale = self.head_dim ** (-0.5)
        self.dropout = dropout

        self.vision_proj = nn.Linear(self.vision_dim, self.embed_dim)
        self.text_proj = nn.Linear(self.text_dim, self.embed_dim)
        self.values_vision_proj = nn.Linear(self.vision_dim, self.embed_dim)
        self.values_text_proj = nn.Linear(self.text_dim, self.embed_dim)

        self.out_vision_proj = nn.Linear(self.embed_dim, self.vision_dim)
        self.out_text_proj = nn.Linear(self.embed_dim, self.text_dim)

    def _shape(self, tensor: torch.Tensor, seq_len: int, batch_size: int):
        return tensor.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        vision_features: torch.FloatTensor,
        text_features: torch.FloatTensor,
        vision_attention_mask: Optional[torch.BoolTensor] = None,
        text_attention_mask: Optional[torch.BoolTensor] = None,
    ) -> Tuple[Tuple[torch.FloatTensor, torch.FloatTensor], Tuple[torch.FloatTensor, torch.FloatTensor]]:
        """Image-to-text and text-to-image cross-attention
        Args:
            vision_features (`torch.FloatTensor` of shape `(batch_size, vision_sequence_length, hidden_dim)`):
                Projected flattened image features generated by the vision backbone.
            text_features (`torch.FloatTensor` of shape `(batch_size, text_sequence_length, hidden_dim)`):
                Projected text features generated by the text encoder.
            vision_attention_mask (`torch.BoolTensor`, **optional**):
                Attention mask for image-to-text cross-attention. False for real tokens and True for padding tokens.
            text_attention_mask (`torch.BoolTensor`, **optional**):
                Attention mask for text-to-image cross-attention. False for real tokens and True for padding tokens.
        Returns:
            `tuple(tuple(torch.FloatTensor), tuple(torch.FloatTensor))` where each inner tuple comprises an attention
            output and weights:
            - **vision_attn_output** (`torch.FloatTensor` of shape `(batch_size, vision_sequence_length, hidden_din)`)
              --
                Output of the image-to-text cross-attention layer.
            - **vision_attn_weights** (`torch.FloatTensor` of shape `(batch_size, num_heads, vision_sequence_length,
              vision_sequence_length)`) --
                Attention weights of the image-to-text cross-attention layer.
            - **text_attn_output** (`torch.FloatTensor` of shape `(batch_size, text_sequence_length, hidden_dim)`) --
                Output of the text-to-image cross-attention layer.
            - **text_attn_weights** (`torch.FloatTensor` of shape `(batch_size, num_heads, text_sequence_length,
              text_sequence_length)`) --
                Attention weights of the text-to-image cross-attention layer.
        """
        batch_size, tgt_len, _ = vision_features.size()

        vision_query_states = self.vision_proj(vision_features) * self.scale
        vision_query_states = self._shape(vision_query_states, tgt_len, batch_size)

        text_key_states = self.text_proj(text_features)
        text_key_states = self._shape(text_key_states, -1, batch_size)

        vision_value_states = self.values_vision_proj(vision_features)
        vision_value_states = self._shape(vision_value_states, -1, batch_size)

        text_value_states = self.values_text_proj(text_features)
        text_value_states = self._shape(text_value_states, -1, batch_size)

        proj_shape = (batch_size * self.num_heads, -1, self.head_dim)

        vision_query_states = vision_query_states.view(*proj_shape)
        text_key_states = text_key_states.view(*proj_shape)
        vision_value_states = vision_value_states.view(*proj_shape)
        text_value_states = text_value_states.view(*proj_shape)

        src_len = text_key_states.size(1)
        attn_weights = torch.bmm(vision_query_states, text_key_states.transpose(1, 2))  # bs*nhead, nimg, ntxt

        if attn_weights.size() != (batch_size * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(batch_size * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"
            )

        attn_weights = attn_weights - attn_weights.max()
        # Do not increase -50000/50000, data type half has quite limited range
        attn_weights = torch.clamp(attn_weights, min=-50000, max=50000)

        attn_weights_transposed = attn_weights.transpose(1, 2)
        text_attn_weights = attn_weights_transposed - torch.max(attn_weights_transposed, dim=-1, keepdim=True)[0]

        # Do not increase -50000/50000, data type half has quite limited range
        text_attn_weights = torch.clamp(text_attn_weights, min=-50000, max=50000)

        # mask vision for language
        if vision_attention_mask is not None:
            vision_attention_mask = (
                vision_attention_mask[:, None, None, :].repeat(1, self.num_heads, 1, 1).flatten(0, 1)
            )
            text_attn_weights.masked_fill_(vision_attention_mask, float("-inf"))

        text_attn_weights = text_attn_weights.softmax(dim=-1)

        # mask language for vision
        if text_attention_mask is not None:
            text_attention_mask = text_attention_mask[:, None, None, :].repeat(1, self.num_heads, 1, 1).flatten(0, 1)
            attn_weights.masked_fill_(text_attention_mask, float("-inf"))
        vision_attn_weights = attn_weights.softmax(dim=-1)

        vision_attn_probs = F.dropout(vision_attn_weights, p=self.dropout, training=self.training)
        text_attn_probs = F.dropout(text_attn_weights, p=self.dropout, training=self.training)

        vision_attn_output = torch.bmm(vision_attn_probs, text_value_states)
        text_attn_output = torch.bmm(text_attn_probs, vision_value_states)

        if vision_attn_output.size() != (batch_size * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`vision_attn_output` should be of size {(batch_size, self.num_heads, tgt_len, self.head_dim)}, but is {vision_attn_output.size()}"
            )

        if text_attn_output.size() != (batch_size * self.num_heads, src_len, self.head_dim):
            raise ValueError(
                f"`text_attn_output` should be of size {(batch_size, self.num_heads, src_len, self.head_dim)}, but is {text_attn_output.size()}"
            )

        vision_attn_output = vision_attn_output.view(batch_size, self.num_heads, tgt_len, self.head_dim)
        vision_attn_output = vision_attn_output.transpose(1, 2)
        vision_attn_output = vision_attn_output.reshape(batch_size, tgt_len, self.embed_dim)

        text_attn_output = text_attn_output.view(batch_size, self.num_heads, src_len, self.head_dim)
        text_attn_output = text_attn_output.transpose(1, 2)
        text_attn_output = text_attn_output.reshape(batch_size, src_len, self.embed_dim)

        vision_attn_output = self.out_vision_proj(vision_attn_output)
        text_attn_output = self.out_text_proj(text_attn_output)

        return (vision_attn_output, vision_attn_weights), (text_attn_output, text_attn_weights)


# Copied from transformers.models.beit.modeling_beit.drop_path
def drop_path(input: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    Comment by Ross Wightman: This is the same as the DropConnect impl I created for EfficientNet, etc networks,
    however, the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for changing the
    layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use 'survival rate' as the
    argument.
    """
    if drop_prob == 0.0 or not training:
        return input
    keep_prob = 1 - drop_prob
    shape = (input.shape[0],) + (1,) * (input.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=input.dtype, device=input.device)
    random_tensor.floor_()  # binarize
    output = input.div(keep_prob) * random_tensor
    return output


# Copied from transformers.models.beit.modeling_beit.BeitDropPath with Beit->GroundingDino
class GroundingDinoDropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: Optional[float] = None) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return drop_path(hidden_states, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


class GroundingDinoFusionLayer(nn.Module):
    def __init__(self, config, init_values=1e-4):
        super().__init__()
        drop_path = config.fusion_droppath

        # pre layer norm
        self.layer_norm_vision = nn.LayerNorm(config.d_model)
        self.layer_norm_text = nn.LayerNorm(config.d_model)
        self.attn = GroundingDinoBiMultiHeadAttention(config)

        # add layer scale for training stability
        self.drop_path = GroundingDinoDropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.vision_param = nn.Parameter(init_values * torch.ones((config.d_model)), requires_grad=True)
        self.text_param = nn.Parameter(init_values * torch.ones((config.d_model)), requires_grad=True)

    def forward(
        self,
        vision_features: torch.FloatTensor,
        text_features: torch.FloatTensor,
        attention_mask_vision: Optional[torch.BoolTensor] = None,
        attention_mask_text: Optional[torch.BoolTensor] = None,
    ) -> Tuple[Tuple[torch.FloatTensor, torch.FloatTensor], Tuple[torch.FloatTensor, torch.FloatTensor]]:
        """Image and text features fusion
        Args:
            vision_features (`torch.FloatTensor` of shape `(batch_size, vision_sequence_length, hidden_dim)`):
                Projected flattened image features generated by the vision backbone.
            text_features (`torch.FloatTensor` of shape `(batch_size, text_sequence_length, hidden_dim)`):
                Projected text features generated by the text encoder.
            attention_mask_vision (`torch.BoolTensor`, **optional**):
                Attention mask for image-to-text cross-attention. False for real tokens and True for padding tokens.
            attention_mask_text (`torch.BoolTensor`, **optional**):
                Attention mask for text-to-image cross-attention. False for real tokens and True for padding tokens.
        Returns:
            `tuple(tuple(torch.FloatTensor), tuple(torch.FloatTensor))` where each inner tuple comprises an enhanced
            feature and attention output and weights:
            - **vision_features** (`torch.FloatTensor` of shape `(batch_size, vision_sequence_length, vision_dim)`) --
                Updated vision features with attention output from image-to-text cross-attention layer.
            - **vision_attn_weights** (`torch.FloatTensor` of shape `(batch_size, num_heads, vision_sequence_length,
              vision_sequence_length)`) --
                Attention weights of the image-to-text cross-attention layer.
            - **text_features** (`torch.FloatTensor` of shape `(batch_size, text_sequence_length, text_dim)`) --
                Updated text features with attention output from text-to-image cross-attention layer.
            - **text_attn_weights** (`torch.FloatTensor` of shape `(batch_size, num_heads, text_sequence_length,
              text_sequence_length)`) --
                Attention weights of the text-to-image cross-attention layer.
        """
        vision_features = self.layer_norm_vision(vision_features)
        text_features = self.layer_norm_text(text_features)
        (delta_v, vision_attn), (delta_t, text_attn) = self.attn(
            vision_features,
            text_features,
            vision_attention_mask=attention_mask_vision,
            text_attention_mask=attention_mask_text,
        )
        vision_features = vision_features + self.drop_path(self.vision_param * delta_v)
        text_features = text_features + self.drop_path(self.text_param * delta_t)

        return (vision_features, vision_attn), (text_features, text_attn)


# NOTE just renamed the class
class GroundingDinoDeformableLayer(nn.Module):
    def __init__(self, config: GroundingDinoConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = GroundingDinoMultiscaleDeformableAttention(
            config, num_heads=config.encoder_attention_heads, n_points=config.encoder_n_points
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_embeddings: torch.Tensor = None,
        reference_points=None,
        spatial_shapes=None,
        level_start_index=None,
        output_attentions: bool = False,
    ):
        """
        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Input to the layer.
            attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
                Attention mask.
            position_embeddings (`torch.FloatTensor`, *optional*):
                Position embeddings, to be added to `hidden_states`.
            reference_points (`torch.FloatTensor`, *optional*):
                Reference points.
            spatial_shapes (`torch.LongTensor`, *optional*):
                Spatial shapes of the backbone feature maps.
            level_start_index (`torch.LongTensor`, *optional*):
                Level start index.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        # Apply Multi-scale Deformable Attention Module on the multi-scale feature maps.
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            output_attentions=output_attentions,
        )

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)

        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        if self.training:
            if torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any():
                clamp_value = torch.finfo(hidden_states.dtype).max - 1000
                hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        return hidden_states, attn_weights


def get_sine_pos_embed(
    pos_tensor: torch.Tensor,
    num_pos_feats: int = 128,
    temperature: int = 10000,
    exchange_xy: bool = True,
) -> Tensor:
    """generate sine position embedding from a position tensor
    Args:
        pos_tensor (torch.Tensor): shape: [..., n].
        num_pos_feats (int): projected shape for each float in the tensor.
        temperature (int): temperature in the sine/cosine function.
        exchange_xy (bool, optional): exchange pos x and pos y. \
            For example, input tensor is [x,y], the results will be [pos(y), pos(x)]. Defaults to True.
    Returns:
        pos_embed (torch.Tensor): shape: [..., n*num_pos_feats].
    """
    scale = 2 * math.pi
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos_tensor.device)
    dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / num_pos_feats)

    def sine_func(x: torch.Tensor):
        sin_x = x * scale / dim_t
        sin_x = torch.stack((sin_x[..., 0::2].sin(), sin_x[..., 1::2].cos()), dim=3).flatten(2)
        return sin_x

    pos_res = [sine_func(x) for x in pos_tensor.split([1] * pos_tensor.shape[-1], dim=-1)]
    if exchange_xy:
        pos_res[0], pos_res[1] = pos_res[1], pos_res[0]
    pos_res = torch.cat(pos_res, dim=-1)
    return pos_res


class GroundingDinoEncoderLayer(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

        self.d_model = config.d_model

        self.text_enhancer_layer = GroundingDinoTextEnhancerLayer(config)
        self.fusion_layer = GroundingDinoFusionLayer(config)
        self.deformable_layer = GroundingDinoDeformableLayer(config)

    def get_text_position_embeddings(
        self, text_features: Tensor, text_position_embedding: Tensor, text_position_ids: Tensor
    ) -> Tensor:
        batch_size, seq_length, _ = text_features.shape
        if text_position_embedding is None and text_position_ids is None:
            text_position_embedding = torch.arange(seq_length, device=text_features.device)
            text_position_embedding = text_position_embedding.float()
            text_position_embedding = text_position_embedding.unsqueeze(0).unsqueeze(-1)
            text_position_embedding = text_position_embedding.repeat(batch_size, 1, 1)
            text_position_embedding = get_sine_pos_embed(
                text_position_embedding, num_pos_feats=self.d_model, exchange_xy=False
            )
        if text_position_ids is not None:
            text_position_embedding = get_sine_pos_embed(
                text_position_ids[..., None], num_pos_feats=self.d_model, exchange_xy=False
            )

        return text_position_embedding

    def forward(
        self,
        vision_features: Tensor,
        vision_position_embedding: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        key_padding_mask: Tensor,
        reference_points: Tensor,
        text_features: Optional[Tensor] = None,
        text_attention_mask: Optional[Tensor] = None,
        text_position_embedding: Optional[Tensor] = None,
        text_self_attention_masks: Optional[Tensor] = None,
        text_position_ids: Optional[Tensor] = None,
    ):
        text_position_embedding = self.get_text_position_embeddings(
            text_features, text_position_embedding, text_position_ids
        ).to(vision_features.dtype)

        (vision_features, vision_fused_attn), (text_features, text_fused_attn) = self.fusion_layer(
            vision_features=vision_features,
            text_features=text_features,
            attention_mask_vision=key_padding_mask,
            attention_mask_text=text_attention_mask,
        )

        (text_features, text_enhanced_attn) = self.text_enhancer_layer(
            hidden_states=text_features,
            attention_masks=~text_self_attention_masks,  # note we use ~ for mask here
            position_embeddings=(text_position_embedding if text_position_embedding is not None else None),
        )

        (vision_features, vision_deformable_attn) = self.deformable_layer(
            hidden_states=vision_features,
            attention_mask=~key_padding_mask,
            position_embeddings=vision_position_embedding,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
        )

        return (
            (vision_features, text_features),
            (vision_fused_attn, text_fused_attn, text_enhanced_attn, vision_deformable_attn),
        )


class GroundingDinoDecoderLayer(nn.Module):
    def __init__(self, config: GroundingDinoConfig):
        super().__init__()
        self.embed_dim = config.d_model

        # self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            batch_first=True,
        )
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        # cross-attention text
        self.encoder_attn_text = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            batch_first=True,
        )
        self.encoder_attn_text_layer_norm = nn.LayerNorm(self.embed_dim)
        # cross-attention
        self.encoder_attn = GroundingDinoMultiscaleDeformableAttention(
            config,
            num_heads=config.decoder_attention_heads,
            n_points=config.decoder_n_points,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        # feedforward neural networks
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def with_pos_embed(self, tensor: torch.Tensor, position_embeddings: Optional[Tensor]):
        return tensor if position_embeddings is None else tensor + position_embeddings

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[torch.Tensor] = None,
        reference_points=None,
        spatial_shapes=None,
        level_start_index=None,
        vision_encoder_hidden_states: Optional[torch.Tensor] = None,
        vision_encoder_attention_mask: Optional[torch.Tensor] = None,
        text_encoder_hidden_states: Optional[torch.Tensor] = None,
        text_encoder_attention_mask: Optional[torch.Tensor] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ):
        residual = hidden_states

        # Self Attention
        q = k = self.with_pos_embed(hidden_states, position_embeddings)
        hidden_states, self_attn_weights = self.self_attn(
            query=q, key=k, value=hidden_states, attn_mask=self_attn_mask, average_attn_weights=False
        )

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        second_residual = hidden_states

        # Cross-Attention Text
        hidden_states, text_cross_attn_weights = self.encoder_attn_text(
            query=self.with_pos_embed(hidden_states, position_embeddings),
            key=text_encoder_hidden_states,
            value=text_encoder_hidden_states,
            key_padding_mask=text_encoder_attention_mask,
            average_attn_weights=False,
        )

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = second_residual + hidden_states
        hidden_states = self.encoder_attn_text_layer_norm(hidden_states)

        third_residual = hidden_states

        # Cross-Attention
        cross_attn_weights = None
        hidden_states, cross_attn_weights = self.encoder_attn(
            hidden_states=hidden_states,
            attention_mask=vision_encoder_attention_mask,
            encoder_hidden_states=vision_encoder_hidden_states,
            encoder_attention_mask=vision_encoder_attention_mask,
            position_embeddings=position_embeddings,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            output_attentions=output_attentions,
        )

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = third_residual + hidden_states
        hidden_states = self.encoder_attn_layer_norm(hidden_states)

        # Fully Connected
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, text_cross_attn_weights, cross_attn_weights)

        return outputs


class GroundingDinoContrastiveEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.max_text_len = config.max_text_len

    def forward(
        self,
        vision_hidden_state: torch.FloatTensor,
        text_hidden_state: torch.FloatTensor,
        text_token_mask: torch.BoolTensor,
    ) -> torch.FloatTensor:
        output = vision_hidden_state @ text_hidden_state.transpose(-1, -2)  # [bs, num_queries, num_patches]
        output.masked_fill_(~text_token_mask[:, None, :], float("-inf"))

        # padding to max_text_len
        new_output = torch.full((*output.shape[:-1], self.max_text_len), float("-inf"), device=output.device)
        new_output[..., : output.shape[-1]] = output  # [bs, num_patches, 256], text_token=256

        return new_output


# Copied from transformers.models.detr.modeling_detr.DetrClassificationHead
class GroundingDinoClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, input_dim: int, inner_dim: int, num_classes: int, pooler_dropout: float):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states


class GroundingDinoPreTrainedModel(PreTrainedModel):
    config_class = GroundingDinoConfig
    base_model_prefix = "model"
    main_input_name = "pixel_values"

    def _init_weights(self, module):
        std = self.config.init_std

        if isinstance(module, GroundingDinoLearnedPositionEmbedding):
            nn.init.uniform_(module.row_embeddings.weight)
            nn.init.uniform_(module.column_embeddings.weight)
        elif isinstance(module, GroundingDinoMultiscaleDeformableAttention):
            module._reset_parameters()
        elif isinstance(module, GroundingDinoBiMultiHeadAttention):
            nn.init.xavier_uniform_(module.vision_proj.weight)
            module.vision_proj.bias.data.fill_(0)
            nn.init.xavier_uniform_(module.text_proj.weight)
            module.text_proj.bias.data.fill_(0)
            nn.init.xavier_uniform_(module.values_vision_proj.weight)
            module.values_vision_proj.bias.data.fill_(0)
            nn.init.xavier_uniform_(module.values_text_proj.weight)
            module.values_text_proj.bias.data.fill_(0)
            nn.init.xavier_uniform_(module.out_vision_proj.weight)
            module.out_vision_proj.bias.data.fill_(0)
            nn.init.xavier_uniform_(module.out_text_proj.weight)
            module.out_text_proj.bias.data.fill_(0)
        elif isinstance(module, (GroundingDinoEncoderLayer, GroundingDinoDecoderLayer)):
            for p in module.parameters():
                if p.dim() > 1:
                    nn.init.normal_(p, mean=0.0, std=std)
        elif isinstance(module, (nn.Linear, nn.Conv2d, nn.BatchNorm2d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, GroundingDinoMLPPredictionHead):
            nn.init.constant_(module.layers[-1].weight.data, 0)
            nn.init.constant_(module.layers[-1].bias.data, 0)

        if hasattr(module, "reference_points") and not self.config.two_stage:
            nn.init.xavier_uniform_(module.reference_points.weight.data, gain=1.0)
            nn.init.constant_(module.reference_points.bias.data, 0.0)
        if hasattr(module, "level_embed"):
            nn.init.normal_(module.level_embed)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, GroundingDinoDecoder):
            module.gradient_checkpointing = value


GROUNDING_DINO_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)
    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.
    Parameters:
        config ([`GroundingDinoConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

GROUNDING_DINO_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Padding will be ignored by default should you provide it.
            Pixel values can be obtained using [`AutoImageProcessor`]. See [`GroundingDinoImageProcessor.__call__`] for
            details.
        pixel_mask (`torch.LongTensor` of shape `(batch_size, height, width)`, *optional*):
            Mask to avoid performing attention on padding pixel values. Mask values selected in `[0, 1]`:
            - 1 for pixels that are real (i.e. **not masked**),
            - 0 for pixels that are padding (i.e. **masked**).
            [What are attention masks?](../glossary#attention-mask)
        input_ids (`torch.LongTensor` of shape `(batch_size, text_sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.
            Indices can be obtained using [`AutoTokenizer`]. See [`GroundingDinoTokenizer.__call__`] for details.
        attention_mask (`torch.LongTensor` of shape `(batch_size, text_sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
            - 1 for tokens that are real (i.e. **not masked**),
            - 0 for tokens that are padding (i.e. **masked**).
            [What are attention masks?](../glossary#attention-mask)
        token_type_ids (`torch.LongTensor` of shape `(batch_size, text_sequence_length)`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
            1]`: 0 corresponds to a `sentence A` token, 1 corresponds to a `sentence B` token
            [What are token type IDs?](../glossary#token-type-ids)
        encoder_outputs (`tuple(tuple(torch.FloatTensor)`, *optional*):
            Tuple consists of (`last_hidden_state_vision`, *optional*: `last_hidden_state_text`, *optional*:
            `hidden_states_vision`, *optional*: `hidden_states_text`, *optional*: `attentions`)
            `last_hidden_state_vision` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) is a sequence
            of hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the
            decoder.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple.
"""


class GroundingDinoEncoder(GroundingDinoPreTrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* deformable attention layers. Each layer is a
    [`GroundingDinoEncoderLayer`].
    The encoder updates the flattened multi-scale feature maps through multiple deformable attention layers.
    Args:
        config: GroundingDinoConfig
    """

    def __init__(self, config: GroundingDinoConfig):
        super().__init__(config)

        self.dropout = config.dropout
        self.layers = nn.ModuleList([GroundingDinoEncoderLayer(config) for _ in range(config.encoder_layers)])

        # Initialize weights and apply final processing
        self.post_init()

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        """
        Get reference points for each feature map. Used in decoder.
        Args:
            spatial_shapes (`torch.LongTensor` of shape `(num_feature_levels, 2)`):
                Spatial shapes of each feature map.
            valid_ratios (`torch.FloatTensor` of shape `(batch_size, num_feature_levels, 2)`):
                Valid ratios of each feature map.
            device (`torch.device`):
                Device on which to create the tensors.
        Returns:
            `torch.FloatTensor` of shape `(batch_size, num_queries, num_feature_levels, 2)`
        """
        reference_points_list = []
        for level, (height, width) in enumerate(spatial_shapes):
            ref_y, ref_x = meshgrid(
                torch.linspace(0.5, height - 0.5, height, dtype=torch.float32, device=device),
                torch.linspace(0.5, width - 0.5, width, dtype=torch.float32, device=device),
                indexing="ij",
            )
            # TODO: valid_ratios could be useless here. check https://github.com/fundamentalvision/Deformable-DETR/issues/36
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, level, 1] * height)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, level, 0] * width)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(
        self,
        vision_features: Tensor,
        vision_attention_mask: Tensor,
        vision_position_embedding: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        valid_ratios=None,
        text_features: Optional[Tensor] = None,
        text_attention_mask: Optional[Tensor] = None,
        text_position_embedding: Optional[Tensor] = None,
        text_self_attention_masks: Optional[Tensor] = None,
        text_position_ids: Optional[Tensor] = None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Args:
            vision_features (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Flattened feature map (output of the backbone + projection layer) that is passed to the encoder.
            vision_attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding pixel features. Mask values selected in `[0, 1]`:
                - 0 for pixel features that are real (i.e. **not masked**),
                - 1 for pixel features that are padding (i.e. **masked**).
                [What are attention masks?](../glossary#attention-mask)
            vision_position_embedding (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Position embeddings that are added to the queries and keys in each self-attention layer.
            spatial_shapes (`torch.LongTensor` of shape `(num_feature_levels, 2)`):
                Spatial shapes of each feature map.
            level_start_index (`torch.LongTensor` of shape `(num_feature_levels)`):
                Starting index of each feature map.
            valid_ratios (`torch.FloatTensor` of shape `(batch_size, num_feature_levels, 2)`):
                Ratio of valid area in each feature level.
            text_features (`torch.FloatTensor` of shape `(batch_size, text_seq_len, hidden_size)`):
                Flattened text features that are passed to the encoder.
            text_attention_mask (`torch.Tensor` of shape `(batch_size, text_seq_len)`, *optional*):
                Mask to avoid performing attention on padding text features. Mask values selected in `[0, 1]`:
                - 0 for text features that are real (i.e. **not masked**),
                - 1 for text features that are padding (i.e. **masked**).
                [What are attention masks?](../glossary#attention-mask)
            text_position_embedding (`torch.FloatTensor` of shape `(batch_size, text_seq_len)`):
                Position embeddings that are added to the queries and keys in each self-attention layer.
            text_self_attention_masks (`torch.BoolTensor` of shape `(batch_size, text_seq_len, text_seq_len)`):
                Masks to avoid performing attention between padding text features. Mask values selected in `[0, 1]`:
                - 1 for text features that are real (i.e. **not masked**),
                - 0 for text features that are padding (i.e. **masked**).
            text_position_ids (`torch.LongTensor` of shape `(batch_size, num_queries)`):
                Position ids for text features.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=vision_features.device)

        encoder_vision_states = () if output_hidden_states else None
        encoder_text_states = () if output_hidden_states else None
        all_attns = () if output_attentions else None
        all_attn_fused_text = () if output_attentions else None
        all_attn_fused_vision = () if output_attentions else None
        all_attn_enhanced_text = () if output_attentions else None
        all_attn_deformable = () if output_attentions else None
        for i, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_vision_states += (vision_features,)
                encoder_text_states += (text_features,)

            (vision_features, text_features), attentions = encoder_layer(
                vision_features=vision_features,
                vision_position_embedding=vision_position_embedding,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                key_padding_mask=vision_attention_mask,
                reference_points=reference_points,
                text_features=text_features,
                text_attention_mask=text_attention_mask,
                text_position_embedding=text_position_embedding,
                text_self_attention_masks=text_self_attention_masks,
                text_position_ids=text_position_ids,
            )

            if output_attentions:
                all_attn_fused_vision += (attentions[0],)
                all_attn_fused_text += (attentions[1],)
                all_attn_enhanced_text += (attentions[2],)
                all_attn_deformable += (attentions[3],)

        if output_hidden_states:
            encoder_vision_states += (vision_features,)
            encoder_text_states += (text_features,)

        if output_attentions:
            all_attns = (all_attn_fused_vision, all_attn_fused_text, all_attn_enhanced_text, all_attn_deformable)

        if not return_dict:
            enc_outputs = [vision_features, text_features, encoder_vision_states, encoder_text_states, all_attns]
            return tuple(v for v in enc_outputs if v is not None)
        return GroundingDinoEncoderOutput(
            last_hidden_state_vision=vision_features,
            last_hidden_state_text=text_features,
            hidden_states_vision=encoder_vision_states,
            hidden_states_text=encoder_text_states,
            attentions=all_attns,
        )


class GroundingDinoDecoder(GroundingDinoPreTrainedModel):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`GroundingDinoDecoderLayer`].
    The decoder updates the query embeddings through multiple self-attention and cross-attention layers.
    Some tweaks for Grounding DINO:
    - `position_embeddings`, `reference_points`, `spatial_shapes` and `valid_ratios` are added to the forward pass.
    - it also returns a stack of intermediate outputs and reference points from all decoding layers.
    Args:
        config: GroundingDinoConfig
    """

    def __init__(self, config: GroundingDinoConfig):
        super().__init__(config)

        self.dropout = config.dropout
        self.layer_norm = nn.LayerNorm(config.d_model)
        self.layers = nn.ModuleList([GroundingDinoDecoderLayer(config) for _ in range(config.decoder_layers)])
        self.reference_points_head = GroundingDinoMLPPredictionHead(
            config.query_dim // 2 * config.d_model, config.d_model, config.d_model, 2
        )
        self.gradient_checkpointing = False

        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.bbox_embed = None
        self.class_embed = None
        self.query_scale = None

        # Initialize weights and apply final processing
        self.post_init()

    def get_proposal_pos_embed(self, proposals: torch.FloatTensor) -> torch.FloatTensor:
        """Get the position embedding of the proposals."""
        num_pos_feats = self.config.d_model // 2
        temperature = 10000
        scale = 2 * math.pi

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / num_pos_feats)
        # batch_size, num_queries
        pos_x = proposals[:, :, 0] * scale
        pos_y = proposals[:, :, 1] * scale
        # batch_size, num_queries, num_pos_feats
        pos_x = pos_x[:, :, None] / dim_t
        pos_y = pos_y[:, :, None] / dim_t
        # batch_size, num_queries, num_pos_feats
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
        pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)

        if proposals.size(-1) == 2:
            # batch_size, num_queries, num_pos_feats * 2
            pos = torch.cat((pos_y, pos_x), dim=2)
        elif proposals.size(-1) == 4:
            w_embed = proposals[:, :, 2] * scale
            pos_w = w_embed[:, :, None] / dim_t
            # batch_size, num_queries, num_pos_feats
            pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)

            h_embed = proposals[:, :, 3] * scale
            pos_h = h_embed[:, :, None] / dim_t
            # batch_size, num_queries, num_pos_feats
            pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)
            # batch_size, num_queries, num_pos_feats * 4
            pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
        else:
            raise ValueError("Unknown proposals shape(-1):{}".format(proposals.size(-1)))
        return pos

    def forward(
        self,
        inputs_embeds,
        vision_encoder_hidden_states,
        mask_features,
        vision_encoder_attention_mask=None,
        text_encoder_hidden_states=None,
        text_encoder_attention_mask=None,
        reference_points=None,
        spatial_shapes=None,
        level_start_index=None,
        valid_ratios=None,
        self_attn_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_size)`):
                The query embeddings that are passed into the decoder.
            vision_encoder_hidden_states (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Last hidden state from encoder related to vision feature map.
            vision_encoder_attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding pixel features. Mask values selected in `[0, 1]`:
                - 1 for pixel features that are real (i.e. **not masked**),
                - 0 for pixel features that are padding (i.e. **masked**).
            text_encoder_hidden_states (`torch.FloatTensor` of shape `(batch_size, text_seq_len, hidden_size)`):
                Last hidden state from encoder related to text features.
            text_encoder_attention_mask (`torch.Tensor` of shape `(batch_size, text_seq_len)`, *optional*):
                Mask to avoid performing attention on padding text features. Mask values selected in `[0, 1]`:
                - 0 for text features that are real (i.e. **not masked**),
                - 1 for text features that are padding (i.e. **masked**).
            reference_points (`torch.FloatTensor` of shape `(batch_size, num_queries, 4)` is `as_two_stage` else `(batch_size, num_queries, 2)` or , *optional*):
                Reference point in range `[0, 1]`, top-left (0,0), bottom-right (1, 1), including padding area.
            spatial_shapes (`torch.FloatTensor` of shape `(num_feature_levels, 2)`):
                Spatial shapes of the feature maps.
            level_start_index (`torch.LongTensor` of shape `(num_feature_levels)`, *optional*):
                Indexes for the start of each feature level. In range `[0, sequence_length]`.
            valid_ratios (`torch.FloatTensor` of shape `(batch_size, num_feature_levels, 2)`, *optional*):
                Ratio of valid area in each feature level.
            self_attn_mask (`torch.BoolTensor` of shape `(batch_size, text_seq_len)`):
                Masks to avoid performing self-attention between vision hidden state. Mask values selected in `[0, 1]`:
                - 1 for queries that are real (i.e. **not masked**),
                - 0 for queries that are padding (i.e. **masked**).
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is not None:
            hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_attns = () if output_attentions else None
        all_cross_attns_vision = () if (output_attentions and vision_encoder_hidden_states is not None) else None
        all_cross_attns_text = () if (output_attentions and text_encoder_hidden_states is not None) else None
        intermediate = ()
        intermediate_reference_points = ()

        for idx, decoder_layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = (
                    reference_points[:, :, None] * torch.cat([valid_ratios, valid_ratios], -1)[:, None]
                )
            else:
                if reference_points.shape[-1] != 2:
                    raise ValueError("Reference points' last dimension must be of size 2")
                reference_points_input = reference_points[:, :, None] * valid_ratios[:, None]
            query_pos = self.get_proposal_pos_embed(reference_points_input[:, :, 0, :]).to(hidden_states.dtype)
            query_pos = self.reference_points_head(query_pos)

            # In original implementation they apply layer norm before outputting intermediate hidden states
            # Though that's not through between layers so the layers use as input the output of the previous layer
            # withtout layer norm
            if output_hidden_states:
                all_hidden_states += (self.layer_norm(hidden_states),)

            gradient_checkpointing = False  # not supported now
            # if self.gradient_checkpointing and self.training:
            if gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    query_pos,
                    reference_points_input,
                    spatial_shapes,
                    level_start_index,
                    vision_encoder_hidden_states,
                    vision_encoder_attention_mask,
                    text_encoder_hidden_states,
                    text_encoder_attention_mask,
                    self_attn_mask,
                    None,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states=hidden_states,
                    position_embeddings=query_pos,
                    reference_points=reference_points_input,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    vision_encoder_hidden_states=vision_encoder_hidden_states,
                    vision_encoder_attention_mask=vision_encoder_attention_mask,
                    text_encoder_hidden_states=text_encoder_hidden_states,
                    text_encoder_attention_mask=text_encoder_attention_mask,
                    self_attn_mask=self_attn_mask,
                    output_attentions=output_attentions,
                )

            hidden_states = layer_outputs[0]

            # hack implementation for iterative bounding box refinement
            if self.bbox_embed is not None:
                tmp = self.bbox_embed[idx](hidden_states)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    if reference_points.shape[-1] != 2:
                        raise ValueError(
                            f"Reference points' last dimension must be of size 2, but is {reference_points.shape[-1]}"
                        )
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            intermediate += (self.layer_norm(hidden_states),)
            intermediate_reference_points += (reference_points,)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if text_encoder_hidden_states is not None:
                    all_cross_attns_text += (layer_outputs[2],)

                if vision_encoder_hidden_states is not None:
                    all_cross_attns_vision += (layer_outputs[3],)

        # Keep batch_size as first dimension
        intermediate = torch.stack(intermediate, dim=1)
        intermediate_reference_points = torch.stack(intermediate_reference_points, dim=1)
        hidden_states = self.layer_norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if output_attentions:
            all_attns += (all_self_attns, all_cross_attns_text, all_cross_attns_vision)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    intermediate,
                    intermediate_reference_points,
                    all_hidden_states,
                    all_attns,
                ]
                if v is not None
            )
        return GroundingDinoDecoderOutput(
            last_hidden_state=hidden_states,
            intermediate_hidden_states=intermediate,
            intermediate_reference_points=intermediate_reference_points,
            hidden_states=all_hidden_states,
            attentions=all_attns,
        )


SPECIAL_TOKENS = [101, 102, 1012, 1029]


def generate_masks_with_special_tokens_and_transfer_map(input_ids: torch.LongTensor) -> Tuple[Tensor, Tensor]:
    """Generate attention mask between each pair of special tokens and positional ids.
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.
    Returns:
        `tuple(torch.Tensor)` comprising attention mask between each special tokens and position_ids:
        - **attention_mask** (`torch.BoolTensor` of shape `(batch_size, sequence_length, sequence_length)`)
        - **position_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`)
    """
    batch_size, num_token = input_ids.shape
    # special_tokens_mask: batch_size, num_token. 1 for special tokens. 0 for normal tokens
    special_tokens_mask = torch.zeros((batch_size, num_token), device=input_ids.device).bool()
    for special_token in SPECIAL_TOKENS:
        special_tokens_mask |= input_ids == special_token

    # idxs: each row is a list of indices of special tokens
    idxs = torch.nonzero(special_tokens_mask)

    # generate attention mask and positional ids
    attention_mask = torch.eye(num_token, device=input_ids.device).bool().unsqueeze(0).repeat(batch_size, 1, 1)
    position_ids = torch.zeros((batch_size, num_token), device=input_ids.device)
    previous_col = 0
    for i in range(idxs.shape[0]):
        row, col = idxs[i]
        if (col == 0) or (col == num_token - 1):
            attention_mask[row, col, col] = True
            position_ids[row, col] = 0
        else:
            attention_mask[row, previous_col + 1 : col + 1, previous_col + 1 : col + 1] = True
            position_ids[row, previous_col + 1 : col + 1] = torch.arange(
                0, col - previous_col, device=input_ids.device
            )

        previous_col = col

    return attention_mask, position_ids.to(torch.long)

def generate_masks_with_text_query_masks(text_query_masks):
    """
    Args:
        text_query_masks (bool): [bs, max_num_patches,], valid is 1
    Returns:
        self_attention_mask (bool): [bs, max_num_patches, max_num_patches], padding is 1
        position_ids (long): [bs, max_num_patches]
    """
    batch_size, num_token = text_query_masks.shape
    self_attention_mask = torch.eye(num_token, device=text_query_masks.device).bool().unsqueeze(0).repeat(batch_size, 1, 1)
    position_ids = torch.zeros((batch_size, num_token), dtype=torch.long, device=text_query_masks.device)

    for batch_idx in range(batch_size):
        num_valid = text_query_masks[batch_idx].sum()  # [num_patches_i,]
        self_attention_mask[batch_idx, :num_valid, :num_valid] = True
        position_ids[batch_idx, :num_valid] = torch.arange(0, num_valid, device=text_query_masks.device)

    return self_attention_mask, position_ids

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

@add_start_docstrings(
    """
    The bare Grounding DINO Model (consisting of a backbone and encoder-decoder Transformer) outputting raw
    hidden-states without any specific head on top.
    """,
    GROUNDING_DINO_START_DOCSTRING,
)
class OVGroundingDinoModel(GroundingDinoPreTrainedModel):
    def __init__(self, config: GroundingDinoConfig):
        super().__init__(config)

        # Create backbone + positional encoding
        config_dict = config.to_dict()
        if config_dict['backbone_config']['model_type'] == 'swin':
            backbone = GroundingDinoConvEncoder(config)
        elif config_dict['backbone_config']['model_type'] == 'internimage-H':
            backbone = GroundingDinoInternImageBackbone(config)
        else:
            raise NotImplementedError
        position_embeddings = build_position_encoding(config)
        self.backbone = GroundingDinoConvModel(backbone, position_embeddings)

        # Create input projection layers
        if config.num_feature_levels > 1:
            # 8x -> 32x
            num_backbone_outs = len(backbone.intermediate_channel_sizes[-3:])
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.intermediate_channel_sizes[-3:][_]
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, config.d_model, kernel_size=1),
                        nn.GroupNorm(32, config.d_model),
                    )
                )
            # input_proj on top vision feature
            for _ in range(config.num_feature_levels - num_backbone_outs): # 1
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, config.d_model, kernel_size=3, stride=2, padding=1),
                        nn.GroupNorm(32, config.d_model),
                    )
                )
                in_channels = config.d_model
            self.input_proj_vision = nn.ModuleList(input_proj_list)
        else:
            self.input_proj_vision = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(backbone.intermediate_channel_sizes[-1], config.d_model, kernel_size=1),
                        nn.GroupNorm(32, config.d_model),
                    )
                ]
            )
        self.num_fpn_levels = max(config.num_feature_levels-len(backbone.intermediate_channel_sizes[-3:]), 1)

        self.mask_dim = config.mask_dim
        
        self.mask_features = Conv2d(
            config.d_model,
            config.mask_dim,
            kernel_size=1,
            stride = 1,
            padding=0
        )
        weight_init.c2_xavier_fill(self.mask_features)

        lateral_convs = []
        output_convs = []
        use_bias = config.norm == ""
        # 4x feature
        for idx, in_channels in enumerate(backbone.intermediate_channel_sizes[:self.num_fpn_levels]):
            lateral_norm = get_norm(config.norm, config.d_model)
            output_norm = get_norm(config.norm, config.d_model)

            lateral_conv = Conv2d(
                in_channels, config.d_model, kernel_size=1, bias=use_bias, norm=lateral_norm
            )
            output_conv = Conv2d(
                config.d_model,
                config.d_model,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=use_bias,
                norm=output_norm,
                activation=F.relu,
            )
            weight_init.c2_xavier_fill(lateral_conv)
            weight_init.c2_xavier_fill(output_conv)
            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)
        
        self.lateral_convs = nn.ModuleList(lateral_convs)
        self.output_convs = nn.ModuleList(output_convs)
        # Create text backbone
        # do not create text backbone since we have passed the text query
        # self.text_backbone = GroundingDinoTextPrenet(config.text_backbone_config)
        # self.text_projection = nn.Linear(config.text_backbone_config.hidden_size, config.d_model)

        if config.embedding_init_target or not config.two_stage:
            self.query_position_embeddings = nn.Embedding(config.num_queries, config.d_model)
        self.mask_embed = GroundingDinoMLPPredictionHead(input_dim=config.d_model, hidden_dim=config.d_model, output_dim=config.mask_dim, num_layers=3)
        self.encoder = GroundingDinoEncoder(config)
        self.decoder = GroundingDinoDecoder(config)

        self.level_embed = nn.Parameter(torch.Tensor(config.num_feature_levels, config.d_model))

        if config.two_stage:
            self.enc_output = nn.Linear(config.d_model, config.d_model)
            self.enc_output_norm = nn.LayerNorm(config.d_model)
            if (
                config.two_stage_bbox_embed_share
                and config.decoder_bbox_embed_share
                and self.decoder.bbox_embed is not None
            ):
                self.encoder_output_bbox_embed = self.decoder.bbox_embed
            else:
                self.encoder_output_bbox_embed = GroundingDinoMLPPredictionHead(
                    input_dim=config.d_model, hidden_dim=config.d_model, output_dim=4, num_layers=3
                )

            self.encoder_output_class_embed = GroundingDinoContrastiveEmbedding(config)
        else:
            self.reference_points = nn.Embedding(config.num_queries, 4)

        self.post_init()

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def freeze_backbone(self):
        for name, param in self.backbone.conv_encoder.model.named_parameters():
            param.requires_grad_(False)

    def unfreeze_backbone(self):
        for name, param in self.backbone.conv_encoder.model.named_parameters():
            param.requires_grad_(True)

    def get_valid_ratio(self, mask):
        """Get the valid ratio of all feature maps."""

        _, height, width = mask.shape
        valid_height = torch.sum(mask[:, :, 0], 1)
        valid_width = torch.sum(mask[:, 0, :], 1)
        valid_ratio_heigth = valid_height.float() / height
        valid_ratio_width = valid_width.float() / width
        valid_ratio = torch.stack([valid_ratio_width, valid_ratio_heigth], -1)
        return valid_ratio

    def get_proposal_pos_embed(self, proposals):
        """Get the position embedding of the proposals."""

        num_pos_feats = self.config.d_model // 2
        temperature = 10000
        scale = 2 * math.pi

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / num_pos_feats)
        # batch_size, num_queries, 4
        proposals = proposals.sigmoid() * scale
        # batch_size, num_queries, 4, 128
        pos = proposals[:, :, :, None] / dim_t
        # batch_size, num_queries, 4, 64, 2 -> batch_size, num_queries, 512
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        return pos

    def gen_encoder_output_proposals(self, enc_output, padding_mask, spatial_shapes):
        """Generate the encoder output proposals from encoded enc_output.
        Args:
            enc_output (`torch.Tensor[batch_size, sequence_length, hidden_size]`): Output of the encoder.
            padding_mask (`torch.Tensor[batch_size, sequence_length]`): Padding mask for `enc_output`.
            spatial_shapes (`torch.Tensor[num_feature_levels, 2]`): Spatial shapes of the feature maps.
        Returns:
            `tuple(torch.FloatTensor)`: A tuple of feature map and bbox prediction.
                - object_query (Tensor[batch_size, sequence_length, hidden_size]): Object query features. Later used to
                  directly predict a bounding box. (without the need of a decoder)
                - output_proposals (Tensor[batch_size, sequence_length, 4]): Normalized proposals, after an inverse
                  sigmoid.
        """
        batch_size = enc_output.shape[0]
        proposals = []
        current_position = 0
        for level, (height, width) in enumerate(spatial_shapes):
            mask_flatten_ = padding_mask[:, current_position : (current_position + height * width)].view(
                batch_size, height, width, 1
            )
            valid_height = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_width = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            grid_y, grid_x = meshgrid(
                torch.linspace(0, height - 1, height, dtype=torch.float32, device=enc_output.device),
                torch.linspace(0, width - 1, width, dtype=torch.float32, device=enc_output.device),
                indexing="ij",
            )
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            scale = torch.cat([valid_width.unsqueeze(-1), valid_height.unsqueeze(-1)], 1).view(batch_size, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(batch_size, -1, -1, -1) + 0.5) / scale
            width_heigth = torch.ones_like(grid) * 0.05 * (2.0**level)
            proposal = torch.cat((grid, width_heigth), -1).view(batch_size, -1, 4)
            proposals.append(proposal)
            current_position += height * width

        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals))  # inverse sigmoid
        output_proposals = output_proposals.masked_fill(padding_mask.unsqueeze(-1), float("inf"))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float("inf"))

        # assign each pixel as an object query
        object_query = enc_output
        object_query = object_query.masked_fill(padding_mask.unsqueeze(-1), float(0))
        object_query = object_query.masked_fill(~output_proposals_valid, float(0))
        object_query = self.enc_output_norm(self.enc_output(object_query))
        return object_query, output_proposals

    def forward_seg_heads(self, output, mask_features):
        mask_embed = self.mask_embed(output)
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)
        return outputs_mask

    @add_start_docstrings_to_model_forward(GROUNDING_DINO_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=GroundingDinoModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: Tensor,
        # input_ids: Tensor,
        # token_type_ids: Tensor = None,
        # attention_mask: Tensor = None,
        pixel_mask: Optional[Tensor] = None,
        text_query=None,         # [bs, max_num_patches, c]
        text_query_masks=None,   # [bs, max_num_patches,], valid is 1
        # ------------------------------
        # for dn training
        input_query_label=None, 
        input_query_bbox=None,
        attn_mask=None,
        dn_meta=None,
        # ------------------------------
        encoder_outputs=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Returns:
        Examples:
        ```python
        >>> from transformers import AutoProcessor, GroundingDinoModel
        >>> from PIL import Image
        >>> import requests
        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> text = "a cat."
        >>> processor = AutoProcessor.from_pretrained("EduardoPacheco/grounding-dino-tiny")
        >>> model = GroundingDinoForObjectDetection.from_pretrained("EduardoPacheco/grounding-dino-tiny")
        >>> inputs = processor(images=image, text=text, return_tensors="pt")
        >>> outputs = model(**inputs)
        >>> last_hidden_states = outputs.last_hidden_state
        >>> list(last_hidden_states.shape)
        [1, 900, 256]
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        """
        e.g.
        text_self_attn: (bool) [bs, len, len], valid is 1
        [[1, 1, 1, 0, 0],
         [1, 1, 1, 0, 0],
         [1, 1, 1, 0, 0],
         [0, 0, 0, 1, 0],
         [0, 0, 0, 0, 1]]
        position_ids: (long) [bs, len]
        [[0, 1, 2, 0, 0]]
        """
        text_self_attention_masks, position_ids = generate_masks_with_text_query_masks(text_query_masks)

        attention_mask = text_query_masks  # [bs, max_num_patches]
        text_token_mask = attention_mask.bool()  # just to avoid renaming everywhere
        # text_token_mask: True for nomask, False for mask
        # text_self_attention_masks: True for nomask, False for mask

        # comment the following codes since we pass the text_query and text_query_masks
        # text_self_attention_masks, position_ids = generate_masks_with_special_tokens_and_transfer_map(input_ids)

        # if attention_mask is None:
        #     attention_mask = torch.ones_like(input_ids) # valid is 1

        # if token_type_ids is None:
        #     token_type_ids = torch.zeros_like(input_ids)

        # text_token_mask = attention_mask.bool()  # just to avoid renaming everywhere

        max_text_len = self.config.max_text_len # 256
        if text_self_attention_masks.shape[1] > max_text_len:
            text_self_attention_masks = text_self_attention_masks[:, :max_text_len, :max_text_len]
            position_ids = position_ids[:, :max_text_len]
            # input_ids = input_ids[:, :max_text_len]
            # token_type_ids = token_type_ids[:, :max_text_len]
            text_token_mask = text_token_mask[:, :max_text_len]

        text_features = text_query  # [bs, max_num_patches, c]

        # comment the following codes since we pass the text_query and text_query_masks
        # Extract text features from text backbone
        # text_features = self.text_backbone(input_ids, text_self_attention_masks, token_type_ids, position_ids)[
        #     "last_hidden_state"
        # ]
        # text_features = self.text_projection(text_features)

        batch_size, num_channels, height, width = pixel_values.shape
        device = pixel_values.device

        if pixel_mask is None:
            pixel_mask = torch.ones(((batch_size, height, width)), dtype=torch.long, device=device)

        # Extract multi-scale feature maps of same resolution `config.d_model` (cf Figure 4 in paper)
        # First, sent pixel_values + pixel_mask through Backbone to obtain the features
        # which is a list of tuples
        vision_features, position_embeddings_list = self.backbone(pixel_values, pixel_mask)
        # Then, apply 1x1 convolution to reduce the channel dimension to d_model (256 by default)
        # FIXME: check here, 8x -> 32x for transformer
        vision_features_input = vision_features[1:]
        position_embeddings_list_input = position_embeddings_list[1:]
        sources = []
        masks = []
        for level, (source, mask) in enumerate(vision_features_input):
            sources.append(self.input_proj_vision[level](source))
            masks.append(mask)
            if mask is None:
                raise ValueError("No attention mask was provided")

        # Lowest resolution feature maps are obtained via 3x3 stride 2 convolutions on the final stage
        if self.config.num_feature_levels > len(sources):
            _len_sources = len(sources)  # 3
            for level in range(_len_sources, self.config.num_feature_levels):
                if level == _len_sources:
                    source = self.input_proj_vision[level](vision_features[-1][0])
                else:
                    source = self.input_proj_vision[level](sources[-1])
                mask = nn.functional.interpolate(pixel_mask[None].float(), size=source.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone.position_embedding(source, mask).to(source.dtype)
                sources.append(source)
                masks.append(mask)
                position_embeddings_list_input.append(pos_l)

        # Create queries
        query_embeds = None
        if self.config.embedding_init_target or self.config.two_stage:
            query_embeds = self.query_position_embeddings.weight

        # Prepare encoder inputs (by flattening)
        source_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for level, (source, mask, pos_embed) in enumerate(zip(sources, masks, position_embeddings_list_input)):
            batch_size, num_channels, height, width = source.shape
            spatial_shape = (height, width)
            spatial_shapes.append(spatial_shape)
            source = source.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[level].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            source_flatten.append(source)
            mask_flatten.append(mask)
        source_flatten = torch.cat(source_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=source_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)
        valid_ratios = valid_ratios.float()

        # Fourth, sent source_flatten + mask_flatten + lvl_pos_embed_flatten (backbone + proj layer output) through encoder
        # Also provide spatial_shapes, level_start_index and valid_ratios
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                vision_features=source_flatten,
                vision_attention_mask=~mask_flatten,
                vision_position_embedding=lvl_pos_embed_flatten,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                text_features=text_features,
                text_attention_mask=~text_token_mask,
                text_position_embedding=None,
                text_self_attention_masks=text_self_attention_masks,
                text_position_ids=position_ids,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a GroundingDinoEncoderOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, GroundingDinoEncoderOutput):
            encoder_outputs = GroundingDinoEncoderOutput(
                last_hidden_state_vision=encoder_outputs[0],
                last_hidden_state_text=encoder_outputs[1],
                hidden_states_vision=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
                hidden_states_text=encoder_outputs[3] if len(encoder_outputs) > 3 else None,
                attentions=encoder_outputs[4] if len(encoder_outputs) > 4 else None,
            )

        # build multi-scale mask features
        y = encoder_outputs.last_hidden_state_vision
        bs = y.shape[0]

        split_size_or_sections = [None] * self.config.num_feature_levels
        for i in range(self.config.num_feature_levels):
            if i < self.config.num_feature_levels - 1:
                split_size_or_sections[i] = level_start_index[i + 1] - level_start_index[i]
            else:
                split_size_or_sections[i] = y.shape[1] - level_start_index[i]
        y = torch.split(y, split_size_or_sections, dim=1)
        out = []
        multi_scale_features = []
        for i, z in enumerate(y):
            out.append(z.transpose(1, 2).view(bs, -1, spatial_shapes[i][0], spatial_shapes[i][1]))
        
        for idx in range(self.num_fpn_levels):  # 1
            x = vision_features[idx][0]
            cur_fpn = self.lateral_convs[idx](x)
            y = cur_fpn + F.interpolate(out[0].float(), size=cur_fpn.shape[-2:],mode='bilinear', align_corners=False).to(cur_fpn)
            y = self.output_convs[idx](y)
            out.insert(0, y)
        
        # for idx, o in enumerate(out):
        #     if idx >= self.num_fpn_levels:
        #         multi_scale_features.append(o)

        mask_features = self.mask_features(out[0])

        # Fifth, prepare decoder inputs
        enc_outputs_class = None
        enc_outputs_coord_logits = None

        if self.config.two_stage:
            object_query_embedding, output_proposals = self.gen_encoder_output_proposals(
                encoder_outputs[0], ~mask_flatten, spatial_shapes
            )

            # hack implementation for two-stage Deformable DETR
            # apply a detection head to each pixel (A.4 in paper)
            # linear projection for bounding box binary classification (i.e. foreground and background)
            enc_outputs_class = self.encoder_output_class_embed(
                object_query_embedding, encoder_outputs[1], text_token_mask
            )  # [bs, \sum(hw), K]
            # 3-layer FFN to predict bounding boxes coordinates (bbox regression branch)
            delta_bbox = self.encoder_output_bbox_embed(object_query_embedding)
            enc_outputs_coord_logits = delta_bbox + output_proposals  # [bs, \sum(hw), 4], unsigmoid

            # only keep top scoring `config.num_queries` proposals
            topk = self.config.num_queries
            topk_logits = enc_outputs_class.max(-1)[0]
            topk_proposals = torch.topk(topk_logits, topk, dim=1)[1]
            topk_coords_logits = torch.gather(
                enc_outputs_coord_logits, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4)
            )

            topk_coords_logits = topk_coords_logits.detach()
            reference_points = topk_coords_logits.sigmoid()
            init_reference_points = reference_points
            enc_topk_coords_logits = torch.gather(
                enc_outputs_coord_logits, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4)
            )
            enc_topk_class_logits = torch.gather(
                enc_outputs_class, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, enc_outputs_class.shape[-1])
            )
            if query_embeds is not None:
                target = query_embeds.unsqueeze(0).repeat(batch_size, 1, 1)  # [num_queries, C] -> [bs, num_queries, c]
                outputs_mask = self.forward_seg_heads(target, mask_features)
                # target_undetach = torch.gather(object_query_embedding, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, self.config.d_model))
                # outputs_mask = self.forward_seg_heads(target_undetach, mask_features)
            else:
                target_undetach = torch.gather(object_query_embedding, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, self.config.d_model))
                outputs_mask = self.forward_seg_heads(target_undetach, mask_features)
                target = target_undetach.detach()
        else:
            target = query_embeds.unsqueeze(0).repeat(batch_size, 1, 1)
            reference_points = self.reference_points.weight.unsqueeze(0).repeat(batch_size, 1, 1).sigmoid()
            init_reference_points = reference_points

        # prepare for dn decoder inputs
        if self.training and dn_meta is not None:
            # dn first
            target = torch.cat([input_query_label, target], dim=1)  # [bs, dn_number + num_queries, C]
            reference_points = torch.cat([input_query_bbox.sigmoid(), reference_points], dim=1)  # [bs, dn_number + num_queries, 4]
            init_reference_points = reference_points



        decoder_outputs = self.decoder(
            inputs_embeds=target,
            mask_features=mask_features,
            vision_encoder_hidden_states=encoder_outputs[0],
            vision_encoder_attention_mask=mask_flatten,
            text_encoder_hidden_states=encoder_outputs[1],
            text_encoder_attention_mask=~text_token_mask,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            self_attn_mask=attn_mask,  # this is for query self-attn
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        if not return_dict:
            enc_outputs = tuple(value for value in [enc_outputs_class, enc_outputs_coord_logits] if value is not None)
            tuple_outputs = (init_reference_points,) + decoder_outputs + encoder_outputs + enc_outputs

            return tuple_outputs

        return GroundingDinoModelOutput(
            init_reference_points=init_reference_points,
            last_hidden_state=decoder_outputs.last_hidden_state,
            intermediate_hidden_states=decoder_outputs.intermediate_hidden_states,
            intermediate_reference_points=decoder_outputs.intermediate_reference_points,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            mask_features = mask_features,
            output_mask_intermediate = outputs_mask,
            encoder_last_hidden_state_vision=encoder_outputs.last_hidden_state_vision,
            encoder_last_hidden_state_text=encoder_outputs.last_hidden_state_text,
            encoder_hidden_states_vision=encoder_outputs.hidden_states_vision,
            encoder_hidden_states_text=encoder_outputs.hidden_states_text,
            encoder_attentions=encoder_outputs.attentions,
            enc_outputs_class=enc_outputs_class,
            enc_outputs_coord_logits=enc_outputs_coord_logits,
            enc_topk_coords_logits = enc_topk_coords_logits,
            enc_topk_class_logits = enc_topk_class_logits
        )


@add_start_docstrings(
    """
    Grounding DINO Model (consisting of a backbone and encoder-decoder Transformer) with object detection heads on top,
    for tasks such as COCO detection.
    """,
    GROUNDING_DINO_START_DOCSTRING,
)
class OVGroundingDinoForObjectDetection(GroundingDinoPreTrainedModel):
    # When using clones, all layers > 0 will be clones, but layer 0 *is* required
    _tied_weights_keys = [r"bbox_embed\.[1-9]\d*"]

    def __init__(self, config: GroundingDinoConfig):
        super().__init__(config)

        # Deformable DETR encoder-decoder model
        self.model = OVGroundingDinoModel(config)

        # Detection heads on top
        _class_embed = GroundingDinoContrastiveEmbedding(config)
        _bbox_embed = GroundingDinoMLPPredictionHead(
            input_dim=config.d_model, hidden_dim=config.d_model, output_dim=4, num_layers=3
        )
        mask_embed = GroundingDinoMLPPredictionHead(input_dim=config.d_model, hidden_dim=config.d_model, output_dim=config.mask_dim, num_layers=3)
        if True:
            self.mask_embed = nn.ModuleList([mask_embed for _ in range(config.decoder_layers)])
        else:
            self.mask_embed = _get_clones(mask_embed, config.decoder_layers)
        if config.decoder_bbox_embed_share:
            self.bbox_embed = nn.ModuleList([_bbox_embed for _ in range(config.decoder_layers)])
        else:
            self.bbox_embed = _get_clones(_bbox_embed, config.decoder_layers)
        self.class_embed = nn.ModuleList([_class_embed for _ in range(config.decoder_layers)])
        # hack implementation for two-stage
        self.model.decoder.bbox_embed = self.bbox_embed
        self.model.decoder.class_embed = self.class_embed

        # for text query
        # use nn.Linear for projection
        # self.patch2query = nn.Linear(config.l_hidden_size, config.d_model) # 4096 -> 256
        # nn.init.xavier_uniform_(self.patch2query.weight)
        # nn.init.constant_(self.patch2query.bias, 0)
        # use MLP for projection
        self.patch2query = MLP(config.l_hidden_size, config.d_model, config.d_model, 3)
        nn.init.xavier_uniform_(self.patch2query.layers[-1].weight.data)
        nn.init.constant_(self.patch2query.layers[-1].bias.data, 0)

        # dn configs
        self.dn_number = 100
        self.label_noise_ratio = 0.5
        self.box_noise_scale = 1.0
        self.dn_labelbook_size = 256  # max_text_len


        # Initialize weights and apply final processing
        self.post_init()

    # for dn 
    def prepare_for_cdn(self, targets, dn_number, label_noise_ratio, box_noise_scale, num_queries, hidden_dim, dn_labelbook_size, label_enc):
        """
        Args:
            targets (list[dict]): training labels
            dn_number (int): dn query number
            label_noise_ratio: 0.5
            box_noise_scale: 1.0
            num_queries: 900
            hidden_dim: 256
            dn_labelbook_size: rand classes label size
            label_enc (tensor): [bs, c]
        """
        max_num_patches = label_enc.shape[1]
        dn_labelbook_size = min(dn_labelbook_size, max_num_patches)  # in case the labelbook size is too large

        if dn_number <= 0:
            return None, None, None, None
        # e.g. class_labels: [[0,1,2], [1,4]]

        # positivie and negative dn queries
        dn_number = dn_number * 2 
        known = [(torch.ones_like(t["class_labels"])).cuda() for t in targets]  # e.g. [(1, 1, 1), (1, 1)]
        batch_size = len(known)
        known_num = [sum(k) for k in known]  # list[int], e.g. [3, 2]
        if int(max(known_num)) == 0:
            return None, None, None, None

        dn_number = dn_number // (int(max(known_num) * 2)) # num of dn-group
        # dn_number is shared in a batch, here dn_number is num_groups

        if dn_number == 0:
            dn_number = 1
        unmask_bbox = unmask_label = torch.cat(known)            # [num_all_gt,], value is 1
        labels = torch.cat([t["class_labels"] for t in targets]) # [num_all_gt,], e.g. [0, 1, 2, 1, 4]
        boxes = torch.cat([t["boxes"] for t in targets])         # [num_all_gt, 4]
        batch_idx = torch.cat(
            [torch.full_like(t["class_labels"].long(), i) for i, t in enumerate(targets)]
        )  # e.g. [0, 0, 0, 1, 1]

        known_indice = torch.nonzero(unmask_label + unmask_bbox)
        known_indice = known_indice.view(-1)  # e.g. [0, 1, 2, 3, 4]

        known_indice = known_indice.repeat(2 * dn_number, 1).view(-1)  # [num_all_gt x 2 x dn_number,], e.g. [0, 1, 2, 3, 4, 0, 1, 2, 3, 4...]
        known_labels = labels.repeat(2 * dn_number, 1).view(-1)        # [num_all_gt x 2 x dn_number,]
        known_bid = batch_idx.repeat(2 * dn_number, 1).view(-1)        # [num_all_gt x 2 x dn_number,]
        known_bboxs = boxes.repeat(2 * dn_number, 1)                   # [num_all_gt x 2 x dn_number, 4]
        known_labels_expaned = known_labels.clone()
        known_bbox_expand = known_bboxs.clone()

        # label jittering
        if label_noise_ratio > 0:
            p = torch.rand_like(known_labels_expaned.float())
            chosen_indice = torch.nonzero(p < (label_noise_ratio * 0.5)).view(-1)  # half of box prob
            new_label = torch.randint_like(
                chosen_indice, 0, dn_labelbook_size
            )  # [num_rand,], randomly put a new class
            known_labels_expaned.scatter_(0, chosen_indice, new_label)  # [num_all_gt x 2 x dn_number,]
        single_padding = int(max(known_num))  # value of max_num_gt in a batch

        pad_size = int(single_padding * 2 * dn_number)
        positive_idx = (
            torch.tensor(range(len(boxes))).long().cuda().unsqueeze(0).repeat(dn_number, 1)
        )  # [num_all_gt x dn_number, 4]
        positive_idx += (torch.tensor(range(dn_number)) * len(boxes) * 2).long().cuda().unsqueeze(1)  
        positive_idx = positive_idx.flatten()
        negative_idx = positive_idx + len(boxes) # [pos, neg, pos, neg,...]
        # for [num_all_gt x 2 x dn_number,], pos, neg, pos, neg...

        # box jittering
        if box_noise_scale > 0:
            known_bbox_ = torch.zeros_like(known_bboxs)  # [num_all_gt x 2 x dn_number, 4]
            known_bbox_[:, :2] = known_bboxs[:, :2] - known_bboxs[:, 2:] / 2 # (x1, y1)
            known_bbox_[:, 2:] = known_bboxs[:, :2] + known_bboxs[:, 2:] / 2 # (x2, y2)

            diff = torch.zeros_like(known_bboxs) # [num_all_gt x 2 x dn_number, 4]
            diff[:, :2] = known_bboxs[:, 2:] / 2 # (0.5w, 0.5h)
            diff[:, 2:] = known_bboxs[:, 2:] / 2 # (0.5w, 0.5h)

            rand_sign = (
                torch.randint_like(known_bboxs, low=0, high=2, dtype=torch.float32) * 2.0 - 1.0
            ) # -1 or 1
            rand_part = torch.rand_like(known_bboxs) # [0, 1]
            rand_part[negative_idx] += 1.0 # negative [1, 2], negatvie boxes noise scale is larger
            rand_part *= rand_sign
            known_bbox_ = known_bbox_ + torch.mul(rand_part, diff).cuda() * box_noise_scale
            known_bbox_ = known_bbox_.clamp(min=0.0, max=1.0)
            # transform back to the cxcywh format
            known_bbox_expand[:, :2] = (known_bbox_[:, :2] + known_bbox_[:, 2:]) / 2
            known_bbox_expand[:, 2:] = known_bbox_[:, 2:] - known_bbox_[:, :2]

        # label_embed is from the corresponding location index of text_query 
        input_label_embed = label_enc[known_bid.long(), known_labels_expaned.long()] # [n, c]
        input_bbox_embed = inverse_sigmoid(known_bbox_expand)  # [n, 4], n = num_gt_all x 2 x dn_number

        padding_label = torch.zeros(pad_size, hidden_dim).cuda()  # [max_num_gt x 2 x dn_number, C]
        padding_bbox = torch.zeros(pad_size, 4).cuda()            # [max_num_gt x 2 x dn_number, C]

        input_query_label = padding_label.repeat(batch_size, 1, 1).to(label_enc.dtype)  # [bs, N, C], N = max_num_gt x 2 x dn_number
        input_query_bbox = padding_bbox.repeat(batch_size, 1, 1).to(label_enc.dtype)    # [bs, N, 4]
        
        map_known_indice = torch.tensor([]).to("cuda")
        if len(known_num):
            map_known_indice = torch.cat(
                [torch.tensor(range(num)) for num in known_num]
            )  # e.g. [0, 1, 2, 0, 1], [num_all_gt,]
            map_known_indice = torch.cat(
                [map_known_indice + single_padding * i for i in range(2 * dn_number)]
            ).long()  # [num_all_gt x 2 x dn_number,]
        if len(known_bid):
            # (batch idx, known_idx)
            input_query_label[(known_bid.long(), map_known_indice)] = input_label_embed
            input_query_bbox[(known_bid.long(), map_known_indice)] = input_bbox_embed  # this the box coordinate cxcywh, not need sigmoid

        tgt_size = pad_size + num_queries  # dn first
        """
        For a binary mask, a ``True`` value indicates that the corresponding position is not allowed to attend. 
        For a byte mask, a non-zero value indicates that the corresponding position is not allowed to attend. 
        For a float mask, the mask values will be added to the attention weight.
        
        Generate attention mask to prevent information leakage from
        different denoising groups and matching parts.
        .. code:: text

                        0 0 0 0 1 1 1 1 0 0 0 0 0
                        0 0 0 0 1 1 1 1 0 0 0 0 0
                        0 0 0 0 1 1 1 1 0 0 0 0 0
                        0 0 0 0 1 1 1 1 0 0 0 0 0
                        1 1 1 1 0 0 0 0 0 0 0 0 0
                        1 1 1 1 0 0 0 0 0 0 0 0 0
                        1 1 1 1 0 0 0 0 0 0 0 0 0
                        1 1 1 1 0 0 0 0 0 0 0 0 0
                        1 1 1 1 1 1 1 1 0 0 0 0 0
                        1 1 1 1 1 1 1 1 0 0 0 0 0
                        1 1 1 1 1 1 1 1 0 0 0 0 0
                        1 1 1 1 1 1 1 1 0 0 0 0 0
                        1 1 1 1 1 1 1 1 0 0 0 0 0
         max_num_target |_|           |_________| num_matching_queries
                        |_____________| num_denoising_queries

               1 -> True  (Masked), means 'can not see'.
               0 -> False (UnMasked), means 'can see'.
        """
        attn_mask = torch.ones(tgt_size, tgt_size).to("cuda") < 0  # False
        # match query can not see the reconstruct
        attn_mask[pad_size:, :pad_size] = True
        # reconstruct cannot see each other
        for i in range(dn_number):
            if i == 0:
                attn_mask[
                    single_padding * 2 * i : single_padding * 2 * (i + 1),
                    single_padding * 2 * (i + 1) : pad_size,
                ] = True
            if i == dn_number - 1:
                attn_mask[
                    single_padding * 2 * i : single_padding * 2 * (i + 1), : single_padding * i * 2
                ] = True
            else:
                attn_mask[
                    single_padding * 2 * i : single_padding * 2 * (i + 1),
                    single_padding * 2 * (i + 1) : pad_size,
                ] = True
                attn_mask[
                    single_padding * 2 * i : single_padding * 2 * (i + 1), : single_padding * 2 * i
                ] = True

        dn_meta = {
            "single_padding": single_padding * 2,
            "dn_num": dn_number,
        }
        return input_query_label, input_query_bbox, attn_mask, dn_meta

    def dn_post_process(self, outputs_class, outputs_coord, dn_metas, outputs_mask=None):
        if dn_metas and dn_metas["single_padding"] > 0:
            padding_size = dn_metas["single_padding"] * dn_metas["dn_num"]
            # dn part
            output_known_class = outputs_class[:, :, :padding_size, :]  # [lvl, bs, dn_number, 256]
            output_known_coord = outputs_coord[:, :, :padding_size, :]  # [lvl, bs, dn_number, 4]
            # matching part
            outputs_class = outputs_class[:, :, padding_size:, :]  # [lvl, bs, num_queries, 256]
            outputs_coord = outputs_coord[:, :, padding_size:, :]  # [lvl, bs, num_queries, 4]
            if outputs_mask is not None:
                output_known_mask = outputs_mask[:, :, :padding_size, :]
                outputs_mask = outputs_mask[:, :, padding_size:, :]
            out = {"logits": output_known_class[-1], "pred_boxes": output_known_coord[-1], "pred_masks": output_known_mask[-1]}
            if self.config.auxiliary_loss:
                out["aux_outputs"] = self._set_aux_loss(output_known_class, output_known_coord, output_known_mask)
            dn_metas["output_known_lbs_bboxes"] = out
        return outputs_class, outputs_coord, outputs_mask, dn_metas


    # taken from https://github.com/facebookresearch/detr/blob/master/models/detr.py
    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_mask):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{"logits": a, "pred_boxes": b, "pred_masks": c} for a, b, c in zip(outputs_class[:-1], outputs_coord[:-1], outputs_mask[:-1])]
    
    def forward_seg_heads(self, output, mask_features, level):
        mask_embed = self.mask_embed[level](output)
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)
        return outputs_mask

    # @add_start_docstrings_to_model_forward(GROUNDING_DINO_INPUTS_DOCSTRING)
    # @replace_return_docstrings(output_type=GroundingDinoObjectDetectionOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        pixel_mask: Optional[torch.BoolTensor] = None,
        encoder_outputs: Optional[Union[GroundingDinoEncoderOutput, Tuple]] = None,
        labels: List[Dict[str, Union[torch.LongTensor, torch.FloatTensor]]] = None,
        text_query = None,          # [bs, max_num_patch, num_embs, c]
        text_query_masks = None,    # [bs, max_num_patch,]
        img_metas = None,           # list[dict], for converting id2index
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        if self.training:
            return self.forward_train(
                pixel_values=pixel_values,
                pixel_mask=pixel_mask,
                encoder_outputs=encoder_outputs,
                labels=labels,
                text_query=text_query,
                text_query_masks=text_query_masks,
                img_metas=img_metas,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            return self.forward_test(
                pixel_values=pixel_values,
                pixel_mask=pixel_mask,
                encoder_outputs=encoder_outputs,
                labels=labels,
                text_query=text_query,
                text_query_masks=text_query_masks,
                img_metas=img_metas,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
    
    def forward_train(
        self,
        pixel_values: torch.FloatTensor,
        pixel_mask: Optional[torch.BoolTensor] = None,
        encoder_outputs: Optional[Union[GroundingDinoEncoderOutput, Tuple]] = None,
        labels: List[Dict[str, Union[torch.LongTensor, torch.FloatTensor]]] = None,
        text_query = None,          # [bs, max_num_patch, num_embs, c]
        text_query_masks = None,    # [bs, max_num_patch,]
        img_metas = None,           # list[dict], for converting id2index
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # convert the class labels during training
        if self.training:
            assert img_metas is not None
            assert labels is not None
            new_labels = []
            for img_meta, label in zip(img_metas, labels):
                # img_meta, label: dict
                id2index = img_meta["id2index"]  # dict, store the class id to location index mapping 
                class_labels = copy.deepcopy(label["class_labels"])  # [num_gt_i,]
                class_labels_list = class_labels.tolist()
                new_class_labels = torch.LongTensor([id2index[label_i] for label_i in class_labels_list]).to(pixel_values.device)
                label["class_labels"] = new_class_labels
                # pad gt_masks, if any
                if "mask_labels" in label.keys():
                    pad_h, pad_w = pixel_values.shape[-2:]
                    mask_labels = copy.deepcopy(label["mask_labels"])  # [num_gt_i, h, w]
                    padded_masks = torch.zeros((mask_labels.shape[0], pad_h, pad_w), dtype=mask_labels.dtype, device=pixel_values.device)
                    padded_masks[:, :mask_labels.shape[1], :mask_labels.shape[2]] = mask_labels
                    label["mask_labels"] = padded_masks
                new_labels.append(label)
            labels = new_labels


        # create text_queries
        # in: text_query [bs, max_num_patches, num_embs, c], text_query_masks: [bs, max_num_patches], valid is 1
        # out: text_query [bs, max_num_patches, 256], text_query_masks: [bs, max_num_patches]
        text_query = self.patch2query(text_query).mean(-2) 

        # if attention_mask is None:
        #     attention_mask = torch.ones_like(input_ids)
        attention_mask = text_query_masks  # [bs, seq_len], valid is 1 for text_query

        # prepare for dn inputs
        if self.training:
            # here, attn_mask is for queries self attn, True locations are not allowed to attend to.
            input_query_label, input_query_bbox, attn_mask, dn_meta = self.prepare_for_cdn(
                labels, self.dn_number, self.label_noise_ratio, self.box_noise_scale, self.config.num_queries,
                self.config.d_model, self.dn_labelbook_size, text_query
            )
        else:
            input_query_label, input_query_bbox = None, None
            attn_mask, dn_meta = None, None

        # First, sent images through Grounding DINO base model to obtain encoder + decoder outputs
        outputs = self.model(
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            text_query=text_query,
            text_query_masks=text_query_masks,
            # ------------------------------
            # for dn training
            input_query_label=input_query_label, 
            input_query_bbox=input_query_bbox,
            attn_mask=attn_mask,
            dn_meta=dn_meta,
            # ------------------------------
            encoder_outputs=encoder_outputs,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # index for encoder_last_hidden_state_text
        idx = 5 + (1 if output_attentions else 0) + (1 if output_hidden_states else 0)

        hidden_states = outputs.intermediate_hidden_states if return_dict else outputs[2]
        enc_text_hidden_state = outputs.encoder_last_hidden_state_text if return_dict else outputs[idx]
        init_reference = outputs.init_reference_points if return_dict else outputs[0]
        inter_references = outputs.intermediate_reference_points if return_dict else outputs[3]

        # class logits + predicted bounding boxes
        outputs_classes = []
        outputs_coords = []
        outputs_masks = []
        for level in range(hidden_states.shape[1]):
            if level == 0:
                reference = init_reference
            else:
                reference = inter_references[:, level - 1]
            reference = inverse_sigmoid(reference)
            output_mask = self.forward_seg_heads(hidden_states[:, level], outputs.mask_features, level)
            outputs_class = self.class_embed[level](
                vision_hidden_state=hidden_states[:, level],
                text_hidden_state=enc_text_hidden_state,
                text_token_mask=attention_mask.bool(),
            ) # [bs, num_queries, 256], text_len=256
            delta_bbox = self.bbox_embed[level](hidden_states[:, level])
            if reference.shape[-1] == 4:
                outputs_coord_logits = delta_bbox + reference
            elif reference.shape[-1] == 2:
                delta_bbox[..., :2] += reference
                outputs_coord_logits = delta_bbox
            else:
                raise ValueError(f"reference.shape[-1] should be 4 or 2, but got {reference.shape[-1]}")
            outputs_coord = outputs_coord_logits.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
            outputs_masks.append(output_mask)

        outputs_class = torch.stack(outputs_classes).to(torch.float32)  # [lvl, bs, num_queries, 256]
        outputs_coord = torch.stack(outputs_coords).to(torch.float32)   # [lvl, bs, num_queries, 4]
        outputs_mask = torch.stack(outputs_masks).to(torch.float32)

        # remove dn part during training
        if self.training and dn_meta is not None:
            outputs_class, outputs_coord, outputs_mask, dn_meta = self.dn_post_process(outputs_class, outputs_coord, dn_meta, outputs_mask)

        logits = outputs_class[-1]
        pred_boxes = outputs_coord[-1]
        pred_masks = outputs_mask[-1]

        loss, loss_dict, auxiliary_outputs = None, None, None
        if labels is not None:
            # First: create the matcher
            matcher = GroundingDinoHungarianMatcher(
                class_cost=self.config.class_cost, 
                bbox_cost=self.config.bbox_cost, 
                giou_cost=self.config.giou_cost,
                dice_cost=self.config.dice_cost,
                mask_cost=self.config.mask_cost
            )
            # Second: create the criterion
            losses = ["labels", "boxes", "masks"]
            criterion = DNGroundingDinoLoss(
                matcher=matcher,
                # num_classes=self.config.num_labels,
                focal_alpha=self.config.focal_alpha,
                losses=losses,
            )
            criterion.to(self.device)
            # Third: compute the losses, based on outputs and labels
            outputs_loss = {}
            outputs_loss["logits"] = logits           # [bs, num_queries, 256]
            outputs_loss["pred_boxes"] = pred_boxes   # [bs, num_queries, 4]
            outputs_loss["pred_masks"] = pred_masks   # [bs, num_queries, h, w]
            outputs_loss["text_query_masks"] = text_query_masks # [bs, max_num_patches], valid is 1
            if self.config.auxiliary_loss:
                auxiliary_outputs = self._set_aux_loss(outputs_class, outputs_coord, outputs_mask)
                outputs_loss["auxiliary_outputs"] = auxiliary_outputs
                for out in outputs_loss["auxiliary_outputs"]:
                    out["text_query_masks"] = text_query_masks
            if self.config.two_stage:
                enc_outputs_coord = outputs.enc_topk_coords_logits.sigmoid()
                outputs_loss["enc_outputs"] = {"logits": outputs.enc_topk_class_logits, "pred_boxes": enc_outputs_coord, "pred_masks": outputs.output_mask_intermediate.to(torch.float32)}
                outputs_loss["enc_outputs"]["text_query_masks"] = text_query_masks
            # add text_query_masks to dn_meta
            if self.training and dn_meta is not None:
                dn_meta["output_known_lbs_bboxes"]["text_query_masks"] = text_query_masks
                for x in dn_meta["output_known_lbs_bboxes"]["aux_outputs"]:
                    x["text_query_masks"] = text_query_masks

            loss_dict = criterion(outputs_loss, labels, dn_meta)
            
            # Fourth: compute total loss, as a weighted sum of the various losses
            weight_dict = {"loss_ce": self.config.class_weight, "loss_bbox": self.config.box_weight, 'loss_mask': self.config.mask_weight, 'loss_dice': self.config.dice_weight}
            weight_dict["loss_giou"] = self.config.giou_weight
            if self.config.auxiliary_loss:
                aux_weight_dict = {}
                for i in range(self.config.decoder_layers - 1):
                    aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
                aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
                weight_dict.update(aux_weight_dict)
            # update dn loss weight
            weight_dict_dn = {"loss_ce_dn": self.config.class_weight, "loss_bbox_dn": self.config.box_weight,
                              "loss_giou_dn": self.config.giou_weight,'loss_mask_dn': self.config.mask_weight, 'loss_dice_dn': self.config.dice_weight}
            aux_weight_dict_dn = {}
            for i in range(self.config.decoder_layers - 1):
                aux_weight_dict_dn.update({k + f"_{i}": v for k, v in weight_dict_dn.items()})
            weight_dict_dn.update(aux_weight_dict_dn)
            weight_dict.update(weight_dict_dn)
            # sum the total loss
            loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        if not return_dict:
            if auxiliary_outputs is not None:
                output = (logits, pred_boxes) + auxiliary_outputs + outputs
            else:
                output = (logits, pred_boxes) + outputs
            tuple_outputs = ((loss, loss_dict) + output) if loss is not None else output

            return tuple_outputs

        dict_outputs = GroundingDinoObjectDetectionOutput(
            loss=loss,
            loss_dict=loss_dict,
            logits=logits,
            pred_boxes=pred_boxes,
            auxiliary_outputs=auxiliary_outputs,
            last_hidden_state=outputs.last_hidden_state,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            encoder_last_hidden_state_vision=outputs.encoder_last_hidden_state_vision,
            encoder_last_hidden_state_text=outputs.encoder_last_hidden_state_text,
            encoder_hidden_states_vision=outputs.encoder_hidden_states_vision,
            encoder_hidden_states_text=outputs.encoder_hidden_states_text,
            encoder_attentions=outputs.encoder_attentions,
            intermediate_hidden_states=outputs.intermediate_hidden_states,
            intermediate_reference_points=outputs.intermediate_reference_points,
            init_reference_points=outputs.init_reference_points,
            enc_outputs_class=outputs.enc_outputs_class,
            enc_outputs_coord_logits=outputs.enc_outputs_coord_logits,
        )

        return dict_outputs

    def forward_test(
        self,
        pixel_values: torch.FloatTensor,
        pixel_mask: Optional[torch.BoolTensor] = None,
        encoder_outputs: Optional[Union[GroundingDinoEncoderOutput, Tuple]] = None,
        labels: List[Dict[str, Union[torch.LongTensor, torch.FloatTensor]]] = None,
        text_query = None,          # [bs, max_num_patch, num_embs, c]
        text_query_masks = None,    # [bs, max_num_patch,]
        img_metas = None,           # list[dict], for converting id2index
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        text_query = self.patch2query(text_query).mean(-2) 
        attention_mask = text_query_masks  # [bs, seq_len], valid is 1 for text_query
        input_query_label, input_query_bbox = None, None
        attn_mask, dn_meta = None, None
        outputs = self.model(
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            text_query=text_query,
            text_query_masks=text_query_masks,
            # ------------------------------
            # for dn training
            input_query_label=input_query_label, 
            input_query_bbox=input_query_bbox,
            attn_mask=attn_mask,
            dn_meta=dn_meta,
            # ------------------------------
            encoder_outputs=encoder_outputs,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        idx = 5 + (1 if output_attentions else 0) + (1 if output_hidden_states else 0)

        hidden_states = outputs.intermediate_hidden_states if return_dict else outputs[2]
        enc_text_hidden_state = outputs.encoder_last_hidden_state_text if return_dict else outputs[idx]
        init_reference = outputs.init_reference_points if return_dict else outputs[0]
        inter_references = outputs.intermediate_reference_points if return_dict else outputs[3]

        # class logits + predicted bounding boxes
        outputs_classes = []
        outputs_coords = []
        outputs_masks = []

        for level in range(hidden_states.shape[1]):
            if level == 0:
                reference = init_reference
            else:
                reference = inter_references[:, level - 1]
            reference = inverse_sigmoid(reference)
            output_mask = self.forward_seg_heads(hidden_states[:, level], outputs.mask_features, level)
            outputs_class = self.class_embed[level](
                vision_hidden_state=hidden_states[:, level],
                text_hidden_state=enc_text_hidden_state,
                text_token_mask=attention_mask.bool(),
            ) # [bs, num_queries, 256], text_len=256
            delta_bbox = self.bbox_embed[level](hidden_states[:, level])
            if reference.shape[-1] == 4:
                outputs_coord_logits = delta_bbox + reference
            elif reference.shape[-1] == 2:
                delta_bbox[..., :2] += reference
                outputs_coord_logits = delta_bbox
            else:
                raise ValueError(f"reference.shape[-1] should be 4 or 2, but got {reference.shape[-1]}")
            outputs_coord = outputs_coord_logits.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
            outputs_masks.append(output_mask)
        outputs_class = torch.stack(outputs_classes).to(torch.float32)  # [lvl, bs, num_queries, 256]
        outputs_coord = torch.stack(outputs_coords).to(torch.float32)   # [lvl, bs, num_queries, 4]
        outputs_mask = torch.stack(outputs_masks).to(torch.float32) 

        logits = outputs_class[-1]
        pred_boxes = outputs_coord[-1]
        pred_masks = outputs_mask[-1]

        output = GroundingDinoObjectDetectionOutput(
            logits=logits,
            pred_boxes=pred_boxes,
            pred_masks=pred_masks
        )

        return output


# Copied from transformers.models.detr.modeling_detr.dice_loss
def dice_loss(inputs, targets, num_boxes):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs (0 for the negative class and 1 for the positive
                 class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_boxes


def token_sigmoid_binary_focal_loss(inputs, targets, alpha=0.25, gamma=2.0, text_mask=None, reduction=True):
    # input: [bs, num_queries, 256]
    # targets: [bs, num_queries, 256]
    # text_mask: [bs, max_num_patches,], max_num_patches < 256
    assert (inputs.dim() == 3)
    assert (targets.dim() == 3)
    assert (text_mask is not None and text_mask.dim() == 2)
    bs, num_queries, max_text_len = inputs.size()
    bs, max_num_patches = text_mask.size()

    text_mask_pad = torch.zeros((bs, max_text_len), dtype=torch.bool, device=inputs.device)
    text_mask_pad[:, :max_num_patches] = text_mask
    text_mask = text_mask_pad

    text_mask = (text_mask > 0).unsqueeze(1)         # (bs, 1, max_seq_len)
    text_mask = text_mask.repeat(1, num_queries, 1)  # [bs, num_queries, max_seq_len]
    inputs = torch.masked_select(inputs, text_mask)
    targets = torch.masked_select(targets, text_mask)

    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    if reduction:
        return loss.sum()
    else:
        return loss



# Copied from transformers.models.detr.modeling_detr.sigmoid_focal_loss
def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs (`torch.FloatTensor` of arbitrary shape):
            The predictions for each example.
        targets (`torch.FloatTensor` with the same shape as `inputs`)
            A tensor storing the binary classification label for each element in the `inputs` (0 for the negative class
            and 1 for the positive class).
        alpha (`float`, *optional*, defaults to `0.25`):
            Optional weighting factor in the range (0,1) to balance positive vs. negative examples.
        gamma (`int`, *optional*, defaults to `2`):
            Exponent of the modulating factor (1 - p_t) to balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    # add modulating factor
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes


# Copied from transformers.models.deformable_detr.modeling_deformable_detr.DeformableDetrLoss with DeformableDetr->GroundingDino
class GroundingDinoLoss(nn.Module):
    """
    This class computes the losses for `GroundingDinoForObjectDetection`. The process happens in two steps: 1) we
    compute hungarian assignment between ground truth boxes and the outputs of the model 2) we supervise each pair of
    matched ground-truth / prediction (supervise class and box).
    Args:
        matcher (`GroundingDinoHungarianMatcher`):
            Module able to compute a matching between targets and proposals.
        num_classes (`int`):
            Number of object categories, omitting the special no-object category.
        focal_alpha (`float`):
            Alpha parameter in focal loss.
        losses (`List[str]`):
            List of all the losses to be applied. See `get_loss` for a list of all available losses.
    """

    def __init__(self, matcher, focal_alpha, losses, num_points=12544, oversample_ratio=3.0, importance_sample_ratio=0.75):
        super().__init__()
        self.matcher = matcher
        # self.num_classes = num_classes  # not used
        self.focal_alpha = focal_alpha
        self.losses = losses
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio

    # removed logging parameter, which was part of the original implementation
    def loss_labels(self, outputs, targets, indices, num_boxes):
        """
        Classification loss (Binary focal loss) targets dicts must contain the key "class_labels" containing a tensor
        of dim [nb_target_boxes]
        """
        if "logits" not in outputs:
            raise KeyError("No logits were found in the outputs")
        source_logits = outputs["logits"]  # [bs, num_queries, 256]

        # TODO: check here, loss_labels according to vl
        idx = self._get_source_permutation_idx(indices)  # (batch_idx, src_idx)
        if len(idx[0]) == 0:
            loss_ce = source_logits[:, :, 0].sum() * 0.0  # in case of -inf in padded classes
            losses = {"loss_ce": loss_ce}
            return losses

        target_classes_onehot = torch.zeros(source_logits.size(),
                    dtype=source_logits.dtype,
                    layout=source_logits.layout,
                    device=source_logits.device
        )  # [bs, num_queries, 256]
        class_labels = [x["class_labels"] for x in targets] # [num_gt_1, num_gt_2,..]
        # loop over batch size
        for batch_idx, (src_idxs, tgt_idxs) in enumerate(indices):
            # loop over each gt object
            for (src_idx, tgt_idx) in zip(src_idxs, tgt_idxs):
                target_classes_onehot[batch_idx, src_idx, class_labels[batch_idx][tgt_idx]] = 1

        loss_ce = token_sigmoid_binary_focal_loss(
            source_logits, target_classes_onehot, text_mask=outputs["text_query_masks"]) / num_boxes
        losses = {"loss_ce": loss_ce}
        return losses

    @torch.no_grad()
    # Copied from transformers.models.detr.modeling_detr.DetrLoss.loss_cardinality
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """
        Compute the cardinality error, i.e. the absolute error in the number of predicted non-empty boxes.
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients.
        """
        logits = outputs["logits"]
        device = logits.device
        target_lengths = torch.as_tensor([len(v["class_labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (logits.argmax(-1) != logits.shape[-1] - 1).sum(1)
        card_err = nn.functional.l1_loss(card_pred.float(), target_lengths.float())
        losses = {"cardinality_error": card_err}
        return losses

    # Copied from transformers.models.detr.modeling_detr.DetrLoss.loss_boxes
    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """
        Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss.
        Targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]. The target boxes
        are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        if "pred_boxes" not in outputs:
            raise KeyError("No predicted boxes found in outputs")
        idx = self._get_source_permutation_idx(indices)
        source_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0).to(torch.float32)

        loss_bbox = nn.functional.l1_loss(source_boxes, target_boxes, reduction="none")

        losses = {}
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(
            generalized_box_iou(center_to_corners_format(source_boxes), center_to_corners_format(target_boxes))
        )
        losses["loss_giou"] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_masks):
        # handle datasets w/o mask annotations
        has_mask_anno = "mask_labels" in targets[0].keys()
        if not has_mask_anno:
            src_masks = outputs["pred_masks"]
            losses = {}
            losses['loss_mask'] = src_masks.sum() * 0.0
            losses['loss_dice'] = src_masks.sum() * 0.0
            return losses

        torch.cuda.empty_cache()
        assert "pred_masks" in outputs
        src_idx = self._get_source_permutation_idx(indices)
        tgt_idx = self._get_target_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["mask_labels"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]
        src_masks = src_masks[:, None]
        target_masks = target_masks[:, None]
        with torch.no_grad():
            # sample point_coords
            point_coords = self.sample_points_using_uncertainty(
                src_masks,
                lambda logits: self.calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            # get gt labels
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1)
        
        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)

        losses = {
            "loss_mask" : sigmoid_cross_entropy_loss(point_logits, point_labels, num_masks),
            "loss_dice" : dice_loss(point_logits, point_labels, num_masks)
        }
        del src_masks
        del target_masks
        return losses

    def sample_points_using_uncertainty(
        self,
        logits: torch.Tensor,
        uncertainty_function,
        num_points: int,
        oversample_ratio: int,
        importance_sample_ratio: float,
    ) -> torch.Tensor:
        """
        This function is meant for sampling points in [0, 1] * [0, 1] coordinate space based on their uncertainty. The
        uncertainty is calculated for each point using the passed `uncertainty function` that takes points logit
        prediction as input.

        Args:
            logits (`float`):
                Logit predictions for P points.
            uncertainty_function:
                A function that takes logit predictions for P points and returns their uncertainties.
            num_points (`int`):
                The number of points P to sample.
            oversample_ratio (`int`):
                Oversampling parameter.
            importance_sample_ratio (`float`):
                Ratio of points that are sampled via importance sampling.

        Returns:
            point_coordinates (`torch.Tensor`):
                Coordinates for P sampled points.
        """

        num_boxes = logits.shape[0]
        num_points_sampled = int(num_points * oversample_ratio)

        # Get random point coordinates
        point_coordinates = torch.rand(num_boxes, num_points_sampled, 2, device=logits.device)
        # Get sampled prediction value for the point coordinates
        point_logits = point_sample(logits, point_coordinates, align_corners=False)
        # Calculate the uncertainties based on the sampled prediction values of the points
        point_uncertainties = uncertainty_function(point_logits)

        num_uncertain_points = int(importance_sample_ratio * num_points)
        num_random_points = num_points - num_uncertain_points

        idx = torch.topk(point_uncertainties[:, 0, :], k=num_uncertain_points, dim=1)[1]
        shift = num_points_sampled * torch.arange(num_boxes, dtype=torch.long, device=logits.device)
        idx += shift[:, None]
        point_coordinates = point_coordinates.view(-1, 2)[idx.view(-1), :].view(num_boxes, num_uncertain_points, 2)

        if num_random_points > 0:
            point_coordinates = torch.cat(
                [point_coordinates, torch.rand(num_boxes, num_random_points, 2, device=logits.device)],
                dim=1,
            )
        return point_coordinates

    def calculate_uncertainty(self, logits: torch.Tensor) -> torch.Tensor:
        """
        In Mask2Former paper, uncertainty is estimated as L1 distance between 0.0 and the logit prediction in 'logits'
        for the foreground class in `classes`.

        Args:
            logits (`torch.Tensor`):
            A tensor of shape (R, 1, ...) for class-specific or class-agnostic, where R is the total number of predicted masks in all images and C is:
            the number of foreground classes. The values are logits.

        Returns:
            scores (`torch.Tensor`): A tensor of shape (R, 1, ...) that contains uncertainty scores with the most
            uncertain locations having the highest uncertainty score.
        """
        uncertainty_scores = -(torch.abs(logits))
        return uncertainty_scores


    # Copied from transformers.models.detr.modeling_detr.DetrLoss._get_source_permutation_idx
    def _get_source_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(source, i) for i, (source, _) in enumerate(indices)])
        source_idx = torch.cat([source for (source, _) in indices])
        return batch_idx, source_idx

    # Copied from transformers.models.detr.modeling_detr.DetrLoss._get_target_permutation_idx
    def _get_target_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(target, i) for i, (_, target) in enumerate(indices)])
        target_idx = torch.cat([target for (_, target) in indices])
        return batch_idx, target_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes):
        loss_map = {
            "labels": self.loss_labels,
            "cardinality": self.loss_cardinality,
            "boxes": self.loss_boxes,
            "masks": self.loss_masks,
        }
        if loss not in loss_map:
            raise ValueError(f"Loss {loss} not supported")
        return loss_map[loss](outputs, targets, indices, num_boxes)

    def forward(self, outputs, targets):
        """
        This performs the loss computation.
        Args:
             outputs (`dict`, *optional*):
                Dictionary of tensors, see the output specification of the model for the format.
             targets (`List[dict]`, *optional*):
                List of dicts, such that `len(targets) == batch_size`. The expected keys in each dict depends on the
                losses applied, see each loss' doc.
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "auxiliary_outputs" and k != "enc_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["class_labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        # (Niels): comment out function below, distributed training to be added
        # if is_dist_avail_and_initialized():
        #     torch.distributed.all_reduce(num_boxes)
        # (Niels) in original implementation, num_boxes is divided by get_world_size()
        num_boxes = torch.clamp(num_boxes, min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))
        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "auxiliary_outputs" in outputs:
            for i, auxiliary_outputs in enumerate(outputs["auxiliary_outputs"]):
                indices = self.matcher(auxiliary_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, auxiliary_outputs, targets, indices, num_boxes)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if "enc_outputs" in outputs:
            enc_outputs = outputs["enc_outputs"]
            indices = self.matcher(enc_outputs, targets)
            for loss in self.losses:
                l_dict = self.get_loss(loss, enc_outputs, targets, indices, num_boxes)
                l_dict = {k + "_enc": v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses

def sigmoid_cross_entropy_loss(inputs: torch.Tensor, labels: torch.Tensor, num_masks: int) -> torch.Tensor:
    r"""
    Args:
        inputs (`torch.Tensor`):
            A float tensor of arbitrary shape.
        labels (`torch.Tensor`):
            A tensor with the same shape as inputs. Stores the binary classification labels for each element in inputs
            (0 for the negative class and 1 for the positive class).

    Returns:
        loss (`torch.Tensor`): The computed loss.
    """
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    cross_entropy_loss = criterion(inputs, labels)

    loss = cross_entropy_loss.mean(1).sum() / num_masks
    return loss

class DNGroundingDinoLoss(GroundingDinoLoss):
    def forward(self, outputs, targets, dn_metas=None):
        losses = super(DNGroundingDinoLoss, self).forward(outputs, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["class_labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        # (Niels): comment out function below, distributed training to be added
        # if is_dist_avail_and_initialized():
        #     torch.distributed.all_reduce(num_boxes)
        # (Niels) in original implementation, num_boxes is divided by get_world_size()
        num_boxes = torch.clamp(num_boxes, min=1).item()

        # compute all the requested losses
        aux_num = 0 
        if "auxiliary_outputs" in outputs:
            aux_num = len(outputs["auxiliary_outputs"])  # aux decoder layer number
        dn_losses = self.compute_dn_loss(dn_metas, targets, aux_num, num_boxes)
        losses.update(dn_losses)
        return losses
    
    def compute_dn_loss(self, dn_metas, targets, aux_num, num_boxes):
        losses = {}
        if dn_metas and "output_known_lbs_bboxes" in dn_metas:
            output_known_lbs_bboxes, dn_num, single_padding = (
                dn_metas["output_known_lbs_bboxes"],
                dn_metas["dn_num"],
                dn_metas["single_padding"],
            )
            dn_idx = []
            # loop over batchsize
            for i in range(len(targets)):
                if len(targets[i]["class_labels"]) > 0:
                    t = torch.arange(0, len(targets[i]["class_labels"])).long().cuda()  # [num_gt_i,]
                    t = t.unsqueeze(0).repeat(dn_num, 1) # shape: (dn_num, n)
                    tgt_idx = t.flatten()
                    output_idx = (
                        torch.tensor(range(dn_num)) * single_padding
                    ).long().cuda().unsqueeze(1) + t
                    output_idx = output_idx.flatten()
                else:
                    output_idx = tgt_idx = torch.tensor([]).long().cuda()
                dn_idx.append((output_idx, tgt_idx))
            l_dict = {}
            for loss in self.losses:
                if loss not in ["labels", "boxes", "masks"]:
                    continue
                l_dict.update(
                    self.get_loss(
                        loss, output_known_lbs_bboxes, targets, dn_idx, num_boxes * dn_num
                    )
                )
            
            l_dict = {k + "_dn": v for k, v in l_dict.items()}
            losses.update(l_dict)
        else:
            losses["loss_bbox_dn"] = torch.as_tensor(0.0).to("cuda")
            losses["loss_giou_dn"] = torch.as_tensor(0.0).to("cuda")
            losses["loss_class_dn"] = torch.as_tensor(0.0).to("cuda")
            losses['loss_mask_dn'] = torch.as_tensor(0.).to('cuda')
            losses['loss_dice_dn'] = torch.as_tensor(0.).to('cuda')

        # dn aux loss
        for i in range(aux_num):
            l_dict = {}
            if dn_metas and "output_known_lbs_bboxes" in dn_metas:
                output_known_lbs_bboxes_aux = output_known_lbs_bboxes["aux_outputs"][i]
                for loss in self.losses:
                    if loss not in ["labels", "boxes", "masks"]:
                        continue
                    l_dict.update(
                        self.get_loss(
                            loss, output_known_lbs_bboxes_aux, targets, dn_idx, num_boxes * dn_num
                        )
                    )
                l_dict = {k + f"_dn_{i}": v for k, v in l_dict.items()}
            else:
                l_dict[f"loss_bbox_dn_{i}"] = torch.as_tensor(0.0).to("cuda")
                l_dict[f"loss_giou_dn_{i}"] = torch.as_tensor(0.0).to("cuda")
                l_dict[f"loss_class_dn_{i}"] = torch.as_tensor(0.0).to("cuda")
                l_dict[f'loss_mask_dn_{i}'] = torch.as_tensor(0.).to('cuda')
                l_dict[f'loss_dice_dn_{i}'] = torch.as_tensor(0.).to('cuda')
            losses.update(l_dict)
        return losses


            



# Copied from transformers.models.detr.modeling_detr.DetrMLPPredictionHead
class GroundingDinoMLPPredictionHead(nn.Module):
    """
    Very simple multi-layer perceptron (MLP, also called FFN), used to predict the normalized center coordinates,
    height and width of a bounding box w.r.t. an image.
    Copied from https://github.com/facebookresearch/detr/blob/master/models/detr.py
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = nn.functional.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

def pair_wise_sigmoid_cross_entropy_loss(inputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    r"""
    A pair wise version of the cross entropy loss, see `sigmoid_cross_entropy_loss` for usage.

    Args:
        inputs (`torch.Tensor`):
            A tensor representing a mask.
        labels (`torch.Tensor`):
            A tensor with the same shape as inputs. Stores the binary classification labels for each element in inputs
            (0 for the negative class and 1 for the positive class).

    Returns:
        loss (`torch.Tensor`): The computed loss between each pairs.
    """

    height_and_width = inputs.shape[1]

    criterion = nn.BCEWithLogitsLoss(reduction="none")
    cross_entropy_loss_pos = criterion(inputs, torch.ones_like(inputs))
    cross_entropy_loss_neg = criterion(inputs, torch.zeros_like(inputs))

    loss_pos = torch.matmul(cross_entropy_loss_pos, labels.T)
    loss_neg = torch.matmul(cross_entropy_loss_neg, (1 - labels).T)
    loss = loss_pos + loss_neg
    loss = loss / height_and_width
    return loss

def pair_wise_dice_loss(inputs: Tensor, labels: Tensor) -> Tensor:
    """
    A pair wise version of the dice loss, see `dice_loss` for usage.

    Args:
        inputs (`torch.Tensor`):
            A tensor representing a mask
        labels (`torch.Tensor`):
            A tensor with the same shape as inputs. Stores the binary classification labels for each element in inputs
            (0 for the negative class and 1 for the positive class).

    Returns:
        `torch.Tensor`: The computed loss between each pairs.
    """
    inputs = inputs.sigmoid().flatten(1)
    numerator = 2 * torch.matmul(inputs, labels.T)
    # using broadcasting to get a [num_queries, NUM_CLASSES] matrix
    denominator = inputs.sum(-1)[:, None] + labels.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss

def point_sample(input, point_coords, **kwargs):
    """
    A wrapper around :function:`torch.nn.functional.grid_sample` to support 3D point_coords tensors.
    Unlike :function:`torch.nn.functional.grid_sample` it assumes `point_coords` to lie inside
    [0, 1] x [0, 1] square.

    Args:
        input (Tensor): A tensor of shape (N, C, H, W) that contains features map on a H x W grid.
        point_coords (Tensor): A tensor of shape (N, P, 2) or (N, Hgrid, Wgrid, 2) that contains
        [0, 1] x [0, 1] normalized point coordinates.

    Returns:
        output (Tensor): A tensor of shape (N, C, P) or (N, C, Hgrid, Wgrid) that contains
            features for points in `point_coords`. The features are obtained via bilinear
            interplation from `input` the same way as :function:`torch.nn.functional.grid_sample`.
    """
    add_dim = False
    if point_coords.dim() == 3:
        add_dim = True
        point_coords = point_coords.unsqueeze(2)
    output = F.grid_sample(input, 2.0 * point_coords - 1.0, **kwargs)
    if add_dim:
        output = output.squeeze(3)
    return output

# Copied from transformers.models.deformable_detr.modeling_deformable_detr.DeformableDetrHungarianMatcher with DeformableDetr->GroundingDino
class GroundingDinoHungarianMatcher(nn.Module):
    """
    This class computes an assignment between the targets and the predictions of the network.
    For efficiency reasons, the targets don't include the no_object. Because of this, in general, there are more
    predictions than targets. In this case, we do a 1-to-1 matching of the best predictions, while the others are
    un-matched (and thus treated as non-objects).
    Args:
        class_cost:
            The relative weight of the classification error in the matching cost.
        bbox_cost:
            The relative weight of the L1 error of the bounding box coordinates in the matching cost.
        giou_cost:
            The relative weight of the giou loss of the bounding box in the matching cost.
    """

    def __init__(self, class_cost: float = 1, bbox_cost: float = 1, giou_cost: float = 1, mask_cost: float = 1, dice_cost: float=1,num_points: int = 12544):
        super().__init__()
        requires_backends(self, ["scipy"])

        self.class_cost = class_cost
        self.bbox_cost = bbox_cost
        self.giou_cost = giou_cost
        self.mask_cost = mask_cost
        self.dice_cost = dice_cost
        self.num_points = num_points
        if class_cost == 0 and bbox_cost == 0 and giou_cost == 0:
            raise ValueError("All costs of the Matcher can't be 0")

    @torch.no_grad()
    def forward(self, outputs, targets):
        """
        Args:
            outputs (`dict`):
                A dictionary that contains at least these entries:
                * "logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                * "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates.
            targets (`List[dict]`):
                A list of targets (len(targets) = batch_size), where each target is a dict containing:
                * "class_labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of
                  ground-truth
                 objects in the target) containing the class labels
                * "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates.
        Returns:
            `List[Tuple]`: A list of size `batch_size`, containing tuples of (index_i, index_j) where:
            - index_i is the indices of the selected predictions (in order)
            - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds: len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        # handle datasets w/o mask annotation
        has_mask_anno = "mask_labels" in targets[0].keys()

        batch_size, num_queries = outputs["logits"].shape[:2]

        indices = []

        for b in range(batch_size):
            out_bbox = outputs['pred_boxes'][b]
            tgt_bbox = targets[b]['boxes'].to(torch.float32)
            # cost of box
            cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
            cost_giou = -generalized_box_iou(center_to_corners_format(out_bbox), center_to_corners_format(tgt_bbox))

            # cost of class
            out_prob = outputs["logits"][b].sigmoid()
            tgt_ids = targets[b]["class_labels"]
            alpha = 0.25
            gamma = 2.0
            neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
            pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
            cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

            # for both det/instseg, use cls+box cost
            cost_matrix = (
                    self.bbox_cost * cost_bbox
                    + self.giou_cost * cost_giou
                    + self.class_cost * cost_class
                )

            # if has_mask_anno:
            #     # cost of mask
            #     out_mask = outputs["pred_masks"][b]
            #     tgt_mask = targets[b]["mask_labels"].to(out_mask)
            #     out_mask = out_mask[:, None]
            #     tgt_mask = tgt_mask[:, None]
            #     point_coords = torch.rand(1, self.num_points, 2, device=out_mask.device)
            #     tgt_mask = point_sample(
            #             tgt_mask,
            #             point_coords.repeat(tgt_mask.shape[0], 1, 1),
            #             align_corners=False,
            #         ).squeeze(1)

            #     out_mask = point_sample(
            #             out_mask,
            #             point_coords.repeat(out_mask.shape[0], 1, 1),
            #             align_corners=False,
            #         ).squeeze(1)
            #     cost_dice = pair_wise_dice_loss(out_mask, tgt_mask)
            #     cost_mask = pair_wise_sigmoid_cross_entropy_loss(out_mask, tgt_mask)
            #     cost_matrix = (
            #         self.bbox_cost * cost_bbox
            #         + self.giou_cost * cost_giou
            #         + self.class_cost * cost_class
            #         + self.mask_cost * cost_mask
            #         + self.dice_cost * cost_dice
            #     )
            # else:
            #     cost_matrix = (
            #         self.bbox_cost * cost_bbox
            #         + self.giou_cost * cost_giou
            #         + self.class_cost * cost_class
            #     )
            cost_matrix = cost_matrix.view(num_queries, -1).cpu()
            indices.append(linear_sum_assignment(cost_matrix))
        
        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]


# Copied from transformers.models.detr.modeling_detr._upcast
def _upcast(t: Tensor) -> Tensor:
    # Protects from numerical overflows in multiplications by upcasting to the equivalent higher type
    if t.is_floating_point():
        return t if t.dtype in (torch.float32, torch.float64) else t.float()
    else:
        return t if t.dtype in (torch.int32, torch.int64) else t.int()


# Copied from transformers.models.detr.modeling_detr.box_area
def box_area(boxes: Tensor) -> Tensor:
    """
    Computes the area of a set of bounding boxes, which are specified by its (x1, y1, x2, y2) coordinates.
    Args:
        boxes (`torch.FloatTensor` of shape `(number_of_boxes, 4)`):
            Boxes for which the area will be computed. They are expected to be in (x1, y1, x2, y2) format with `0 <= x1
            < x2` and `0 <= y1 < y2`.
    Returns:
        `torch.FloatTensor`: a tensor containing the area for each box.
    """
    boxes = _upcast(boxes)
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


# Copied from transformers.models.detr.modeling_detr.box_iou
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    left_top = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    right_bottom = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    width_height = (right_bottom - left_top).clamp(min=0)  # [N,M,2]
    inter = width_height[:, :, 0] * width_height[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


# Copied from transformers.models.detr.modeling_detr.generalized_box_iou
def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/. The boxes should be in [x0, y0, x1, y1] (corner) format.
    Returns:
        `torch.FloatTensor`: a [N, M] pairwise matrix, where N = len(boxes1) and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    if not (boxes1[:, 2:] >= boxes1[:, :2]).all():
        raise ValueError(f"boxes1 must be in [x0, y0, x1, y1] (corner) format, but got {boxes1}")
    if not (boxes2[:, 2:] >= boxes2[:, :2]).all():
        raise ValueError(f"boxes2 must be in [x0, y0, x1, y1] (corner) format, but got {boxes2}")
    iou, union = box_iou(boxes1, boxes2)

    top_left = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    bottom_right = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    width_height = (bottom_right - top_left).clamp(min=0)  # [N,M,2]
    area = width_height[:, :, 0] * width_height[:, :, 1]

    return iou - (area - union) / area


# Copied from transformers.models.detr.modeling_detr._max_by_axis
def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


# Copied from transformers.models.detr.modeling_detr.NestedTensor
class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


# Copied from transformers.models.detr.modeling_detr.nested_tensor_from_tensor_list
def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    if tensor_list[0].ndim == 3:
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        batch_shape = [len(tensor_list)] + max_size
        batch_size, num_channels, height, width = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((batch_size, height, width), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], : img.shape[2]] = False
    else:
        raise ValueError("Only 3-dimensional tensors are supported")
    return NestedTensor(tensor, mask)


# Copied from transformers.models.bert.modeling_bert.BertEmbeddings with Bert->GroundingDinoText
class GroundingDinoTextEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )
        self.register_buffer(
            "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values_length: int = 0,
    ) -> torch.Tensor:
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


# Copied from transformers.models.bert.modeling_bert.BertSelfAttention with Bert->GroundingDinoText
class GroundingDinoTextSelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        self.is_decoder = config.is_decoder

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        use_cache = past_key_value is not None
        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            query_length, key_length = query_layer.shape[2], key_layer.shape[2]
            if use_cache:
                position_ids_l = torch.tensor(key_length - 1, dtype=torch.long, device=hidden_states.device).view(
                    -1, 1
                )
            else:
                position_ids_l = torch.arange(query_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(key_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r

            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in GroundingDinoTextModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs


# Copied from transformers.models.bert.modeling_bert.BertSelfOutput with Bert->GroundingDinoText
class GroundingDinoTextSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertAttention with Bert->GroundingDinoText
class GroundingDinoTextAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        self.self = GroundingDinoTextSelfAttention(config, position_embedding_type=position_embedding_type)
        self.output = GroundingDinoTextSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


# Copied from transformers.models.bert.modeling_bert.BertIntermediate with Bert->GroundingDinoText
class GroundingDinoTextIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertOutput with Bert->GroundingDinoText
class GroundingDinoTextOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertLayer with Bert->GroundingDinoText
class GroundingDinoTextLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = GroundingDinoTextAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = GroundingDinoTextAttention(config, position_embedding_type="absolute")
        self.intermediate = GroundingDinoTextIntermediate(config)
        self.output = GroundingDinoTextOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


# Copied from transformers.models.bert.modeling_bert.BertEncoder with Bert->GroundingDinoText
class GroundingDinoTextEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([GroundingDinoTextLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


# Copied from transformers.models.bert.modeling_bert.BertPooler with Bert->GroundingDinoText
class GroundingDinoTextPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class GroundingDinoTextPrenet(GroundingDinoPreTrainedModel):
    config_class = GroundingDinoTextConfig

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = GroundingDinoTextEmbeddings(config)
        self.encoder = GroundingDinoTextEncoder(config)

        self.pooler = GroundingDinoTextPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPooling]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
    

# -------------- InternImage ---------------
    #InternImage
class to_channels_first(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 3, 1, 2)


class to_channels_last(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 2, 3, 1)


def build_norm_layer(dim,
                     norm_layer,
                     in_format='channels_last',
                     out_format='channels_last',
                     eps=1e-6):
    layers = []
    if norm_layer == 'BN':
        if in_format == 'channels_last':
            layers.append(to_channels_first())
        layers.append(nn.BatchNorm2d(dim))
        if out_format == 'channels_last':
            layers.append(to_channels_last())
    elif norm_layer == 'LN':
        if in_format == 'channels_first':
            layers.append(to_channels_last())
        layers.append(nn.LayerNorm(dim, eps=eps))
        if out_format == 'channels_first':
            layers.append(to_channels_first())
    else:
        raise NotImplementedError(
            f'build_norm_layer does not support {norm_layer}')
    return nn.Sequential(*layers)


def build_act_layer(act_layer):
    if act_layer == 'ReLU':
        return nn.ReLU(inplace=True)
    elif act_layer == 'SiLU':
        return nn.SiLU(inplace=True)
    elif act_layer == 'GELU':
        return nn.GELU()

    raise NotImplementedError(f'build_act_layer does not support {act_layer}')


class StemLayer(nn.Module):
    r""" Stem layer of InternImage
    Args:
        in_chans (int): number of input channels
        out_chans (int): number of output channels
        act_layer (str): activation layer
        norm_layer (str): normalization layer
    """

    def __init__(self,
                 in_chans=3,
                 out_chans=96,
                 act_layer='GELU',
                 norm_layer='BN'):
        super().__init__()
        self.conv1 = nn.Conv2d(in_chans,
                               out_chans // 2,
                               kernel_size=3,
                               stride=2,
                               padding=1)
        self.norm1 = build_norm_layer(out_chans // 2, norm_layer,
                                      'channels_first', 'channels_first')
        self.act = build_act_layer(act_layer)
        self.conv2 = nn.Conv2d(out_chans // 2,
                               out_chans,
                               kernel_size=3,
                               stride=2,
                               padding=1)
        self.norm2 = build_norm_layer(out_chans, norm_layer, 'channels_first',
                                      'channels_last')

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.norm2(x)
        return x


class DownsampleLayer(nn.Module):
    r""" Downsample layer of InternImage
    Args:
        channels (int): number of input channels
        norm_layer (str): normalization layer
    """

    def __init__(self, channels, norm_layer='LN'):
        super().__init__()
        self.conv = nn.Conv2d(channels,
                              2 * channels,
                              kernel_size=3,
                              stride=2,
                              padding=1,
                              bias=False)
        self.norm = build_norm_layer(2 * channels, norm_layer,
                                     'channels_first', 'channels_last')

    def forward(self, x):
        x = self.conv(x.permute(0, 3, 1, 2))
        x = self.norm(x)
        return x


class MLPLayer(nn.Module):
    r""" MLP layer of InternImage
    Args:
        in_features (int): number of input features
        hidden_features (int): number of hidden features
        out_features (int): number of output features
        act_layer (str): activation layer
        drop (float): dropout rate
    """

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer='GELU',
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = build_act_layer(act_layer)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class InternImageLayer(nn.Module):
    r""" Basic layer of InternImage
    Args:
        core_op (nn.Module): core operation of InternImage
        channels (int): number of input channels
        groups (list): Groups of each block.
        mlp_ratio (float): ratio of mlp hidden features to input channels
        drop (float): dropout rate
        drop_path (float): drop path rate
        act_layer (str): activation layer
        norm_layer (str): normalization layer
        post_norm (bool): whether to use post normalization
        layer_scale (float): layer scale
        offset_scale (float): offset scale
        with_cp (bool): whether to use checkpoint
    """

    def __init__(self,
                 core_op,
                 channels,
                 groups,
                 mlp_ratio=4.,
                 drop=0.,
                 drop_path=0.,
                 act_layer='GELU',
                 norm_layer='LN',
                 post_norm=False,
                 layer_scale=None,
                 offset_scale=1.0,
                 with_cp=False,
                 dw_kernel_size=None, # for InternImage-H/G
                 res_post_norm=False, # for InternImage-H/G
                 center_feature_scale=False): # for InternImage-H/G
        super().__init__()
        self.channels = channels
        self.groups = groups
        self.mlp_ratio = mlp_ratio
        self.with_cp = with_cp

        self.norm1 = build_norm_layer(channels, 'LN')
        self.post_norm = post_norm
        self.dcn = core_op(
            channels=channels,
            kernel_size=3,
            stride=1,
            pad=1,
            dilation=1,
            group=groups,
            offset_scale=offset_scale,
            act_layer=act_layer,
            norm_layer=norm_layer,
            dw_kernel_size=dw_kernel_size, # for InternImage-H/G
            center_feature_scale=center_feature_scale) # for InternImage-H/G
        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        self.norm2 = build_norm_layer(channels, 'LN')
        self.mlp = MLPLayer(in_features=channels,
                            hidden_features=int(channels * mlp_ratio),
                            act_layer=act_layer,
                            drop=drop)
        self.layer_scale = layer_scale is not None
        if self.layer_scale:
            self.gamma1 = nn.Parameter(layer_scale * torch.ones(channels),
                                       requires_grad=True)
            self.gamma2 = nn.Parameter(layer_scale * torch.ones(channels),
                                       requires_grad=True)
        self.res_post_norm = res_post_norm
        if res_post_norm:
            self.res_post_norm1 = build_norm_layer(channels, 'LN')
            self.res_post_norm2 = build_norm_layer(channels, 'LN')

    def forward(self, x):

        def _inner_forward(x):
            if not self.layer_scale:
                if self.post_norm:
                    x = x + self.drop_path(self.norm1(self.dcn(x)))
                    x = x + self.drop_path(self.norm2(self.mlp(x)))
                elif self.res_post_norm: # for InternImage-H/G
                    x = x + self.drop_path(self.res_post_norm1(self.dcn(self.norm1(x))))
                    x = x + self.drop_path(self.res_post_norm2(self.mlp(self.norm2(x))))
                else:
                    x = x + self.drop_path(self.dcn(self.norm1(x)))
                    x = x + self.drop_path(self.mlp(self.norm2(x)))
                return x
            if self.post_norm:
                x = x + self.drop_path(self.gamma1 * self.norm1(self.dcn(x)))
                x = x + self.drop_path(self.gamma2 * self.norm2(self.mlp(x)))
            else:
                x = x + self.drop_path(self.gamma1 * self.dcn(self.norm1(x)))
                x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
            return x

        if self.with_cp and x.requires_grad:
            x = checkpoint.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x


class InternImageBlock(nn.Module):
    r""" Block of InternImage
    Args:
        core_op (nn.Module): core operation of InternImage
        channels (int): number of input channels
        depths (list): Depth of each block.
        groups (list): Groups of each block.
        mlp_ratio (float): ratio of mlp hidden features to input channels
        drop (float): dropout rate
        drop_path (float): drop path rate
        act_layer (str): activation layer
        norm_layer (str): normalization layer
        post_norm (bool): whether to use post normalization
        layer_scale (float): layer scale
        offset_scale (float): offset scale
        with_cp (bool): whether to use checkpoint
    """

    def __init__(self,
                 core_op,
                 channels,
                 depth,
                 groups,
                 downsample=True,
                 mlp_ratio=4.,
                 drop=0.,
                 drop_path=0.,
                 act_layer='GELU',
                 norm_layer='LN',
                 post_norm=False,
                 offset_scale=1.0,
                 layer_scale=None,
                 with_cp=False,
                 dw_kernel_size=None, # for InternImage-H/G
                 post_norm_block_ids=None, # for InternImage-H/G
                 res_post_norm=False, # for InternImage-H/G
                 center_feature_scale=False): # for InternImage-H/G
        super().__init__()
        self.channels = channels
        self.depth = depth
        self.post_norm = post_norm
        self.center_feature_scale = center_feature_scale

        self.blocks = nn.ModuleList([
            InternImageLayer(
                core_op=core_op,
                channels=channels,
                groups=groups,
                mlp_ratio=mlp_ratio,
                drop=drop,
                drop_path=drop_path[i] if isinstance(
                    drop_path, list) else drop_path,
                act_layer=act_layer,
                norm_layer=norm_layer,
                post_norm=post_norm,
                layer_scale=layer_scale,
                offset_scale=offset_scale,
                with_cp=with_cp,
                dw_kernel_size=dw_kernel_size, # for InternImage-H/G
                res_post_norm=res_post_norm, # for InternImage-H/G
                center_feature_scale=center_feature_scale # for InternImage-H/G
            ) for i in range(depth)
        ])
        if not self.post_norm or center_feature_scale:
            self.norm = build_norm_layer(channels, 'LN')
        self.post_norm_block_ids = post_norm_block_ids
        if post_norm_block_ids is not None: # for InternImage-H/G
            self.post_norms = nn.ModuleList(
                [build_norm_layer(channels, 'LN', eps=1e-6) for _ in post_norm_block_ids]
            )
        self.downsample = DownsampleLayer(
            channels=channels, norm_layer=norm_layer) if downsample else None

    def forward(self, x, return_wo_downsample=False):
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if (self.post_norm_block_ids is not None) and (i in self.post_norm_block_ids):
                index = self.post_norm_block_ids.index(i)
                x = self.post_norms[index](x) # for InternImage-H/G
        if not self.post_norm or self.center_feature_scale:
            x = self.norm(x)
        if return_wo_downsample:
            x_ = x
        if self.downsample is not None:
            x = self.downsample(x)

        if return_wo_downsample:
            return x, x_
        return x


@BACKBONES.register_module()
class InternImage(nn.Module):
    r""" InternImage
        A PyTorch impl of : `InternImage: Exploring Large-Scale Vision Foundation Models with Deformable Convolutions`  -
          https://arxiv.org/pdf/2103.14030
    Args:
        core_op (str): Core operator. Default: 'DCNv3'
        channels (int): Number of the first stage. Default: 64
        depths (list): Depth of each block. Default: [3, 4, 18, 5]
        groups (list): Groups of each block. Default: [3, 6, 12, 24]
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        drop_rate (float): Probability of an element to be zeroed. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        act_layer (str): Activation layer. Default: 'GELU'
        norm_layer (str): Normalization layer. Default: 'LN'
        layer_scale (bool): Whether to use layer scale. Default: False
        cls_scale (bool): Whether to use class scale. Default: False
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
        dw_kernel_size (int): Size of the dwconv. Default: None
        level2_post_norm (bool): Whether to use level2 post norm. Default: False
        level2_post_norm_block_ids (list): Indexes of post norm blocks. Default: None
        res_post_norm (bool): Whether to use res post norm. Default: False
        center_feature_scale (bool): Whether to use center feature scale. Default: False
    """

    def __init__(self,
                 core_op='DCNv3',
                 channels=64,
                 depths=[3, 4, 18, 5],
                 groups=[3, 6, 12, 24],
                 mlp_ratio=4.,
                 drop_rate=0.,
                 drop_path_rate=0.2,
                 drop_path_type='linear',
                 act_layer='GELU',
                 norm_layer='LN',
                 layer_scale=None,
                 offset_scale=1.0,
                 post_norm=False,
                 with_cp=False,
                 dw_kernel_size=None,  # for InternImage-H/G
                 level2_post_norm=False,  # for InternImage-H/G
                 level2_post_norm_block_ids=None,  # for InternImage-H/G
                 res_post_norm=False,  # for InternImage-H/G
                 center_feature_scale=False,  # for InternImage-H/G
                 out_indices=(0, 1, 2, 3),
                 init_cfg=None,
                 **kwargs):
        super().__init__()
        self.core_op = core_op
        self.num_levels = len(depths)
        self.depths = depths
        self.channels = channels
        self.num_features = [int(channels * 2 ** i) for i in range(self.num_levels)]
        self.post_norm = post_norm
        self.mlp_ratio = mlp_ratio
        self.init_cfg = init_cfg
        self.out_indices = out_indices
        self.level2_post_norm_block_ids = level2_post_norm_block_ids
        logger = get_root_logger()
        logger.info(f'using core type: {core_op}')
        logger.info(f'using activation layer: {act_layer}')
        logger.info(f'using main norm layer: {norm_layer}')
        logger.info(f'using dpr: {drop_path_type}, {drop_path_rate}')
        logger.info(f"level2_post_norm: {level2_post_norm}")
        logger.info(f"level2_post_norm_block_ids: {level2_post_norm_block_ids}")
        logger.info(f"res_post_norm: {res_post_norm}")

        in_chans = 3
        self.patch_embed = StemLayer(in_chans=in_chans,
                                     out_chans=channels,
                                     act_layer=act_layer,
                                     norm_layer=norm_layer)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]
        if drop_path_type == 'uniform':
            for i in range(len(dpr)):
                dpr[i] = drop_path_rate

        self.levels = nn.ModuleList()
        for i in range(self.num_levels):
            post_norm_block_ids = level2_post_norm_block_ids if level2_post_norm and (
                i == 2) else None # for InternImage-H/G
            level = InternImageBlock(
                core_op=getattr(opsm, core_op),
                channels=int(channels * 2**i),
                depth=depths[i],
                groups=groups[i],
                mlp_ratio=self.mlp_ratio,
                drop=drop_rate,
                drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                act_layer=act_layer,
                norm_layer=norm_layer,
                post_norm=post_norm,
                downsample=(i < self.num_levels - 1),
                layer_scale=layer_scale,
                offset_scale=offset_scale,
                with_cp=with_cp,
                dw_kernel_size=dw_kernel_size,  # for InternImage-H/G
                post_norm_block_ids=post_norm_block_ids, # for InternImage-H/G
                res_post_norm=res_post_norm, # for InternImage-H/G
                center_feature_scale=center_feature_scale # for InternImage-H/G
            )
            self.levels.append(level)

        self.num_layers = len(depths)
        self.apply(self._init_weights)
        self.apply(self._init_deform_weights)

    def init_weights(self):
        logger = get_root_logger()
        if self.init_cfg is None:
            logger.warn(f'No pre-trained weights for '
                        f'{self.__class__.__name__}, '
                        f'training start from scratch')
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, 1.0)
        else:
            assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                  f'specify `Pretrained` in ' \
                                                  f'`init_cfg` in ' \
                                                  f'{self.__class__.__name__} '
            ckpt = _load_checkpoint(self.init_cfg.checkpoint,
                                    logger=logger,
                                    map_location='cpu')
            if 'state_dict' in ckpt:
                _state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                _state_dict = ckpt['model']
            else:
                _state_dict = ckpt

            state_dict = OrderedDict()
            for k, v in _state_dict.items():
                if k.startswith('backbone.'):
                    state_dict[k[9:]] = v
                else:
                    state_dict[k] = v

            # strip prefix of state_dict
            if list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k[7:]: v for k, v in state_dict.items()}

            # load state_dict
            meg = self.load_state_dict(state_dict, False)
            logger.info(meg)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _init_deform_weights(self, m):
        if isinstance(m, getattr(opsm, self.core_op)):
            m._reset_parameters()

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)

        seq_out = []
        for level_idx, level in enumerate(self.levels):
            x, x_ = level(x, return_wo_downsample=True)
            if level_idx in self.out_indices:
                seq_out.append(x_.permute(0, 3, 1, 2).contiguous())
        return seq_out

def build_internimage_h(cfg_hf):
    cfg = dict(
        core_op='DCNv3',
        channels=320,
        depths=[6, 6, 32, 6],
        groups=[10, 20, 40, 80],
        mlp_ratio=4.,
        drop_path_rate=0.,
        norm_layer='LN',
        layer_scale=None,
        offset_scale=1.0,
        post_norm=False,
        dw_kernel_size=5,  # for InternImage-H/G
        res_post_norm=True,  # for InternImage-H/G
        level2_post_norm=True,  # for InternImage-H/G
        level2_post_norm_block_ids=[5, 11, 17, 23, 29],  # for InternImage-H/G
        center_feature_scale=True,  # for InternImage-H/G
        with_cp=True,
        out_indices=(0, 1, 2, 3),
        init_cfg=None # dict(type='Pretrained', checkpoint=pretrained)
    )
    if isinstance(cfg_hf, dict):
        cfg.update(cfg_hf)

    load_path = cfg.get('load_path', None)
    if load_path is not None:
        print(f'--------------backbone load path: {load_path} ----------------')
        cfg['init_cfg'] = dict(type='Pretrained', checkpoint=load_path)
    backbone = InternImage(**cfg)
    backbone.num_features = [320,640,1280,2560]
    return backbone

class GroundingDinoInternImageBackbone(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        #backbone = AutoBackbone.from_config(config.backbone_config)
        backbone = build_internimage_h(config.backbone_config)

        # # replace batch norm by frozen batch norm
        # with torch.no_grad():
        #     replace_batch_norm(backbone)
        self.model = backbone
        self.intermediate_channel_sizes = self.model.num_features

    def forward(self, pixel_values: torch.Tensor, pixel_mask: torch.Tensor):
        # send pixel_values through the model to get list of feature maps
        features = self.model(pixel_values)

        out = []
        for feature_map in features:
            # downsample pixel_mask to match shape of corresponding feature_map
            mask = nn.functional.interpolate(pixel_mask[None].float(), size=feature_map.shape[-2:]).to(torch.bool)[0]
            out.append((feature_map, mask))
        return out