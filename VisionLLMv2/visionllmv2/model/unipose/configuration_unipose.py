import os
import copy
from typing import Union, Dict

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from transformers.models.auto import CONFIG_MAPPING

logger = logging.get_logger(__name__)

class UniPoseConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`GroundingDinoModel`]. It is used to instantiate a
    Grounding DINO model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the Grounding DINO
    [EduardoPacheco/grounding-dino-tiny](https://huggingface.co/EduardoPacheco/grounding-dino-tiny) architecture.
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    Args:
        backbone_config (`PretrainedConfig` or `dict`, *optional*, defaults to `ResNetConfig()`):
            The configuration of the backbone model.
        text_backbone_config (`str`, *optional*, defaults to `GroundingDinoTextConfig()`):
            The configuration of the text backbone model. Should be a BERT-like config.
        num_queries (`int`, *optional*, defaults to 900):
            Number of object queries, i.e. detection slots. This is the maximal number of objects
            [`GroundingDinoModel`] can detect in a single image.
        encoder_layers (`int`, *optional*, defaults to 6):
            Number of encoder layers.
        encoder_ffn_dim (`int`, *optional*, defaults to 2048):
            Dimension of the "intermediate" (often named feed-forward) layer in decoder.
        encoder_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads for each attention layer in the Transformer encoder.
        decoder_layers (`int`, *optional*, defaults to 6):
            Number of decoder layers.
        decoder_ffn_dim (`int`, *optional*, defaults to 2048):
            Dimension of the "intermediate" (often named feed-forward) layer in decoder.
        decoder_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads for each attention layer in the Transformer decoder.
        is_encoder_decoder (`bool`, *optional*, defaults to `True`):
            Whether the model is used as an encoder/decoder or not.
        activation_function (`str` or `function`, *optional*, defaults to `"relu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        d_model (`int`, *optional*, defaults to 256):
            Dimension of the layers.
        dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        activation_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for activations inside the fully connected layer.
        auxiliary_loss (`bool`, *optional*, defaults to `False`):
            Whether auxiliary decoding losses (loss at each decoder layer) are to be used.
        position_embedding_type (`str`, *optional*, defaults to `"sine"`):
            Type of position embeddings to be used on top of the image features. One of `"sine"` or `"learned"`.
        num_feature_levels (`int`, *optional*, defaults to 4):
            The number of input feature levels.
        encoder_n_points (`int`, *optional*, defaults to 4):
            The number of sampled keys in each feature level for each attention head in the encoder.
        decoder_n_points (`int`, *optional*, defaults to 4):
            The number of sampled keys in each feature level for each attention head in the decoder.
        two_stage (`bool`, *optional*, defaults to `True`):
            Whether to apply a two-stage deformable DETR, where the region proposals are also generated by a variant of
            Grounding DINO, which are further fed into the decoder for iterative bounding box refinement.
        class_cost (`float`, *optional*, defaults to 1.0):
            Relative weight of the classification error in the Hungarian matching cost.
        bbox_cost (`float`, *optional*, defaults to 5.0):
            Relative weight of the L1 error of the bounding box coordinates in the Hungarian matching cost.
        giou_cost (`float`, *optional*, defaults to 2.0):
            Relative weight of the generalized IoU loss of the bounding box in the Hungarian matching cost.
        bbox_loss_coefficient (`float`, *optional*, defaults to 5.0):
            Relative weight of the L1 bounding box loss in the object detection loss.
        giou_loss_coefficient (`float`, *optional*, defaults to 2.0):
            Relative weight of the generalized IoU loss in the object detection loss.
        focal_alpha (`float`, *optional*, defaults to 0.25):
            Alpha parameter in the focal loss.
        disable_custom_kernels (`bool`, *optional*, defaults to `False`):
            Disable the use of custom CUDA and CPU kernels. This option is necessary for the ONNX export, as custom
            kernels are not supported by PyTorch ONNX export.
        max_text_len (`int`, *optional*, defaults to 256):
            The maximum length of the text input.
        text_enhancer_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the text enhancer.
        fusion_droppath (`float`, *optional*, defaults to 0.1):
            The droppath ratio for the fusion module.
        fusion_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the fusion module.
        embedding_init_target (`bool`, *optional*, defaults to `True`):
            Whether to initialize the target with Embedding weights.
        query_dim (`int`, *optional*, defaults to 4):
            The dimension of the query vector.
        decoder_bbox_embed_share (`bool`, *optional*, defaults to `True`):
            Whether to share the bbox embedding between the decoder and the two-stage bbox generator.
        two_stage_bbox_embed_share (`bool`, *optional*, defaults to `False`):
            Whether to share the bbox embedding between the two-stage bbox generator and the region proposal
            generation.
        positional_embedding_temperature (`float`, *optional*, defaults to 20):
            The temperature for Sine Positional Embedding that is used together with vision backbone.
        init_std (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
    Examples:
    ```python
    >>> from transformers import GroundingDinoConfig, GroundingDinoModel
    >>> # Initializing a Grounding DINO EduardoPacheco/grounding-dino-tiny style configuration
    >>> configuration = GroundingDinoConfig()
    >>> # Initializing a model (with random weights) from the EduardoPacheco/grounding-dino-tiny style configuration
    >>> model = GroundingDinoModel(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "unipose"
    attribute_map = {
        "hidden_size": "d_model",
        "num_attention_heads": "encoder_attention_heads",
    }

    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)
        #args: argparse Namespace
        #transform args to dict   
        # if args is not None:
        #     args = vars(args)
        #     #print(args)
        #     for k, v in args['_cfg_dict'].items():
        #         setattr(self, k, v)
        for k, v in kwargs.items():
            setattr(self, k, v)

    @property
    def num_attention_heads(self) -> int:
        return self.encoder_attention_heads

    @property
    def hidden_size(self) -> int:
        return self.d_model

    def to_dict(self) -> Dict[str, any]:
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        #output["backbone_config"] = self.backbone_config.to_dict()
        #output["text_backbone_config"] = self.text_backbone_config.to_dict()
        #output["model_type"] = self.__class__.model_type
        return output

