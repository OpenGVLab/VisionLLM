""" 
VisionLLMv2 configuration
"""

import copy
from re import M

from transformers import LlamaConfig, CLIPVisionConfig
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

# atom tools
from .grounding_dino.configuration_grounding_dino import GroundingDinoConfig
from .unipose.configuration_unipose import UniPoseConfig
from .stable_diffusion.modeling_sd import StableDiffusionWithLLMEmbConfig
from .instruct_pix2pix.modeling_instruct_pix2pix import InstructPix2PixWithLLMEmbConfig

logger = logging.get_logger(__name__)

class VisionLLMv2Config(PretrainedConfig):
    model_type = "visionllmv2"
    is_composition = True
    
    def __init__(
            self,
            vis_encoder_config=None,
            llm_config=None,
            pretrained_vl_bridge=None,
            use_llm_lora=False,           # llm lora
            vl_bridge_type="mlp2x_gelu",  # same as llava
            vis_output_layer=-2,          # same as llava
            use_pixelshuffle=False,
            num_embs=4,
            num_embs_gen=64,
            # region encoder
            use_region_encoder=False,
            # -------------------- atom tools ---------------------
            use_gdino=False,
            gdino_config=None,
            use_unipose=False,
            unipose_config=None,
            use_sd=False,
            sd_config=None,
            use_ip2p=False,
            ip2p_config=None,
            **kwargs
        ):
        super().__init__(**kwargs)

        if vis_encoder_config is None:
            vis_encoder_config = {}
            logger.info("vis_encoder_config is None. Initializing the CLIPVisionConfig with default values.")

        if llm_config is None:
            llm_config = {}
            logger.info("llm_config is None. Initializing the LlamaConfig with default values.")
        
        self.vis_encoder_config = CLIPVisionConfig(**vis_encoder_config)
        self.llm_config = LlamaConfig(**llm_config)

        self.pretrained_vl_bridge = pretrained_vl_bridge
        self.use_llm_lora = use_llm_lora
        self.vl_bridge_type = vl_bridge_type
        self.vis_output_layer = vis_output_layer
        self.use_pixelshuffle = use_pixelshuffle

        # [EMB]
        self.num_embs = num_embs
        self.num_embs_gen = num_embs_gen

        # region encoder
        self.use_region_encoder = use_region_encoder

        # -------------------- atom tools -----------------------
        self.use_gdino = use_gdino
        if self.use_gdino:
            if gdino_config is None:
                gdino_config = {}
                logger.info("gdino_config is None. Initializing the GroundingDinoConfig with default values.")
            self.gdino_config = GroundingDinoConfig(**gdino_config)

        self.use_unipose = use_unipose
        if self.use_unipose:
            if unipose_config is None:
                unipose_config = {}
                logger.info("unipose_config is None. Initializing the UniPoseConfig with default values.")
            self.unipose_config = UniPoseConfig(**unipose_config)

        self.use_sd = use_sd
        if self.use_sd:
            if sd_config is None:
                sd_config = {}
                logger.info("sd_config is None. Initializing the StableDiffusionWithLLMEmbConfig with default values.")
            self.sd_config = StableDiffusionWithLLMEmbConfig(**sd_config)

        self.use_ip2p = use_ip2p
        if self.use_ip2p:
            if ip2p_config is None:
                ip2p_config = {}
                logger.info("ip2p_config is None. Initializing the InstructPix2PixWithLLMEmbConfig with default values.")
            self.ip2p_config = InstructPix2PixWithLLMEmbConfig(**ip2p_config)


    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        output["vis_encoder_config"] = self.vis_encoder_config.to_dict()
        output["llm_config"] = self.llm_config.to_dict()
        # ------------------- atom tools ---------------------
        output["gdino_config"] = self.gdino_config.to_dict() if self.use_gdino else None
        output["unipose_config"] = self.unipose_config.to_dict() if self.use_unipose else None
        output['sd_config'] = self.sd_config.to_dict() if self.use_sd else None
        output['ip2p_config'] = self.ip2p_config.to_dict() if self.use_ip2p else None
        output["model_type"] = self.__class__.model_type
        return output
