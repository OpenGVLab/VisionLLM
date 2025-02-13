"""
VisionLLMv2 model
"""

from dataclasses import dataclass
import json

import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn import CrossEntropyLoss
from typing import List, Optional, Tuple, Union, Dict, Any
import itertools

from transformers.utils import logging, ModelOutput
from transformers import LlamaModel, LlamaForCausalLM
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers import AutoConfig, AutoModel, \
    PretrainedConfig, PreTrainedModel, CLIPVisionModel, CLIPImageProcessor
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType

# vision / llm
from .internvit.modeling_intern_vit import InternVisionModel
from .internlm2.modeling_internlm2 import InternLM2ForCausalLM

# det/grd/seg
from .grounding_dino.modeling_ov_grounding_dino_mask_dn import GroundingDinoObjectDetectionOutput
from .grounding_dino.modeling_ov_grounding_dino_mask_dn import OVGroundingDinoForObjectDetection

# pose
from .unipose.modeling_unipose import UniPose
from .unipose.modeling_unipose import UniPoseOutput

# gen/edit
from .stable_diffusion.modeling_sd import StableDiffusionWithLLMEmb, StableDiffusionWithLLMEmbOutput
from .instruct_pix2pix.modeling_instruct_pix2pix import InstructPix2PixWithLLMEmb, InstructPix2PixWithLLMEmbOutput

from ..constant import IGNORE_INDEX, DEFAULT_TOKENS
from ..util.box_ops import box_cxcywh_to_xyxy
from ..util.misc import nested_tensor_from_tensor_list

from visionllmv2.train.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
from visionllmv2.train.llama_forward_monkey_patch import replace_llama_rmsnorm_with_fused_rmsnorm, replace_llama_forward_with_custom_forward

from .configuration_visionllmv2 import VisionLLMv2Config
from .region_encoder import RegionEncoder

replace_llama_rmsnorm_with_fused_rmsnorm()
# replace_llama_forward_with_custom_forward()  

logger = logging.get_logger(__name__)


@dataclass
class VisionLLMv2ModelOutput(ModelOutput):
    """
    Class defining the outputs of [`VisionLLMv2`].
    """
    # CausalLMOutputWithPast
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

    # -------------------- atom tools --------------------------
    loss_gdino: Optional[torch.FloatTensor] = None
    gdino_outputs: Optional[GroundingDinoObjectDetectionOutput] = None

    loss_unipose: Optional[torch.FloatTensor] = None
    unipose_outputs: Optional[UniPoseOutput] = None

    loss_sd: Optional[torch.FloatTensor] = None
    sd_outputs: Optional[StableDiffusionWithLLMEmbOutput] = None

    loss_ip2p: Optional[torch.FloatTensor] = None
    ip2p_outputs: Optional[InstructPix2PixWithLLMEmbOutput] = None



class VisionLLMv2PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = VisionLLMv2Config
    base_model_prefix = "visionllmv2"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if hasattr(module, "bias") and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value
        # if isinstance(module, CLIPVisionModel):
        #     module.gradient_checkpointing = value
        # if isinstance(module, LlamaModel):
        #     module.gradient_checkpointing = value



class VisionLLMv2Model(VisionLLMv2PreTrainedModel):
    config_class = VisionLLMv2Config
    supports_gradient_checkpointing = True

    def __init__(self, config: VisionLLMv2Config, 
                vis_encoder=None, llm=None, 
                tokenizer=None,
                # ----------- atom tools ----------------
                gdino=None, unipose=None,
                sd=None, ip2p=None
        ):
        super().__init__(config)

        # vis encoder and llm
        if vis_encoder is not None:
            self.vis_encoder = vis_encoder
        else:
            if config.vis_encoder_config.architectures == ['InternVisionModel']:
                self.vis_encoder = InternVisionModel(config.vis_encoder_config)
            else:
                self.vis_encoder = CLIPVisionModel(config.vis_encoder_config)

        if llm is not None:
            self.llm = llm
        else:
            if config.llm_config.architectures == ['InternLM2ForCausalLM']:
                self.llm = InternLM2ForCausalLM(config.llm_config)
            else:
                self.llm = LlamaForCausalLM(config.llm_config)

        # wrap lora for llm
        if config.use_llm_lora:
            self.wrap_llm_lora()

        # build region encoder
        self.use_region_encoder = config.use_region_encoder
        if config.use_region_encoder:
            self.build_region_encoder()

        # tokenizer
        self.tokenizer = tokenizer

        self.use_pixelshuffle = config.use_pixelshuffle

        # vl bridge
        self.v_hidden_size = self.vis_encoder.config.hidden_size   # 1024
        self.l_hidden_size = self.llm.config.hidden_size           # 4096
        vl_bridge_type = getattr(config, 'vl_bridge_type', 'linear')
        v_hidden_size = self.v_hidden_size * 4 if self.use_pixelshuffle else self.v_hidden_size
        if vl_bridge_type == "linear":
            self.vl_bridge = nn.Linear(v_hidden_size, self.l_hidden_size)
        elif vl_bridge_type == 'internvl_mlp' or vl_bridge_type == 'internvl':
            self.vl_bridge = nn.Sequential(
                nn.LayerNorm(v_hidden_size),
                nn.Linear(v_hidden_size, self.l_hidden_size),
                nn.GELU(),
                nn.Linear(self.l_hidden_size, self.l_hidden_size)
            )
        else:
            mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu*', vl_bridge_type)
            if mlp_gelu_match:
                mlp_depth = int(mlp_gelu_match.group(1))
                modules = []
                modules.append(nn.Linear(v_hidden_size, self.l_hidden_size))
                for _ in range(1, mlp_depth):
                    modules.append(nn.GELU())
                    modules.append(nn.Linear(self.l_hidden_size, self.l_hidden_size))
                self.vl_bridge = nn.Sequential(*modules)
            else:
                raise NotImplementedError(f"{vl_bridge_type} not supported yet.")
        if config.pretrained_vl_bridge is not None:
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}
            print("Load vl_bridge weights from {}.".format(config.pretrained_vl_bridge))
            vl_bridge_weights = torch.load(config.pretrained_vl_bridge, map_location='cpu')
            self.vl_bridge.load_state_dict(get_w(vl_bridge_weights, 'vl_bridge'), strict=True)

        # ------------------------ atom tools -------------------------------
        self.use_gdino = config.use_gdino
        if self.use_gdino:
            if gdino is not None:
                self.gdino = gdino
            else:
                self.gdino = OVGroundingDinoForObjectDetection(config.gdino_config)

        self.use_unipose = config.use_unipose
        if self.use_unipose:
            if unipose is not None:
                self.unipose = unipose
            else:
                self.unipose = UniPose(config.unipose_config)

        self.use_sd = config.use_sd
        if self.use_sd:
            if sd is not None:
                self.sd = sd
            else:
                self.sd = StableDiffusionWithLLMEmb(config.sd_config)

        self.use_ip2p = config.use_ip2p
        if self.use_ip2p:
            if ip2p is not None:
                self.ip2p = ip2p
            else:
                self.ip2p = InstructPix2PixWithLLMEmb(config.ip2p_config)

        # emb_embeddings
        self.num_embs = config.num_embs
        self.num_embs_gen = config.num_embs_gen
        self.emb_embeddings_det = nn.Embedding(self.num_embs, self.l_hidden_size)
        self.emb_embeddings_pose = nn.Embedding(self.num_embs, self.l_hidden_size)
        self.emb_embeddings_gen = nn.Embedding(self.num_embs_gen, self.l_hidden_size)
        self.emb_embeddings_edit = nn.Embedding(self.num_embs_gen, self.l_hidden_size)
        

        # self.post_init()

    def init_special_token_ids(self, tokenizer):
        self.pad_token_id = tokenizer.pad_token_id
        self.img_token_id = tokenizer.convert_tokens_to_ids([DEFAULT_TOKENS['img']])[0]  # <image>
        self.imp_token_id = tokenizer.convert_tokens_to_ids([DEFAULT_TOKENS['imp']])[0]  # <im_patch>
        self.reg_token_id = tokenizer.convert_tokens_to_ids([DEFAULT_TOKENS['reg']])[0]  # <region>
        self.emb_token_id = tokenizer.convert_tokens_to_ids([DEFAULT_TOKENS['emb']])[0]  # start token id of embs
        # atom tools id
        self.det_tool_id = tokenizer.convert_tokens_to_ids([DEFAULT_TOKENS['det']])[0]
        self.grd_tool_id = tokenizer.convert_tokens_to_ids([DEFAULT_TOKENS['grd']])[0]
        self.seg_tool_id = tokenizer.convert_tokens_to_ids([DEFAULT_TOKENS['seg']])[0]
        self.pose_tool_id = tokenizer.convert_tokens_to_ids([DEFAULT_TOKENS['pose']])[0]
        self.gen_tool_id = tokenizer.convert_tokens_to_ids([DEFAULT_TOKENS['gen']])[0]
        self.edit_tool_id = tokenizer.convert_tokens_to_ids([DEFAULT_TOKENS['edit']])[0]
        return 
    
    def build_region_encoder(self):
        embed_dim = self.vis_encoder.config.hidden_size  # 1024
        out_dim = self.llm.config.hidden_size            # 4096
        patch_size = self.vis_encoder.config.patch_size
        self.region_encoder = RegionEncoder(hidden_dim=256, embed_dim=embed_dim, out_dim=out_dim, 
                                patch_size=patch_size, mask_pool_type='grid_sample')


    def wrap_llm_lora(self, r=32, lora_alpha=64, lora_dropout=0.05):
        if self.config.llm_config.architectures == ['InternLM2ForCausalLM']:
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                target_modules=["wqkv", "wo", "w2", "w3"],
                inference_mode=False,
                r=r, 
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout
            )
        else:  # llama
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                "mlp.down_proj", "mlp.up_proj"],
                inference_mode=False,
                r=r, 
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout
            )
        self.llm = get_peft_model(self.llm, lora_config)
        # input/output_embeddings requires grad
        self.llm.get_input_embeddings().weight.requires_grad_(True)
        self.llm.get_output_embeddings().weight.requires_grad_(True)
        self.llm.print_trainable_parameters()


    def _freeze_params(self, module):
        for param in module.parameters():
            param.requires_grad_(False)

    def freeze_vis_encoder(self):
        self.vis_encoder.requires_grad_(False)

    def freeze_llm(self):
        self.llm.requires_grad_(False)

    def freeze_vl_bridge(self):
        self.vl_bridge.requires_grad_(False)

    def freeze_region_encoder(self):
        if getattr(self, 'region_encoder', None) is not None:
            self.region_encoder.requires_grad_(False)

    def freeze_emb_embeddings(self):
        self.emb_embeddings_det.requires_grad_(False)
        self.emb_embeddings_pose.requires_grad_(False)
        self.emb_embeddings_gen.requires_grad_(False)
        self.emb_embeddings_edit.requires_grad_(False)

    def get_vis_encoder(self):
        return getattr(self, 'vis_encoder', None)

    def get_llm(self):
        return getattr(self, 'llm', None)

    def get_vl_bridge(self):
        return getattr(self, 'vl_bridge', None)
    
    def get_region_encoder(self):
        return getattr(self, 'region_encoder', None)

    # --------------------- atom tools -------------------
    def get_gdino(self):
        return getattr(self, 'gdino', None)
    
    def freeze_gdino(self):
        if getattr(self, 'gdino', None) is not None:
            self.gdino.requires_grad_(False)

    def get_unipose(self):
        return getattr(self, 'unipose', None)
    
    def freeze_unipose(self):
        if getattr(self, 'unipose', None) is not None:
            self.unipose.requires_grad_(False)

    def get_sd(self):
        return getattr(self, 'sd', None)
    
    def freeze_sd(self):
        if getattr(self, 'sd', None) is not None:
            self.sd.requires_grad_(False)

    def get_ip2p(self):
        return getattr(self, 'ip2p', None)
    
    def freeze_ip2p(self):
        if getattr(self, 'ip2p', None) is not None:
            self.ip2p.requires_grad_(False)

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past

    def prepare_inputs_for_generation(
            self,
            input_ids,
            past_key_values=None,
            attention_mask=None,
            inputs_embeds=None,
            **kwargs):

        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:  # for the following geration steps
            model_inputs = {"input_ids": input_ids}

        model_inputs.update({
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "use_cache": kwargs.get("use_cache"),
            "images": kwargs.get("images", None),
            "regions": kwargs.get("regions", None),  
        })
        return model_inputs

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(n, int(h * scale_factor), int(w * scale_factor),
                   int(c / (scale_factor * scale_factor)))
        # N,  W * scale, H * scale,  C // (scale ** 2)
        x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            images: Optional[list] = None,   # 'pad': [bs, 3, h, w], 'anyres': list[tensor], bs x [1 + n_split, 3, h, w]
            regions: Optional[list] = None,  # this is for region referring or visual prompt location. List[tensor], bs x [n_region, h, w], 0-1 mask.
            num_splits: Optional[List] = None,  # list[list[int]]: bs x n_image x [n_split], used when using mmic data
            # ----------------------------------
            images_aug: Optional[list] = None,
            targets: Optional[list] = None,
            img_metas: Optional[list] = None,
            # ----------------------------------
            input_images: Optional[list] = None,  # [bs, 3, h, w]
            output_images: Optional[list] = None, # [bs, 3, h, w]
            captions: Optional[list] = None,      # list[str]
            # ----------------------------------
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            use_cache: Optional[bool] = False,
            output_attentions: Optional[bool] = False,
            output_hidden_states: Optional[bool] = False,
            return_dict: Optional[bool] = False,
    ) -> Union[Tuple, VisionLLMv2ModelOutput]:
        # TODO beautify (init images_pos within forward)
        if inputs_embeds is None:
            inputs_embeds = self.llm.get_input_embeddings()(input_ids)
        
        # ------------------------------------------------------------
        # NOTE: special operation for the [emb] tokens, this works well for both train and generation (use_cache=True)
        # replace with tool emb_embeddings
        # and concat emb ids after special tool ids
        if self.det_tool_id in input_ids[0] or self.seg_tool_id in input_ids[0] or self.grd_tool_id in input_ids[0] or \
            self.pose_tool_id in input_ids[0] or self.gen_tool_id in input_ids[0] or self.edit_tool_id in input_ids[0]:
            if self.emb_token_id not in input_ids[0]:  # for generation, generate tokens 1 by 1
                gap_len, gap_len_gen = 0, 0
            else:   # for training, we have added the [EMB] tokens in the input_ids
                gap_len, gap_len_gen = self.num_embs, self.num_embs_gen
        emb_ids = torch.tensor([x for x in range(self.emb_token_id, self.emb_token_id + self.num_embs)], dtype=torch.long).to(input_ids.device)
        emb_embeddings_det = self.emb_embeddings_det.weight.unsqueeze(0).repeat(inputs_embeds.shape[0], 1, 1)    # [bs, num_embeds, c]
        emb_embeddings_pose = self.emb_embeddings_pose.weight.unsqueeze(0).repeat(inputs_embeds.shape[0], 1, 1)  # [bs, num_embeds, c]
        emb_ids_gen = self.emb_token_id * torch.ones(self.num_embs_gen, dtype=torch.long).to(input_ids.device)   
        emb_embeddings_gen = self.emb_embeddings_gen.weight.unsqueeze(0).repeat(inputs_embeds.shape[0], 1, 1)    # [bs, num_embs_gen, c]
        emb_embeddings_edit = self.emb_embeddings_edit.weight.unsqueeze(0).repeat(inputs_embeds.shape[0], 1, 1)  # [bs, num_embs_gen, c]
        new_inputs_embeds, new_input_ids = [], []
        # for each batch
        for cur_input_ids, cur_input_embeds, cur_emb_embeddings_det, cur_emb_embeddings_pose, cur_emb_embeddings_gen, cur_emb_embeddings_edit in zip(
                input_ids, inputs_embeds, emb_embeddings_det, emb_embeddings_pose, emb_embeddings_gen, emb_embeddings_edit):
            emb_start_pos_det = torch.where(cur_input_ids==self.det_tool_id)[0]
            emb_start_pos_seg = torch.where(cur_input_ids==self.seg_tool_id)[0]
            emb_start_pos_grd = torch.where(cur_input_ids==self.grd_tool_id)[0]
            emb_start_pos_det = torch.cat([emb_start_pos_det, emb_start_pos_seg, emb_start_pos_grd], dim=0) # using gdino
            emb_start_pos_pose = torch.where(cur_input_ids==self.pose_tool_id)[0]  # using unipose
            emb_start_pos_gen = torch.where(cur_input_ids==self.gen_tool_id)[0]    # using sd
            emb_start_pos_edit = torch.where(cur_input_ids==self.edit_tool_id)[0]  # using ip2p
            cur_new_input_ids = cur_input_ids
            cur_new_input_embeds = cur_input_embeds
            # using gdino
            for i, _start_pos in enumerate(emb_start_pos_det):
                # concat emb ids
                cur_new_input_ids = torch.cat(
                    [
                        cur_new_input_ids[: _start_pos + 1],
                        emb_ids,
                        cur_new_input_ids[_start_pos + gap_len + 1 :]
                    ], dim=0
                )
                # repalce with emb embeddings
                cur_new_input_embeds = torch.cat(
                    [
                        cur_new_input_embeds[: _start_pos + 1],
                        cur_emb_embeddings_det,
                        cur_new_input_embeds[_start_pos + gap_len + 1 :]
                    ], dim=0
                ).contiguous()  # replace with self.emb_embeddings
            # using unipose
            for i, _start_pos in enumerate(emb_start_pos_pose):
                # concat emb ids
                cur_new_input_ids = torch.cat(
                    [
                        cur_new_input_ids[: _start_pos + 1],
                        emb_ids,
                        cur_new_input_ids[_start_pos + gap_len + 1 :]
                    ], dim=0
                )
                # repalce with emb embeddings
                cur_new_input_embeds = torch.cat(
                    [
                        cur_new_input_embeds[: _start_pos + 1],
                        cur_emb_embeddings_pose,
                        cur_new_input_embeds[_start_pos + gap_len + 1 :]
                    ], dim=0
                ).contiguous()  # replace with self.emb_embeddings
            # using sd
            for i, _start_pos in enumerate(emb_start_pos_gen):
                # concat emb ids
                cur_new_input_ids = torch.cat(
                    [
                        cur_new_input_ids[: _start_pos + 1],
                        emb_ids_gen,
                        cur_new_input_ids[_start_pos + gap_len_gen + 1 :]
                    ], dim=0
                )
                # repalce with emb embeddings
                cur_new_input_embeds = torch.cat(
                    [
                        cur_new_input_embeds[: _start_pos + 1],
                        cur_emb_embeddings_gen,
                        cur_new_input_embeds[_start_pos + gap_len_gen + 1 :]
                    ], dim=0
                ).contiguous()  # replace with self.emb_embeddings
            # using ip2p
            for i, _start_pos in enumerate(emb_start_pos_edit):
                # concat emb ids
                cur_new_input_ids = torch.cat(
                    [
                        cur_new_input_ids[: _start_pos + 1],
                        emb_ids_gen,
                        cur_new_input_ids[_start_pos + gap_len_gen + 1 :]
                    ], dim=0
                )
                # repalce with emb embeddings
                cur_new_input_embeds = torch.cat(
                    [
                        cur_new_input_embeds[: _start_pos + 1],
                        cur_emb_embeddings_edit,
                        cur_new_input_embeds[_start_pos + gap_len_gen + 1 :]
                    ], dim=0
                ).contiguous()  # replace with self.emb_embeddings
            # assert cur_new_input_embeds.shape[0] == cur_input_embeds.shape[0]
            new_input_ids.append(cur_new_input_ids)
            new_inputs_embeds.append(cur_new_input_embeds)
        input_ids = torch.stack(new_input_ids, dim=0)
        inputs_embeds = torch.stack(new_inputs_embeds, dim=0)
        # --------------------------------------------------------
        # for generate
        if past_key_values is not None:  # following generation steps
            assert attention_mask.shape[0] == 1, "Only support bs=1 generation"
            past_length = past_key_values[0][0].shape[2]
            if attention_mask.shape[1] != past_length + 1:  # because having [emb] tokens hidden_states in the input_ids, pad
                add_length = past_length + 1 - attention_mask.shape[1]
                attention_mask = torch.cat(
                    [attention_mask, attention_mask.new_ones((attention_mask.shape[0], add_length))], dim=-1
                )
        else:  # useful for multi-round chat
            total_length = input_ids.shape[1]
            if attention_mask.shape[1] != total_length: 
                add_length = total_length - attention_mask.shape[1]
                attention_mask = torch.cat(
                    [attention_mask, attention_mask.new_ones((attention_mask.shape[0], add_length))], dim=-1
                )
        if past_key_values is not None:
            # having [emb] in this generation step
            if input_ids.shape[1] != 1:  
                if self.gen_tool_id in input_ids or self.edit_tool_id in input_ids:  # visual generation
                    attention_mask = torch.cat(
                        [attention_mask, attention_mask.new_ones((attention_mask.shape[0], self.num_embs_gen))], dim=-1
                    )
                else:  # visual perception
                    attention_mask = torch.cat(
                        [attention_mask, attention_mask.new_ones((attention_mask.shape[0], self.num_embs))], dim=-1
                    )
        # ------------------------------------------------------------

        # for the 1st step generation
        if past_key_values is None:
            with torch.no_grad():
                if type(images) == list:  # 'anyres'
                    images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images] # list[tensor], bs x [n_split, 3, h , w]
                    split_sizes = [image.shape[0] for image in images]              # list[int]
                    concat_images = torch.cat([image for image in images], dim=0)   # [n_all_image_splits, 3, h, w]
                    image_forward_outs = self.vis_encoder(concat_images, output_hidden_states=True) 
                else:  # 'pad'
                    # here images: after clip preprocess, [b, 3, h, w]
                    image_forward_outs = self.vis_encoder(images, output_hidden_states=True)  
                select_hidden_state_layer = getattr(self.config, "vis_output_layer", -2)
                select_hidden_state = image_forward_outs.hidden_states[select_hidden_state_layer]
                image_features_ori = select_hidden_state[:, 1:].to(self.llm.dtype)  # [bs, img_len, 1024] or [n_all_image_splits, img_len, 1024]
            
            # pixel shuffle
            if self.use_pixelshuffle:
                h = w = int(image_features_ori.shape[1] ** 0.5)
                image_features_ori = image_features_ori.reshape(image_features_ori.shape[0], h, w, -1)
                image_features_ori = self.pixel_shuffle(image_features_ori, scale_factor=0.5)
                image_features_ori = image_features_ori.reshape(image_features_ori.shape[0], -1, image_features_ori.shape[-1])
            image_features = self.vl_bridge(image_features_ori).to(inputs_embeds.dtype)  # [bs, img_len, 4096] or [n_all_image_splits, img_len, 4096]

            # replace image patch token for multi-modal/nlp datasets
            B, L, C = inputs_embeds.shape 
            inputs_embeds = inputs_embeds.reshape(B * L, C)
            selected = input_ids == self.imp_token_id             
            has_image = selected.sum(-1) != 0                   
            if type(images) == list:
                has_image = [has_image[i][None].repeat(split_sizes[i]) for i in range(B)]
                has_image = torch.cat(has_image, dim=0)
            selected = selected.reshape(-1)                           # [B*L]
            # handle interleaved data when num(<images>) != num(images)
            try:
                vit_embeds = image_features[has_image].reshape(-1, C)
                inputs_embeds[selected] = inputs_embeds[selected] * 0.0 + vit_embeds
                ignore_flag = False
            except Exception as e:
                vit_embeds = image_features[has_image].reshape(-1, C) 
                print(f'warning: {e}, inputs_embeds[selected].shape={inputs_embeds[selected].shape}, '
                  f'vit_embeds.shape={vit_embeds.shape}')
                n_selected_token = selected.sum()
                n_vit_token = vit_embeds.shape[0]
                vit_embeds = vit_embeds.repeat(n_selected_token // n_vit_token, 1) if n_selected_token > n_vit_token \
                                else vit_embeds[:n_vit_token]
                inputs_embeds[selected] = inputs_embeds[selected] * 0.0  + vit_embeds
                ignore_flag = True
            inputs_embeds = inputs_embeds.reshape(B, L, C)


            # deal with region/non-region data joint train with zero2/3
            if self.use_region_encoder:
                if regions is not None:   # region data, list[tensor], bs x [n_region, h, w]
                    # concat regions in the batch dimension
                    # regions: list[tensor], bs x [n_region, h, w]
                    num_regions = [len(regions_per_batch) for regions_per_batch in regions]      
                    all_regions = [regions_per_batch[:, None] for regions_per_batch in regions] 
                    all_regions = torch.cat(all_regions, dim=0)  
                    # deal with model.generate() when num_beams > 1
                    num_beams = len(images) // len(regions)
                    if num_beams > 1:
                        num_regions = num_regions * num_beams
                        all_regions = all_regions.repeat(num_beams, 1, 1, 1)
                        
                    #########################################################
                    # all_images
                    # repeat image and image_features, [bs, 3, h, w], [bs, 1024, c]
                    if num_splits is not None:  # mmic-data
                        all_images = []
                        for i, (images_per_batch, num_splits_per_batch) in enumerate(zip(images, num_splits)):
                            # image_per_batch: [n_images_and_splits, 3, h, w]
                            # num_splits_per_batch: list[int]
                            cumu_num_splits_per_batch = list(itertools.accumulate(num_splits_per_batch))
                            cumu_num_splits_per_batch = [x-1 for x in cumu_num_splits_per_batch] # start from 0
                            cumu_num_splits_per_batch = torch.as_tensor(cumu_num_splits_per_batch, dtype=torch.long, device=input_ids.device)
                            images_per_batch = images_per_batch[cumu_num_splits_per_batch]  # [n_images, 3, h, w]
                            images_per_batch = images_per_batch[:num_regions[i]]  # [n_region, 3, h, w], mmic-data, len(images)==len(regions) in a sample
                            all_images.append(images_per_batch)
                    else:
                        if type(images) == list:  # 'anyres', last split is the global image
                                all_images = [images[i][-1][None].repeat_interleave(num_regions[i], dim=0) for i in range(len(images))]
                        else:  # 'pad'
                            all_images = [images[i][None].repeat_interleave(num_regions[i], dim=0) for i in range(len(images))]  # list[tensor], bs x [n_region, 3, h, w]
                    all_images = torch.cat(all_images, dim=0)    # [n_all_regions, 3, h, w]
                    #########################################################
                    # all_image_features
                    # multi-scale of last 3 levels image features for region encoder
                    mlvl_image_features = image_forward_outs.hidden_states[-3:] 
                    if num_splits is not None:  # mmic-data
                        all_image_features = []
                        # for each level
                        for image_features_per_level in mlvl_image_features:  # [n_all_image_and_splits, 1 + img_len, 1024]
                            image_features_per_level = torch.split(image_features_per_level, split_sizes, dim=0)  # bs x [n_image_and_splits, 1 + img_len, 1024]
                            # for each batch
                            all_image_features_per_level = []
                            for i, (image_features_per_level_per_batch, num_splits_per_batch) in enumerate(zip(image_features_per_level, num_splits)):
                                # image_features_per_level_per_batch: [n_image_and_splits, 1 + img_len, 1024]
                                # num_splits_per_batch: list[int]
                                cumu_num_splits_per_batch = list(itertools.accumulate(num_splits_per_batch))
                                cumu_num_splits_per_batch = [x-1 for x in cumu_num_splits_per_batch] # start from 0
                                cumu_num_splits_per_batch = torch.as_tensor(cumu_num_splits_per_batch, dtype=torch.long, device=input_ids.device)
                                image_features_per_level_per_batch = image_features_per_level_per_batch[cumu_num_splits_per_batch, 1:] # [n_images, img_len, 1024]
                                image_features_per_level_per_batch = image_features_per_level_per_batch[:num_regions[i]] # [n_region, img_len, 1024], mmic-data, len(images)==len(regions)
                                all_image_features_per_level.append(image_features_per_level_per_batch)
                            all_image_features_per_level = torch.cat(all_image_features_per_level, dim=0)  # [n_all_regions, img_len, 1024]
                            all_image_features.append(all_image_features_per_level)
                    else:
                        if type(images) == list:  # 'anyres', last split is global image
                            new_mlvl_image_features = []
                            for image_features_per_level in mlvl_image_features:  # [n_all_image_splits, 1 + img_len, 1024]
                                image_features_per_level = torch.split(image_features_per_level, split_sizes, dim=0)  # bs x [n_image_splits, 1 + img_len, 1024]
                                image_features_per_level = [x[-1, 1:] for x in image_features_per_level]  
                                image_features_per_level = torch.stack(image_features_per_level, dim=0)  # [bs, img_len, 1024]
                                new_mlvl_image_features.append(image_features_per_level)  
                            mlvl_image_features = new_mlvl_image_features  # list[tensor], 3 x [bs, img_len, 1024]
                            del new_mlvl_image_features
                        else: # 'pad'
                            mlvl_image_features = [mlvl_image_feature[:, 1:] for mlvl_image_feature in mlvl_image_features]  # list[tensor], 3 x [bs, img_len, 1024]
                        all_image_features = []                      
                        for image_features_per_level in mlvl_image_features:
                            # [bs, img_len, 1024]
                            all_image_features_per_level = [image_features_per_level[i][None].repeat_interleave(num_regions[i], dim=0) for i in range(len(images))]
                            all_image_features_per_level = torch.cat(all_image_features_per_level)
                            all_image_features.append(all_image_features_per_level)  # 3 x [n_all_regions, img_len, 1024]
                    #########################################################
                    
                    # all_images:  [n_all_regions, 3, h, w]
                    # all_regions: [n_all_regions, 1, h, w]
                    # all_image_features: 3 x [n_all_regions, img_len, 1024]
                    all_region_features = self.region_encoder(all_images, all_regions, all_image_features) 
                    all_region_features = all_region_features.to(inputs_embeds.dtype)

                    # replace <region> token
                    inputs_embeds = inputs_embeds.reshape(B*L, C)
                    region_mask = input_ids == self.reg_token_id    
                    region_mask = region_mask.reshape(-1)          
                    temp_embeds = torch.zeros_like(inputs_embeds) 
                    temp_embeds[region_mask] = all_region_features     
                    region_mask = region_mask.to(inputs_embeds.dtype).unsqueeze(-1) 
                    inputs_embeds = inputs_embeds * (1 - region_mask) + temp_embeds * region_mask
                    inputs_embeds = inputs_embeds.reshape(B, L, C)
                else:  # regions is None
                    if type(images) == list:  # 'anyres'
                        H, W = images[0][0].shape[-2:]
                        dummy_all_images = torch.zeros((1, 3, H, W), dtype=inputs_embeds.dtype, device=inputs_embeds.device) # [1, 3, h, w]
                        dummy_all_regions = torch.ones((1, 1, H, W), dtype=inputs_embeds.dtype, device=inputs_embeds.device) # [1, 1, h, w]
                        dummy_all_image_features = torch.zeros_like(image_forward_outs.hidden_states[-1][:1, 1:])  # [1, img_len, 1024]
                        dummy_all_image_features = [dummy_all_image_features] * 3  # multi-scale image features, 3 x [1, img_len, 1024]
                    else:  # 'pad'
                        B, _, H, W = images.shape
                        dummy_all_images = torch.zeros_like(images)  # [b, 3, h, w]
                        dummy_all_regions = torch.ones((B, 1, H, W), dtype=images.dtype, device=images.device)    # [b, 1, h, w]
                        dummy_all_image_features = torch.zeros_like(image_forward_outs.hidden_states[-1][:, 1:])  # [b, img_len, 1024]
                        dummy_all_image_features = [dummy_all_image_features] * 3 # multi-scale image features, 3 x [b, img_len, 1024]
                    # dummy forward for region encoder
                    dummy_all_region_features = self.region_encoder(dummy_all_images, dummy_all_regions, dummy_all_image_features)
                    dummy_all_region_features = dummy_all_region_features.to(inputs_embeds.dtype)
                    inputs_embeds = inputs_embeds + (dummy_all_region_features * 0.).sum()


        output_attentions = output_attentions if output_attentions is not None else self.config.llm_config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.llm_config.output_hidden_states)
        return_dict = return_dict if return_dict is not None else self.config.llm_config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.llm(
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=True,
                return_dict=True,
        )
        hidden_states = outputs.hidden_states[-1]    # [bs, seq_len, 4096]
        if self.config.llm_config.architectures[0] == 'InternLM2ForCausalLM':
            logits = self.llm.output(hidden_states)
        else:
            logits = self.llm.lm_head(hidden_states)     
        logits = logits.float()  # [bs, seq_len, vocab_size], augmented
            
        loss = None
        if labels is not None:
            # ignore the emb_tokens for labels
            emb_select = (labels >= self.emb_token_id) & (labels <= self.emb_token_id + self.num_embs - 1)  # [B, L]
            labels[emb_select] = IGNORE_INDEX

            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.llm.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            if ignore_flag:
                loss = loss * 0.0
            

        # ---------------------- atom tools ------------------------------
        if img_metas is not None:
            task = img_metas[0]['task']  # each gpu has samples from one dataset
        else: 
            task = None
            

        # det/grd/seg
        # gdino
        if self.use_gdino:
            if task in ['det', 'det_cap', 'grd', 'seg', 'count_text', 'count_visual', 'interactive', 'ic_mask'] and images_aug is not None:
                images_aug = nested_tensor_from_tensor_list(images_aug, size_divisibility=32)
                pixel_values, pixel_mask = images_aug.tensors, ~images_aug.mask  # [bs, 3, h, w], [bs, h, w]
                pixel_mask = pixel_values[:, 0, :, :] != 0  # valid is 1
                # select the corresponding [EMB] hidden states as text_query
                batch_size, seq_len, hidden_size = inputs_embeds.shape
                emb_select = (input_ids >= self.emb_token_id) & (input_ids <= self.emb_token_id + self.num_embs - 1)  # [bs, seq_len]
                # if have [EMB] tokens
                if emb_select.sum() != 0:
                    num_patches = emb_select.sum(-1) // self.num_embs 
                    max_num_patches = num_patches.max()
                    text_query = torch.zeros((batch_size, max_num_patches, self.num_embs, hidden_size), dtype=hidden_states.dtype, device=hidden_states.device)
                    text_query_masks = torch.zeros(batch_size, max_num_patches, dtype=torch.bool, device=hidden_states.device)     
                    for batch_idx in range(batch_size):
                        if num_patches[batch_idx] != 0:
                            text_query_i = hidden_states[batch_idx, emb_select[batch_idx], :].reshape(-1, self.num_embs, hidden_size) 
                            text_query[batch_idx, :num_patches[batch_idx]] = text_query_i
                            text_query_masks[batch_idx, :num_patches[batch_idx]] = 1
                    gdino_outputs = self.gdino(pixel_values, pixel_mask=pixel_mask, text_query=text_query, text_query_masks=text_query_masks, img_metas=img_metas, labels=targets)
                    if targets is not None:
                        loss_gdino = gdino_outputs.loss
                        loss += loss_gdino
            else:  # for generate
                loss_gdino, gdino_outputs = None, None
        else:
            loss_gdino, gdino_outputs = None, None

        # pose
        # unipose
        if self.use_unipose:
            if task in ['pose'] and images_aug is not None:
                images_aug = nested_tensor_from_tensor_list(images_aug, size_divisibility=32)
                # select the corresponding [EMB] hidden states as text_query
                batch_size, seq_len, hidden_size = inputs_embeds.shape
                emb_select = (input_ids >= self.emb_token_id) & (input_ids <= self.emb_token_id + self.num_embs - 1)  # [bs, seq_len]
                # if have [EMB] tokens
                if emb_select.sum() != 0:
                    num_patches = emb_select.sum(-1) // self.num_embs 
                    # this is for obj class patches
                    max_num_obj_patches = 100
                    # this is for pose class patches
                    max_num_kpt_patches = 100

                    # [bs, max_num_obj/kpt_patches, num_embs, c], [bs, max_num_obj/kpt_patches]
                    obj_querys = torch.zeros((batch_size, max_num_obj_patches, self.num_embs, hidden_size), dtype=hidden_states.dtype, device=hidden_states.device)
                    obj_query_masks = torch.zeros((batch_size, max_num_obj_patches), dtype=torch.bool, device=hidden_states.device)
                    kpt_querys = torch.zeros((batch_size, max_num_kpt_patches, self.num_embs, hidden_size), dtype=hidden_states.dtype, device=hidden_states.device)
                    kpt_query_masks = torch.zeros((batch_size, max_num_kpt_patches), dtype=torch.bool, device=hidden_states.device)
                    for batch_idx in range(batch_size):
                        num_objcls = len(img_metas[batch_idx]['id2index'])
                        num_kpts = num_patches[batch_idx] - num_objcls
                        if num_objcls != 0 and num_kpts != 0:
                            text_query_i = hidden_states[batch_idx, emb_select[batch_idx], :].reshape(-1, self.num_embs, hidden_size)
                            obj_querys[batch_idx, :num_objcls] = text_query_i[:num_objcls, ...] 
                            obj_query_masks[batch_idx, :num_objcls] = 1                
                            kpt_querys[batch_idx, :num_kpts] = text_query_i[num_objcls:, ...]  
                            kpt_query_masks[batch_idx, :num_kpts] = 1               

                    text_query = dict(
                        obj_querys=obj_querys,
                        obj_query_masks=obj_query_masks,
                        kpt_querys=kpt_querys,
                        kpt_query_masks=kpt_query_masks
                    )
                    unipose_outputs = self.unipose(images_aug, targets, text_query=text_query, img_metas=img_metas)
                    if targets is not None:
                        loss_unipose = unipose_outputs.loss
                        loss += loss_unipose
            else:  # for geneate
                loss_unipose, unipose_outputs = None, None
        else:
            loss_unipose, unipose_outputs = None, None

        # gen
        # sd
        if self.use_sd:
            if task in ['t2i'] and output_images is not None:
                sd_outputs = self.sd(input_ids, hidden_states, output_images=output_images, captions=captions)
                if self.training:
                    loss_sd = sd_outputs.loss 
                    loss += loss_sd
            else:  # for generate
                loss_sd, sd_outputs = None, None
        else:
            loss_sd, sd_outputs = None, None

        # edit
        # ip2p
        if self.use_ip2p:
            if task in ['edit'] and output_images is not None:
                ip2p_outputs = self.ip2p(input_ids, hidden_states, input_images=input_images, output_images=output_images, captions=captions)
                if self.training:
                    loss_ip2p = ip2p_outputs.loss 
                    loss += loss_ip2p
            else:  # for generate
                loss_ip2p, ip2p_outputs = None, None
        else:
            loss_ip2p, ip2p_outputs = None, None

        # for inference
        if not self.training:
            loss_gdino, loss_unipose, loss_sd, loss_ip2p = None, None, None, None

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return VisionLLMv2ModelOutput(
            # CausalLMOutputWithPast
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            # atom tools
            loss_gdino=loss_gdino.detach() if loss_gdino is not None else None,
            gdino_outputs=gdino_outputs,
            loss_unipose=loss_unipose.detach() if loss_unipose is not None else None,
            unipose_outputs=unipose_outputs,
            loss_sd=loss_sd.detach() if loss_sd is not None else None,
            sd_outputs=sd_outputs,
            loss_ip2p=loss_ip2p.detach() if loss_ip2p is not None else None,
            ip2p_outputs=ip2p_outputs,
        )


AutoConfig.register("visionllmv2", VisionLLMv2Config)
AutoModel.register(VisionLLMv2Config, VisionLLMv2Model)
