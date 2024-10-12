"""
[EMB] tokens are not supervised by llm ce loss,
and [EMB] embeddings are initialized with nn.Embedding
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


# from .grounding_dino.modeling_ov_grounding_dino_dn import GroundingDinoObjectDetectionOutput
# from .grounding_dino.modeling_ov_grounding_dino_dn import OVGroundingDinoForObjectDetection

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
            # self.vl_bridge.load_state_dict({k.split('.')[-1]: v for k, v in vl_bridge_weights.items() if "vl_bridge" in k})

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
        # TODO: may add more special token id in the future
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
        # replace emb embeddings with predefined self.emb_embeddings, 
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
                    images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images] # list[tensor], bs x [n_split, 3, h, w]
                    split_sizes = [image.shape[0] for image in images]              # list[int]
                    concat_images = torch.cat([image for image in images], dim=0)   # [n_all_image_splits, 3, h, w]
                    image_forward_outs = self.vis_encoder(concat_images, output_hidden_states=True)  # [n_decoder, n_all_image_splits, 1 + img_len, 1024]
                else:  # 'pad'
                    # here images: after clip preprocess, [b, 3, h, w]
                    image_forward_outs = self.vis_encoder(images, output_hidden_states=True)  # [n_decoder, bs, 1 + img_len, 1024]
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
            selected = input_ids == self.imp_token_id                 # [B, L]
            has_image = selected.sum(-1) != 0                         # [B,]
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
                    #######################################################
                    # all_regions
                    # concat regions in the batch dimension
                    # regions: list[tensor], bs x [n_region, h, w]
                    num_regions = [len(regions_per_batch) for regions_per_batch in regions]      # list[int]
                    all_regions = [regions_per_batch[:, None] for regions_per_batch in regions]  # list[tensor], bs x [n_region, 1, h, w]
                    all_regions = torch.cat(all_regions, dim=0)  # [n_all_regions, 1, h, w]
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
                     # tuple[tensor]: 3 x [bs, 1 + img_len, 1024] or 3 x [n_all_image_splits, 1 + img_len, 1024]
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
                    all_region_features = self.region_encoder(all_images, all_regions, all_image_features)  # [n_all_regions, 4096]
                    all_region_features = all_region_features.to(inputs_embeds.dtype)

                    # replace <region> token
                    inputs_embeds = inputs_embeds.reshape(B*L, C)
                    region_mask = input_ids == self.reg_token_id    # [B, L]
                    region_mask = region_mask.reshape(-1)           # [B*L]
                    temp_embeds = torch.zeros_like(inputs_embeds)   # [B*L, C]
                    temp_embeds[region_mask] = all_region_features     
                    region_mask = region_mask.to(inputs_embeds.dtype).unsqueeze(-1) # [B*L, 1]
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
                # NOTE: valid location is 1 for pixel mask, opposite to the original implementation in officical repo
                pixel_values, pixel_mask = images_aug.tensors, ~images_aug.mask  # [bs, 3, h, w], [bs, h, w]
                # padding pixel values is 0 for ade20k
                pixel_mask = pixel_values[:, 0, :, :] != 0  # valid is 1
                # select the corresponding [EMB] hidden states as text_query
                batch_size, seq_len, hidden_size = inputs_embeds.shape
                emb_select = (input_ids >= self.emb_token_id) & (input_ids <= self.emb_token_id + self.num_embs - 1)  # [bs, seq_len]
                # if have [EMB] tokens
                if emb_select.sum() != 0:
                    num_patches = emb_select.sum(-1) // self.num_embs  # [bs,]
                    max_num_patches = num_patches.max()
                    text_query = torch.zeros((batch_size, max_num_patches, self.num_embs, hidden_size), dtype=hidden_states.dtype, device=hidden_states.device) # [bs, max_num_patches, num_embs, c]
                    text_query_masks = torch.zeros(batch_size, max_num_patches, dtype=torch.bool, device=hidden_states.device)       # [bs, max_num_patches], valid is 1
                    for batch_idx in range(batch_size):
                        if num_patches[batch_idx] != 0:
                            text_query_i = hidden_states[batch_idx, emb_select[batch_idx], :].reshape(-1, self.num_embs, hidden_size)  # [num_patch_i*num_embs, c] -> [num_patch_i, num_embs, c]
                            text_query[batch_idx, :num_patches[batch_idx]] = text_query_i
                            text_query_masks[batch_idx, :num_patches[batch_idx]] = 1
                    gdino_outputs = self.gdino(pixel_values, pixel_mask=pixel_mask, text_query=text_query, text_query_masks=text_query_masks, img_metas=img_metas, labels=targets)
                    # import ipdb; ipdb.set_trace()
                    # debug_data(pixel_values.float(), targets, img_metas)
                    # debug_predictions(pixel_values, gdino_outputs, img_metas)
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
                            # [num_patch_i*num_embs, c] -> [num_patch_i, num_embs, c], including obj and kpt embs
                            obj_querys[batch_idx, :num_objcls] = text_query_i[:num_objcls, ...]  # [1, 4, c]
                            obj_query_masks[batch_idx, :num_objcls] = 1                 # [1,]
                            kpt_querys[batch_idx, :num_kpts] = text_query_i[num_objcls:, ...]  # [17, 4, c]
                            kpt_query_masks[batch_idx, :num_kpts] = 1                     # [17,]

                    text_query = dict(
                        obj_querys=obj_querys,
                        obj_query_masks=obj_query_masks,
                        kpt_querys=kpt_querys,
                        kpt_query_masks=kpt_query_masks
                    )
                    unipose_outputs = self.unipose(images_aug, targets, text_query=text_query, img_metas=img_metas)
                    # import ipdb; ipdb.set_trace()
                    # debug_data(images_aug.tensors.float(), targets, img_metas)
                    # debug_predictions_unipose(images_aug.tensors, unipose_outputs, img_metas)
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

    # TODO: this is for atom tools test
    @torch.no_grad()
    def forward_test(
            self,
            input_ids: torch.LongTensor = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            images: Optional[list] = None,
            bboxes: Optional[list] = None,
            images_aug: Optional[list] = None,
            targets: Optional[list] = None,
            img_metas: Optional[list] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            use_cache: Optional[bool] = False,
            output_attentions: Optional[bool] = False,
            output_hidden_states: Optional[bool] = False,
            return_dict: Optional[bool] = False,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        # TODO: 
        # 现在要测性能，需要用gt 的output_ids
        # 一个修改
        # 1. visionllmv2/eval/eval_det.py里， model.forward_test(**data) 改为 model(**data)


        # TODO: delete this
        assert len(input_ids) == 1  # batch_size = 1
        if labels is not None:
            instruction_len =  (labels[0] == -100).sum()
            new_input_ids = input_ids[:, :instruction_len]
            input_ids = new_input_ids

        # import ipdb; ipdb.set_trace()

        # Here, generate func only forward the VLLM.
        with torch.no_grad():
            outputs = self.generate(
                input_ids=input_ids,
                images=images,
                generation_config=self.get_llm().generation_config,
                do_sample=False,  # greedy search
                temperature=0.,
                max_new_tokens=1024,
                use_cache=True,
                output_hidden_states=True,
                output_scores=True,
                return_dict_in_generate=True,
                )
            # inference, bs=1
            # we denote L = in_len + out_len
            # outputs.hidden_states (tuple[tuple[tensor]]): out_len x num_decoder_layers x [1, *, C]
            #       for the first element, hidden_states[0][0], size of [1, in_len, C]
            #       for the subsequent elements, hidden_states[*][0], size of [1, 1, C] 
            # outputs.sequences (tensor): [1, L]
            # outputs.scores (tuple[tensor]): out_len x [1, vocab_size]
            output_hidden_states = torch.cat([out[-1] for out in outputs.hidden_states[1:]], dim=1)  # [B, out_len-1, C], -1 for the end_token
            output_ids = outputs.sequences   # [B, L]
            
        # convert output_ids to string
        input_token_len = input_ids.shape[1]
        outputs_str = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=False)[0]
        outputs_str = outputs_str.strip() # str

        output_ids = output_ids[:, input_token_len:]  # [B, out_len], the output_ids do not contain [emb] tokens
        # -----------------------------------------------
        # NOTE: special operation for the [emb] tokens
        def insert_ids(output_ids, insert_positions, emb_ids):
            """
            Args:
            output_ids: torch.Tensor, original ids
            insert_positions: torch.Tensor, positions for inserting ids
            emb_ids: torch.Tensor, the ids to be inserted

            Returns:
            new_output_ids: torch.Tensor, output_ids
            """
            device = output_ids.device

            # calculate new length
            new_length = output_ids.size(0) + len(insert_positions) * emb_ids.size(0)
            # creat a new output_ids 
            new_output_ids = torch.zeros(new_length, dtype=torch.long).to(device)

            output_index = 0
            new_output_index = 0
            for i in range(output_ids.size(0)):
                new_output_ids[new_output_index] = output_ids[output_index]
                new_output_index += 1

                # if current pos need insert ids
                if output_index in insert_positions:
                    for emb_id in emb_ids:
                        new_output_ids[new_output_index] = emb_id
                        new_output_index += 1

                output_index += 1
            
            # the last pos whether need insert ids
            if output_index in insert_positions:
                for emb_id in emb_ids:
                    new_output_ids[new_output_index] = emb_id
            return new_output_ids
        
        emb_ids = torch.tensor([x for x in range(self.emb_token_id, self.emb_token_id + self.num_embs)], dtype=torch.long).to(output_ids.device)
        new_output_ids = []
        for cur_output_ids in output_ids:  # inference, bs=1
            cur_new_output_ids = cur_output_ids
            emb_start_pos_det = torch.where(cur_output_ids==self.det_tool_id)[0]
            emb_start_pos_seg = torch.where(cur_output_ids==self.seg_tool_id)[0]
            emb_start_pos_grd = torch.where(cur_output_ids==self.grd_tool_id)[0]
            emb_start_pos_pose = torch.where(cur_output_ids==self.pose_tool_id)[0]
            emb_start_pos = torch.cat([emb_start_pos_det, emb_start_pos_seg, emb_start_pos_grd, emb_start_pos_pose], dim=0)
            emb_start_pos = torch.sort(emb_start_pos)[0]
            cur_new_output_ids = insert_ids(cur_new_output_ids, emb_start_pos, emb_ids)
            new_output_ids.append(cur_new_output_ids)
        output_ids = torch.stack(new_output_ids, dim=0)
        # import ipdb; ipdb.set_trace()


        # FIXME: beautify the code
        name2id = img_metas[0]['name2id']  # dict
        cats = re.split('([\,\.\?\!])', outputs_str)
        cats = [cat for cat in cats if '[DET]' in cat]
        cats = [cat[:cat.index('[DET]')] for cat in cats]
        cat_names = []
        for i, cat in enumerate(cats):
            cat = cat.strip().split(' ')
            # 1 word or 2 words for coco
            if cat[-1] in name2id: # 1 word
                cat = cat[-1]
            else: # 2 words
                cat = cat[-2] + ' ' + cat[-1]
            cat_names.append(cat)
        cat_name_ids = [name2id[name] for name in cat_names]  # list[int]
        cat_name_ids = torch.as_tensor(cat_name_ids, device=input_ids.device).long()

        # import ipdb; ipdb.set_trace()


        # -------------------------- atom tools -------------------------------
        # gdino
        if self.use_gdino:
            if images_aug is not None:
                images_aug = nested_tensor_from_tensor_list(images_aug, size_divisibility=32)
                # NOTE: valid location is 1 for pixel mask, opposite to the original implementation in officical repo
                pixel_values, pixel_mask = images_aug.tensors, ~images_aug.mask  # [bs, 3, h, w], [bs, h, w]
                # padding pixel values is 0 for ade20k
                pixel_mask = pixel_values[:, 0, :, :] != 0  # valid is 1
                # select the corresponding [EMB] hidden states as text_query
                batch_size, seq_len, hidden_size = output_hidden_states.shape
                emb_select = (output_ids[:, :-1] >= self.emb_token_id) & (output_ids[:, :-1] <= self.emb_token_id + self.num_embs - 1)  # [bs, seq_len]
                # if have [EMB] tokens
                if emb_select.sum() != 0:
                    num_patches = emb_select.sum(-1) // self.num_embs  # [bs,]
                    max_num_patches = num_patches.max()
                    text_query = torch.zeros((batch_size, max_num_patches, self.num_embs, hidden_size), dtype=output_hidden_states.dtype, device=output_hidden_states.device) # [bs, max_num_patches, num_embs, c]
                    text_query_masks = torch.zeros(batch_size, max_num_patches, dtype=torch.bool, device=output_hidden_states.device)       # [bs, max_num_patches], valid is 1
                    for batch_idx in range(batch_size):
                        if num_patches[batch_idx] != 0:
                            text_query_i = output_hidden_states[batch_idx, emb_select[batch_idx], :].reshape(-1, self.num_embs, hidden_size)  # [num_patch_i*num_embs, c] -> [num_patch_i, num_embs, c]
                            text_query[batch_idx, :num_patches[batch_idx]] = text_query_i
                            text_query_masks[batch_idx, :num_patches[batch_idx]] = 1
                    gdino_outputs = self.gdino(pixel_values, pixel_mask=pixel_mask, text_query=text_query, text_query_masks=text_query_masks, img_metas=img_metas, labels=targets)
                    # convert gdino logits to original class id order
                    gdino_logits = gdino_outputs.logits[:, :, :max_num_patches]
                    logits = -torch.tensor(float('inf')) * torch.ones_like(gdino_outputs.logits)
                    logits[:, :, cat_name_ids] = gdino_logits
                    gdino_outputs.logits = logits
                    # debug_predictions(pixel_values.float(), gdino_outputs, img_metas)
                    # import ipdb; ipdb.set_trace()
                    
        
        return VisionLLMv2ModelOutput(
            # CausalLMOutputWithPast
            loss=None,
            logits=None,           # outputs.logits
            past_key_values=None, # outputs.past_key_values
            hidden_states=None,   # outputs.hidden_states
            attentions=None,      # outputs.attentions
            # atom tools
            loss_gdino=None,
            gdino_outputs=gdino_outputs,
            loss_unipose=None,
            unipose_outputs=None,  # TODO
        )


AutoConfig.register("visionllmv2", VisionLLMv2Config)
AutoModel.register(VisionLLMv2Config, VisionLLMv2Model)


# --------------------------------------------------------------------------------
# debug visualiza purpose
import numpy as np
import cv2

def colormap(rgb=False):
    color_list = np.array(
        [
            0.000, 0.447, 0.741,
            0.850, 0.325, 0.098,
            0.929, 0.694, 0.125,
            0.494, 0.184, 0.556,
            0.466, 0.674, 0.188,
            0.301, 0.745, 0.933,
            0.635, 0.078, 0.184,
            0.300, 0.300, 0.300,
            0.600, 0.600, 0.600,
            1.000, 0.000, 0.000,
            1.000, 0.500, 0.000,
            0.749, 0.749, 0.000,
            0.000, 1.000, 0.000,
            0.000, 0.000, 1.000,
            0.667, 0.000, 1.000,
            0.333, 0.333, 0.000,
            0.333, 0.667, 0.000,
            0.333, 1.000, 0.000,
            0.667, 0.333, 0.000,
            0.667, 0.667, 0.000,
            0.667, 1.000, 0.000,
            1.000, 0.333, 0.000,
            1.000, 0.667, 0.000,
            1.000, 1.000, 0.000,
            0.000, 0.333, 0.500,
            0.000, 0.667, 0.500,
            0.000, 1.000, 0.500,
            0.333, 0.000, 0.500,
            0.333, 0.333, 0.500,
            0.333, 0.667, 0.500,
            0.333, 1.000, 0.500,
            0.667, 0.000, 0.500,
            0.667, 0.333, 0.500,
            0.667, 0.667, 0.500,
            0.667, 1.000, 0.500,
            1.000, 0.000, 0.500,
            1.000, 0.333, 0.500,
            1.000, 0.667, 0.500,
            1.000, 1.000, 0.500,
            0.000, 0.333, 1.000,
            0.000, 0.667, 1.000,
            0.000, 1.000, 1.000,
            0.333, 0.000, 1.000,
            0.333, 0.333, 1.000,
            0.333, 0.667, 1.000,
            0.333, 1.000, 1.000,
            0.667, 0.000, 1.000,
            0.667, 0.333, 1.000,
            0.667, 0.667, 1.000,
            0.667, 1.000, 1.000,
            1.000, 0.000, 1.000,
            1.000, 0.333, 1.000,
            1.000, 0.667, 1.000,
            0.167, 0.000, 0.000,
            0.333, 0.000, 0.000,
            0.500, 0.000, 0.000,
            0.667, 0.000, 0.000,
            0.833, 0.000, 0.000,
            1.000, 0.000, 0.000,
            0.000, 0.167, 0.000,
            0.000, 0.333, 0.000,
            0.000, 0.500, 0.000,
            0.000, 0.667, 0.000,
            0.000, 0.833, 0.000,
            0.000, 1.000, 0.000,
            0.000, 0.000, 0.167,
            0.000, 0.000, 0.333,
            0.000, 0.000, 0.500,
            0.000, 0.000, 0.667,
            0.000, 0.000, 0.833,
            0.000, 0.000, 1.000,
            0.000, 0.000, 0.000,
            0.143, 0.143, 0.143,
            0.286, 0.286, 0.286,
            0.429, 0.429, 0.429,
            0.571, 0.571, 0.571,
            0.714, 0.714, 0.714,
            0.857, 0.857, 0.857,
            1.000, 1.000, 1.000
        ]
    ).astype(np.float32)
    color_list = color_list.reshape((-1, 3)) * 255
    if not rgb:
        color_list = color_list[:, ::-1]
    return color_list

def vis_add_mask(img, mask, color):
    # visaulize one mask
    # origin_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    origin_img = img
    mask = mask.reshape(mask.shape[0], mask.shape[1]).astype('uint8')

    alpha = 0.5
    for c in range(3):
        origin_img[:, :, c] = np.where(mask == 1,
                origin_img[:, :, c] * (1 - alpha) + alpha * color[c],
                origin_img[:, :, c])
    return origin_img


def debug_data_regions(images, masks, img_metas=None):
    # images: [bs, c, h, w], resize to 336x336
    # masks (i.e., regions): list[tensor], bs x [n, h, w]
    # img_metas: list[dict]
    import numpy as np
    import copy
    import cv2
    import torch.distributed as dist
    import sys
    import time
    # mean = np.array([123.675, 116.280, 103.530])
    # std = np.array([58.395, 57.120, 57.375])
    # clip mean std
    mean=[0.48145466 * 255, 0.4578275 * 255, 0.40821073 * 255]
    std=[0.26862954 * 255, 0.26130258 * 255, 0.27577711 * 255]
    default_color = (255,255,255)
    color_list = colormap().tolist()
    num_color = len(color_list)

    device = images.device

    assert len(images) == len(masks)
    for i in range(len(images)):
        # image
        image = images[i].permute((1,2,0)).cpu().numpy() * std + mean # [H, W, 3], including padding
        image = np.ascontiguousarray(image[:, :, ::-1]).clip(0, 255)
        # mask
        # masks: list[tensor], bs x [n, h, w]
        num_inst = len(masks[i])
        for j in range(num_inst):
            mask = masks[i][j].float().cpu().numpy()  # [h, w]
            if mask.shape != image.shape[:2]:  # images have padding while mask not, plot padded images
                ori_h, ori_w = mask.shape[:2]
                mask_new = np.zeros(image.shape[:2])
                mask_new[:ori_h, :ori_w] = mask[:ori_h, :ori_w]
            else:
                mask_new = mask
            # plot
            color = color_list[j%num_color]
            image = vis_add_mask(image, mask_new, color)

        cv2.imwrite("rank_0_batch_%d_img.jpg"%(i), image)
    return
        




def debug_data(images, targets=None, img_metas=None):
    # gt_bbox: xyxy in image size
    import numpy as np
    import copy
    import cv2
    import torch.distributed as dist
    import sys
    import time
    mean = np.array([123.675, 116.280, 103.530])
    std = np.array([58.395, 57.120, 57.375])
    default_color = (255,255,255)
    color_list = colormap().tolist()
    num_color = len(color_list)

    device = images.device

    if targets is not None:  # list[dict]
        assert len(images) == len(targets)
    # for each image
    for i in range(len(images)):
        image = images[i].permute((1,2,0)).cpu().numpy() * std + mean # [H, W, 3], including padding
        image = np.ascontiguousarray(image[:, :, ::-1]).clip(0, 255)
        if targets is None:
            cv2.imwrite("rank_0_batch_%d_img.jpg"%(i), image)
        else: # also plot targets
            labels = targets[i]['class_labels']
            # bboxes
            if "boxes" in targets[i].keys():
                bboxes = targets[i]["boxes"].float()
                assert len(bboxes) == len(labels)
                image_size = img_metas[i]['img_shape'][:2]  # [h, w], no padding
                bboxes = box_cxcywh_to_xyxy(bboxes) # box cxcywh [0, 1] -> xyxy in image size
                im_h, im_w = image_size
                scale_fct = torch.as_tensor([im_w, im_h, im_w, im_h], dtype=torch.float32).to(bboxes.device)
                bboxes = bboxes * scale_fct[None, :]
                bboxes = bboxes.cpu().numpy()   # [num_target, 4], xyxy in image size
                labels = labels.cpu().numpy()   # [num_target,]
                num_inst = bboxes.shape[0]
                for j in range(num_inst):
                    label = labels[j]
                    x1, y1, x2, y2 = bboxes[j]
                    color = color_list[j%num_color]
                    # plot
                    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness=4)
                    cv2.putText(image, "{}".format(int(label)), (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 4)
            # masks
            if "mask_labels" in targets[i].keys():
                masks = targets[i]["mask_labels"].float()
                assert len(masks) == len(labels)
                masks = masks.cpu().numpy()  # [num_target, H, W]
                num_inst = masks.shape[0]
                for j in range(num_inst):
                    label = labels[j]
                    mask = masks[j]
                    if mask.shape != image.shape[:2]:  # images have padding while mask not, plot padded images
                        ori_h, ori_w = mask.shape[:2]
                        mask_new = np.zeros(image.shape[:2])
                        mask_new[:ori_h, :ori_w] = mask[:ori_h, :ori_w]
                    else:
                        mask_new = mask
                    # plot
                    color = color_list[j%num_color]
                    image = vis_add_mask(image, mask_new, color)

            # keypoints
            if "keypoints" in targets[i].keys():
                keypoints = targets[i]["keypoints"].float()  # [num_gt, 17*3], xyxyzz
                keypoints = keypoints[:, :17*2].reshape(-1, 17, 2) 
                scale_fct = torch.as_tensor([im_w, im_h], dtype=torch.float32).to(device)
                keypoints = keypoints * scale_fct[None, None, :]
                keypoints = keypoints.cpu().numpy()  # [num_gt, 17, 2]
                num_inst = keypoints.shape[0]
                for j in range(num_inst):
                    keypoint = keypoints[j]  # [17, 2]
                    color = color_list[j%num_color]
                    for k in range(17):
                        x, y = keypoint[k]
                        cv2.circle(image, (int(x), int(y)), 5, color, -1)

            cv2.imwrite("rank_0_batch_%d_img.jpg"%(i), image)
    return 


def debug_predictions(images, outputs, img_metas):
    import numpy as np
    import copy
    import cv2
    import torch.distributed as dist
    import sys
    import time
    mean = np.array([123.675, 116.280, 103.530])
    std = np.array([58.395, 57.120, 57.375])
    default_color = (255,255,255)
    color_list = colormap().tolist()
    num_color = len(color_list)

    # image and masks have padding, size_divisibility=32
    for i in range(len(images)):
        image = images[i].permute((1,2,0)).cpu().numpy() * std + mean # [H, W, 3], including padding
        image = np.ascontiguousarray(image[:, :, ::-1]).clip(0, 255)
        # predictions
        # [nq, K], [nq, 4], [nq, H/4, W/4]
        out_logits, out_boxes, out_masks = outputs.logits[i], outputs.pred_boxes[i], outputs.pred_masks[i] 
        choose = out_logits.sigmoid().max(-1)[0] > 0.5  # max score for a query > 0.5
        out_logits  = out_logits[choose]
        out_boxes = out_boxes[choose]
        out_masks = out_masks[choose]
        # convert scores
        out_scores = out_logits.sigmoid().max(-1)[0]
        # convert boxes and masks
        image_size = img_metas[i]['img_shape'][:2]  # [h, w], no padding
        out_boxes = box_cxcywh_to_xyxy(out_boxes) # box cxcywh [0, 1] -> xyxy in image size
        im_h, im_w = image_size
        scale_fct = torch.as_tensor([im_w, im_h, im_w, im_h], dtype=torch.float32).to(out_boxes.device)
        out_boxes = out_boxes * scale_fct[None, :]
        boxes = out_boxes.cpu().numpy()
        # convert masks
        H, W = out_masks.shape[-2:]
        out_masks = F.interpolate(out_masks[:, None], size=(H*4, W*4), mode='bilinear', align_corners=False)[:, 0]
        masks = out_masks.sigmoid() > 0.5  # bool
        # for each inst
        num_inst = out_boxes.shape[0]
        for j in range(num_inst):
            color = color_list[j % num_color]
            # box
            x1, y1, x2, y2 = boxes[j]
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness=2)
            # mask
            mask = masks[j].cpu().float().numpy()  # [H, W]
            image = vis_add_mask(image, mask, color=color)
            # score 
            score = out_scores[j].cpu().numpy()
            cv2.putText(image, "{:.2f}".format(score), (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color=color, thickness=2)
        cv2.imwrite("rank_0_batch_%d_img.jpg"%(i), image)
    return

            

def debug_predictions_unipose(images, outputs, img_metas, num_keypoints=17): # coco, num_keypoints=17
    import numpy as np
    import copy
    import cv2
    import torch.distributed as dist
    import sys
    import time
    mean = np.array([123.675, 116.280, 103.530])
    std = np.array([58.395, 57.120, 57.375])
    default_color = (255,255,255)
    color_list = colormap().tolist()
    num_color = len(color_list)

    # image and masks have padding, size_divisibility=32
    import ipdb; ipdb.set_trace()
    for i in range(len(images)):
        image = images[i].permute((1,2,0)).cpu().numpy() * std + mean # [H, W, 3], including padding
        image = np.ascontiguousarray(image[:, :, ::-1]).clip(0, 255)
        # predictions
        # [nq, K], [nq, 4], [nq, H/4, W/4]
        out_logits, out_boxes, out_keypoints = outputs.pred_logits[i], outputs.pred_boxes[i], outputs.pred_keypoints[i] 
        choose = out_logits.sigmoid().max(-1)[0] > 0.5  # max score for a query > 0.5
        out_logits  = out_logits[choose]
        out_boxes = out_boxes[choose]          # [nq, 4]
        out_keypoints = out_keypoints[choose]  # [nq, 68*3], xyxyzz
        # convert scores
        out_scores = out_logits.sigmoid().max(-1)[0]
        # convert boxes 
        image_size = img_metas[i]['img_shape'][:2]  # [h, w], no padding
        out_boxes = box_cxcywh_to_xyxy(out_boxes) # box cxcywh [0, 1] -> xyxy in image size
        im_h, im_w = image_size
        scale_fct = torch.as_tensor([im_w, im_h, im_w, im_h], dtype=torch.float32).to(out_boxes.device)
        out_boxes = out_boxes * scale_fct[None, :]
        boxes = out_boxes.cpu().numpy()
        # convert keypoints
        out_keypoints = out_keypoints[:, :2*num_keypoints].reshape(-1, num_keypoints, 2)  # [nq, 17, 2], coco 
        scale_fct = torch.as_tensor([im_w, im_h], dtype=torch.float32).to(out_boxes.device)
        out_keypoints = out_keypoints * scale_fct[None, None, :]
        keypoints = out_keypoints.cpu().numpy()

        # for each inst
        num_inst = out_boxes.shape[0]
        for j in range(num_inst):
            color = color_list[j % num_color]
            # box
            x1, y1, x2, y2 = boxes[j]
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness=2)
            # keypoint
            keypoint = keypoints[j]  # [17 x 2]
            for k in range(num_keypoints):
                x, y = keypoint[k]
                cv2.circle(image, (int(x), int(y)), 5, color, -1)
            # score 
            score = out_scores[j].cpu().numpy()
            cv2.putText(image, "{:.2f}".format(score), (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color=color, thickness=2)

        cv2.imwrite("rank_0_batch_%d_img.jpg"%(i), image)
    return