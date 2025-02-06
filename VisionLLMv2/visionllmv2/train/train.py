#!/usr/bin/env python
# coding=utf-8
# Copyright Qing-Long Zhang. All rights reserved.
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
"""
Fine-tuning the library models for sequence to sequence.
"""

import logging
import os
import sys
import math
import warnings

from dataclasses import dataclass, field
from typing import Optional
import pathlib

from PIL import PngImagePlugin, Image, ImageFile

import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

import transformers
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    LlamaTokenizer,
    AutoTokenizer,
    Trainer,
    set_seed,
    default_data_collator,
    DataCollatorForSeq2Seq,
)

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_int8_training,
)

from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

from transformers.utils.logging import (
    set_verbosity_info,
    set_verbosity,
    enable_default_handler,
    enable_explicit_format,
)

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoImageProcessor
from transformers import CLIPVisionConfig, CLIPVisionModel, CLIPImageProcessor
from transformers import LlamaConfig, LlamaForCausalLM

from visionllmv2 import conversation as conversation_lib
from visionllmv2.model.configuration_visionllmv2 import VisionLLMv2Config
from visionllmv2.model.modeling_visionllmv2 import VisionLLMv2Model
from visionllmv2.train.visionllmv2_trainer import VisionLLMv2Trainer
from visionllmv2.datasets.build import build_dataset, build_multi_datasets
from visionllmv2.datasets.collator import DataCollatorForHybridDetSegPoseGenDataset
from visionllmv2.constant import IGNORE_INDEX, DEFAULT_TOKENS

# ----------------------------- atom tools ---------------------------------
# det/grd/seg
from visionllmv2.model.grounding_dino.configuration_grounding_dino import GroundingDinoConfig
from visionllmv2.model.grounding_dino.modeling_ov_grounding_dino_mask_dn import OVGroundingDinoForObjectDetection

# pose
from visionllmv2.model.unipose.configuration_unipose import UniPoseConfig
from visionllmv2.model.unipose.modeling_unipose import UniPose

# eval
import time
import mmcv
from mmcv import Config
from mmcv.runner import get_dist_info
from mmdet.apis.test import collect_results_cpu
from mmdet.core import bbox2result

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from visionllmv2.dist_utils import init_dist
from ..eval.eval_det import eval_det
from ..eval.eval_semseg import eval_semseg
from ..eval.eval_pose import eval_pose
from ..eval.eval_sod import eval_sod
from ..eval.eval_visual_prompt import eval_visual_prompt


Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2 ** 20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.29.0.dev0")
# require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/translation/requirements.txt")

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

# os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "true"


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=None)
    llm_path: Optional[str] = field(default=None)
    vis_encoder_path: Optional[str] = field(default=None)         
    pretrained_vl_bridge: Optional[str] = field(default=None) 
    use_llm_lora: Optional[bool] = field(default=False)         # whether use lora for llm
    vl_bridge_type: Optional[str] = field(default="mlp2x_gelu") # same as llava
    vis_output_layer: Optional[int] = field(default=-2)         # same as llava
    use_pixelshuffle: Optional[bool] = field(default=False)     # whether using pixelshuffle
    num_embs: int=field(default=4)                              # [EMB] token number
    num_embs_gen: int=field(default=64)                         # [EMB] token number
    # ------------------- region encoder -------------------
    use_region_encoder: Optional[bool] = field(default=False)
    # ------------------- atom tools ---------------------
    use_gdino: Optional[bool] = field(default=False)
    gdino_path: Optional[str] = field(default=None)
    use_unipose: Optional[bool] = field(default=False)
    unipose_path: Optional[str] = field(default=None)
    use_sd: Optional[bool] = field(default=False)
    sd_path: Optional[str] = field(default=None)                # sd config path
    use_ip2p: Optional[bool] = field(default=False)
    ip2p_path: Optional[str] = field(default=None)              # ip2p config path


@dataclass
class DataArguments:
    data_path: str = field(default='data')
    dataset_config: str = field(default='visionllmv2/datasets/configs/llava_stage1.py')
    lazy_preprocess: bool = False                      # not used
    version: Optional[str] = field(default="v0")       # conversation version
    image_size: int = field(default=336)               # clip336
    image_max_tile: int = field(default=6)             # image max split number
    image_token_len: int = field(default=576)          # 336/14 x 336/14, not used in new update 
    image_folder: Optional[str] = field(default=None)  # not used
    image_aspect_ratio: str = 'square'                 # 'square', 'pad', 'anyres'
    use_im_start_end: bool = field(default=False)      # False
    multi_datasets: bool = field(default=False)        # False
    

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    freeze_vis_encoder: bool = field(default=True)
    freeze_llm: bool = field(default=False)
    freeze_vl_bridge: bool = field(default=False)
    freeze_region_encoder: bool = field(default=False)
    freeze_emb_embeddings: bool = field(default=False)
    freeze_backbone: bool = field(default=False)           # whether freeze backbone in atom tools
    tune_llm_embed: bool = field(default=False)            # only tune input/output embeddings for llm
    tune_vl_bridge: bool = field(default=False)            # only tune vl_bridge
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    force_fsdp: bool = field(default=False)
    model_max_length: int = field(default=512)
    group_by_modality_length: bool = field(default=False)  # llava-v1.5
    group_by_data_source: bool = field(default=False)      # multi-dataset loader
    group_by_task_data_source: bool = field(default=False) # multi-dataset loader, only using one tool in one iter
    eval_only: bool = field(default=False)                 # for eval
    # lr multiplier
    lr_llm_multiplier: Optional[float] = field(default=1.0)
    lr_multiplier: Optional[float] = field(default=0.1)


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param

# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return

def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return

def get_vl_bridge_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "tune_vl_bridge", False):
        # only save vl_bridge
        keys_to_match = ['vl_bridge']

        weight_to_save = get_vl_bridge_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        print("weight to save:", weight_to_save.keys())
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                vl_bridge_folder = os.path.join(parent_folder, "vl_bridge")
                os.makedirs(vl_bridge_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(vl_bridge_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'vl_bridge.bin'))
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa

def train(eval_only=False):
    # 1. Parse input arguments
    # if use deepspeed zero3, init_dist before HFArgumentParse
    # uncomment following line if use slurm
    # init_dist(launcher='slurm', backend='nccl', port=29503)
    # See all possible arguments in src/transformers/training_args.py
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()    
    training_args._frozen = False  # compatible with transformers==4.32.0
    data_args._frozen = False  # compatible with transformers==4.32.0
    model_args._frozen = False  # compatible with transformers==4.32.0
    data_args.num_embs = model_args.num_embs
    data_args.num_embs_gen = model_args.num_embs_gen
    data_args.use_pixelshuffle = model_args.use_pixelshuffle

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    # send_example_telemetry("visionllmv2", model_args, data_args)

    # 2. Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    set_verbosity(log_level)
    enable_default_handler()
    enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # 3. Detecting last checkpoint and eventually continue from last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)


    # 4. Load pretrained model, tokenizer, and image processor
    #
    # Distributed training: The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    # Loading tokenizer
    print(f"Loading tokenizer: {model_args.llm_path or model_args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.llm_path or model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
        add_eos_token=False,
        trust_remote_code=True
    )
    # add new tokens
    if not training_args.eval_only:
        tokenizer.pad_token = tokenizer.unk_token
        num_new_tokens = tokenizer.add_tokens(list(DEFAULT_TOKENS.values()), special_tokens=True)
    else:
        num_new_tokens = 0

    # Loading model
    data_args.img_processor = CLIPImageProcessor.from_pretrained(model_args.vis_encoder_path)
    if model_args.model_name_or_path is not None:
        # vision tasks
        print("Loading VisionLLMv2Model...")
        visionllmv2_config = VisionLLMv2Config.from_pretrained(model_args.model_name_or_path)
        if visionllmv2_config.llm_config.architectures[0] == 'InternLM2ForCausalLM':
            visionllmv2_config.llm_config.attn_implementation = 'flash_attention_2' 
        else:
            visionllmv2_config.llm_config._flash_attn_2_enabled = True  # llama, transformers=4.34
        # no need to load the pretrained_vl_bridge because we have already had the weights of the whole VLLM model
        visionllmv2_config.pretrained_vl_bridge = None
        # overwrite the configs if training:
        if not training_args.eval_only:
            visionllmv2_config.vl_bridge_type = model_args.vl_bridge_type
            visionllmv2_config.vis_output_layer = model_args.vis_output_layer
            visionllmv2_config.num_embs = model_args.num_embs   
            visionllmv2_config.num_embs_gen = model_args.num_embs_gen
            visionllmv2_config.use_pixelshuffle = model_args.use_pixelshuffle
        model = VisionLLMv2Model.from_pretrained(model_args.model_name_or_path, torch_dtype=torch.bfloat16, config=visionllmv2_config)
    else:
        # LLaVA pretraning and finetune
        print("Loading CLIPVisionModel...")
        if 'InternViT' in model_args.vis_encoder_path or 'intern_vit' in model_args.vis_encoder_path:  # InternViT-6B
            vis_encoder_config = AutoConfig.from_pretrained(model_args.vis_encoder_path, trust_remote_code=True)
            vis_encoder = AutoModel.from_pretrained(model_args.vis_encoder_path, torch_dtype=torch.bfloat16, config=vis_encoder_config, trust_remote_code=True)
        else:  # CLIP-L
            vis_encoder_config = CLIPVisionConfig.from_pretrained(model_args.vis_encoder_path)
            vis_encoder = CLIPVisionModel.from_pretrained(model_args.vis_encoder_path, torch_dtype=torch.bfloat16, config=vis_encoder_config)
        print("Loading LLaMA...")
        if 'internlm' in model_args.llm_path:
            llm_config = AutoConfig.from_pretrained(model_args.llm_path, trust_remote_code=True)
            llm_config.attn_implementation = 'flash_attention_2' 
            llm = AutoModelForCausalLM.from_pretrained(model_args.llm_path, torch_dtype=torch.bfloat16, config=llm_config, trust_remote_code=True)
        else:
            llm_config = LlamaConfig.from_pretrained(model_args.llm_path)
            llm_config._flash_attn_2_enabled = True  # llama, transformers=4.34
            llm = LlamaForCausalLM.from_pretrained(model_args.llm_path, torch_dtype=torch.bfloat16, config=llm_config)
        print("Building VisionLLMv2Model...")
        visionllmv2_config = VisionLLMv2Config(
            llm_config=llm_config.to_dict(),
            vis_encoder_config=vis_encoder_config.to_dict(),
            pretrained_vl_bridge=model_args.pretrained_vl_bridge,  # if any, load the vl_bridge weights
            use_llm_lora=model_args.use_llm_lora,
            vl_bridge_type=model_args.vl_bridge_type,
            vis_output_layer=model_args.vis_output_layer,
            use_pixelshuffle=model_args.use_pixelshuffle,
            num_embs=model_args.num_embs,
            num_embs_gen=model_args.num_embs_gen,
            # atom tools: False
        )
        model = VisionLLMv2Model(visionllmv2_config, vis_encoder=vis_encoder, llm=llm)
        model.get_llm().config.use_cache = False
        model.generation_config = model.get_llm().generation_config
    # init special token ids
    model.tokenizer = tokenizer
    model.init_special_token_ids(tokenizer)

    # ---------------------- atom tools --------------------------
    # build atom models during training
    if not training_args.eval_only and model_args.use_gdino and not model.config.use_gdino:
        print("Loading OVGroundingDino...")
        gdino_config = GroundingDinoConfig.from_pretrained(model_args.gdino_path)
        gdino_config.auxiliary_loss = True       # default is False
        # matching cost
        gdino_config.class_cost = 2.0        
        gdino_config.dice_cost = 5.0
        gdino_config.mask_cost = 5.0
        gdino_config.box_cost = 5.0
        gdino_config.giou_cost = 2.0
        # loss weight
        gdino_config.class_weight = 2.0
        gdino_config.dice_weight = 5.0
        gdino_config.mask_weight = 5.0
        gdino_config.box_weight = 5.0
        gdino_config.giou_weight = 2.0
        # for mask
        gdino_config.mask_dim = 256
        gdino_config.norm = 'GN'
        gdino_config.l_hidden_size = visionllmv2_config.llm_config.hidden_size
        gdino_config.num_embs = visionllmv2_config.num_embs
        gdino = OVGroundingDinoForObjectDetection.from_pretrained(
            model_args.gdino_path, torch_dtype=torch.bfloat16, config=gdino_config, ignore_mismatched_sizes=True
        )
        # update visionllmv2
        model.config.use_gdino = True
        model.config.gdino_config = gdino_config
        model.use_gdino = True
        model.gdino = gdino 
    if not training_args.eval_only and model_args.use_unipose and not model.config.use_unipose:
        print("Loading UniPose...")
        unipose_config = UniPoseConfig.from_pretrained(model_args.unipose_path)
        unipose_config.aux_loss = True  # default is False
        unipose_config.dn_number = 100
        # matching cost, 2, 5, 2, 10, 4
        unipose_config.set_cost_class = 2.0
        unipose_config.set_cost_bbox = 5.0
        unipose_config.set_cost_giou = 2.0
        unipose_config.set_cost_keypoint = 10.0
        unipose_config.set_cost_oks = 4.0
        # loss weight
        unipose_config.cls_loss_coef = 2.0
        unipose_config.bbox_loss_coef = 5.0
        unipose_config.giou_loss_coef = 2.0
        unipose_config.keypoint_loss_coef = 10.0
        unipose_config.oks_loss_coef = 4.0
        unipose_config.num_embs = visionllmv2_config.num_embs  # not used
        unipose_config.l_hidden_size = visionllmv2_config.llm_config.hidden_size
        unipose = UniPose.from_pretrained(
            model_args.unipose_path, torch_dtype=torch.bfloat16, config=unipose_config, ignore_mismatched_sizes=True
        )
        # update visionllmv2
        model.config.use_unipose = True
        model.config.unipose_config = unipose_config
        model.use_unipose = True
        model.unipose = unipose
    if not training_args.eval_only and model_args.use_sd and not model.config.use_sd:
        print("Loading StableDiffusion...")
        sd_config = AutoConfig.from_pretrained(model_args.sd_path)
        sd_config.llm_hidden_size = visionllmv2_config.llm_config.hidden_size
        sd_config.num_embed_tokens = visionllmv2_config.num_embs_gen
        sd_config.trigger_token_id = model.gen_tool_id
        sd = AutoModel.from_config(sd_config)
        # update visionllmv2
        model.config.use_sd = True
        model.config.sd_config = sd_config
        model.use_sd = True
        model.sd = sd
    if not training_args.eval_only and model_args.use_ip2p and not model.config.use_ip2p:
        print("Loading InstructPix2Pix...")
        ip2p_config = AutoConfig.from_pretrained(model_args.ip2p_path)
        ip2p_config.llm_hidden_size = visionllmv2_config.llm_config.hidden_size
        ip2p_config.num_embed_tokens = visionllmv2_config.num_embs_gen
        ip2p_config.trigger_token_id = model.edit_tool_id
        ip2p = AutoModel.from_config(ip2p_config)
        # update visionllmv2
        model.config.use_ip2p = True
        model.config.ip2p_config = ip2p_config
        model.use_ip2p = True
        model.ip2p = ip2p
    

    # ------------------ other components -----------------
    # add region encoder, if any
    if not training_args.eval_only and model_args.use_region_encoder and not model.config.use_region_encoder:
        model.config.use_region_encoder = True
        model.use_region_encoder = True
        model.build_region_encoder()


    print("Building Model Finished.")

    # initialized input/output embeddings for new tokens
    if num_new_tokens > 0:
        print(f"Add {num_new_tokens} new tokens.")
        model.llm.resize_token_embeddings(len(tokenizer))
        input_embeddings = model.llm.get_input_embeddings().weight.data
        output_embeddings = model.llm.get_output_embeddings().weight.data
        #
        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        #
        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

    model.config.llm_config.vocab_size = len(tokenizer)
    model.llm.config.vocab_size = len(tokenizer)
        
    # add lora tune, if any
    if not training_args.eval_only and model_args.use_llm_lora:
        # https://zhuanlan.zhihu.com/p/628232317
        # https://link.zhihu.com/?target=https%3A//github.com/huggingface/peft/issues/137
        model.llm.enable_input_require_grads()
        if not model.config.use_llm_lora:
            model.wrap_llm_lora()
            model.config.use_llm_lora = model_args.use_llm_lora

    # freeze model, if any
    if training_args.freeze_vis_encoder:
        model.freeze_vis_encoder()
    if training_args.freeze_vl_bridge:
        model.freeze_vl_bridge()
    if training_args.freeze_llm:
        model.freeze_llm()
    if training_args.freeze_region_encoder:
        model.freeze_region_encoder()
    if training_args.freeze_emb_embeddings:
        model.freeze_emb_embeddings()
    if training_args.freeze_backbone:
        print("Freeze atom tools backbone.")
        for n, p in model.named_parameters():
            if 'backbone' in n:
                p.requires_grad = False
    if training_args.tune_llm_embed:
        print('Only tune input/output embeddings for llm.')
        model.llm.requires_grad_(False)
        model.llm.get_input_embeddings().weight.requires_grad_(True)
        model.llm.get_output_embeddings().weight.requires_grad_(True)
    if training_args.tune_vl_bridge:
        print("Only tune vl bridge.")
        model.requires_grad_(False)
        for p in model.get_vl_bridge().parameters():
            p.requires_grad = True


    # set default conversation
    # move to datasets/llava_data.py preprocess()
    # if data_args.version in conversation_lib.conv_templates:
    #     conversation_lib.default_conversation = conversation_lib.conv_templates[data_args.version]
    # else:
    #     conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    # 5. Get the datasets
    # you can either provide your own JSON training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.

    # evaluation
    if training_args.eval_only:
        eval_dataset_config = Config.fromfile(data_args.dataset_config)
        num_eval_datasets = len(eval_dataset_config.datasets)
        for dataset_idx in range(num_eval_datasets):
            eval_dataset = build_dataset(eval_dataset_config.datasets[dataset_idx], 
                            tokenizer=tokenizer,
                            data_args=data_args
            )
            data_collator = DataCollatorForHybridDetSegPoseGenDataset(tokenizer, data_args.image_aspect_ratio)
            sampler = DistributedSampler(eval_dataset, shuffle=False, drop_last=False)
            # dataloader eval batch_size=1
            eval_dataloader = DataLoader(eval_dataset, sampler=sampler, collate_fn=data_collator, batch_size=1)
            # get task and dataset_name
            task = eval_dataset.task
            dataset_name = eval_dataset.dataset_name
            # gdino
            if model.use_gdino and task in ['det', 'grd', 'seg', 'count_text', 'count_visual', 'interactive']:
                # det
                if task == 'det':
                    if dataset_name == 'coco':
                        num_classes, topk = 80, 100
                        eval_det(model, eval_dataloader, num_classes=num_classes, topk=topk, with_mask=eval_dataset.with_mask)
                    elif dataset_name == 'crowdhuman':
                        num_classes, topk = 1, 500
                        eval_det(model, eval_dataloader, num_classes=num_classes, topk=topk, with_mask=eval_dataset.with_mask, jsonfile_prefix='crowdhuman')
                    elif dataset_name == 'odinw':
                        num_classes, topk = eval_dataset.num_classes, 100
                        eval_det(model, eval_dataloader, num_classes=num_classes, topk=topk, with_mask=eval_dataset.with_mask)
                    elif dataset_name in ['sod', 'cod']:
                        num_classes, topk = 1, 1
                        # sod
                        if 'DUTS' in eval_dataset.ann_file:
                            save_dir, gt_dir = 'results/DUTS', 'data/sod/DUTS/DUTS-TE-Mask'
                        elif 'DUT-OMRON' in eval_dataset.ann_file:
                            save_dir, gt_dir = 'results/DUT-OMRON', 'data/sod/DUT-OMRON/DUT-OMRON-mask'
                        elif 'ecssd' in eval_dataset.ann_file:
                            save_dir, gt_dir = 'results/ecssd', 'data/sod/ecssd/ground_truth_mask'
                        elif 'PASCAL-S' in eval_dataset.ann_file:
                            save_dir, gt_dir = 'results/PASCAL-S', 'data/sod/PASCAL-S/Masks'
                        elif 'HKU-IS' in eval_dataset.ann_file:
                            save_dir, gt_dir = 'results/HKU-IS', 'data/sod/HKU-IS/gt'
                        # cod
                        if 'CAMO' in eval_dataset.ann_file:
                            save_dir, gt_dir = 'results/CAMO', 'data/cod/CAMO-V.1.0_CVIU2019/CAMO-V.1.0-CVIU2019/GT'
                        elif 'COD10K' in eval_dataset.ann_file:
                            save_dir, gt_dir = 'results/COD10k', 'data/cod/COD10K-v3/Test/GT_Object'
                        eval_sod(model, eval_dataloader, num_classes=num_classes, topk=topk, save_dir=save_dir, gt_dir=gt_dir)
                # grd
                elif task == 'grd':
                    if dataset_name == 'refcoco' or dataset_name == 'reasonseg':
                        num_classes, topk = 1, 1
                    eval_det(model, eval_dataloader, num_classes=num_classes, topk=topk, with_mask=eval_dataset.with_mask)
                # semseg
                elif task == 'seg':
                    if dataset_name == 'ade20k':
                        num_classes, topk = 150, 100
                        sem_seg_postprocess_before_inference = True
                    elif dataset_name == 'cityscape':
                        num_classes, topk = 19, 100
                        sem_seg_postprocess_before_inference = True
                    elif dataset_name == 'mapillary':
                        num_classes, topk = 66, 100
                        sem_seg_postprocess_before_inference = False
                    elif dataset_name == 'loveda':
                        num_classes, topk = 7, 100
                        sem_seg_postprocess_before_inference = True
                    elif dataset_name == 'medical_mr':
                        num_classes, topk = 3, 100
                        sem_seg_postprocess_before_inference = True
                    eval_semseg(model, eval_dataloader, num_classes=num_classes, topk=topk, \
                                sem_seg_postprocess_before_inference=sem_seg_postprocess_before_inference)
                # interactive
                elif task == 'interactive':
                    num_classes, topk = None, 300  # num_classes will be determined by gt number in each image
                    eval_visual_prompt(model, eval_dataloader, num_classes=num_classes, topk=topk, with_mask=eval_dataset.with_mask)
            # unipose
            elif model.use_unipose and task in ['pose']:
                if dataset_name == 'coco':
                    num_classes, topk, num_body_points = 1, 100, 17
                elif dataset_name == 'crowdpose':
                    num_classes, topk, num_body_points = 1, 100, 14
                elif dataset_name == 'unikpt':
                    num_classes, topk, num_body_points = eval_dataset.num_classes, 100, eval_dataset.num_keypoints
                eval_pose(model, eval_dataloader, num_classes=num_classes, topk=topk, num_body_points=num_body_points, dataset_name=dataset_name, ann_file=eval_dataset.ann_file)
        return 

    if data_args.multi_datasets:
        train_datasets = build_multi_datasets(
            data_args.dataset_config,
            tokenizer=tokenizer,
            data_args=data_args
        )
    else:
        dataset_cfg = Config.fromfile(data_args.dataset_config)
        dataset_cfg = dataset_cfg.datasets[0]
        train_datasets = build_dataset(
            dataset_cfg,
            tokenizer=tokenizer,
            data_args=data_args
        )
    data_collator = DataCollatorForHybridDetSegPoseGenDataset(tokenizer, data_args.image_aspect_ratio)

    # set seed for torch dataloaders
    set_seed(training_args.seed)

    
    # 6. Initialize our Trainer
    trainer = VisionLLMv2Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_datasets,
        eval_dataset=None,
        data_collator=data_collator)
    trainer.fsdp_ignore_frozen_params()

    # 9. Training
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        train_result = trainer.train(resume_from_checkpoint=True)
    else:
        train_result = trainer.train()

    metrics = train_result.metrics
    metrics["train_samples"] = len(train_datasets)

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # save models
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

if __name__ == "__main__":
    train()