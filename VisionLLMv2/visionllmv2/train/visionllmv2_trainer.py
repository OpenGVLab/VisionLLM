import numpy as np
import random
from typing import List, Iterator, Optional, Sized

from transformers import Trainer
from transformers.trainer import (
    has_length,
)
from transformers.utils import (
    is_sagemaker_mp_enabled, is_torch_tpu_available, 
    is_accelerate_available, logging,
    is_datasets_available
)
from transformers.trainer_pt_utils import get_parameter_names
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer_utils import ShardedDDPOption, PREFIX_CHECKPOINT_DIR
from transformers.trainer_pt_utils import reissue_pt_warnings
from transformers.training_args import ParallelMode

from packaging import version


import os
import glob
import torch
import torch.nn as nn
from torch.utils.data import Sampler
from typing import List, Dict, Any, Set, Optional
import warnings

from accelerate.utils import DeepSpeedSchedulerWrapper

logger = logging.get_logger(__name__)

# Name of the files used for checkpointing
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
OPTIMIZER_NAME_BIN = "optimizer.bin"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, 'no ignore status')
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_vl_bridge_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return

# LengthGroupedSampler from LLaVA
def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    assert len(mm_indices) > 0, "Should have at least one multimodal sample."
    assert len(lang_indices) > 0, "Should have at least one language sample."

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) >= megabatch_size:
        megabatches = [additional_batch[:megabatch_size]] + megabatches
        additional_batch = additional_batch[megabatch_size:]

    if len(additional_batch) > 0:
        megabatches.append(additional_batch)

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        else:
            indices = get_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)


class RandomSourcedBatchSampler(Sampler):
    data_source: Sized
    replacement: bool

    def __init__(
        self, 
        data_source: Sized, 
        batch_size: int,
        dataset_sizes: List[int],
        generator=None
    ) -> None:

        self.data_source = data_source
        self.batch_size = batch_size
        self.dataset_sizes = dataset_sizes
        self.generator = generator

    @property
    def num_samples(self) -> int:
        return len(self.data_source)

    def __iter__(self) -> Iterator[int]:
        n = len(self.data_source)
        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator

        indices = torch.arange(n)  
        chunked_indices = torch.split(indices, self.dataset_sizes)  
        # sample-level permutation in each dataset
        inner_perm_indices = [x[torch.randperm(len(x), generator=generator)] for x in chunked_indices]
        inner_perm_indices = [x[:len(x) - len(x) % self.batch_size] for x in inner_perm_indices]
        # split into batches
        outer_perm_indices = [torch.split(x, self.batch_size) for x in inner_perm_indices]
        outer_perm_indices = [y for x in outer_perm_indices for y in x]
        # batch-level permutation
        outer_perm_indices = [outer_perm_indices[i] for i in torch.randperm(len(outer_perm_indices), generator=generator)]
        outer_perm_indices = torch.cat(outer_perm_indices, dim=0)
        yield from outer_perm_indices.tolist()

    def __len__(self) -> int:
        return self.num_samples

# In each iter, datasets from tasks that being used for one tool (e.g. gdino, unipose, ...)
class RandomTaskSourcedBatchSampler(Sampler):
    data_source: Sized
    replacement: bool

    def __init__(
        self, 
        data_source: Sized,         # ConcatDataset
        batch_size: int,            # batch size per gpu
        total_batch_size: int,      # total batch size in an iter
        dataset_sizes: List[int],   # dataset size for each dataset
        tasks: List[str],           # dataset task for each dataset
        generator=None
    ) -> None:

        self.data_source = data_source
        self.batch_size = batch_size
        self.total_batch_size = total_batch_size
        self.dataset_sizes = dataset_sizes
        self.tasks = tasks
        self.generator = generator

        self.gdino_tasks = ['det', 'det_cap', 'grd', 'seg', 'count_text', 'count_visual', 'interactive', 'ic_mask']
        self.unipose_tasks = ['pose']
        self.sd_tasks = ['t2i']
        self.ip2p_tasks = ['edit']
        tool_tasks = self.gdino_tasks + self.unipose_tasks + self.sd_tasks + self.ip2p_tasks
        self.vlm_tasks = list(set([t for t in self.tasks if t not in tool_tasks]))


    @property
    def num_samples(self) -> int:
        return len(self.data_source)

    def __iter__(self) -> Iterator[int]:
        n = len(self.data_source)
        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator

        indices = torch.arange(n)  
        chunked_indices = torch.split(indices, self.dataset_sizes)  
        # sample-level permutation in each dataset
        inner_perm_indices = [x[torch.randperm(len(x), generator=generator)] for x in chunked_indices]
        inner_perm_indices = [x[:len(x) - len(x) % self.batch_size] for x in inner_perm_indices] # list[tensor], tensor size of dataset_size_i
        # group by tool indices
        assert len(inner_perm_indices) == len(self.tasks)
        vlm_indices, gdino_indices, unipose_indices, sd_indices, ip2p_indices = [], [], [], [], []
        for inner_indices, task in zip(inner_perm_indices, self.tasks): # for each dataset
            if task in self.vlm_tasks:
                vlm_indices.append(inner_indices)
            if task in self.gdino_tasks:
                gdino_indices.append(inner_indices)
            if task in self.unipose_tasks:
                unipose_indices.append(inner_indices)
            if task in self.sd_tasks:
                sd_indices.append(inner_indices)
            if task in self.ip2p_tasks:
                ip2p_indices.append(inner_indices)
        # permutation indices for each tool
        all_indices = (vlm_indices, gdino_indices, unipose_indices, sd_indices, ip2p_indices)
        outer_perm_indices = []
        for inner_indices in all_indices:
            if len(inner_indices) == 0:
                continue
            # inner_indices: list[tensor], tensor size of dataset_size_i
            # split into per_batch_size:
            inner_indices = [torch.split(x, self.batch_size) for x in inner_indices]
            inner_indices = [y for x in inner_indices for y in x] 
            # batch-level permutation
            inner_indices = [inner_indices[i] for i in torch.randperm(len(inner_indices), generator=generator)]
            inner_indices = torch.cat(inner_indices, dim=0)  # all_dataset_size for a tool
            # split into total_batch_size
            inner_indices = inner_indices[:len(inner_indices) - len(inner_indices) % self.total_batch_size]
            inner_indices = list(torch.split(inner_indices, self.total_batch_size))  # list[tensor], tensor size of [totol_batch_size,]
            outer_perm_indices += inner_indices
        # tool-level permutation
        outer_perm_indices = [outer_perm_indices[i] for i in torch.randperm(len(outer_perm_indices), generator=generator)]
        outer_perm_indices = torch.cat(outer_perm_indices, dim=0)
        yield from outer_perm_indices.tolist()


    def __len__(self) -> int:
        return self.num_samples

def match_name_keywords(n, name_keywords):
    out = False
    for b in name_keywords:
        if b in n:
            out = True
            break
    return out

ALL_LR_LAYERS = ['backbone', 'sampling_offsets', 'reference_points']


class VisionLLMv2Trainer(Trainer):
    def fsdp_ignore_frozen_params(self):
        if self.is_fsdp_enabled:
            frozen_params = [p for p in self.model.parameters() if p.requires_grad == False]
            self.accelerator.state.fsdp_plugin.ignored_parameters = iter(frozen_params)
        return

    # sampler
    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                # self.args.train_batch_size * self.args.gradient_accumulation_steps, # TODO: seems that we should not have gradient_accumulation_steps
                self.args.train_batch_size,
                world_size=self.args.world_size,
                lengths=lengths,
                group_by_modality=True,
            )
        elif self.args.group_by_data_source:
            cumu_sizes = self.train_dataset.cumulative_sizes
            dataset_sizes = [cumu_sizes[0]] + [cumu_sizes[i]-cumu_sizes[i-1] for i in range(1,len(cumu_sizes))]
            return RandomSourcedBatchSampler(
                self.train_dataset,
                self.args.train_batch_size,
                dataset_sizes
            )
        elif self.args.group_by_task_data_source:
            cumu_sizes = self.train_dataset.cumulative_sizes
            dataset_sizes = [cumu_sizes[0]] + [cumu_sizes[i]-cumu_sizes[i-1] for i in range(1,len(cumu_sizes))]
            assert self.args.gradient_accumulation_steps == 1
            total_batch_size = self.args.train_batch_size * self.args.world_size
            datasets = self.train_dataset.datasets  # list[dataset]
            tasks = [getattr(d, 'task', 'chat') for d in datasets]
            return RandomTaskSourcedBatchSampler(
                self.train_dataset,
                self.args.train_batch_size,
                total_batch_size,
                dataset_sizes,
                tasks
            )
        else:
            return super()._get_train_sampler()


    # set lr for specific layers
    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model

        if self.optimizer is None:
            # weight decay
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            # lr layers
            lr_parameters = [name for name in ALL_LR_LAYERS]  # x lr_multiplier
            lr_llm_parameters = ['llm', 'region_encoder', 'vl_bridge']  # x lr_llm_multiplier
            optimizer_grouped_parameters = [
                # weight_decay = self.args.weight_decay
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if 
                            n in decay_parameters and not match_name_keywords(n, lr_parameters) and 
                            not match_name_keywords(n, lr_llm_parameters) and p.requires_grad            
                    ],
                    "weight_decay": self.args.weight_decay, "lr": self.args.learning_rate,
                    "name": "decay_parameters"
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if 
                            n in decay_parameters and match_name_keywords(n, lr_parameters) and p.requires_grad            
                    ],
                    "weight_decay": self.args.weight_decay, "lr": self.args.learning_rate * self.args.lr_multiplier,
                    "name": "decay_parameters_backbone"
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if 
                            n in decay_parameters and match_name_keywords(n, lr_llm_parameters) and p.requires_grad            
                    ],
                    "weight_decay": self.args.weight_decay, "lr": self.args.learning_rate * self.args.lr_llm_multiplier,
                    "name": "decay_parameters_llm"
                },
                # weight_decay = 0.0
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if 
                            n not in decay_parameters and not match_name_keywords(n, lr_parameters) and 
                            not match_name_keywords(n, lr_llm_parameters) and p.requires_grad            
                    ],
                    "weight_decay": 0.0, "lr": self.args.learning_rate,
                    "name": "no_decay_parameters"
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if 
                            n not in decay_parameters and match_name_keywords(n, lr_parameters) and p.requires_grad            
                    ],
                    "weight_decay": 0.0, "lr": self.args.learning_rate * self.args.lr_multiplier,
                    "name": "no_decay_parameters_backbone"
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if 
                            n not in decay_parameters and match_name_keywords(n, lr_llm_parameters) and p.requires_grad            
                    ],
                    "weight_decay": 0.0, "lr": self.args.learning_rate * self.args.lr_llm_multiplier,
                    "name": "no_decay_parameters_llm"
                },
            ]
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            if self.sharded_ddp == ShardedDDPOption.SIMPLE:
                from fairscale.optim import OSS
                self.optimizer = OSS(
                    params=optimizer_grouped_parameters,
                    optim=optimizer_cls,
                    **optimizer_kwargs,
                )
            else:
                self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
                if optimizer_cls.__name__ == "Adam8bit":
                    import bitsandbytes

                    manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                    skipped = 0
                    for module in opt_model.modules():
                        if isinstance(module, nn.Embedding):
                            skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                            logger.info(f"skipped {module}: {skipped/2**20}M params")
                            manager.register_module_override(module, "weight", {"optim_bits": 32})
                            logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                    logger.info(f"skipped: {skipped/2**20}M params")

        if is_sagemaker_mp_enabled():
            import smdistributed.modelparallel.torch as smp
            self.optimizer = smp.DistributedOptimizer(self.optimizer)

        return self.optimizer

    # transformer 4.34 has fixed resume lr bug