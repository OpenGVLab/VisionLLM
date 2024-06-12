# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
# Make it more memory efficient by monkey patching the LLaMA model with FlashAttn.

# Need to call this before importing transformers.
from visionllmv2.train.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
from visionllmv2.train.llama_forward_monkey_patch import replace_llama_rmsnorm_with_fused_rmsnorm, replace_llama_forward_with_custom_forward

# NOTE: use_cache=True is not supported for flash-attn, so disabled during training
# during inferece, please comment this line
# replace_llama_attn_with_flash_attn()  
# move to model/visionllmv2.py
# replace_llama_rmsnorm_with_fused_rmsnorm()
# replace_llama_forward_with_custom_forward()

from visionllmv2.train.train import train

if __name__ == "__main__":
    train()
