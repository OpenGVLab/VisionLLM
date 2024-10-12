from typing import Optional
import random
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import StableDiffusionInstructPix2PixPipeline, AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from transformers import PretrainedConfig, PreTrainedModel, AutoModel, AutoConfig, CLIPTextModel, CLIPTokenizer
from transformers.modeling_outputs import ModelOutput


@dataclass
class InstructPix2PixWithLLMEmbOutput(ModelOutput):
    loss: torch.FloatTensor = None
    image_loss: Optional[torch.FloatTensor] = None
    caption_loss: Optional[torch.FloatTensor] = None


class InstructPix2PixWithLLMEmbConfig(PretrainedConfig):
    model_type = "instructpix2pix_with_llm_emb"
    is_composition = True

    def __init__(
        self,
        llm_hidden_size=4096,
        sd_hidden_size=768,
        num_queries=77,
        num_encoder_layers=1,
        num_decoder_layers=1,
        sd_model_id="timbrooks/instruct-pix2pix",
        trigger_token="[EDIT]",
        trigger_token_id=None,
        num_embed_tokens=64,
        embed_tokens=None,
        cfg_drop_rate=0.0,
        cfg_scale=7.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.llm_hidden_size = llm_hidden_size
        self.sd_hidden_size = sd_hidden_size
        self.num_queries = num_queries
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.sd_model_id = sd_model_id
        self.trigger_token = trigger_token
        self.trigger_token_id = trigger_token_id
        self.num_embed_tokens = num_embed_tokens
        self.embed_tokens = embed_tokens
        self.cfg_drop_rate = cfg_drop_rate
        self.cfg_scale = cfg_scale

        # if len(embed_tokens) != num_embed_tokens:
        #     raise ValueError(f'number of embed token {num_embed_tokens} is not matched with {embed_tokens}')


class InstructPix2PixWithLLMEmbPreTrainedModel(PreTrainedModel):
    config_class = InstructPix2PixWithLLMEmbConfig
    base_model_prefix = "model"


class InstructPix2PixWithLLMEmb(InstructPix2PixWithLLMEmbPreTrainedModel):
    def __init__(self, config: InstructPix2PixWithLLMEmbConfig):
        super().__init__(config)
        llm_hidden_size = config.llm_hidden_size
        sd_hidden_size = config.sd_hidden_size
        self.emb_proj = nn.Sequential(
            nn.Linear(llm_hidden_size, sd_hidden_size),
            nn.GELU(),
            nn.Linear(sd_hidden_size, sd_hidden_size),
        )
        self.llm2sd_mapper_queries = torch.nn.Parameter(torch.randn((1, config.num_queries, sd_hidden_size))) # [1, 77, c]
        self.llm2sd_mapper = nn.Transformer(
            batch_first=True,
            norm_first=True,
            d_model=sd_hidden_size,
            num_encoder_layers=config.num_encoder_layers,
            num_decoder_layers=config.num_decoder_layers,
            dim_feedforward=sd_hidden_size * 4,
            dropout=0.0,
        )

        # sd
        self.sd_text_encoder = CLIPTextModel.from_pretrained(config.sd_model_id, subfolder="text_encoder", local_files_only=True, torch_dtype=torch.bfloat16)
        self.sd_tokenizer = CLIPTokenizer.from_pretrained(config.sd_model_id, subfolder="tokenizer", local_files_only=True, torch_dtype=torch.bfloat16)
        self.sd_vae = AutoencoderKL.from_pretrained(config.sd_model_id, subfolder="vae", local_files_only=True, torch_dtype=torch.bfloat16)
        self.sd_unet = UNet2DConditionModel.from_pretrained(config.sd_model_id, subfolder="unet", local_files_only=True, torch_dtype=torch.bfloat16)
        self.sd_pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            config.sd_model_id,
            vae=self.sd_vae,
            unet=self.sd_unet,
            tokenizer=self.sd_tokenizer,
            text_encoder=self.sd_text_encoder,
            safety_checker=None,
            local_files_only=True,
            torch_dtype=torch.bfloat16
        )
        self.sd_noise_scheduler = DDPMScheduler.from_config(self.sd_pipeline.scheduler.config)

        # Freeze vae and text_encoder
        self.sd_vae.requires_grad_(False)
        self.sd_text_encoder.requires_grad_(False)
        # self.sd_unet.requires_grad_(False)

        self.post_init()

    def forward(
        self,
        input_ids,
        hidden_states,
        input_images=None,
        output_images=None,
        captions=None,
        **kwargs,
    ):
        special_token_index = (input_ids == self.config.trigger_token_id).nonzero()
        if len(special_token_index) == 0:
            return InstructPix2PixWithLLMEmbOutput(loss=0)

        # ---------- compute image loss ---------- #
        last_hidden_state = hidden_states

        t2i_input_embedding = []
        for i in range(len(special_token_index)):
            bs_id, seq_id = special_token_index[i]
            t2i_input_embedding.append(last_hidden_state[bs_id: bs_id + 1, seq_id + 1: seq_id + 1 + self.config.num_embed_tokens, :])
        t2i_input_embedding = torch.cat(t2i_input_embedding, dim=0)

        img_token_bs = t2i_input_embedding.shape[0]
        t2i_input_embedding = self.emb_proj(t2i_input_embedding)
        mapping_feature = self.llm2sd_mapper(src=t2i_input_embedding, tgt=self.llm2sd_mapper_queries.repeat(img_token_bs, 1, 1))
        image_loss = self.compute_image_loss(mapping_feature, input_images[special_token_index[:, 0]], output_images[special_token_index[:, 0]])
        loss = image_loss
        
        # ---------- compute caption loss ---------- #
        caption_loss = None
        if captions is not None and not any([c is None for c in captions]):
            caption_feature = []
            for i in range(len(special_token_index)):
                bs_id, seq_id = special_token_index[i]
                caption_feature.append(self.encode_caption(captions[bs_id], self.sd_tokenizer.model_max_length))

            caption_feature = torch.cat(caption_feature, dim=0)
            caption_loss = F.mse_loss(mapping_feature, caption_feature)
            loss = loss + caption_loss * 0.1

        return InstructPix2PixWithLLMEmbOutput(
            loss=loss,
            image_loss=image_loss,
            caption_loss=caption_loss,
        )

    def compute_image_loss(self, encoder_hidden_states, input_images, output_images):
        latents = self.sd_vae.encode(output_images).latent_dist.sample()
        latents = latents.to(output_images.dtype)
        latents = latents * self.sd_vae.config.scaling_factor

        noise = torch.randn_like(latents)
        bsz = latents.shape[0]

        # Sample a random timestep for each image
        timesteps = torch.randint(0, self.sd_noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
        timesteps = timesteps.long()

        noisy_latents = self.sd_noise_scheduler.add_noise(latents, noise, timesteps)
        original_image_embeds = self.sd_vae.encode(input_images).latent_dist.mode()

        # Conditioning dropout to support classifier-free guidance during inference. For more details
        # check out the section 3.2.1 of the original paper https://arxiv.org/abs/2211.09800.
        if self.config.cfg_drop_rate > 0:
            random_p = torch.rand(bsz, device=latents.device)
            # Sample masks for the edit prompts.
            prompt_mask = random_p < 2 * self.config.cfg_drop_rate
            prompt_mask = prompt_mask.reshape(bsz, 1, 1)
            # Final text conditioning.
            null_conditioning = self.encode_caption([""], self.config.num_queries)
            encoder_hidden_states = torch.where(prompt_mask, null_conditioning, encoder_hidden_states)

            # Sample masks for the original images.
            image_mask_dtype = original_image_embeds.dtype
            image_mask = 1 - (
                (random_p >= self.config.cfg_drop_rate).to(image_mask_dtype)
                * (random_p < 3 * self.config.cfg_drop_rate).to(image_mask_dtype)
            )
            image_mask = image_mask.reshape(bsz, 1, 1, 1)
            # Final image conditioning.
            original_image_embeds = image_mask * original_image_embeds

        concatenated_noisy_latents = torch.cat([noisy_latents, original_image_embeds], dim=1)

        model_pred = self.sd_unet(concatenated_noisy_latents, timesteps, encoder_hidden_states).sample

        target = noise
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        return loss

    def encode_caption(self, caption, length):
        text_inputs = self.sd_tokenizer(
            caption,
            padding="max_length",
            max_length=length,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        prompt_embeds = self.sd_text_encoder(text_inputs.input_ids)[0]
        return prompt_embeds

    @torch.no_grad()
    def extract_features(self, input_ids, hidden_states):
        special_token_index = (input_ids == self.config.trigger_token_id).nonzero()
        # special_token_index = (input_ids == 32003).nonzero()
        last_hidden_state = hidden_states # [bs, l, c]

        t2i_input_embedding = []
        for i in range(len(special_token_index)):
            bs_id, seq_id = special_token_index[i]
            t2i_input_embedding.append(last_hidden_state[bs_id: bs_id + 1, seq_id + 1: seq_id + 1 + self.config.num_embed_tokens, :])
        t2i_input_embedding = torch.cat(t2i_input_embedding, dim=0)

        img_token_bs = t2i_input_embedding.shape[0]
        t2i_input_embedding = self.emb_proj(t2i_input_embedding)
        mapping_feature = self.llm2sd_mapper(src=t2i_input_embedding, tgt=self.llm2sd_mapper_queries.repeat(img_token_bs, 1, 1))
        return mapping_feature

    @torch.no_grad()
    def run(self, input_ids, hidden_states, **kwargs):
        mapping_feature = self.extract_features(input_ids, hidden_states)
        sd_pipeline = self.sd_pipeline.to(mapping_feature.device, dtype=mapping_feature.dtype)
        predicted_image = sd_pipeline(prompt_embeds=mapping_feature, **kwargs).images
        return predicted_image


def concat_cache_hidden_states(hidden_states_token_list):
    hidden_states_layer_list_cat = [[] for _ in range(len(hidden_states_token_list[0]))]
    for hidden_states_layer_list in hidden_states_token_list:
        for layer, hidden_states in enumerate(hidden_states_layer_list):
            hidden_states_layer_list_cat[layer].append(hidden_states)
    return [torch.cat(x, dim=1) for x in hidden_states_layer_list_cat]


AutoConfig.register("instructpix2pix_with_llm_emb", InstructPix2PixWithLLMEmbConfig)
AutoModel.register(InstructPix2PixWithLLMEmbConfig, InstructPix2PixWithLLMEmb)