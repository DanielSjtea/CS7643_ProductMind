import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from peft import LoraConfig, get_peft_model
from PIL import Image
import requests

##### Abbrievations used:
# B – batch size

class CrossAttentionBlock(nn.Module):
    """
    Cross Attention blocks (Flamingo-styled)
    """
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.mh_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dtype=torch.float16)
        self.lnorm1 = nn.LayerNorm(embed_dim, dtype=torch.float16)
        self.lnorm2 = nn.LayerNorm(embed_dim, dtype=torch.float16)
        ## 4x expansion typical in BERT, GPT – used for wider intermediate representation
        ## GELU for non-linearity (instead of ReLU)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim, dtype=torch.float16),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim, dtype=torch.float16)
        )

    def forward(self, t: torch.Tensor, v: torch.Tensor):
        """
        :param t: text tokens/embedding sequence (Query)
        :param v: visual feature sequence (Key/Value)
        """
        mh_attn_out, _ = self.mh_attn(t, v, v) ## cross attn
        output = self.lnorm1(t + mh_attn_out) ## residual + layernorm
        output = self.lnorm2(output + self.mlp(output)) ## MLP + residual + layernorm
        return output
    
class PerceiverResampler(nn.Module):
    """
    Perceiver Resampler adapted from Flamingo (scaled down version)
    Takes as input a variable number of image features from image encoder and produces a fixed number of visual outputs (num_latents)
    """
    def __init__(self, d_model: int, num_latents: int = 64, num_layers: int = 2, num_heads: int = 8, dtype=torch.float16):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, d_model, dtype=dtype))
        self.layers = nn.ModuleList(
            PerceiverResamplerLayer(d_model, num_heads) for _ in range(num_layers)
        )
        self.dtype = dtype

    def forward(self, patch_embeds: torch.Tensor) -> torch.Tensor:
        ## Get batch size
        B = patch_embeds.size(0)

        lat = self.latents.unsqueeze(0).expand(B, -1, -1).to(patch_embeds.dtype) # (B, num_latents, d_model)

        ## apply the Perceiver Resampler layers
        for layer in self.layers:
            lat = layer(lat, patch_embeds)
        
        return lat

class PerceiverResamplerLayer(nn.Module):
    """
    Repeated layer in the PerceiverResampler
    """
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True, dtype=torch.float16)
        self.ffw = nn.Sequential(
            nn.Linear(d_model, 4 * d_model, dtype=torch.float16),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model, dtype=torch.float16),
        )
        self.lnorm_attn = nn.LayerNorm(d_model, dtype=torch.float16)
        self.lnorm_ffw = nn.LayerNorm(d_model, dtype=torch.float16)
    
    def forward(self, x: torch.Tensor, x_f: torch.Tensor) -> torch.Tensor:
        """
        :param x: learned latents of shape (num_latents, d_model)
        :param x_f: visual features (B, S, d_model)
        """
        kv = torch.cat([x_f, x], dim=1)

        # attention with residual
        attn_out, _ = self.attn(x, kv, kv)
        x = x + attn_out
        x = self.lnorm_attn(x)

        # feed forward with residual
        ffw_out = self.ffw(x)
        x = x + ffw_out
        x = self.lnorm_ffw(x)

        return x
    
class DecoderLayerWithCrossAttn(nn.Module):
    def __init__(self, base_layer, cross_attn_block: CrossAttentionBlock | None):
        super().__init__()
        self.base_layer = base_layer
        self.cross_attn_block = cross_attn_block
        self.v_tokens = None  # set externally

    def forward(self, *args, **kwargs):
        # base_layer will return a tuple:
        # (hidden_states, present_key_value) or (hidden_states, present_key_value, attn_weights)
        outputs = self.base_layer(*args, **kwargs)

        hidden_states = outputs[0]

        if self.cross_attn_block is not None and self.v_tokens is not None:
            hidden_states = self.cross_attn_block(hidden_states, self.v_tokens)

        # return same tuple length as input
        if len(outputs) == 2:
            return (hidden_states, outputs[1])
        elif len(outputs) == 3:
            return (hidden_states, outputs[1], outputs[2])
        else:
            raise ValueError("Unexpected number of outputs from base decoder layer.")
    
class HybridBlip2Flamingo(nn.Module):
    """
    Modification on BLIP-2 to include Flamingo-style cross-attention blocks

    Frozen components:
    - Image Encoder
    - Q-Former
    - LLM (OPT)

    Trainable Params:
    - Cross-attention between input text, visual and language model (OPT)
        - MLP layers
    - PerceiverResampler
        - Attn layers
        - FFW layers
    """
    def __init__(
        self,
        n_attn_heads: int = 4,
        *,
        attn_layers_at: tuple[int, ...] | None = None,
        attn_every_n: int | None = 4,
    ):
        super().__init__()
        self.model_blip2 = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16
        )
        self.processor = Blip2Processor.from_pretrained(
            "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16
        )

        ## get sub-modules from BLIP-2
        self.vision_blip2 = self.model_blip2.vision_model.eval().requires_grad_(False)
        self.qformer_blip2 = self.model_blip2.qformer.eval().requires_grad_(False)
        self.language_blip2 = self.model_blip2.language_model
        self.language_proj_blip2 = self.model_blip2.language_projection.eval().requires_grad_(False)

        ## Get language model dimensions
        d_model = self.language_blip2.config.hidden_size
        n_layers = len(self.language_blip2.model.decoder.layers)

        ## Perceiver Resampler from Flamingo to use later for cross attn
        d_image_encoder = self.vision_blip2.config.hidden_size
        self.perceiver_resampler = PerceiverResampler(d_model=d_image_encoder)
        self.v_proj = nn.Linear(d_image_encoder, d_model, dtype=torch.float16)

        ## Decide which layers to add cross-attention
        if attn_layers_at:
            self.cross_idx = set(attn_layers_at)
        elif attn_layers_at and attn_every_n > 0:
            self.cross_idx = set(range(attn_every_n - 1, n_layers, attn_every_n))
        else:
            self.cross_idx = set()

        self.cross_attn_blocks = nn.ModuleDict()
        for i in range(n_layers):
            if i in self.cross_idx:
                cross_attn = CrossAttentionBlock(d_model, n_attn_heads)
                self.cross_attn_blocks[str(i)] = cross_attn
                wrapped_layer = DecoderLayerWithCrossAttn(
                    self.language_blip2.model.decoder.layers[i],
                    cross_attn
                )
            else:
                wrapped_layer = DecoderLayerWithCrossAttn(
                    self.language_blip2.model.decoder.layers[i],
                    None
                )
            self.language_blip2.model.decoder.layers[i] = wrapped_layer

        self.blip2_query_tokens = self.model_blip2.query_tokens

        ## Freeze all BLIP-2 weights
        ## Set new model parameters for added cross attention layers to train
        self.model_blip2.requires_grad_(False)
        self.perceiver_resampler.requires_grad_(True)
        self.v_proj.requires_grad_(True)
        for cross_attn_block in self.cross_attn_blocks.values():
            cross_attn_block.requires_grad_(True)

    def _encode_image(self, images):
        """
        Image encoder using BLIP-2 vision model
        """
        if isinstance(images, torch.Tensor):
            pixel = images
        else:
            pixel = self.processor(images=images, return_tensors="pt").pixel_values
        pixel = pixel.to(next(self.parameters()).device)
        return self.vision_blip2(pixel, return_dict=True).last_hidden_state
    
    def forward(self, images, texts, labels=None):
        device = next(self.parameters()).device

        v_image_encodings = self._encode_image(images)

        ## standard BLIP-2 approach
        B = v_image_encodings.size(0) ## batch size
        q_tokens = self.blip2_query_tokens.expand(B, -1, -1)
        q_out = self.qformer_blip2(query_embeds=q_tokens, encoder_hidden_states=v_image_encodings, return_dict=True)
        q_features = self.language_proj_blip2(q_out.last_hidden_state.to(dtype=torch.float16)) # (B, Q, d_model)

        ## Visual tokens for cross-attention
        v_tokens = self.v_proj(self.perceiver_resampler(v_image_encodings))

        ## text encodings using default BLIP-2
        text_enc = self.processor.tokenizer(texts, padding=True, return_tensors="pt").to(device)

        ## get text embeddings
        text_embeds = self.language_blip2.model.decoder.embed_tokens(text_enc.input_ids)

        ## add position embeddings
        seq_len = text_enc.input_ids.size(1)
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand_as(text_enc.input_ids)
        position_embeds = self.language_blip2.model.decoder.embed_positions(position_ids)
        text_embeds = text_embeds + position_embeds

        ## default BLIP-2 concat Q-Former tokens to text
        input_embeds = torch.cat([q_features, text_embeds], dim = 1)
        attention_mask = torch.cat([
            torch.ones(B, q_features.size(1), dtype=torch.bool, device=device),
            text_enc.attention_mask.bool()
        ], dim=1)

        # Set v_tokens for cross-attn blocks
        for i in self.cross_idx:
            self.cross_attn_blocks[str(i)].v_tokens = v_tokens

        ## Manual layer processing for Cross Attention layers
        hidden = input_embeds
        for layer in self.language_blip2.model.decoder.layers:
            hidden = layer(hidden, attention_mask=attention_mask)[0]

        ## standard BLIP-2 final processing
        hidden = self.language_blip2.model.decoder.final_layer_norm(hidden)
        logits = self.language_blip2.lm_head(hidden)

        ## loss calculation using CrossEntropy
        if labels is not None:
            lab = self.processor(labels, padding=True, return_tensors="pt").input_ids.to(device)

            # Ensure logits and labels are the same length
            logits = logits[:, -lab.size(1):, :]  # truncate any excess tokens

            # Shift for CLM: predict token t+1 using token t
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = lab[:, 1:].contiguous()

            loss = nn.CrossEntropyLoss(ignore_index=self.processor.tokenizer.pad_token_id)(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )

        return {"logits": logits, "loss": loss}
    
    @torch.no_grad()
    def generate(
        self,
        images: list[Image.Image] | torch.Tensor,
        prompts: list[str] | str = "",
        max_new_tokens: int = 30,
        temperature: float = 1.0,
        **generate_kwargs,
    ) -> list[str]:
        """
        BLIP‑2‑style conditional generation with Flamingo cross‑attention.

        Args:
            images: Either a tensor (B,3,H,W) **already** on the target device or a
                    list/tuple of PIL images.
            prompts: A single string or a list of length B. Empty string == “caption”.
            max_new_tokens: Same meaning as in HF `generate`.
            temperature: Passed straight to HF `generate` (works with sampling).
            **generate_kwargs: Anything else accepted by `OPTForCausalLM.generate`
                            (e.g. top_k, top_p, num_beams, repetition_penalty …).

        Returns:
            A `list[str]` of length `batch_size`, *only* the text generated after the
            user prompt.
        """
        device = next(self.parameters()).device
        if "prompt" in generate_kwargs:
            prompts = generate_kwargs.pop("prompt")
        if isinstance(prompts, str):
            prompts = [prompts]

        # ---------------
        # 1) IMAGE → PREFIX
        # ---------------
        # Vision encoder
        image_feats = self._encode_image(images)                                # (B, Sv, d_img)
        B, _, d_img = image_feats.shape

        # Q‑Former (frozen, fp32) → down‑cast back to fp16
        q_tokens = self.blip2_query_tokens.expand(B, -1, -1)                    # (B, Q, d_img)
        q_out = self.qformer_blip2(
            query_embeds=q_tokens,
            encoder_hidden_states=image_feats,
            return_dict=True,
        ).last_hidden_state.to(dtype=torch.float16)

        img_prefix = self.language_proj_blip2(q_out)                            # (B, Q, d_model)

        # Perceiver‑Resampler → visual tokens used inside cross‑att blocks.
        # We *cache* them so that every forward inside HF decoding sees the
        # same `v_tokens` without recomputing.
        self._cached_v_tokens = self.v_proj(self.perceiver_resampler(image_feats))  # (B, L, d_model)
        # Inject v_tokens into decoder cross-attn blocks
        for i in self.cross_idx:
            self.cross_attn_blocks[str(i)].v_tokens = self._cached_v_tokens

        # ---------------
        # 2) TEXT PROMPT → TOKENS
        # ---------------
        tok_out = self.processor.tokenizer(
            prompts, padding=True, return_tensors="pt"
        ).to(device)
        input_ids = tok_out.input_ids                                           # (B, T)
        attn_mask = tok_out.attention_mask                                      # (B, T)

        # Word embeddings (OPT): (B,T,d_model)
        txt_embeds = self.language_blip2.model.decoder.embed_tokens(input_ids)

        # Add learned positional embeddings
        pos_ids = torch.arange(txt_embeds.size(1), device=device).unsqueeze(0)
        txt_embeds = txt_embeds + self.language_blip2.model.decoder.embed_positions(
            attn_mask, position_ids=pos_ids
        )

        # ---------------
        # 3) CONCAT prefix + prompt   →   inputs_embeds / attention_mask
        # ---------------
        inputs_embeds = torch.cat([img_prefix, txt_embeds], dim=1)              # (B, Q+T, d)
        prefix_mask   = torch.ones(B, img_prefix.size(1), dtype=attn_mask.dtype, device=device)
        attention_mask = torch.cat([prefix_mask, attn_mask], dim=1)             # (B, Q+T)

        # ---------------
        # 4) HF GENERATE (OPT)
        # ---------------
        # NB: Because our Flamingo cross‑attn blocks sit *outside* the wrapped
        # OPT model, we expose them via a tiny wrapper that simply calls the
        # normal OPT decoder layer *then* injects cross‑attention.  That wrapper
        # is already what `forward()` does, so `prepare_inputs_for_generation`
        # just needs to re‑route `inputs_embeds` through *this* module.
        #
        # Easiest: register ourselves as the generation model temporarily.
        #
        # -> call self.language_blip2.generate with `inputs_embeds` **but**
        #    override `forward` & `prepare_inputs_for_generation` through
        #    `patch_generate_forward()`.  For clarity (and less metaprogramming)
        #    we instead wrap everything in our own loop based on
        #    `torch.utils.checkpoint` and incremental kv‑cache — that is quite
        #    verbose, so below we stick to the pragmatic approach:
        outputs = self.language_blip2.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            **generate_kwargs,
        )                                                                       # (B, Q+T+new)

        # ---------------
        # 5) POST‑PROCESS (strip prefix & prompt)
        # ---------------
        # We want only tokens *after* the user‑supplied prompt, not the prefix.
        generated_texts = []
        for seq, prompt in zip(outputs, prompts):
            # Convert to list[int], drop bos if present
            seq = seq.tolist()
            if seq and seq[0] == self.processor.tokenizer.bos_token_id:
                seq = seq[1:]

            # Decode everything, then remove the original prompt
            decoded = self.processor.tokenizer.decode(seq, skip_special_tokens=True)
            generated_texts.append(decoded[len(prompt):].lstrip())

        # Clean cache
        del self._cached_v_tokens
        for block in self.cross_attn_blocks.values():
            block.v_tokens = None

        return generated_texts


if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = HybridBlip2Flamingo(attn_layers_at=(10, 20, 30)).to(device)

    img_url = "https://github.com/salesforce/LAVIS/blob/main/docs/_static/merlion.png?raw=true"
    image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")

    caption = model.generate([image], prompts="A famous statue in ")
    print("Generated caption:", caption)