import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import Blip2Processor, Blip2Model, Blip2Config
from transformers import AutoModelForCausalLM, AutoTokenizer

class CrossAttentionBlock(nn.Module):
    """
    Cross Attention blocks (Flamingo-styled)
    """
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x, context):
        # x: (batch, seq, dim), context: (batch, context_seq, dim)
        attn_output, _ = self.cross_attn(x, context, context)
        x = x + attn_output
        x = self.norm(x)
        mlp_output = self.mlp(x)
        x = x + mlp_output
        x = self.norm2(x)
        return x


class PerceiverResampler(nn.Module):
    """
    Perceiver Resampler adapted from Flamingo (scaled down version)
    Takes as input a variable number of image features from image encoder and produces a fixed number of visual outputs (num_latents)
    """
    def __init__(self, dim, num_latents=64, latent_dim=2560, num_layers=2):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=latent_dim, nhead=8)
            for _ in range(num_layers)
        ])
        self.proj_in = nn.Linear(768, latent_dim)

    def forward(self, x):
        # x: (batch, seq, dim)
        batch_size = x.size(0)
        x_proj = self.proj_in(x)
        latents = self.latents.unsqueeze(0).expand(batch_size, -1, -1)
        hidden = latents + 0
        for layer in self.layers:
            latents = layer(latents)
        return latents  

        for layer in self.layers:
            hidden = layer(hidden)
        out = self.proj_out(hidden)
        return out 


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
    def __init__(self, 
                 blip2_model_name="Salesforce/blip2-opt-2.7b", 
                 llm_model_name="facebook/opt-2.7b", 
                 cross_attn_layers=2,
                 visual_dim = 768, 
                 text_dim=2560, 
                 num_resampler_latents=64):
        super().__init__()

        # Load BLIP-2
        self.processor = Blip2Processor.from_pretrained(blip2_model_name)
        self.blip2 = Blip2Model.from_pretrained(blip2_model_name)
        for p in self.blip2.parameters():
            p.requires_grad = False  # Freeze BLIP2 (including image encoder & Q-Former)

        # Visual resampler (Flamingo style)
        self.perceiver_resampler = PerceiverResampler(
            dim=visual_dim,
            num_latents=num_resampler_latents,
            latent_dim=text_dim,
            num_layers=2
        )

        # LLM (OPT)
        self.llm = AutoModelForCausalLM.from_pretrained(llm_model_name)
        for p in self.llm.parameters():
            p.requires_grad = False  # Freeze OPT
        
        # Cross-attention blocks
        self.cross_attn_blocks = nn.ModuleList([
            CrossAttentionBlock(dim=text_dim, num_heads=8)
            for _ in range(cross_attn_layers)
        ])

        self.visual_proj = nn.Identity()

    def forward(
        self, 
        pixel_values,     # (batch, channels, H, W)
        input_ids,        # (batch, seq)
        attention_mask=None,
        labels=None
    ):
        # 1. Get visual tokens
        blip2_outputs = self.blip2(pixel_values=pixel_values, input_ids=input_ids, return_dict=True)
        image_embeds = blip2_outputs.qformer_outputs.last_hidden_state  # (batch, n_visual, visual_dim)
        visual_tokens = self.perceiver_resampler(image_embeds)
        visual_tokens = self.visual_proj(visual_tokens)  # (batch, n_visual, text_dim)

        batch_size, n_visual, _ = visual_tokens.shape
        _, seq_len = input_ids.shape

        # 2. Embed input_ids (text tokens) and concat with visual tokens
        text_embeds = self.llm.model.decoder.embed_tokens(input_ids)
        x = torch.cat([visual_tokens, text_embeds], dim=1)  # [batch, n_visual + seq_len, text_dim]

        print("input_ids.shape:", input_ids.shape)
        print("labels.shape:", labels.shape if labels is not None else None)
        print("x.shape:", x.shape)

        # 3. Labels for loss (ignore visual tokens)
        if labels is not None:
            new_labels = torch.full((batch_size, n_visual + seq_len), -100, dtype=labels.dtype, device=labels.device)
            new_labels[:, n_visual:] = labels
        else:
            new_labels = None

        # 4. Attention mask
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, n_visual + seq_len), dtype=torch.long, device=x.device)
        else:
            prefix_mask = torch.ones((batch_size, n_visual), dtype=attention_mask.dtype, device=attention_mask.device)
            attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        print("attention_mask.shape:", attention_mask.shape)

        # 5. Pass to LLM
        outputs = self.llm(
            inputs_embeds=x,
            attention_mask=attention_mask,
            labels=new_labels
        )
        return outputs

    @torch.no_grad()
    def generate(
        self,
        pixel_values,
        input_ids,
        attention_mask=None,
        max_new_tokens=32,
        num_beams=1
    ):

        batch_size = pixel_values.shape[0]

        # 1. Get visual tokens
        dummy_input_ids = torch.full((batch_size, 1), self.processor.tokenizer.bos_token_id, device=pixel_values.device)
        outputs = self.blip2(pixel_values=pixel_values, input_ids=dummy_input_ids, return_dict=True)
        image_embeds = outputs.qformer_outputs.last_hidden_state
        visual_tokens = self.perceiver_resampler(image_embeds)
        visual_tokens = self.visual_proj(visual_tokens)
        
        text_embeds = self.llm.model.decoder.embed_tokens(input_ids)
        x = torch.cat([visual_tokens, text_embeds], dim=1)

        n_visual = visual_tokens.shape[1]
        batch_size, seq_len = input_ids.shape
   
        if attention_mask is not None:
            prefix_mask = torch.ones((batch_size, n_visual), dtype=attention_mask.dtype, device=attention_mask.device)
            gen_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        else:
            gen_mask = torch.ones((batch_size, n_visual + seq_len), device=x.device)
  
        # Generate with LLM, using visual-injected embeddings
        generated_ids = self.llm.generate(
                  inputs_embeds=x,
                  attention_mask=gen_mask,
                  max_new_tokens=max_new_tokens,
                  num_beams=num_beams
        )
        # Decode predictions to strings!
        if isinstance(generated_ids, torch.Tensor):
            generated_ids = generated_ids.cpu()
        decoded_preds = [
            self.processor.tokenizer.decode(ids, skip_special_tokens=True)
            for ids in generated_ids
        ]
        return decoded_preds
