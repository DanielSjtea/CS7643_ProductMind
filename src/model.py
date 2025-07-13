from math import sqrt
import types

import torch
import torch.nn as nn
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from peft import LoraConfig, TaskType, get_peft_model
from PIL import Image
import requests

class CrossAttention(nn.Module):
    """
    Cross Attention Layer

    We use the text to attend to visual features: 
    query Q comes from text and key K, value V comes from visual image

    Abbreviations:
    N: batch size
    T: text sequence length
    S: visual sequence length
    d_model: hidden dimensions of the entire model (e.g. hidden_dim_model)
    d_head: hidden dimensions of each attention head (e.g. hidden_dim_head)
    """
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_head = d_model // n_heads
        self.scale = 1 / sqrt(self.d_head)

        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.output_proj = nn.Linear(d_model, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, text: torch.Tensor, visual: torch.Tensor) -> torch.Tensor:
        """
        :param text: (N, T, d) text token representations
        :param visual: (N, S, d) visual token representations
        :returns outputs:
        """
        N, T, d = text.shape

        ##### TODO:
        ### Layer Norm before attention seems to be the "new" standard?
        ### Assignments do layer norm after attention, but this seems to be an older approach
        ### Decide on either pre-norm or post-norm and we'll stick to one
        norm_text = self.layer_norm(text)
        
        q = self.q_proj(norm_text) # (N, T, d)
        k = self.k_proj(visual) # (N, S, d)
        v = self.v_proj(visual) # (N, S, d)

        # Reshape for multi-head attention
        q = q.view(N, -1, self.n_heads, self.d_head).transpose(1, 2) # (N, n_heads, T, d_head)
        k = k.view(N, -1, self.n_heads, self.d_head).transpose(1, 2) # (N, n_heads, S, d_head)
        v = v.view(N, -1, self.n_heads, self.d_head).transpose(1, 2) # (N, n_heads, S, d_head)

        attn = (q @ k.transpose(-1, -2)) * self.scale # (N, n_heads, T, S)
        attn = self.softmax(attn)
        context = attn @ v # (N, n_heads, T, d_head)
        context = context.transpose(1, 2).reshape(N, T, -1) # (N, T, d)

        output = self.output_proj(context) # (N, T, d)
        output = text + output
        return output
        
class OPTLayerWithCrossAttention(nn.Module):
    """
    Wrapper class to apply Cross Attention to the OPT language model in BLIP-2
    """
    def __init__(self, original_layer, d_model: int, n_heads: int, add_cross_attention: bool = True):
        super().__init__()
        self.original_layer = original_layer
        self.add_cross_attention = add_cross_attention
        
        if add_cross_attention:
            self.cross_attention = CrossAttention(d_model, n_heads)
        
    def forward(self, hidden_states, attention_mask=None, visual_features=None, **kwargs):
        ## Apply original OPT layer
        outputs = self.original_layer(hidden_states, attention_mask=attention_mask, **kwargs)
        hidden_states = outputs[0]
        
        # Apply cross-attention if visual features provided
        if self.add_cross_attention and visual_features is not None:
            hidden_states = self.cross_attention(hidden_states, visual_features)
        
        return (hidden_states,) + outputs[1:]

class BLIP2CrossAttention(nn.Module):
    """
    Modification on BLIP-2 to include Flamingo-style cross-attention blocks

    Frozen components:
    - Image Encoder
    - Q-Former
    - LLM (OPT)

    Trainable Params:
    - Cross-attention between input text, visual and language model (OPT)
    """
    def __init__(self, n_attn_heads: int = 4, attn_every_n: int = 4, lora_r: int = 16):
        super().__init__()
        self.model_blip2 = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16
        )
        self.processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")

        ## get sub-modules from BLIP-2
        self.vision_blip2 = self.model_blip2.vision_model.eval().requires_grad_(False)
        self.qformer_blip2 = self.model_blip2.qformer.eval().requires_grad_(False)
        self.language_blip2 = self.model_blip2.language_model
        self.language_proj_blip2 = self.model_blip2.language_projection.eval().requires_grad_(False)

        ## freeze all original parameters to prevent retraining
        # for param in self.vision_blip2.parameters():
        #     param.requires_grad = False
        # for param in self.qformer_blip2.parameters():
        #     param.requires_grad = False
        for param in self.language_blip2.parameters():
            param.requires_grad = False

        ## get model dimensions
        d_model = self.language_blip2.config.hidden_size

        ## add cross-attention layers to the LLM (OPT)
        self._add_cross_attention_layers(attn_every_n, d_model, n_attn_heads)
        
        ## apply LoRA only to cross-attention layers
        self._apply_lora_to_cross_attention(lora_r)
        
        ## patch the language model to accept visual features
        self._patch_language_model()

    def _add_cross_attention_layers(self, attn_every_n: int, d_model: int, n_heads: int):
        """Add cross-attention to every n-th layer"""
        original_layers = self.language_blip2.model.decoder.layers
        
        for i, layer in enumerate(original_layers):
            should_add_cross_attention = (i + 1) % attn_every_n == 0
            
            original_layers[i] = OPTLayerWithCrossAttention(
                original_layer=layer,
                d_model=d_model,
                n_heads=n_heads,
                add_cross_attention=should_add_cross_attention
            )        

    def _apply_lora_to_cross_attention(self, r: int):
        """Apply LoRA specifically to cross-attention layers"""
        lora_config = LoraConfig(
            r=r,
            lora_alpha=2 * r,
            target_modules=["q_proj", "k_proj", "v_proj", "output_proj"],
            lora_dropout=0.1,
        )
        
        ## apply LoRA only to cross-attention layers
        for layer in self.language_blip2.model.decoder.layers:
            if hasattr(layer, 'cross_attention'):
                layer.cross_attention = get_peft_model(layer.cross_attention, lora_config)

    def _patch_language_model(self):
        """Patch the language model to accept visual_features"""
        
        def custom_forward(self, input_ids=None, attention_mask=None, inputs_embeds=None, 
                         labels=None, visual_features=None, **kwargs):
            ## get embeddings
            if inputs_embeds is None:
                inputs_embeds = self.model.decoder.embed_tokens(input_ids)
            
            ## pass through decoder layers
            hidden_states = inputs_embeds
            
            for layer in self.model.decoder.layers:
                layer_outputs = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    visual_features=visual_features,
                    **kwargs
                )
                hidden_states = layer_outputs[0]
            
            ## apply final layer norm
            if hasattr(self.model.decoder, 'final_layer_norm'):
                hidden_states = self.model.decoder.final_layer_norm(hidden_states)
            
            ## get logits
            logits = self.lm_head(hidden_states)
            
            ## calculate loss if labels provided
            loss = None
            if labels is not None:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            return {
                'loss': loss,
                'logits': logits,
                'hidden_states': hidden_states
            }
        
        ## apply patch
        self.language_blip2.forward = types.MethodType(custom_forward, self.language_blip2)

    @torch.no_grad()
    def _visual_tokens(self, pixel_values: torch.Tensor) -> torch.Tensor:
        ## get visual features from vision model
        vision_outputs = self.vision_blip2(pixel_values=pixel_values)
        vision_features = vision_outputs.last_hidden_state  # (B, num_patches, vision_hidden_size)
        
        ## get Q-former features (768-dimensional)
        qformer_outputs = self.qformer_blip2(
            query_embeds=self.qformer_blip2.query_tokens.expand(pixel_values.shape[0], -1, -1),
            encoder_hidden_states=vision_features  # raw vision features as input
        )
        qformer_features = qformer_outputs.last_hidden_state  # (B, 32, 768)
        
        projected_features = self.language_proj_blip2(qformer_features)  # (B, 32, d_model)
            
        return projected_features
    
    def forward(self, pixel_values, input_ids, attention_mask, labels=None):
        ## prepare visual features   
        visual_features = self._visual_tokens(pixel_values)
        
        ## forward pass through LLM (OPT) with visual features
        return self.language_blip2(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            visual_features=visual_features,
        )
    
    @torch.no_grad()
    def infer(self, image_or_url: str, max_new_tokens=30):
        self.eval()

        # Load image
        if isinstance(image_or_url, str):
            if image_or_url.startswith("http"):
                image = Image.open(requests.get(image_or_url, stream=True).raw).convert("RGB")
            else:
                image = Image.open(image_or_url).convert("RGB")
        else:
            image = image_or_url.convert("RGB")

        # Preprocess image
        inputs = self.processor(images=image, return_tensors="pt").to(torch.float16)
        pixel_values = inputs["pixel_values"]

        # Prepare dummy input text
        text_inputs = self.processor(text=[""], return_tensors="pt")
        input_ids = text_inputs["input_ids"]
        attention_mask = text_inputs["attention_mask"]

        # Greedy decoding loop (basic version)
        visual_features = self._visual_tokens(pixel_values)
        generated = input_ids

        for _ in range(max_new_tokens):
            outputs = self.language_blip2(
                input_ids=generated,
                attention_mask=torch.ones_like(generated),
                visual_features=visual_features,
            )
            next_token = outputs["logits"][:, -1, :].argmax(-1, keepdim=True)
            if next_token.item() == self.processor.tokenizer.eos_token_id:
                break
            generated = torch.cat([generated, next_token], dim=-1)

        caption = self.processor.tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()
        return caption

class TestBlip2(nn.Module):
    """
    BLIP-2 with both Q-former and cross-attention blocks (adapting from Flamingo)
    """
    def __init__(self):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor_blip2 = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.model_blip2 = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16
        ).to(self.device)

        ## get sub-modules from BLIP-2
        self.vision_blip2 = self.model_blip2.vision_model.eval().requires_grad_(False)
        self.qformer_blip2 = self.model_blip2.qformer.eval().requires_grad_(False)
        self.language_blip2 = self.model_blip2.language_model

    def infer(self):
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(url, stream=True).raw)
        inputs = self.processor_blip2(images=image, return_tensors="pt").to(self.device, torch.float16)

        generated_ids = self.model_blip2.generate(**inputs)
        generated_text = self.processor_blip2.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        print(generated_text)

if __name__ == "__main__":
    # model = TestBlip2().infer()
    model = BLIP2CrossAttention()
    caption = model.infer("http://images.cocodataset.org/val2017/000000039769.jpg")
    print(caption)
