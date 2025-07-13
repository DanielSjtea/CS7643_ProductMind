from typing import List
import torch
import torch.nn as nn
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from peft import LoraConfig, TaskType, get_peft_model
from PIL import Image
import requests

class BLIP2CrossAttention(nn.Module):
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
    model = BLIP2CrossAttention().infer()
