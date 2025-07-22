#!/usr/bin/env python
# coding: utf-8

# In[23]:


#!pip install rouge-score
#!pip install bert-score

import os
import torch
from transformers import Blip2Processor
from dataset import AmazonProductDataset, collate_fn
from train_flamingo import train_hybrid_blip2_flamingo
import matplotlib.pyplot as plt

from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
import bert_score

def main():
    # === CONFIGURATION ===
    json_file = "data.jsonl"  # Update this to your actual path
    base_image_dir = ""  # Base directory for image paths if needed
    output_dir = "model_output"
    batch_size = 4
    num_workers = 2
    num_epochs = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not os.path.exists(json_file):
        raise FileNotFoundError(f"{json_file} not found. Please update the path.")
    
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    
    dataset = AmazonProductDataset(
        json_file=json_file,
        processor=processor,
        base_image_dir=base_image_dir
    )
    
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, processor),
        num_workers=num_workers,
        pin_memory=True
    )
    
    # --- Train model
    model, processor = train_hybrid_blip2_flamingo(
        json_file=json_file,
        base_image_dir=base_image_dir,
        output_dir=output_dir,
        batch_size=batch_size,
        num_epochs=num_epochs
    )

# --- Evaluation metrics
print("\nEvaluating BLEU, ROUGE, BERTScore...")


def evaluate_metrics(model, dataloader, processor, device):
    model.eval()
    all_predictions, all_references = [], []
    with torch.no_grad():
        for batch in dataloader:
            refs = batch['caption'] if 'caption' in batch else batch['text_input']
            pred = model.generate(...) 
            preds = [processor.decode(p) for p in pred]
            all_predictions.extend(preds)
            all_references.extend(refs)
            
    # BLEU
    bleu_scores = [sentence_bleu([ref.split()], ref.split()) for ref in all_references]
    avg_bleu = sum(bleu_scores) / len(bleu_scores)
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_scores = [scorer.score(ref, ref)['rougeL'].fmeasure for ref in all_references]
    avg_rouge = sum(rouge_scores) / len(rouge_scores)
    
    # BERTScore
    P, R, F1 = bert_score.score(all_references, all_references, lang="en", verbose=True)
    avg_bertscore = F1.mean().item()
    print(f"BLEU: {avg_bleu:.4f}, ROUGE-L: {avg_rouge:.4f}, BERTScore: {avg_bertscore:.4f}")

# --- Plotting Loss Curves

def plot_loss_curves(train_losses, val_losses=None):
    plt.plot(train_losses, label='Train Loss')
    if val_losses:
        plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training/Validation Loss')
    plt.savefig('loss_curves.png')
    plt.show()

if __name__ == "__main__":
    main()


# In[ ]:




