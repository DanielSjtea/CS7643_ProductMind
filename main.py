import matplotlib.pyplot as plt
import pandas as pd
import os
import torch
import random
from data_.dataset import AmazonProductDataset, collate_fn
from torch.utils.data import DataLoader, Subset
from src.model import HybridBlip2Flamingo
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import bert_score

def main():
    # Configurations
    json_file = "/data_/data.jsonl"
    base_image_dir = "/data_"
    batch_size = 4
    num_workers = 2
    num_epochs = 6
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = HybridBlip2Flamingo().to(device)
    model = model.float().to(device)

    # Freeze all except specific layers
    for name, param in model.named_parameters():
        if not (
            "cross_attn_blocks" in name or
            "perceiver_resampler" in name or
            "v_proj" in name
        ):
            param.requires_grad = False

    # Unfreeze the final language model head if exists
    if hasattr(model, 'language_blip2'):
        for param in model.language_blip2.lm_head.parameters():
            param.requires_grad = True

    # Data
    dataset = AmazonProductDataset(
        json_file=json_file,
        processor=model.processor,
        base_image_dir=base_image_dir
    )

    # Subset of the full sample
    all_indices = list(range(len(dataset)))
    random.shuffle(all_indices)
    subset_indices = all_indices[:5000]
    small_dataset = Subset(dataset, subset_indices)

    train_loader = DataLoader(
        small_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, model.processor),
        num_workers=num_workers,
        pin_memory=True
    )

    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-7)
    train_losses = []
    learning_rates = []
    bleu_history = []
    rouge_history = []
    bert_history = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()
            images = batch["pixel_values"].to(device)
            labels = batch["input_ids"].to(device)
            # Shift input/labels for decoder
            input_ids = labels[:, :-1]
            label_ids = labels[:, 1:]
            output = model(pixel_values=images, input_ids=input_ids, labels=label_ids)
            loss = output['loss'] if isinstance(output, dict) else output.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_epoch_loss)
        learning_rates.append(optimizer.param_groups[0]['lr'])
        print(f"Epoch {epoch}, Avg Loss: {avg_epoch_loss:.4f}")

        # Evaluate metrics at end of each epoch
        bleu, rouge, bert = evaluate_metrics(model, train_loader, model.processor, device)
        bleu_history.append(bleu)
        rouge_history.append(rouge)
        bert_history.append(bert)
        print(f"[Epoch {epoch}] BLEU: {bleu:.4f}, ROUGE-L: {rouge:.4f}, BERTScore: {bert:.4f}")

def evaluate_metrics(model, dataloader, processor, device):
    model.eval()
    all_predictions, all_references = [], []
    with torch.no_grad():
        for batch in dataloader:
            images = batch["pixel_values"].to(device)
            refs = [processor.tokenizer.decode(ids, skip_special_tokens=True) for ids in batch["input_ids"]]
            try:
                preds = model.generate(pixel_values=images, input_ids=batch["input_ids"].to(device))
            except Exception as e:
                print("Error in generation:", e)
                preds = [""] * len(refs)
            all_predictions.extend([p.strip() for p in preds])
            all_references.extend([r.strip() for r in refs])
    # BLEU
    smooth_fn = SmoothingFunction().method1
    bleu_scores = [
        sentence_bleu([ref.split()], pred.split(), smoothing_function=smooth_fn)
        for ref, pred in zip(all_references, all_predictions)
    ]
    avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0
    print(f"BLEU: {avg_bleu:.4f}")
    # ROUGE-L
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_scores = [scorer.score(ref, pred)['rougeL'].fmeasure for ref, pred in zip(all_references, all_predictions)]
    avg_rouge = sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0.0
    print(f"ROUGE-L: {avg_rouge:.4f}")
    # BERTScore
    P, R, F1 = bert_score.score(all_predictions, all_references, lang="en", verbose=True)
    avg_bertscore = F1.mean().item()
    print(f"BERTScore: {avg_bertscore:.4f}")


    # 1. Plot the Loss Curve
    plt.figure()
    plt.plot(range(num_epochs), train_losses, marker='o', label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.xticks(range(num_epochs))
    plt.legend()
    plt.savefig('loss_curve.png')
    plt.show()

    #2. Plot BERTscore Curve
    plt.figure(figsize=(7,5))
    plt.plot(epochs, bertscore_per_epoch, marker='o', color='red', label='BERTScore')
    plt.xlabel('Epoch')
    plt.ylabel('BERTScore')
    plt.title('BERTScore')
    plt.legend()
    plt.savefig('bertscore_curve.png')
    plt.show()
  
    # 3. Plot Length histograms
    gen_lens = [len(pred.split()) for pred in all_predictions]
    ref_lens = [len(ref.split()) for ref in all_references]
    plt.figure()
    plt.hist(gen_lens, alpha=0.5, label='Generated', bins=20)
    plt.hist(ref_lens, alpha=0.5, label='Reference', bins=20)
    plt.xlabel('Sentence Length')
    plt.ylabel('Count')
    plt.legend()
    plt.title('Length Distribution')
    plt.savefig('length_histogram.png')
    plt.close()

    # 4. Scatter: Reference vs Generated length
    plt.figure()
    plt.scatter(ref_lens, gen_lens)
    plt.xlabel('Reference Length')
    plt.ylabel('Generated Length')
    plt.title('Ref vs Gen Sentence Length')
    plt.savefig('length_scatter.png')
    plt.close()

    return avg_bleu, avg_rouge, avg_bertscore


    # Evaluation Metrics
    print("\n=== Running Evaluation Metrics on Subset ===")
    bleu, rouge, bert = evaluate_metrics(model, train_loader, model.processor, device)
    print(f"Final BLEU: {bleu:.4f}, ROUGE-L: {rouge:.4f}, BERTScore: {bert:.4f}")
    
    print("\n=== Evaluation Metrics Per Epoch ===")
    for epoch in range(num_epochs):
        print(f"Epoch {epoch}: BLEU={bleu_history[epoch]:.4f}, ROUGE-L={rouge_history[epoch]:.4f}, BERTScore={bert_history[epoch]:.4f}")

    print("=== DONE ===")

if __name__ == "__main__":
    main()