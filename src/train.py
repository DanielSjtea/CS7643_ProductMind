import os
import json
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
import matplotlib.pyplot as plt
from transformers import Blip2Processor, get_scheduler
from model import HybridBlip2Flamingo


def plot_loss_curve(train_losses, output_dir):
    plt.figure()
    plt.plot(range(1, len(train_losses) + 1), train_losses, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Loss Curve")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "loss_curve.png"))
    plt.close()


def train(
    dataset_class, # A class, not an instance
    dataset_args: dict, # Arguments to initialize the dataset
    output_dir: str,
    collate_fn=None,
    num_epochs: int = 5,
    batch_size: int = 4,
    learning_rate: float = 2e-5,
    val_split: float = 0.1,
    eval_fn=None, # Optional: function(model, val_loader, processor, device)
    scheduler_type: str = "linear",
    warmup_steps: int = 0,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    ## --- Initialize Model & Processor ---
    ## TODO: Model Parameters have to be adjusted here
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = HybridBlip2Flamingo(attn_layers_at=(10, 20, 30)).to(device)

    ## --- Dataset ---
    full_dataset = dataset_class(**dataset_args)
    val_size = int(val_split * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    ## --- Optimizer & Scheduler ---
    ## TODO: Can change optimizer here
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    scheduler = get_scheduler(
        name=scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_epochs * len(train_loader)
    )

    ## --- Training ---
    model.train()
    train_losses = []
    metrics_by_epoch = []

    for epoch in range(num_epochs):
        total_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for batch in progress_bar:
            images = batch["images"].to(device)
            prompts = batch["prompts"]
            labels = batch["labels"].to(device)

            outputs = model(images, prompts, labels)
            loss = outputs["loss"]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"\nEpoch {epoch+1} complete. Avg Loss: {avg_loss:.4f}")

        ## --- Validation ---
        if eval_fn:
            metrics = eval_fn(model, val_loader, processor, device)
            metrics_by_epoch.append(metrics)
            print(f"Validation Metrics @ Epoch {epoch+1}:")
            for k, v in metrics.items():
                print(f"  {k}: {v:.4f}")

        ## --- Checkpoint ---
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch+1}.pt")
        torch.save(model.state_dict(), model_path)

    ## --- Save Loss Logs ---
    plot_loss_curve(train_losses, output_dir)
    if eval_fn:
        with open(os.path.join(output_dir, "metrics_by_epoch.json"), "w") as f:
            json.dump(metrics_by_epoch, f, indent=2)

    print("Training complete")

#######################################################################
############ TEMPLATES FOR DATASET, COLLATE_FN, EVALUATION (not done) ############
#### Delete when other parts are complete
## TODO: IMPORTANT TO DELETE

import os
import json
from PIL import Image
from torch.utils.data import Dataset

class TempDataset(Dataset):
    """
    A simple dataset for training vision-language models like HybridBlip2Flamingo.

    Expected structure of the JSON file:
    [
        {
            "image": "img123.jpg",
            "prompt": "A photo of a",
            "label": "cat"
        },
        ...
    ]
    """

    def __init__(self, data_dir, json_path, processor=None):
        self.data_dir = data_dir
        self.processor = processor  # Optional, used in collate_fn
        with open(json_path, "r") as f:
            self.entries = json.load(f)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        image_path = os.path.join(self.data_dir, entry["image"])
        image = Image.open(image_path).convert("RGB")

        return {
            "image": image,
            "prompt": entry["prompt"],
            "label": entry["label"]
        }

def temp_collate_fn(batch, processor):
    """
    Collate function for batching image-text-label examples.

    Returns:
        {
            "images": Tensor of shape (B, 3, H, W),
            "prompts": list of strings,
            "labels": Tensor of token IDs (B, L)
        }
    """
    images = [item["image"] for item in batch]
    prompts = [item["prompt"] for item in batch]
    labels = [item["label"] for item in batch]

    # Image preprocessing
    pixel_values = processor(images=images, return_tensors="pt").pixel_values

    # Label tokenization
    label_ids = processor(text=labels, padding=True, return_tensors="pt").input_ids

    return {
        "images": pixel_values,
        "prompts": prompts,
        "labels": label_ids
    }

if __name__ == "__main__":
    train(
        dataset_class=TempDataset,
        dataset_args={"data_dir": "images/", "json_path": "meta.json"},
        output_dir="./checkpoints",
        collate_fn=temp_collate_fn,
        # eval_fn=evaluate_model,
        num_epochs=5,
        batch_size=4,
        learning_rate=1e-5
    )