import os
import torch
from data_.dataset import AmazonProductDataset, collate_fn
from src.model import HybridBlip2Flamingo

def main():
    # Configurations
    json_file = "data_/data.jsonl"      
    batch_size = 4
    num_workers = 2
    num_epochs = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Load model
    model = HybridBlip2Flamingo().to(device)

    # Data
    dataset = AmazonProductDataset(
        json_file=json_file,
        processor=model.tokenizer,   # use model's tokenizer if needed
        base_image_dir=""
    )
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, model.tokenizer),
        num_workers=num_workers,
        pin_memory=True
    )

    # Loss/optimizer
    params_to_train = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params_to_train, lr=1e-4)

    for epoch in range(num_epochs):
        model.train()
        for batch in train_loader:
            images = batch["image"].to(device)
            labels = batch["input_ids"].to(device)
            output = model(images=images, labels=labels)
            loss = output.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

if __name__ == "__main__":
    main()




