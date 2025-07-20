import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import transformers
from typing import Dict, List, Optional, Tuple, Union

class AmazonProductDataset(Dataset):
    """
    Dataset for loading Amazon product data with images and text for BLIP2/Flamingo models.
    """
    def __init__(
        self,
        json_file: str,
        processor: transformers.Blip2Processor,
        base_image_dir: str = "",
        max_length: int = 77,
    ):
        """
        Initialize the dataset.
        
        Args:
            json_file: Path to the JSON file containing the data
            processor: BLIP2 processor for processing images and text
            base_image_dir: Base directory for image paths (if image paths in JSON are relative)
            max_length: Maximum length for text tokenization
        """
        self.base_image_dir = base_image_dir
        self.processor = processor
        self.max_length = max_length
        
        # Load data from JSON file
        self.data = []
        with open(json_file, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                self.data.append(item)
        
        print(f"Loaded {len(self.data)} samples from {json_file}")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a sample from the dataset.
        
        Returns a dictionary with:
            - image: PIL Image
            - text_input: Text prompt (title + description)
            - item_id: Item ID
            - brand: Brand name (can be used as label)
        """
        item = self.data[idx]
        
        # Get image path and load image
        image_path = item["image_path"]
        if self.base_image_dir:
            image_path = os.path.join(self.base_image_dir, image_path)
        
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a blank image as fallback
            image = Image.new("RGB", (224, 224), color="white")
        
        # Combine title and description as the text prompt
        title = item.get("title", "")
        description = item.get("description", "")
        text_input = f"{title}. {description}"
        
        # Get label (using brand as label)
        label = item.get("brand", "")
        
        return {
            "image": image,
            "text_input": text_input,
            "item_id": item.get("item_id", ""),
            "brand": label
        }


def collate_fn(batch: List[Dict], processor: transformers.Blip2Processor) -> Dict[str, torch.Tensor]:
    """
    Collate function for DataLoader that processes a batch of samples using BLIP2Processor.
    
    Args:
        batch: List of samples from the dataset
        processor: BLIP2 processor for processing images and text
    
    Returns:
        Dictionary with processed inputs for the model
    """
    images = [item["image"] for item in batch]
    text_inputs = [item["text_input"] for item in batch]
    
    # Process images and text using the BLIP2 processor
    inputs = processor(
        images=images,
        text=text_inputs,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    
    # Add additional information to the batch
    inputs["item_ids"] = [item["item_id"] for item in batch]
    inputs["brands"] = [item["brand"] for item in batch]
    
    return inputs


# Example usage
def create_dataloaders(
    json_file: str,
    processor: transformers.Blip2Processor,
    base_image_dir: str = "",
    batch_size: int = 8,
    num_workers: int = 4,
    shuffle: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.
    
    Args:
        json_file: Path to the JSON file
        processor: BLIP2 processor
        base_image_dir: Base directory for image paths
        batch_size: Batch size
        num_workers: Number of workers for DataLoader
        shuffle: Whether to shuffle the data
    
    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    # Create dataset
    dataset = AmazonProductDataset(
        json_file=json_file,
        processor=processor,
        base_image_dir=base_image_dir
    )
    
    # Split into train and validation sets (90/10 split)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda batch: collate_fn(batch, processor)
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda batch: collate_fn(batch, processor)
    )
    
    return train_dataloader, val_dataloader


# Example of how to use the dataset with HybridBlip2Flamingo
def example_usage():
    from transformers import Blip2Processor, Blip2ForConditionalGeneration
    
    # Initialize processor and model
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")
    
    # Create dataloaders
    train_dataloader, val_dataloader = create_dataloaders(
        json_file="data.jsonl",
        processor=processor,
        base_image_dir="",  # Set this to your base image directory if needed
        batch_size=4
    )
    
    # Example training loop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    for batch in train_dataloader:
        # Move inputs to device
        input_ids = batch.pop("input_ids").to(device)
        pixel_values = batch.pop("pixel_values").to(device)
        attention_mask = batch.pop("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        
        # Forward pass
        outputs = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask
        )
        
        # Process outputs
        logits = outputs.logits
        
        # Here you would typically compute loss and perform backpropagation
        # ...

if __name__ == "__main__":
    # This is just to demonstrate usage
    # example_usage()
    pass
