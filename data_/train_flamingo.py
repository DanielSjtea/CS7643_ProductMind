import os
import torch
from torch.optim import AdamW
from transformers import Blip2Processor, get_linear_schedule_with_warmup
from data_.dataset import create_dataloaders

def train_hybrid_blip2_flamingo(
    json_file: str,
    base_image_dir: str = "",
    output_dir: str = "model_output",
    batch_size: int = 4,
    learning_rate: float = 5e-5,
    num_epochs: int = 3,
    warmup_steps: int = 100,
    save_steps: int = 500,
):
    """
    Train a HybridBlip2Flamingo model on the Amazon product dataset.
    
    Args:
        json_file: Path to the JSON file with data
        base_image_dir: Base directory for image paths
        output_dir: Directory to save model checkpoints
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        num_epochs: Number of training epochs
        warmup_steps: Number of warmup steps for learning rate scheduler
        save_steps: Save model checkpoint every N steps
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize BLIP2 processor
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    
    # Create dataloaders
    train_dataloader, val_dataloader = create_dataloaders(
        json_file=json_file,
        processor=processor,
        base_image_dir=base_image_dir,
        batch_size=batch_size
    )
    
    # Initialize HybridBlip2Flamingo model
    # Note: This is a placeholder. You would need to implement or import the actual model
    # For example, you might use open_flamingo library or a custom implementation
    try:
        # Try to import from open_flamingo if available
        from open_flamingo import create_model_and_transforms
        
        # Initialize the model (this is an example, adjust parameters as needed)
        model, _ = create_model_and_transforms(
            clip_vision_encoder_path="ViT-L-14",
            clip_vision_encoder_pretrained="openai",
            lang_encoder_path="facebook/opt-2.7b",
            tokenizer_path="facebook/opt-2.7b",
            cross_attn_every_n_layers=4
        )
    except ImportError:
        # Fallback to using BLIP2 model if open_flamingo is not available
        from transformers import Blip2ForConditionalGeneration
        print("Warning: open_flamingo not found, using BLIP2 model instead")
        model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")
        
    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Initialize optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    # Calculate total training steps
    total_steps = len(train_dataloader) * num_epochs
    
    # Initialize learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps, 
        num_training_steps=total_steps
    )
    
    # Training loop
    global_step = 0
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, batch in enumerate(train_dataloader):
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
                attention_mask=attention_mask,
                labels=input_ids  # Using input_ids as labels for generative training
            )
            
            loss = outputs.loss
            epoch_loss += loss.item()
            
            # Backward pass
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            global_step += 1
            
            # Print progress
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}/{len(train_dataloader)} - Loss: {loss.item():.4f}")
            
            # Save checkpoint
            if global_step % save_steps == 0:
                checkpoint_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
                os.makedirs(checkpoint_dir, exist_ok=True)
                model.save_pretrained(checkpoint_dir)
                processor.save_pretrained(checkpoint_dir)
                print(f"Saved checkpoint to {checkpoint_dir}")
        
        # Print epoch stats
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        print(f"Epoch {epoch+1} - Average Loss: {avg_epoch_loss:.4f}")
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_dataloader:
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
                    attention_mask=attention_mask,
                    labels=input_ids
                )
                
                val_loss += outputs.loss.item()
            
            avg_val_loss = val_loss / len(val_dataloader)
            print(f"Validation Loss: {avg_val_loss:.4f}")
    
    # Save final model
    final_dir = os.path.join(output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)
    print(f"Saved final model to {final_dir}")
    
    return model, processor


if __name__ == "__main__":
    # Example usage
    train_hybrid_blip2_flamingo(
        json_file="data.jsonl",
        base_image_dir="",  # Set this to your base image directory if needed
        output_dir="model_output",
        batch_size=4,
        num_epochs=3
    )
