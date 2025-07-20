# HybridBlip2Flamingo Dataset and Training

This repository contains code for creating a PyTorch Dataset for training a HybridBlip2Flamingo model on Amazon product data. The dataset handles images and text data from a JSON file and provides utilities for processing them with the BLIP2 processor.

## Files

- `dataset.py`: Contains the PyTorch Dataset implementation and collate function
- `train_flamingo.py`: Contains the training loop for the HybridBlip2Flamingo model
- `example.py`: Contains example code for using the dataset and training the model

## Dataset

The `AmazonProductDataset` class in `dataset.py` loads data from a JSON file with the following structure:

```json
{
  "item_id": "123456",
  "image_path": "abo-images-small/00/0000529.jpg",
  "title": "Product Title",
  "description": "Product Description",
  "brand": "Brand Name"
}
```

The dataset loads the images from the specified paths and combines the title and description as the text input.

## Collate Function

The `collate_fn` function in `dataset.py` processes a batch of samples using the BLIP2 processor. It handles both the images and text inputs, and returns a dictionary with the processed inputs for the model.

## Training

The `train_hybrid_blip2_flamingo` function in `train_flamingo.py` provides a training loop for the HybridBlip2Flamingo model. It handles:

- Loading the dataset and creating dataloaders
- Initializing the model (either using open_flamingo or falling back to BLIP2)
- Setting up the optimizer and learning rate scheduler
- Training and validation loops
- Saving checkpoints and the final model

## Example Usage

The `example.py` file demonstrates how to use the dataset and training code. It includes examples for:

1. Loading and exploring the dataset
2. Training the model
3. Using the model for inference

## Requirements

- PyTorch
- transformers (for BLIP2 processor)
- PIL (for image processing)
- open_flamingo (optional, for HybridBlip2Flamingo model)

## Getting Started

1. Install the required packages:
   ```
   pip install torch transformers pillow
   ```

2. (Optional) Install open_flamingo for the HybridBlip2Flamingo model:
   ```
   pip install open_flamingo
   ```

3. Prepare your data in the required JSON format.

4. Run the example script:
   ```
   python example.py
   ```

## Customization

You can customize the dataset and training process by modifying the parameters in the respective functions:

- `AmazonProductDataset.__init__`: Customize the dataset loading process
- `create_dataloaders`: Customize the dataloader creation
- `train_hybrid_blip2_flamingo`: Customize the training process

## Notes

- The code assumes that the images are accessible at the paths specified in the JSON file. If the paths are relative, you can provide a `base_image_dir` parameter to the dataset.
- The training code includes a fallback to using the BLIP2 model if open_flamingo is not available.
- For actual training, you will need a GPU with sufficient memory to handle the model and batch size.
