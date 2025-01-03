import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os

from ContrastiveModel import ContrastiveSentenceTransformerModel
from DataLoader import load_data
from ReviewDataset import ReviewDataset, custom_collate_fn
from test_prediction import predict
from training_loop import train_model

def initialize_device() -> str:
    """Initialize the device for computation (CUDA or CPU)."""
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Number of CUDA Devices: {torch.cuda.device_count()}")
        print(f"Current CUDA Device: {torch.cuda.current_device()}")
        print(f"Device Name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        print("CUDA is not available.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    return device

def prepare_data(frac_of_train_set):
    """Load and prepare data for training, validation, and testing."""
    return load_data(
        train_path="./train",
        val_path="./val",
        test_path="./test",
        frac_of_train_set=frac_of_train_set
    )

def save_model(model, path="trained_model.pth"):
    """Save the trained model to a local file."""
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(model_class, model_path="trained_model.pth", device="cpu"):
    """Load a model from a local file."""
    model = model_class("sentence_transformer_model/").to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Model loaded from {model_path}")
    return model

def train_contrastive_model(batch_size, device, train_users, train_reviews, train_matches, accommodation_amount):
    """Train the contrastive model."""

    # Initialize model, criterion, optimizer
    model_path = "sentence_transformer_model/"
    model = ContrastiveSentenceTransformerModel(model_path).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.01)

    # Learning rate scheduler for warm-up
    total_steps = (len(train_users)/batch_size) * 7  # Total steps for 7 epochs
    warmup_steps = int(total_steps * 0.05)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=warmup_steps
    )

    all_accommodation_ids = train_users["accommodation_id"].unique().tolist()

    # Train model
    epochs = 7
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        trained_model = train_model(
            batch_size=batch_size,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            all_accommodation_ids=all_accommodation_ids,
            accommodation_amount=accommodation_amount
        )
        print(f"  --> Epoch {epoch + 1} \n")

    save_model(trained_model)
    return trained_model

def run_inference(trained_model, test_users, test_reviews):
    """Run inference on the test set and save predictions to a CSV file."""
    result_df = predict(trained_model, test_users, test_reviews, batch_size=64)
    result_df.to_csv("submission.csv", index=False)
    print("Saved predictions to submission.csv")

def main():

    # TODO: more training data -> 0.2 (they used 100 accommodation ids)
    # TODO: try adding another FC layer
    # TODO: try FT on embedders

    frac_of_train_set = 0.1
    batch_size = 8
    accommodation_amount = 1000
    only_inference = False

    # Initialize device
    device = initialize_device()

    # Load data
    train_users, train_reviews, train_matches, val_users, val_reviews, val_matches, test_users, test_reviews = prepare_data(frac_of_train_set)

    if not only_inference:
        # Train the model
        trained_model = train_contrastive_model(batch_size=batch_size, device=device,
                                                train_users=train_users, train_reviews=train_reviews, train_matches=train_matches,
                                                accommodation_amount=accommodation_amount)
    else:
        # Load pre-trained model for inference
        model_path = "trained_model.pth"
        if os.path.exists(model_path):
            trained_model = load_model(model_class=ContrastiveSentenceTransformerModel, model_path=model_path, device=device)
            print("Loaded model")
        else:
            print(f"Model file {model_path} not found. Please train the model first.")
            return

    # Predict on the test set
    # The speed of inference is very correlated to type of GPU allocated from BGU cluster
    run_inference(trained_model, test_users, test_reviews)

if __name__ == "__main__":
    main()
