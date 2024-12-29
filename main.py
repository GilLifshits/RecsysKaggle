import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os

from ContrastiveModel import ContrastiveModel
from DataLoader import load_data
from ReviewDataset import ReviewDataset, custom_collate_fn
from test_prediction import predict
from training_loop import train_model

def initialize_device():
    """Initialize the device for computation (CUDA or CPU)."""
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Number of CUDA Devices: {torch.cuda.device_count()}")
        print(f"Current CUDA Device: {torch.cuda.current_device()}")
        print(f"Device Name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        print("CUDA is not available.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device

def prepare_data():
    """Load and prepare data for training, validation, and testing."""
    return load_data(
        train_path="./train",
        val_path="./val",
        test_path="./test",
        frac_of_train_set=0.1
    )

def save_model(model, path="trained_model.pth"):
    """Save the trained model to a local file."""
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(model_class, user_feature_size, path="trained_model.pth", device="cpu"):
    """Load a model from a local file."""
    model = model_class(user_feature_size).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    print(f"Model loaded from {path}")
    return model

def train_contrastive_model(device, train_users, train_reviews, train_matches, user_feature_size):
    """Train the contrastive model."""
    # Prepare dataset and dataloader
    train_dataset = ReviewDataset(train_users, train_reviews, train_matches)
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=custom_collate_fn,
        pin_memory=(device.type == 'cuda')
    )

    # Initialize model, criterion, optimizer
    model = ContrastiveModel(user_feature_size).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)

    # Train model
    epochs = 10
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        epoch_loss, trained_model = train_model(train_loader, model, criterion, optimizer, device)
        print(f"  --> Epoch {epoch + 1} Loss: {epoch_loss:.4f}\n")

    save_model(trained_model)
    return trained_model

def run_inference(trained_model, test_users, test_reviews):
    """Run inference on the test set and save predictions to a CSV file."""
    result_df = predict(trained_model, test_users, test_reviews, batch_size=64)
    result_df.to_csv("submission.csv", index=False)
    print("Saved predictions to submission.csv")

def main():
    # Initialize device
    device = initialize_device()

    # Load data
    train_users, train_reviews, train_matches, val_users, val_reviews, val_matches, test_users, test_reviews = prepare_data()

    # Set user feature size
    user_feature_size = 13

    # Choose mode (training or inference)
    only_inference = True

    if not only_inference:
        # Train the model
        trained_model = train_contrastive_model(device, train_users, train_reviews, train_matches, user_feature_size)
    else:
        # Load pre-trained model for inference
        model_path = "trained_model.pth"
        if os.path.exists(model_path):
            trained_model = load_model(ContrastiveModel, user_feature_size, model_path, device)
            print("Loaded model")
        else:
            print(f"Model file {model_path} not found. Please train the model first.")
            return

    # Predict on the test set
    run_inference(trained_model, test_users, test_reviews)

if __name__ == "__main__":
    main()
