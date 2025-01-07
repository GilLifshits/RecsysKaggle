import os
import warnings

import torch
from transformers import AutoTokenizer

warnings.filterwarnings("ignore", category=UserWarning)
from ContrastiveModel import ContrastiveTransformerModel
from DataLoader import load_data
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


def prepare_data():
    """Load and prepare data for training, validation, and testing."""
    return load_data(
        train_path="./train",
        val_path="./val",
        test_path="./test"
    )


def save_model(model, path="trained_model.pth"):
    """Save the trained model to a local file."""
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def load_model(model_class, model_path="trained_model.pth", device="cpu"):
    """Load a model from a local file."""
    embed_size = 512
    output_embed_size = 128
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    model = model_class(tokenizer.vocab_size, embed_size, output_embed_size).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Model loaded from {model_path}")
    return model


def train_contrastive_model(batch_size, device, train_users):
    """Train the models."""

    all_accommodation_ids = train_users["accommodation_id"].unique().tolist()

    # Train models
    users_model, reviews_model = train_model(
        batch_size=batch_size,
        device=device,
        all_accommodation_ids=all_accommodation_ids,
        epochs=5
    )

    # Save models
    save_model(model=users_model, path="users_model.pth")
    save_model(model=reviews_model, path="reviews_model.pth")
    return users_model, reviews_model


def run_inference(users_trained_model, reviews_trained_model, test_users, test_reviews):
    """Run inference on the test set and save predictions to a CSV file."""
    result_df = predict(users_trained_model, reviews_trained_model, test_users, test_reviews, batch_size=64)
    result_df.to_csv("submission.csv", index=False)
    print("Saved predictions to submission.csv")


def main():

    batch_size = 64
    only_inference = True

    device = initialize_device()

    # Load data
    train_users, train_reviews, train_matches, val_users, val_reviews, val_matches, test_users, test_reviews = prepare_data()

    if not only_inference:
        # Train the model
        users_trained_model, reviews_trained_model = train_contrastive_model(batch_size=batch_size, device=device,
                                                train_users=train_users)
    else:
        # Load trained model
        users_model_path = "users_model.pth"
        reviews_model_path = "reviews_model.pth"
        if os.path.exists(users_model_path) and os.path.exists(reviews_model_path):
            users_trained_model = load_model(model_class=ContrastiveTransformerModel, model_path=users_model_path,
                                             device=device)
            reviews_trained_model = load_model(model_class=ContrastiveTransformerModel, model_path=reviews_model_path,
                                               device=device)
            print("Loaded model")
        else:
            print(f"Model file {users_model_path} or {reviews_model_path} not found. Please train the model first.")
            return

    # Predict on the test set
    run_inference(users_trained_model, reviews_trained_model, test_users, test_reviews)


if __name__ == "__main__":

    # Sapir Anidgar: 322229410
    # Gil lifshits: 322597717
    # Group name: Gil & Sapir
    main()
