import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ContrastiveModel import ContrastiveModel
from DataLoader import load_data
from ReviewDataset import ReviewDataset, custom_collate_fn
from test_prediction import predict
from training_loop import train_model

if __name__ == "__main__":

    # trained_model = ContrastiveModel(13).to("cuda")
    #
    # sim, user_embedding, review_embedding = trained_model(user_features="gil", review_content="rev")
    # print(sim)
    # print(user_embedding)
    # print(review_embedding)
    #
    # exit(0)

    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Number of CUDA Devices: {torch.cuda.device_count()}")
        print(f"Current CUDA Device: {torch.cuda.current_device()}")
        print(f"Device Name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        print("CUDA is not available.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    (
        train_users,
        train_reviews,
        train_matches,
        val_users,
        val_reviews,
        val_matches,
        test_users,
        test_reviews
    ) = load_data(train_path="./train", val_path="./val", test_path="./test", frac_of_train_set=0.1)

    # Set user_feature_size to the number of features (excluding user_id and accommodation_id)
    user_feature_size = 13

    only_inference = True

    if not only_inference:

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
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        # Train model
        epochs = 1
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            epoch_loss, trained_model = train_model(train_loader, model, criterion, optimizer, device)
            print(f"  --> Epoch {epoch + 1} Loss: {epoch_loss:.4f}\n")
    else:
        trained_model = ContrastiveModel(user_feature_size).to(device)

    # Predict on the test set
    result_df = predict(trained_model, test_users, test_reviews, batch_size=512)
    result_df.to_csv("submission.csv", index=False)
    print("Saved predictions to submission.csv")
