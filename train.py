import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel


# Load only 10% of the train data
def load_data(train_path, val_path, test_path):
    train_users = pd.read_csv(f"{train_path}/train_users.csv")
    train_reviews = pd.read_csv(f"{train_path}/train_reviews.csv")
    train_matches = pd.read_csv(f"{train_path}/train_matches.csv")

    # Sample 10% of training data
    frac = 0.1
    train_users = train_users.sample(frac=frac, random_state=42).reset_index(drop=True)
    train_reviews = train_reviews.sample(frac=frac, random_state=42).reset_index(drop=True)
    train_matches = train_matches.sample(frac=frac, random_state=42).reset_index(drop=True)
    print(f"Loaded {frac} of the training data")

    val_users = pd.read_csv(f"{val_path}/val_users.csv")
    val_reviews = pd.read_csv(f"{val_path}/val_reviews.csv")
    val_matches = pd.read_csv(f"{val_path}/val_matches.csv")
    print("Loaded validation data")

    test_users = pd.read_csv(f"{test_path}/test_users.csv")
    test_reviews = pd.read_csv(f"{test_path}/test_reviews.csv")
    print("Loaded test data")

    return (train_users, train_reviews, train_matches,
            val_users, val_reviews, val_matches,
            test_users, test_reviews)


class ReviewDataset(Dataset):
    def __init__(self, users, reviews, matches):
        # Keep references
        self.users = users
        self.reviews = reviews

        # Only keep matches that exist in both users & reviews
        valid_user_ids = set(self.users['user_id'])
        valid_accommodation_ids = set(self.users['accommodation_id'])
        valid_review_ids = set(self.reviews['review_id'])

        self.matches = matches[
            matches['user_id'].isin(valid_user_ids)
            & matches['accommodation_id'].isin(valid_accommodation_ids)
            & matches['review_id'].isin(valid_review_ids)
        ].reset_index(drop=True)

    def __len__(self):
        return len(self.matches)

    def __getitem__(self, idx):
        match = self.matches.iloc[idx]
        user_id = match['user_id']
        accommodation_id = match['accommodation_id']
        review_id = match['review_id']

        user_features = self.users[
            (self.users['user_id'] == user_id) &
            (self.users['accommodation_id'] == accommodation_id)
        ].iloc[0]

        review_content = self.reviews[
            self.reviews['review_id'] == review_id
        ].iloc[0]

        return {
            'user_features': user_features.to_dict(),
            'review_content': f"{review_content['review_positive']} {review_content['review_negative']}"
        }


class ContrastiveModel(nn.Module):
    def __init__(self, user_feature_size):
        super(ContrastiveModel, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("model")
        self.bert = AutoModel.from_pretrained("model")

        # Use the actual hidden size from the model config rather than hardcoding
        hidden_size = self.bert.config.hidden_size

        self.fc_user = nn.Linear(user_feature_size, 256)
        self.fc_review = nn.Linear(hidden_size, 256)

        # Cosine similarity for user and review embeddings
        self.cos = nn.CosineSimilarity(dim=1)

    def forward(self, user_features, review_content):
        # user_features: (batch_size, user_feature_size)
        # review_content: list of strings (length batch_size)

        user_embedding = torch.relu(self.fc_user(user_features))  # -> (batch_size, 256)

        # Tokenize reviews
        review_tokens = self.tokenizer(
            review_content,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        ).to(user_features.device)

        # If the batch sizes don't match for some reason, expand (rare corner case).
        # Typically, you won't need this if your collate_fn is correct.
        if review_tokens['input_ids'].shape[0] != user_features.shape[0]:
            review_tokens = {
                key: val.expand(user_features.shape[0], -1) for key, val in review_tokens.items()
            }

        # Obtain the [CLS] token representation
        review_output = self.bert(**review_tokens).last_hidden_state[:, 0, :]
        review_embedding = torch.relu(self.fc_review(review_output))  # -> (batch_size, 256)

        # Cosine similarity in [-1, +1]
        similarity = self.cos(user_embedding, review_embedding)

        # Scale similarity to a more logit-friendly range for BCEWithLogitsLoss
        logits = similarity * 5.0  # transforms [-1,1] to roughly [-5,5]
        return logits


def custom_collate_fn(batch):
    user_features = []
    review_content = []

    for item in batch:
        features = []
        for value in item['user_features'].values():
            # Convert everything to float if possible, else zero
            try:
                features.append(float(value))
            except ValueError:
                features.append(0)
        user_features.append(features)
        review_content.append(item['review_content'])

    user_features_tensor = torch.tensor(user_features, dtype=torch.float32)
    return {
        'user_features': user_features_tensor,
        'review_content': review_content
    }


def train_model(train_loader, model, criterion, optimizer, device):
    """
    Train the model for one epoch. Logs loss after each batch.
    """
    model.train()
    total_loss = 0.0

    for batch_idx, batch in enumerate(train_loader):
        user_features = batch['user_features'].to(device)
        review_content = batch['review_content']

        optimizer.zero_grad()
        logits = model(user_features, review_content)

        # For demonstration, all pairs are "similar" => label = 1.0
        targets = torch.ones(len(logits), device=device)

        loss = criterion(logits, targets)
        loss.backward()

        # (Optional) Gradient clipping
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        total_loss += loss.item()

        # Log loss after each batch
        current_loss = total_loss / (batch_idx + 1)
        print(f"  [Batch {batch_idx + 1}/{len(train_loader)}] Loss: {current_loss:.4f}")

    epoch_loss = total_loss / len(train_loader)
    return epoch_loss


@torch.no_grad()
def predict(model, test_users, test_reviews, top_k=10):
    """
    Predict top-k most similar reviews for each user in test_users.
    """
    model.eval()
    results = []

    device = next(model.parameters()).device

    for _, user_row in test_users.iterrows():
        # Exclude columns 'user_id' & 'accommodation_id'
        # Assuming user_row is [user_id, accommodation_id, f1, f2, ..., fN]
        user_features_vals = [
            float(value) if isinstance(value, (int, float)) else 0
            for value in user_row.values[2:]
        ]

        user_features = torch.tensor(
            user_features_vals,
            dtype=torch.float32
        ).unsqueeze(0).to(device)

        similarities = []
        for _, review_row in test_reviews.iterrows():
            review_content = review_row['review_positive'] + ' ' + review_row['review_negative']
            logits = model(user_features, [review_content])
            prob = torch.sigmoid(logits).item()  # Convert logits to probability
            similarities.append((review_row['review_id'], prob))

        # Sort by similarity descending
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_reviews = [review_id for (review_id, _) in similarities[:top_k]]
        results.append((user_row['accommodation_id'], user_row['user_id'], *top_reviews))

    result_df = pd.DataFrame(
        results,
        columns=["accommodation_id", "user_id"] + [f"review_{i}" for i in range(1, top_k + 1)]
    )
    # Add an 'ID' column if it doesn't exist
    if 'ID' not in result_df.columns:
        result_df.insert(0, 'ID', range(1, len(result_df) + 1))

    return result_df


if __name__ == "__main__":
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
    ) = load_data("./train", "./val", "./test")

    # Set user_feature_size to the number of features (excluding user_id and accommodation_id)
    user_feature_size = 13

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
        epoch_loss = train_model(train_loader, model, criterion, optimizer, device)
        print(f"  --> Epoch {epoch + 1} Loss: {epoch_loss:.4f}\n")

    # Predict on the test set
    result_df = predict(model, test_users, test_reviews)
    result_df.to_csv("submission.csv", index=False)
    print("Saved predictions to submission.csv")
