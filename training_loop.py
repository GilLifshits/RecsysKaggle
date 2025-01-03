import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from ReviewDataset import custom_collate_fn, ReviewDataset


def train_model(batch_size, model, optimizer, scheduler, device, all_accommodation_ids, accommodation_amount):
    model.train()

    # Define train loder for each accommodation data:
    for accommodation_id in tqdm(all_accommodation_ids[:accommodation_amount], desc="Processing accommodations"):

        # get all data for specific accommodation_id
        train_users = pd.read_csv(f"in_accommodation_datasets/train_users_{accommodation_id}.csv")
        train_reviews = pd.read_csv(f"in_accommodation_datasets/train_review_{accommodation_id}.csv")
        train_matches = pd.read_csv(f"in_accommodation_datasets/train_matches_{accommodation_id}.csv")

        train_dataset = ReviewDataset(train_users, train_reviews, train_matches)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=custom_collate_fn,
            pin_memory=(device == 'cuda')
        )

        total_accommodation_loss = 0.0

        # train per accommodation id
        for batch_idx, batch in enumerate(train_loader):
            user_features = batch['user_features']
            review_content = batch['review_content']

            # Forward pass for positive pairs
            scaled_similarity, user_embedding, review_embedding = model(user_features, review_content)

            # Generate negative samples by shuffling review_content
            shuffle_indices = torch.randperm(len(review_content))
            negative_review_content = [review_content[i] for i in shuffle_indices.tolist()]

            # Forward pass for negative pairs
            negative_scaled_similarity, _, _ = model(user_features, negative_review_content)

            # Combine positive and negative similarities into logits
            logits = torch.cat([
                scaled_similarity.unsqueeze(1),  # Positive similarity
                negative_scaled_similarity.unsqueeze(1)  # Negative similarity
            ], dim=1)  # Shape: [batch_size, 2]

            # Generate labels: index 0 corresponds to positive samples
            targets = torch.zeros(len(scaled_similarity), dtype=torch.long, device=device)

            # Compute InfoNCE loss
            loss = F.cross_entropy(logits, targets)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update scheduler
            if scheduler:
                scheduler.step()

            total_accommodation_loss += loss.item()
            batch_loss = total_accommodation_loss / (batch_idx + 1)
            # print(
            #     f"  [Accommodation {accommodation_id}] [Batch {batch_idx + 1}/{len(train_loader)}] Loss: {batch_loss:.4f}")

    return model
