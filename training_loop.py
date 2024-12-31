import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


def train_model(train_loader, model, criterion, optimizer, device):
    """
    Train the model for one epoch. Logs loss after each batch.
    """
    model.train()
    total_loss = 0.0

    class ContrastiveLoss(torch.nn.Module):
        def __init__(self, margin):
            super(ContrastiveLoss, self).__init__()
            self.margin = margin

        def forward(self, outputs, labels):
            distances = 1.0 - outputs  # Assuming outputs are similarity scores
            losses = labels * distances ** 2 + (1 - labels) * F.relu(self.margin - distances) ** 2
            return losses.mean()

    criterion = ContrastiveLoss(margin=0.1)

    for batch_idx, batch in enumerate(train_loader):
        user_features = batch['user_features'].to(device)
        review_content = batch['review_content']

        # Forward pass for positive pairs
        scaled_similarity, user_embedding, review_embedding = model(user_features, review_content)

        # Generate negative samples by shuffling review_content
        shuffle_indices = torch.randperm(len(review_content))

        negative_review_content = [review_content[i] for i in shuffle_indices.tolist()]

        # Forward pass for negative pairs
        negative_scaled_similarity, _, _ = model(user_features, negative_review_content)

        # Create labels: 1 for positive pairs, 0 for negative pairs
        positive_targets = torch.ones(len(scaled_similarity), device=device)
        negative_targets = torch.zeros(len(negative_scaled_similarity), device=device)

        # Combine positive and negative pairs
        combined_similarity = torch.cat([scaled_similarity, negative_scaled_similarity], dim=0)
        combined_targets = torch.cat([positive_targets, negative_targets], dim=0)

        # Compute contrastive loss
        loss = criterion(combined_similarity, combined_targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Log loss
        total_loss += loss.item()
        current_loss = total_loss / (batch_idx + 1)
        print(f"  [Batch {batch_idx + 1}/{len(train_loader)}] Loss: {current_loss:.4f}")
    # for batch_idx, batch in enumerate(train_loader):
    #     user_features = batch['user_features'].to(device)
    #     review_content = batch['review_content']
    #
    #     optimizer.zero_grad()
    #     scaled_similarity, user_embedding, review_embedding = model(user_features, review_content)
    #
    #     # For demonstration, all pairs are "similar" => label = 1.0
    #     targets = torch.ones(len(scaled_similarity), device=device)
    #
    #     loss = criterion(scaled_similarity, targets)
    #     loss.backward()
    #
    #     optimizer.step()
    #     total_loss += loss.item()
    #
    #     # Log loss after each batch
    #     current_loss = total_loss / (batch_idx + 1)
    #     print(f"  [Batch {batch_idx + 1}/{len(train_loader)}] Loss: {current_loss:.4f}")

    epoch_loss = total_loss / len(train_loader)
    return epoch_loss, model
