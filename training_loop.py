import torch
import torch.nn.functional as F

def train_model(train_loader, model, optimizer, scheduler, device):
    """
    Train the model for one epoch using InfoNCE loss. Logs loss after each batch.
    """
    model.train()
    total_loss = 0.0

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

        # Combine positive and negative similarities into logits
        logits = torch.cat([scaled_similarity.unsqueeze(1), negative_scaled_similarity.unsqueeze(1)], dim=1)

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

        # Log loss
        total_loss += loss.item()
        current_loss = total_loss / (batch_idx + 1)
        print(f"  [Batch {batch_idx + 1}/{len(train_loader)}] Loss: {current_loss:.4f}")

    epoch_loss = total_loss / len(train_loader)
    return epoch_loss, model
