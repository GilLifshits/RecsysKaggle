import torch


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
        scaled_similarity, user_embedding, review_embedding = model(user_features, review_content)

        # For demonstration, all pairs are "similar" => label = 1.0
        targets = torch.ones(len(scaled_similarity), device=device)

        loss = criterion(scaled_similarity, targets)
        loss.backward()

        optimizer.step()
        total_loss += loss.item()

        # Log loss after each batch
        current_loss = total_loss / (batch_idx + 1)
        print(f"  [Batch {batch_idx + 1}/{len(train_loader)}] Loss: {current_loss:.4f}")

    epoch_loss = total_loss / len(train_loader)
    return epoch_loss, model
