import os

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer
from ContrastiveModel import ContrastiveTransformerModel
from ReviewDataset import custom_collate_fn, ReviewDataset

def train_model(batch_size, device, all_accommodation_ids, epochs):
    embed_size = 512
    output_embed_size = 128
    max_seq_len = 64

    # Initialize tokenizer and encoders
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    user_features_encoder = ContrastiveTransformerModel(tokenizer.vocab_size, embed_size, output_embed_size).to(device)
    review_content_encoder = ContrastiveTransformerModel(tokenizer.vocab_size, embed_size, output_embed_size).to(device)

    optimizer = torch.optim.AdamW(
        list(user_features_encoder.parameters()) + list(review_content_encoder.parameters()),
        lr=1e-5, weight_decay=0.01
    )
    loss_fn = torch.nn.CrossEntropyLoss()

    scaler = torch.cuda.amp.GradScaler()  # Mixed precision training

    # Pre-load CSV files to memory
    print("Loading dataset to memory")
    datasets_cache = {}

    for accommodation_id in tqdm(all_accommodation_ids, desc="Loading datasets"):
        try:
            # Define file paths
            users_file = f"in_accommodation_datasets/train_users_{accommodation_id}.csv"
            reviews_file = f"in_accommodation_datasets/train_review_{accommodation_id}.csv"
            matches_file = f"in_accommodation_datasets/train_matches_{accommodation_id}.csv"

            # Check if files are empty
            if (os.path.getsize(users_file) == 0 or
                    os.path.getsize(reviews_file) == 0 or
                    os.path.getsize(matches_file) == 0):
                print(f"Skipping accommodation_id {accommodation_id}: one or more files are empty.")
                continue

            # Load files into DataFrame
            datasets_cache[accommodation_id] = {
                "users": pd.read_csv(users_file),
                "reviews": pd.read_csv(reviews_file),
                "matches": pd.read_csv(matches_file),
            }
        except Exception as e:
            print(f"Error processing accommodation_id {accommodation_id}: {e}")
    print(f"Loaded {len(datasets_cache)} accommodations")


    # Training loop
    for epoch in range(epochs):

        print(f"Epoch {epoch + 1}/{epochs}")

        accommodation_counter = 0
        running_loss = 0
        batch_counter = 0

        for accommodation_idx, accommodation_id in tqdm(enumerate(all_accommodation_ids), desc="Processing accommodations"):

            if accommodation_id not in datasets_cache:
                continue

            accommodation_counter += 1

            # Get preloaded data
            data = datasets_cache[accommodation_id]
            train_dataset = ReviewDataset(data["users"], data["reviews"], data["matches"])
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=custom_collate_fn,
                pin_memory=(device == 'cuda')
            )

            for batch_idx, batch in enumerate(train_loader):
                batch_counter += 1

                # Tokenize and move to device
                user_features_tok = tokenizer(
                    batch["user_features"], padding=True, truncation=True, return_tensors='pt', max_length=max_seq_len
                ).to(device)
                review_content_tok = tokenizer(
                    batch["review_content"], padding=True, truncation=True, return_tensors='pt', max_length=max_seq_len
                ).to(device)

                # Mixed precision for forward and backward
                with torch.cuda.amp.autocast():
                    user_features_embed = user_features_encoder(user_features_tok)
                    review_content_embed = review_content_encoder(review_content_tok)

                    similarity_scores = user_features_embed @ review_content_embed.T
                    target = torch.arange(user_features_embed.shape[0], dtype=torch.long).to(device)
                    loss = loss_fn(similarity_scores, target)
                    running_loss += loss.item()

                # Gradient scaling and optimization
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            # Log loss every 5 accommodations
            if accommodation_counter % 5 == 0:
                avg_loss = running_loss / batch_counter
                print(f"Average loss after {accommodation_counter} accommodations: {avg_loss:.4f}")

    return user_features_encoder, review_content_encoder
