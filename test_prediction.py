import pandas as pd
import torch
from tqdm import tqdm

@torch.no_grad()
def predict(model, test_combinations, test_reviews, top_k=10, batch_size=512):
    """
    For each (accommodation_id, user_id) pair in test_combinations,
    return a ranked list of the top 'top_k' review IDs predicted by the model.

    Uses mini-batching for reviews to leverage GPU more efficiently.

    The output DataFrame will have 12 columns total:
    ID, accommodation_id, user_id, review_1, ..., review_10
    """
    model.eval()
    device = next(model.parameters()).device

    # Pre-build the list of all reviews (strings) and review_ids
    all_review_texts = [
        f"{row['review_positive']} {row['review_negative']}"
        for _, row in test_reviews.iterrows()
    ]
    all_review_ids = test_reviews["review_id"].tolist()
    n_reviews = len(all_review_texts)

    results = []

    # Initialize outer progress bar
    with tqdm(total=len(test_combinations), desc="Processing Combinations", dynamic_ncols=True) as comb_pbar:
        for _, comb_row in test_combinations.iterrows():
            # Convert row values to float (or handle them however your model expects)
            comb_features_vals = [
                float(value) if isinstance(value, (int, float)) else 0
                for value in comb_row.values[:]
            ]

            # Make a single tensor of shape (1, num_features) on the correct device
            comb_features_tensor = torch.tensor(
                comb_features_vals, dtype=torch.float32
            ).unsqueeze(0).to(device)

            # We'll accumulate probabilities in a list, then sort once at the end
            probs_list = []

            # Process the reviews in mini-batches with progress bar for reviews
            with tqdm(total=n_reviews, desc="Processing Reviews", dynamic_ncols=True, leave=False) as review_pbar:
                for start_idx in range(0, n_reviews, batch_size):
                    end_idx = start_idx + batch_size

                    # Slice the texts for this mini-batch
                    texts_batch = all_review_texts[start_idx:end_idx]
                    batch_size_actual = len(texts_batch)

                    # Repeat the combination features to match this mini-batch size
                    comb_features_batch = comb_features_tensor.repeat(batch_size_actual, 1)

                    # Forward pass for this mini-batch
                    logits = model(comb_features_batch, texts_batch)  # shape: (batch_size,) or (batch_size, 1)
                    probs = torch.sigmoid(logits).squeeze(dim=-1)     # shape: (batch_size,)

                    # Move to CPU once per mini-batch
                    probs_cpu = probs.detach().cpu()
                    probs_list.append(probs_cpu)

                    # Update review progress bar
                    review_pbar.update(batch_size_actual)

            # Concatenate probabilities for all mini-batches: shape (n_reviews,)
            all_probs = torch.cat(probs_list).numpy()

            # Pair each review_id with the predicted probability
            similarities = list(zip(all_review_ids, all_probs))

            # Sort in descending order by probability
            similarities.sort(key=lambda x: x[1], reverse=True)

            # Take top_k
            top_review_ids = [review_id for (review_id, _) in similarities[:top_k]]

            # Collect this row's results
            results.append(
                (
                    comb_row["accommodation_id"],
                    comb_row["user_id"],
                    *top_review_ids
                )
            )

            # Update outer progress bar
            comb_pbar.update(1)

    # Build the final DataFrame
    columns = ["accommodation_id", "user_id"] + [f"review_{i}" for i in range(1, top_k + 1)]
    result_df = pd.DataFrame(results, columns=columns)

    # Insert 'ID' as the first column (index from 1)
    result_df.insert(0, "ID", range(1, len(result_df) + 1))

    return result_df