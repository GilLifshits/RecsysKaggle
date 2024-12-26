import pandas as pd
import torch
from tqdm import tqdm


@torch.no_grad()
def predict(model, test_combinations, test_reviews, top_k=10, batch_size=512):
    """
    Optimized for GPU inference: predicts top 'top_k' review IDs for each test combination.

    Parameters:
    - model: PyTorch model.
    - test_combinations: DataFrame with (accommodation_id, user_id) pairs and features.
    - test_reviews: DataFrame with review_id, review_positive, and review_negative columns.
    - top_k: Number of top reviews to return for each combination.
    - batch_size: Number of reviews to process in a single batch.

    Returns:
    - result_df: DataFrame with ranked top_k reviews for each test combination.
    """
    model.eval()  # Ensure the model is in evaluation mode
    device = next(model.parameters()).device  # Get the device of the model

    # Pre-build tensors for all reviews
    all_review_texts = [
        f"{row['review_positive']} {row['review_negative']}"
        for _, row in test_reviews.iterrows()
    ]
    all_review_ids = test_reviews["review_id"].tolist()
    n_reviews = len(all_review_texts)

    results = []

    # Outer progress bar for combinations
    with tqdm(total=len(test_combinations), desc="Processing Combinations", dynamic_ncols=True) as comb_pbar:
        for _, comb_row in test_combinations.iterrows():
            # Convert row values to float and prepare combination feature tensor
            comb_features_vals = [
                float(value) if isinstance(value, (int, float)) else 0.0
                for value in comb_row.values[:]
            ]
            comb_features_tensor = torch.tensor(
                comb_features_vals, dtype=torch.float32, device=device
            ).unsqueeze(0)  # Shape: (1, num_features)

            # Batch-wise processing of reviews
            probs_list = []
            with tqdm(total=n_reviews, desc="Processing Reviews", dynamic_ncols=True, leave=False) as review_pbar:
                for start_idx in range(0, n_reviews, batch_size):
                    end_idx = min(start_idx + batch_size, n_reviews)

                    # Extract batch review texts
                    texts_batch = all_review_texts[start_idx:end_idx]
                    batch_size_actual = len(texts_batch)

                    # Duplicate combination tensor for batch size
                    comb_features_batch = comb_features_tensor.repeat(batch_size_actual, 1)

                    # Forward pass: predict logits
                    logits = model(comb_features_batch, texts_batch)  # Shape: (batch_size,)
                    probs = torch.sigmoid(logits).squeeze(dim=-1)  # Shape: (batch_size,)

                    # Append probabilities (detached to CPU)
                    probs_list.append(probs.cpu())

                    # Update inner progress bar
                    review_pbar.update(batch_size_actual)

            # Concatenate all probabilities
            all_probs = torch.cat(probs_list).numpy()

            # Pair review IDs with probabilities and sort
            similarities = sorted(zip(all_review_ids, all_probs), key=lambda x: x[1], reverse=True)

            # Extract top_k review IDs
            top_review_ids = [review_id for review_id, _ in similarities[:top_k]]

            # Append result row
            results.append((
                comb_row["accommodation_id"],
                comb_row["user_id"],
                *top_review_ids
            ))

            # Update outer progress bar
            comb_pbar.update(1)

    # Create the final DataFrame
    columns = ["accommodation_id", "user_id"] + [f"review_{i}" for i in range(1, top_k + 1)]
    result_df = pd.DataFrame(results, columns=columns)

    # Add an ID column
    result_df.insert(0, "ID", range(1, len(result_df) + 1))

    return result_df
