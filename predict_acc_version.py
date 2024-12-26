import pandas as pd
import torch
from tqdm import tqdm
from math import ceil
from torch.cuda.amp import autocast

@torch.no_grad()
def predict(
    model,
    test_combinations: pd.DataFrame,
    test_reviews: pd.DataFrame,
    top_k: int = 10,
    batch_size: int = 512,
    comb_chunk_size: int = 64
):
    """
    Refactored prediction function with optimizations:
    1. torch.topk instead of full sort
    2. Vectorized/batched combinations
    3. Mixed-precision inference
    4. Minimized CPU-GPU transfers

    Parameters:
    - model: PyTorch model (accepting (comb_features, text_batch) inputs).
    - test_combinations: DataFrame with (accommodation_id, user_id, ...) feature columns.
    - test_reviews: DataFrame with review_id, review_positive, review_negative columns.
    - top_k: Number of top reviews to return for each combination.
    - batch_size: Number of reviews processed at a time (review batch size).
    - comb_chunk_size: Number of combination rows processed at a time (combination chunk size).

    Returns:
    - result_df: DataFrame with ranked top_k reviews for each test combination.
    """

    model.eval()
    device = next(model.parameters()).device

    # 1) Move combination features to a single GPU tensor
    #    Convert each row of test_combinations into a float32 feature vector.
    comb_features_list = []
    for _, row in test_combinations.iterrows():
        # Convert all columns to floats (or 0 if not numeric)
        comb_values = [float(val) if isinstance(val, (int, float)) else 0.0
                       for val in row.values]
        comb_features_list.append(comb_values)

    comb_features_tensor = torch.tensor(
        comb_features_list, dtype=torch.float32, device=device
    )  # Shape: (num_combinations, num_features)

    # 2) Gather all reviews in memory.
    #    Keep text on CPU (usually too large for GPU), but store IDs on CPU, too.
    all_review_ids = test_reviews["review_id"].tolist()
    all_review_texts = [
        f"{row['review_positive']} {row['review_negative']}"
        for _, row in test_reviews.iterrows()
    ]
    n_reviews = len(all_review_texts)
    n_combinations = comb_features_tensor.shape[0]

    results = []

    # Outer loop: process combinations in chunks to avoid OOM
    with tqdm(total=n_combinations, desc="Processing Combinations", dynamic_ncols=True) as comb_pbar:
        for start_idx in range(0, n_combinations, comb_chunk_size):
            end_idx = min(start_idx + comb_chunk_size, n_combinations)
            current_chunk_size = end_idx - start_idx

            # Slice out the chunk of combination features
            # Shape: (comb_chunk_size, num_features)
            comb_chunk = comb_features_tensor[start_idx:end_idx]

            # Allocate a (C x R) tensor to store probabilities for this chunk,
            # where C = current_chunk_size, R = n_reviews.
            # We'll keep it on GPU to minimize transfers.
            all_probs_chunk = torch.empty(
                (current_chunk_size, n_reviews),
                dtype=torch.float32,
                device=device
            )

            # Inner loop: process reviews in batches
            # Use mixed-precision inference to speed up on GPUs with tensor cores
            with autocast(dtype=torch.float16):
                for rev_start in range(0, n_reviews, batch_size):
                    rev_end = min(rev_start + batch_size, n_reviews)
                    review_batch_size = rev_end - rev_start

                    # Extract batch texts from CPU
                    texts_batch = all_review_texts[rev_start:rev_end]

                    # Repeat combination chunk for each text in the batch
                    # This yields a shape (C * review_batch_size, num_features)
                    comb_chunk_repeated = comb_chunk.unsqueeze(1).repeat(
                        1, review_batch_size, 1
                    ).view(current_chunk_size * review_batch_size, -1)

                    # Replicate each text for all combinations in the chunk
                    repeated_texts = []
                    for t in texts_batch:
                        repeated_texts.extend([t] * current_chunk_size)
                    # repeated_texts has length = (C * review_batch_size)

                    # Forward pass on model
                    logits = model(comb_chunk_repeated, repeated_texts)
                    probs = torch.sigmoid(logits).squeeze(dim=-1)
                    # probs shape: (C * review_batch_size,)

                    # Reshape back to (C, review_batch_size) so we can place it
                    # into the all_probs_chunk
                    probs_2d = probs.view(current_chunk_size, review_batch_size)
                    all_probs_chunk[:, rev_start:rev_end] = probs_2d

            # After we have all probabilities in the chunk, do top_k per combination row
            # This is much faster than sorting all reviews.
            topk_probs, topk_indices = torch.topk(all_probs_chunk, k=top_k, dim=1)
            # topk_indices is (C, top_k), all on GPU

            # Move indices to CPU for final ID lookups (all at once)
            topk_indices_cpu = topk_indices.detach().cpu()

            # Build output rows
            for i in range(current_chunk_size):
                row_top_indices = topk_indices_cpu[i].tolist()
                top_review_ids = [all_review_ids[idx] for idx in row_top_indices]

                # Get the actual row from the original DataFrame for IDs
                comb_row = test_combinations.iloc[start_idx + i]
                results.append((
                    comb_row["accommodation_id"],
                    comb_row["user_id"],
                    *top_review_ids
                ))

            # Update progress
            comb_pbar.update(current_chunk_size)

    # Build final DataFrame
    columns = ["accommodation_id", "user_id"] + [f"review_{i}" for i in range(1, top_k + 1)]
    result_df = pd.DataFrame(results, columns=columns)
    result_df.insert(0, "ID", range(1, len(result_df) + 1))

    return result_df
