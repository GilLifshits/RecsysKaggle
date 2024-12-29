import pandas as pd
import torch
from tqdm import tqdm

@torch.no_grad()
def predict(model, test_combinations, test_reviews, top_k=10, batch_size=512):
    """
    Predict the top_k reviews for each (accommodation_id, user_id) pair in test_combinations,
    filtering only by matching accommodation_id.

    Args:
        model: Trained PyTorch model for predictions.
        test_combinations: DataFrame with columns ['accommodation_id', 'user_id'].
        test_reviews: DataFrame with columns ['accommodation_id', 'review_id', 'review_positive', 'review_negative'].
        top_k: Number of top reviews to return for each combination.
        batch_size: Batch size for processing reviews.

    Returns:
        DataFrame with columns ['ID', 'accommodation_id', 'user_id', 'review_1', ..., 'review_top_k'].
    """
    device = next(model.parameters()).device
    model.eval()

    results = []

    for _, comb_row in tqdm(test_combinations.iterrows(), total=len(test_combinations), desc="Processing Combinations"):
        acc_id = comb_row["accommodation_id"]
        user_id = comb_row["user_id"]

        # Filter reviews by accommodation_id
        reviews_subset = test_reviews[test_reviews["accommodation_id"] == acc_id]
        if reviews_subset.empty:
            results.append((acc_id, user_id, *([None] * top_k)))
            continue

        # Prepare review texts and IDs
        all_review_texts = [
            f"{row['review_positive']} {row['review_negative']}"
            for _, row in reviews_subset.iterrows()
        ]
        all_review_ids = reviews_subset["review_id"].tolist()

        # Prepare combination features tensor
        comb_features_vals = [
            float(value) if isinstance(value, (int, float)) else 0
            for value in comb_row.values[:]
        ]
        comb_features = torch.tensor(
            comb_features_vals, dtype=torch.float32
        ).unsqueeze(0).to(device)

        # Compute probabilities in batches
        probs = []
        for i in range(0, len(all_review_texts), batch_size):
            batch_texts = all_review_texts[i:i + batch_size]
            batch_features = comb_features.repeat(len(batch_texts), 1)
            logits = model(batch_features, batch_texts)
            probs.append(torch.sigmoid(logits).squeeze(-1).cpu().numpy())

        # Get top_k reviews by probability
        all_probs = torch.cat([torch.tensor(p) for p in probs]).numpy()
        top_review_ids = [rid for rid, _ in sorted(zip(all_review_ids, all_probs), key=lambda x: x[1], reverse=True)[:top_k]]

        # Append results
        results.append((acc_id, user_id, *top_review_ids))

    # Create final DataFrame
    columns = ["accommodation_id", "user_id"] + [f"review_{i}" for i in range(1, top_k + 1)]
    result_df = pd.DataFrame(results, columns=columns)
    result_df.insert(0, "ID", range(1, len(result_df) + 1))
    return result_df
