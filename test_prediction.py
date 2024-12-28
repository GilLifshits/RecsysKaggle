import pandas as pd
import torch
from tqdm import tqdm

@torch.no_grad()
def predict(model, test_combinations, test_reviews, top_k=10, batch_size=512):
    """
    For each (accommodation_id, user_id) pair in test_combinations,
    return a ranked list of the top 'top_k' review IDs predicted by the model,
    comparing *only* to reviews that have the same accommodation_id.

    The output DataFrame will have the following columns:
      ID, accommodation_id, user_id, review_1, ..., review_top_k
    """

    model.eval()
    device = next(model.parameters()).device

    results = []

    # Outer progress bar: iterate over each row in test_combinations
    with tqdm(total=len(test_combinations), desc="Processing Combinations", dynamic_ncols=True) as comb_pbar:
        for _, comb_row in test_combinations.iterrows():
            # 1) Filter reviews by accommodation_id
            acc_id = comb_row["accommodation_id"]
            reviews_subset = test_reviews[test_reviews["accommodation_id"] == acc_id]  # TODO: fill out to give 10 review

            # 2) If no reviews for this accommodation_id, handle accordingly
            if reviews_subset.empty:
                # Fill top_k columns with None (or handle differently as needed)
                top_review_ids = [None]*top_k
                results.append(
                    (acc_id, comb_row["user_id"], *top_review_ids)
                )
                comb_pbar.update(1)
                continue

            # 3) Build the list of review texts and IDs for this accommodation_id
            all_review_texts = [
                f"{row['review_positive']} {row['review_negative']}"
                for _, row in reviews_subset.iterrows()
            ]
            all_review_ids = reviews_subset["review_id"].tolist()
            n_reviews = len(all_review_texts)

            # 4) Convert the combination features into a tensor (shape: (1, num_features))
            comb_features_vals = [
                float(value) if isinstance(value, (int, float)) else 0
                for value in comb_row.values[:]
            ]
            comb_features_tensor = torch.tensor(
                comb_features_vals, dtype=torch.float32
            ).unsqueeze(0).to(device)

            # 5) Loop over the reviews in mini-batches, accumulating probabilities
            probs_list = []

            with tqdm(total=n_reviews, desc="Processing Reviews", dynamic_ncols=True, leave=False) as review_pbar:
                for start_idx in range(0, n_reviews, batch_size):
                    end_idx = start_idx + batch_size
                    texts_batch = all_review_texts[start_idx:end_idx]
                    batch_size_actual = len(texts_batch)

                    if batch_size_actual == 0:
                        break  # In case there's an off-by-one, just skip

                    # Repeat the combination features to match the mini-batch size
                    comb_features_batch = comb_features_tensor.repeat(batch_size_actual, 1)

                    # Forward pass
                    logits = model(comb_features_batch, texts_batch)

                    # -- Ensure logits is at least 1-D (e.g., shape=(B,)) --
                    #    If your model returns shape=() for a single example,
                    #    unsqueeze so that we have (1,) not ().
                    if logits.dim() == 0:
                        logits = logits.unsqueeze(0)

                    # Apply sigmoid and remove any trailing dimension
                    probs = torch.sigmoid(logits).squeeze(dim=-1)

                    # Move to CPU to store
                    probs_cpu = probs.detach().cpu()
                    probs_list.append(probs_cpu)

                    review_pbar.update(batch_size_actual)

            # 6) Once we've processed all batches, concatenate probabilities
            if len(probs_list) == 0:
                # Means no valid batches; fill with None
                top_review_ids = [None]*top_k
            else:
                probs_list = [p.unsqueeze(0) if p.dim() == 0 else p for p in probs_list]
                all_probs = torch.cat(probs_list).numpy()

                # Pair each review_id with the predicted probability
                similarities = list(zip(all_review_ids, all_probs))

                # Sort by probability descending
                similarities.sort(key=lambda x: x[1], reverse=True)

                # Take top_k IDs
                top_review_ids = [review_id for (review_id, _) in similarities[:top_k]]

            # 7) Collect the results for this row
            results.append(
                (acc_id, comb_row["user_id"], *top_review_ids)
            )

            comb_pbar.update(1)

    # 8) Build final DataFrame: 'ID' + 'accommodation_id' + 'user_id' + top_k review columns
    columns = ["accommodation_id", "user_id"] + [f"review_{i}" for i in range(1, top_k + 1)]
    result_df = pd.DataFrame(results, columns=columns)
    result_df.insert(0, "ID", range(1, len(result_df) + 1))

    return result_df
