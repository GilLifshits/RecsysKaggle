import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer


@torch.no_grad()
def predict(users_trained_model, reviews_trained_model, test_combinations, test_reviews, top_k=10, batch_size=64):
    reviews_trained_model.eval()
    users_trained_model.eval()
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    max_seq_len = 64

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    # Dictionary to store user embeddings
    review_embeddings_dict = {}

    # Process data in batches
    num_samples = len(test_reviews)
    for start_idx in tqdm(range(0, num_samples, batch_size), desc="Processing Batches"):
        end_idx = min(start_idx + batch_size, num_samples)
        batch = test_reviews.iloc[start_idx:end_idx]

        batch_size = len(batch)  # Assuming batch is a collection like a list or DataFrame

        review_scores = [str(element) for element in batch['review_score']]

        reviews = ('Review title: ' + batch['review_title'].fillna('') + '. ' +
                   'Review positive: ' + batch['review_positive'].fillna('') + '. ' +
                   'Review negative: ' + batch['review_negative'].fillna('') + '. ' +
                   'Review score: ' + review_scores).tolist()

        # Forward pass through the model
        review_content_tok = tokenizer(reviews, padding=True, truncation=True, return_tensors='pt',
                                       max_length=max_seq_len).to("cuda")

        review_content_embed = reviews_trained_model(review_content_tok)

        # Save user embeddings to the dictionary
        for i, review_id in enumerate(batch['review_id']):
            review_embeddings_dict[review_id] = review_content_embed[i].cpu().numpy()

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    # Dictionary to store user embeddings
    user_embeddings_dict = {}

    # Process data in batches
    num_samples = len(test_combinations)
    for start_idx in tqdm(range(0, num_samples, batch_size), desc="Processing Batches"):
        end_idx = min(start_idx + batch_size, num_samples)
        batch = test_combinations.iloc[start_idx:end_idx]

        # Prepare batch combination features tensor
        room_nights = [str(element) for element in batch['room_nights']]
        month = [str(element) for element in batch['month']]
        comb_features = (('Guest type: ' + batch['guest_type'].fillna('') + '. ' +
                          'Guest country: ' + batch['guest_country'].fillna('') + '. ' +
                          'Room nights: ' + room_nights + '. ' +
                          'Month: ' + month + '. ' +
                          'Accommodation type: ' + batch['accommodation_type'].fillna('') + '. ' +
                          'Accommodation country: ' + batch['accommodation_country'].fillna(''))
                         .tolist())

        user_features_tok = tokenizer(comb_features, padding=True, truncation=True, return_tensors='pt',
                                      max_length=max_seq_len).to("cuda")
        user_features_embed = users_trained_model(user_features_tok)

        # Save user embeddings to the dictionary
        for i, user_id in enumerate(batch['user_id']):
            user_embeddings_dict[user_id] = user_features_embed[i].cpu().numpy()

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    results = []

    for _, comb in tqdm(test_combinations.iterrows(), total=len(test_combinations), desc="Processing combinations"):
        user_id, accommodation_id = comb["user_id"], comb["accommodation_id"]

        sub_test_reviews = test_reviews.loc[test_reviews["accommodation_id"] == accommodation_id]
        review_ids = sub_test_reviews["review_id"].tolist()
        sub_test_reviews_embeddings = [
            review_embeddings_dict[review_id] for review_id in review_ids
        ]

        current_user_embedding = torch.tensor(user_embeddings_dict[user_id]).unsqueeze(0)

        if sub_test_reviews_embeddings:  # Ensure there is at least one embedding
            sub_test_reviews_embeddings = torch.tensor(sub_test_reviews_embeddings)

            similarities = []
            for i in range(0, len(sub_test_reviews_embeddings), batch_size):
                batch_embeddings = sub_test_reviews_embeddings[i:i + batch_size]
                batch_review_ids = review_ids[i:i + batch_size]
                similarity = F.cosine_similarity(
                    current_user_embedding, batch_embeddings
                )
                similarities.extend(zip(batch_review_ids, similarity.tolist()))

            # Get top 10 reviews by similarity
            top_reviews = sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]
            top_review_ids = [review_id for review_id, _ in top_reviews]
            results.append([accommodation_id, user_id] + top_review_ids)

    # Create final DataFrame
    columns = ["accommodation_id", "user_id"] + [f"review_{i}" for i in range(1, top_k + 1)]
    result_df = pd.DataFrame(results, columns=columns)
    result_df.insert(0, "ID", range(1, len(result_df) + 1))

    return result_df
