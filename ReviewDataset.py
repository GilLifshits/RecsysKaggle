import torch
from torch.utils.data import Dataset


class ReviewDataset(Dataset):
    def __init__(self, users, reviews, matches):
        # Keep references
        self.users = users
        self.reviews = reviews

        # Only keep matches that exist in both users & reviews
        valid_user_ids = set(self.users['user_id'])
        valid_accommodation_ids = set(self.users['accommodation_id'])
        valid_review_ids = set(self.reviews['review_id'])

        self.matches = matches[
            matches['user_id'].isin(valid_user_ids)
            & matches['accommodation_id'].isin(valid_accommodation_ids)
            & matches['review_id'].isin(valid_review_ids)
            ].reset_index(drop=True)

    def __len__(self):
        return len(self.matches)

    def __getitem__(self, idx):
        match = self.matches.iloc[idx]
        user_id = match['user_id']
        accommodation_id = match['accommodation_id']
        review_id = match['review_id']

        # make sure they exist
        user_features = self.users[(self.users['user_id'] == user_id) & (self.users['accommodation_id'] == accommodation_id)].iloc[0]
        review_content = self.reviews[self.reviews['review_id'] == review_id].iloc[0]

        # user features textual representation
        user_features_textual_representation = (f"Guest type: {user_features['guest_type']} ."
                                         f"Guest country: {user_features['guest_country']} ."
                                         f"Room nights: {user_features['room_nights']} ."
                                         f"Month: {user_features['month']} ."
                                         f"Accommodation type: {user_features['accommodation_type']} ."
                                         f"Accommodation country: {user_features['accommodation_country']}.")

        # Review textual representation
        review_textual_representation = (f"Review title: {review_content['review_title']} ."
                                         f"Review positive: {review_content['review_positive']} ."
                                         f"Review negative: {review_content['review_negative']} ."
                                         f"Review score: {review_content['review_score']}.")

        return {
            'user_features': user_features_textual_representation,
            'review_content': review_textual_representation
        }


def custom_collate_fn(batch):
    user_features, review_content = [], []

    for item in batch:
        user_features.append(item['user_features'])
        review_content.append(item['review_content'])

    # user_features_tensor = torch.tensor(user_features, dtype=torch.float32)
    return {
        'user_features': user_features,
        'review_content': review_content
    }

