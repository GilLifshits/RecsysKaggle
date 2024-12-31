import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel


class ContrastiveModel(nn.Module):
    def __init__(self, user_feature_size):
        super(ContrastiveModel, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("bert_model")
        self.bert = AutoModel.from_pretrained("bert_model")

        # Use the actual hidden size from the model config rather than hardcoding
        hidden_size = self.bert.config.hidden_size

        self.fc_user = nn.Linear(user_feature_size, 256)
        self.fc_review = nn.Linear(hidden_size, 256)

        # Cosine similarity for user and review embeddings
        self.cos = nn.CosineSimilarity(dim=1)

        # Initialize weights to avoid extreme values
        nn.init.xavier_uniform_(self.fc_user.weight)
        nn.init.xavier_uniform_(self.fc_review.weight)

    def forward(self, user_features, review_content):
        # user_features: (batch_size, user_feature_size)
        # review_content: list of strings (length batch_size)

        # Ensure user features are valid
        user_features = torch.nan_to_num(user_features, nan=0.0, posinf=1.0, neginf=-1.0)

        user_embedding = torch.relu(self.fc_user(user_features))  # -> (batch_size, 256)

        # review_content = [str(review) for review in review_content]

        # Tokenize reviews
        review_tokens = self.tokenizer(
            review_content,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt"
        ).to(user_features.device)

        # Handle mismatched batch sizes
        if review_tokens['input_ids'].shape[0] != user_features.shape[0]:
            review_tokens = {
                key: val.expand(user_features.shape[0], -1) for key, val in review_tokens.items()
            }

        # Obtain the [CLS] token representation
        review_output = self.bert(**review_tokens).last_hidden_state[:, 0, :]
        review_embedding = torch.relu(self.fc_review(review_output))  # -> (batch_size, 256)

        # Normalize embeddings to avoid extreme cosine values
        user_embedding = nn.functional.normalize(user_embedding, p=2, dim=1)
        review_embedding = nn.functional.normalize(review_embedding, p=2, dim=1)

        # Cosine similarity in [-1, +1]
        similarity = self.cos(user_embedding, review_embedding)

        # Clamp similarity to avoid NaN during scaling
        similarity = torch.clamp(similarity, min=-1.0, max=1.0)

        # Scale similarity to a logit-friendly range for BCEWithLogitsLoss
        # scaled_similarity = similarity * 5.0  # transforms [-1,1] to roughly [-5,5]
        scaled_similarity = similarity

        return scaled_similarity, user_embedding, review_embedding
