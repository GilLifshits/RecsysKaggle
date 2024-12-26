import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel


class ContrastiveModel(nn.Module):
    def __init__(self, user_feature_size):
        super(ContrastiveModel, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("model")
        self.bert = AutoModel.from_pretrained("model")

        # Use the actual hidden size from the model config rather than hardcoding
        hidden_size = self.bert.config.hidden_size

        self.fc_user = nn.Linear(user_feature_size, 256)
        self.fc_review = nn.Linear(hidden_size, 256)

        # Cosine similarity for user and review embeddings
        self.cos = nn.CosineSimilarity(dim=1)

    def forward(self, user_features, review_content):
        # user_features: (batch_size, user_feature_size)
        # review_content: list of strings (length batch_size)

        user_embedding = torch.relu(self.fc_user(user_features))  # -> (batch_size, 256)

        # Tokenize reviews
        review_tokens = self.tokenizer(
            review_content,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        ).to(user_features.device)

        # If the batch sizes don't match for some reason, expand (rare corner case).
        # Typically, you won't need this if your collate_fn is correct.
        if review_tokens['input_ids'].shape[0] != user_features.shape[0]:
            review_tokens = {
                key: val.expand(user_features.shape[0], -1) for key, val in review_tokens.items()
            }

        # Obtain the [CLS] token representation
        review_output = self.bert(**review_tokens).last_hidden_state[:, 0, :]
        review_embedding = torch.relu(self.fc_review(review_output))  # -> (batch_size, 256)

        # Cosine similarity in [-1, +1]
        similarity = self.cos(user_embedding, review_embedding)

        # Scale similarity to a more logit-friendly range for BCEWithLogitsLoss
        logits = similarity * 5.0  # transforms [-1,1] to roughly [-5,5]
        return logits
