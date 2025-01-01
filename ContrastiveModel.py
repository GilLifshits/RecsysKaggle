import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer


class ContrastiveSentenceTransformerModel(nn.Module):
    def __init__(self, model_path, fine_tune_transformers=False):
        super(ContrastiveSentenceTransformerModel, self).__init__()

        # Load SentenceTransformer models from local directories
        self.context_transformer = SentenceTransformer(model_path)
        self.review_transformer = SentenceTransformer(model_path)

        # Optionally enable fine-tuning for SentenceTransformer models
        if fine_tune_transformers:
            for param in self.context_transformer.parameters():
                param.requires_grad = True

            for param in self.review_transformer.parameters():
                param.requires_grad = True

        # Fine-tuning layers for embeddings
        embedding_dim = self.context_transformer.get_sentence_embedding_dimension()
        self.fc_context = nn.Linear(embedding_dim, 256)
        self.fc_review = nn.Linear(embedding_dim, 256)

        # Initialize weights
        nn.init.xavier_uniform_(self.fc_context.weight)
        nn.init.xavier_uniform_(self.fc_review.weight)

        self.sigmoid = nn.Sigmoid()

    def forward(self, context_text, review_text):
        # context_text: list of strings
        # review_text: list of strings

        # Get embeddings from SentenceTransformer
        context_embeddings = torch.tensor(
            self.context_transformer.encode(context_text, convert_to_tensor=True)
        ).to(self.fc_context.weight.device)

        review_embeddings = torch.tensor(
            self.review_transformer.encode(review_text, convert_to_tensor=True)
        ).to(self.fc_review.weight.device)

        # Apply fine-tuning layers
        context_embeddings = torch.relu(self.fc_context(context_embeddings))
        review_embeddings = torch.relu(self.fc_review(review_embeddings))

        # Normalize embeddings for numerical stability
        context_embeddings = nn.functional.normalize(context_embeddings, p=2, dim=1)
        review_embeddings = nn.functional.normalize(review_embeddings, p=2, dim=1)

        # Compute dot product similarity
        dot_product = torch.sum(context_embeddings * review_embeddings, dim=1)

        # Apply sigmoid for bounded similarity
        similarity = self.sigmoid(dot_product)

        return similarity, context_embeddings, review_embeddings
