import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from ContrastiveModel import ContrastiveSentenceTransformerModel
from ReviewDataset import custom_collate_fn, ReviewDataset


def train_model(batch_size, device, all_accommodation_ids, accommodation_amount):

    embed_size = 512
    output_embed_size = 128
    max_seq_len = 64

    # define the encoders
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    user_features_encoder = ContrastiveSentenceTransformerModel(tokenizer.vocab_size, embed_size, output_embed_size)
    review_content_encoder = ContrastiveSentenceTransformerModel(tokenizer.vocab_size, embed_size, output_embed_size)

    optimizer = torch.optim.AdamW(
        list(user_features_encoder.parameters()) + list(review_content_encoder.parameters()),
        lr=1e-5, weight_decay=0.01
    )

    loss_fn = torch.nn.CrossEntropyLoss()

    running_loss = 0
    accommodation_counter = 0
    batch_counter = 0

    # Define train loader for each accommodation data:
    for accommodation_idx, accommodation_id in enumerate(
        # tqdm(all_accommodation_ids[:], desc="Processing accommodations")
        tqdm(all_accommodation_ids[:accommodation_amount], desc="Processing accommodations")
    ):
        # Increment accommodation counter
        accommodation_counter += 1

        # Get all data for specific accommodation_id
        train_users = pd.read_csv(f"in_accommodation_datasets/train_users_{accommodation_id}.csv")
        train_reviews = pd.read_csv(f"in_accommodation_datasets/train_review_{accommodation_id}.csv")
        train_matches = pd.read_csv(f"in_accommodation_datasets/train_matches_{accommodation_id}.csv")

        train_dataset = ReviewDataset(train_users, train_reviews, train_matches)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=custom_collate_fn,
            pin_memory=(device == 'cuda')
        )

        # Train per accommodation id
        for batch_idx, batch in enumerate(train_loader):

            batch_counter+=1

            # Tokenize the question/answer pairs (each is a batch of 32 questions and 32 answers)
            user_features, review_content = batch["user_features"], batch["review_content"]
            user_features_tok = tokenizer(user_features, padding=True, truncation=True, return_tensors='pt',
                                     max_length=max_seq_len)
            review_content_tok = tokenizer(review_content, padding=True, truncation=True, return_tensors='pt', max_length=max_seq_len)

            # Compute the embeddings: the output is of dim = 32 x 128
            user_features_embed = user_features_encoder(user_features_tok)
            review_content_embed = review_content_encoder(review_content_tok)

            # Compute similarity scores: a 32x32 matrix
            # row[N] reflects similarity between question[N] and answers[0...31]
            similarity_scores = user_features_embed @ review_content_embed.T

            # we want to maximize the values in the diagonal
            target = torch.arange(user_features_embed.shape[0], dtype=torch.long)
            loss = loss_fn(similarity_scores, target)
            running_loss += loss.item()

            # this is where the magic happens
            optimizer.zero_grad()  # reset optimizer so gradients are all-zero
            loss.backward()
            optimizer.step()

        # Print loss every 5 accommodation IDs
        if accommodation_counter % 5 == 0:
            avg_loss = running_loss / batch_counter
            print(f"Average loss after {accommodation_counter} accommodations: {avg_loss:.4f}")

    return user_features_encoder, review_content_encoder




# import pandas as pd
# import torch
# from torch.utils.data import DataLoader
# from tqdm import tqdm
# from transformers import AutoTokenizer
# from ContrastiveModel import ContrastiveSentenceTransformerModel
# from ReviewDataset import custom_collate_fn, ReviewDataset
#
# def train_model(batch_size, device, all_accommodation_ids, accommodation_amount):
#     embed_size = 512
#     output_embed_size = 128
#     max_seq_len = 64
#
#     # Initialize tokenizer and encoders
#     tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
#     user_features_encoder = ContrastiveSentenceTransformerModel(tokenizer.vocab_size, embed_size, output_embed_size).to(device)
#     review_content_encoder = ContrastiveSentenceTransformerModel(tokenizer.vocab_size, embed_size, output_embed_size).to(device)
#
#     optimizer = torch.optim.AdamW(
#         list(user_features_encoder.parameters()) + list(review_content_encoder.parameters()),
#         lr=1e-5, weight_decay=0.01
#     )
#     loss_fn = torch.nn.CrossEntropyLoss()
#
#     scaler = torch.cuda.amp.GradScaler()  # Mixed precision training
#     running_loss = 0
#     accommodation_counter = 0
#     batch_counter = 0
#
#     # Pre-load CSV files to memory if possible (optional for smaller datasets)
#     datasets_cache = {
#         accommodation_id: {
#             "users": pd.read_csv(f"in_accommodation_datasets/train_users_{accommodation_id}.csv"),
#             "reviews": pd.read_csv(f"in_accommodation_datasets/train_review_{accommodation_id}.csv"),
#             "matches": pd.read_csv(f"in_accommodation_datasets/train_matches_{accommodation_id}.csv"),
#         }
#         for accommodation_id in all_accommodation_ids[:accommodation_amount]
#     }
#
#     for accommodation_idx, accommodation_id in tqdm(enumerate(all_accommodation_ids[:accommodation_amount]), desc="Processing accommodations"):
#         accommodation_counter += 1
#
#         # Get preloaded data
#         data = datasets_cache[accommodation_id]
#         train_dataset = ReviewDataset(data["users"], data["reviews"], data["matches"])
#         train_loader = DataLoader(
#             train_dataset,
#             batch_size=batch_size,
#             shuffle=True,
#             collate_fn=custom_collate_fn,
#             pin_memory=(device == 'cuda')
#         )
#
#         for batch_idx, batch in enumerate(train_loader):
#             batch_counter += 1
#
#             # Tokenize and move to device
#             user_features_tok = tokenizer(
#                 batch["user_features"], padding=True, truncation=True, return_tensors='pt', max_length=max_seq_len
#             ).to(device)
#             review_content_tok = tokenizer(
#                 batch["review_content"], padding=True, truncation=True, return_tensors='pt', max_length=max_seq_len
#             ).to(device)
#
#             # Mixed precision for forward and backward
#             with torch.cuda.amp.autocast():
#                 user_features_embed = user_features_encoder(user_features_tok)
#                 review_content_embed = review_content_encoder(review_content_tok)
#
#                 similarity_scores = user_features_embed @ review_content_embed.T
#                 target = torch.arange(user_features_embed.shape[0], dtype=torch.long).to(device)
#                 loss = loss_fn(similarity_scores, target)
#                 running_loss += loss.item()
#
#             # Gradient scaling and optimization
#             scaler.scale(loss).backward()
#             scaler.step(optimizer)
#             scaler.update()
#             optimizer.zero_grad()
#
#         # Log loss every 5 accommodations
#         if accommodation_counter % 5 == 0:
#             avg_loss = running_loss / batch_counter
#             print(f"Average loss after {accommodation_counter} accommodations: {avg_loss:.4f}")
#
#     return user_features_encoder, review_content_encoder


