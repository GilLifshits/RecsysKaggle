
# Recsys Kaggle competition

**Files Structure**:

- **prepare_in_accommodation_dataset.py**:
  - Creating a DataFrame per accommodation_id for (train_users/train_review/train_matches)
  - This is used to train on batches that are within the same accommodation, and not randomly sampled. 


- **main.py**: 
  - Pick if you want only inference or also training.
  - Assuming that prepare_in_accommodation_dataset is finished (created the datasets)


- **DataLoader.py**:
  - simple file to load train/val/test data


- **ReviewDataset.py**:
  - For each match of user and review, returns user_features_textual_representation and review_textual_representation.


- **ContrastiveModel.py**:
  - Defines the Contrastive Transformer Model and forward pass.
  - More details in presentation.


- **training_loop.py**:
  - Pre-load CSV files (per accommodation id) to memory
  - Train both encoders using contrastive learning for 5 epochs on all the training data.


- **test_prediction.py**:
  - Embed all user features
  - Embed all reviews
  - Find top 10 reviews for each (user, accommodation) using cosine similarity