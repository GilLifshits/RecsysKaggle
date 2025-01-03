from main import prepare_data
import pandas as pd
import os
from tqdm import tqdm

# Prepare data
train_users, train_reviews, train_matches, val_users, val_reviews, val_matches, test_users, test_reviews = prepare_data(1)

# Create directory for saving CSVs
output_dir = "in_accommodation_datasets"
os.makedirs(output_dir, exist_ok=True)

# Grouping by accommodation_id for train_users
grouped_train_users = train_users.groupby("accommodation_id")

# Creating a DataFrame per accommodation_id for train_users
grouped_train_users_dict = {
    accommodation_id: group.reset_index(drop=True)
    for accommodation_id, group in grouped_train_users
}

# Grouping by accommodation_id for train_review
grouped_train_review = train_reviews.groupby("accommodation_id")

# Creating a DataFrame per accommodation_id for train_review
grouped_train_review_dict = {
    accommodation_id: group.reset_index(drop=True)
    for accommodation_id, group in grouped_train_review
}

# Grouping by accommodation_id for train_matches
grouped_train_matches = train_matches.groupby("accommodation_id")

# Creating a DataFrame per accommodation_id for train_matches
grouped_train_matches_dict = {
    accommodation_id: group.reset_index(drop=True)
    for accommodation_id, group in grouped_train_matches
}

# Save each grouped DataFrame as a CSV for train_users
for accommodation_id, df in tqdm(grouped_train_users_dict.items(), desc="Saving train_users CSVs"):
    df.to_csv(os.path.join(output_dir, f"train_users_{accommodation_id}.csv"), index=False)

# Save each grouped DataFrame as a CSV for train_review
for accommodation_id, df in tqdm(grouped_train_review_dict.items(), desc="Saving train_review CSVs"):
    df.to_csv(os.path.join(output_dir, f"train_review_{accommodation_id}.csv"), index=False)

# Save each grouped DataFrame as a CSV for train_matches
for accommodation_id, df in tqdm(grouped_train_matches_dict.items(), desc="Saving train_matches CSVs"):
    df.to_csv(os.path.join(output_dir, f"train_matches_{accommodation_id}.csv"), index=False)

# Example: Accessing a specific accommodation_id DataFrame from train_users
example_accommodation_id = -1109473678
train_users_accommodation_df = grouped_train_users_dict.get(example_accommodation_id)

# Print the DataFrame for the example accommodation_id
print(f"Train Users DataFrame for accommodation_id {example_accommodation_id}:\n")
print(train_users_accommodation_df)