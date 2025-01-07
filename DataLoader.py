import pandas as pd


def load_data(train_path, val_path, test_path):
    train_users = pd.read_csv(f"{train_path}/train_users.csv")
    train_reviews = pd.read_csv(f"{train_path}/train_reviews.csv")
    train_matches = pd.read_csv(f"{train_path}/train_matches.csv")
    print("Loaded training data")

    val_users = pd.read_csv(f"{val_path}/val_users.csv")
    val_reviews = pd.read_csv(f"{val_path}/val_reviews.csv")
    val_matches = pd.read_csv(f"{val_path}/val_matches.csv")
    print("Loaded validation data")

    test_users = pd.read_csv(f"{test_path}/test_users.csv")
    test_reviews = pd.read_csv(f"{test_path}/test_reviews.csv")
    print("Loaded test data")

    return (train_users, train_reviews, train_matches,
            val_users, val_reviews, val_matches,
            test_users, test_reviews)
