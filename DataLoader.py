import pandas as pd


def load_data(train_path, val_path, test_path, frac_of_train_set):
    train_users = pd.read_csv(f"{train_path}/train_users.csv")
    train_reviews = pd.read_csv(f"{train_path}/train_reviews.csv")
    train_matches = pd.read_csv(f"{train_path}/train_matches.csv")

    train_users = train_users.sample(frac=frac_of_train_set, random_state=42).reset_index(drop=True)
    train_reviews = train_reviews.sample(frac=frac_of_train_set, random_state=42).reset_index(drop=True)
    train_matches = train_matches.sample(frac=frac_of_train_set, random_state=42).reset_index(drop=True)
    print(f"Loaded {frac_of_train_set} of the training data")

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
