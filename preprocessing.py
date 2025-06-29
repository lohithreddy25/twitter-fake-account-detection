from pathlib import Path
import pandas as pd
from tqdm import tqdm
from transformers import RobertaTokenizer, RobertaModel
import torch
import numpy as np
import os
from torch_geometric.data import HeteroData

CRESCI_2015_PATH = Path("../../datasets/cresci-2015")

def fast_merge():
    dataset_dir = CRESCI_2015_PATH
    print("Loading data files...")

    # Load data
    node_info = pd.read_json(dataset_dir / "node.json")
    label = pd.read_csv(dataset_dir / "label.csv")
    split = pd.read_csv(dataset_dir / "split.csv")

    print(f"Loaded: {node_info.shape[0]} nodes, {label.shape[0]} labels, {split.shape[0]} splits")

    # Split users and tweets efficiently
    print("Separating users and tweets...")
    user_mask = node_info['id'].str.startswith('u', na=False)
    tweet_mask = node_info['id'].str.startswith('t', na=False)

    user = node_info[user_mask].copy()
    tweet = node_info[tweet_mask].copy()

    print(f"Found {len(user)} users and {len(tweet)} tweets")

    # Create efficient mappings
    print("Creating label mappings...")
    label_dict = dict(zip(label['id'], label['label']))
    split_dict = dict(zip(split['id'], split['split']))

    # Add labels and splits to users
    print("Adding labels and splits to users...")
    user['label'] = user['id'].map(label_dict).fillna('None')
    user['split'] = user['id'].map(split_dict).fillna('None')

    # Filter out users without labels (keep only labeled users)
    labeled_users = user[user['label'] != 'None'].copy()
    print(f"Found {len(labeled_users)} labeled users")

    print("Fast merge completed!")
    return labeled_users, tweet



@torch.no_grad()
def hetero_graph_vectorize_users_and_tweets(include_node_feature=False, include_numerical_features=False, include_categorical_features=False):
    dataset_dir = CRESCI_2015_PATH

    # Load user and tweet data
    user, tweet = fast_merge()

    # Load additional data for numerical/categorical features
    node_info = pd.read_json(dataset_dir / "node.json")
    edge = pd.read_csv(dataset_dir / "edge.csv")

    # Create mappings for users
    user_index_to_uid = list(user.id)
    uid_to_user_index = {x: i for i, x in enumerate(user_index_to_uid)}

    # Create mappings for tweets
    tweet_index_to_tid = list(tweet.id)
    tid_to_tweet_index = {x: i for i, x in enumerate(tweet_index_to_tid)}

    # Extract text data
    user_text = [text for text in user.description]
    tweet_text = [text for text in tweet.text]

    # Process numerical features if requested
    num_properties_tensor = None
    if include_numerical_features:
        print('Processing numerical properties...')

        # Extract numerical features from node_info
        followers_count = []
        following_count = []
        active_days = []
        screen_name_length = []
        statuses_count = []

        for i, each in enumerate(node_info['public_metrics']):
            if i == len(user):
                break
            if each is not None and isinstance(each, dict):
                followers_count.append(each.get('followers_count', 0))
                following_count.append(each.get('following_count', 0))
            else:
                followers_count.append(0)
                following_count.append(0)

        # Calculate active days and screen name length
        for i in range(len(user)):
            if i < len(node_info):
                # Active days calculation
                created_at = node_info.iloc[i].get('created_at')
                if created_at and pd.notna(created_at):
                    try:
                        from datetime import datetime
                        created_date = pd.to_datetime(created_at)
                        active_days.append((datetime.now() - created_date).days)
                    except:
                        active_days.append(0)
                else:
                    active_days.append(0)

                # Screen name length
                screen_name = node_info.iloc[i].get('username', '')
                screen_name_length.append(len(str(screen_name)) if screen_name else 0)

                # Status count
                public_metrics = node_info.iloc[i].get('public_metrics')
                if public_metrics and isinstance(public_metrics, dict):
                    statuses_count.append(public_metrics.get('tweet_count', 0))
                else:
                    statuses_count.append(0)
            else:
                active_days.append(0)
                screen_name_length.append(0)
                statuses_count.append(0)

        # Convert to tensors and normalize
        followers_count = torch.tensor(followers_count, dtype=torch.float32).unsqueeze(1)
        following_count = torch.tensor(following_count, dtype=torch.float32).unsqueeze(1)
        active_days = torch.tensor(active_days, dtype=torch.float32).unsqueeze(1)
        screen_name_length = torch.tensor(screen_name_length, dtype=torch.float32).unsqueeze(1)
        statuses_count = torch.tensor(statuses_count, dtype=torch.float32).unsqueeze(1)

        # Normalize features
        def normalize_tensor(tensor):
            mean = tensor.mean()
            std = tensor.std()
            return (tensor - mean) / (std + 1e-8)  # Add small epsilon to avoid division by zero

        followers_count = normalize_tensor(followers_count)
        following_count = normalize_tensor(following_count)
        active_days = normalize_tensor(active_days)
        screen_name_length = normalize_tensor(screen_name_length)
        statuses_count = normalize_tensor(statuses_count)

        # Combine all numerical features
        num_properties_tensor = torch.cat([
            followers_count, following_count, active_days,
            screen_name_length, statuses_count
        ], dim=1)

    # Process categorical features if requested
    cat_properties_tensor = None
    if include_categorical_features:
        print('Processing categorical properties...')

        # Define all categorical properties to extract
        categorical_properties = []
        properties = ['protected', 'geo_enabled', 'verified', 'contributors_enabled',
                     'is_translator', 'is_translation_enabled', 'profile_background_tile',
                     'profile_use_background_image', 'has_extended_profile',
                     'default_profile', 'default_profile_image']

        for i in range(len(user)):
            prop = []

            if i < len(node_info):
                # Get profile data for this user
                user_data = node_info.iloc[i]

                # Extract boolean properties from public_metrics or direct fields
                public_metrics = user_data.get('public_metrics', {})

                # Process each categorical property
                for property_name in properties:
                    if property_name == 'default_profile_image':
                        # Special handling for profile image
                        profile_image_url = user_data.get('profile_image_url', '')
                        if profile_image_url:
                            # Check for default profile image URLs
                            default_urls = [
                                'default_profile_0_normal.png',
                                'default_profile_1_normal.png',
                                'default_profile_2_normal.png',
                                'default_profile_3_normal.png',
                                'default_profile_4_normal.png',
                                'default_profile_5_normal.png',
                                'default_profile_6_normal.png'
                            ]
                            is_default = any(url in str(profile_image_url) for url in default_urls)
                            prop.append(1 if is_default else 0)
                        else:
                            prop.append(1)  # No image = default

                    elif property_name == 'verified':
                        # Check verification status
                        verified = user_data.get('verified', False)
                        prop.append(1 if verified else 0)

                    elif property_name == 'protected':
                        # Check if account is protected/private
                        protected = user_data.get('protected', False)
                        prop.append(1 if protected else 0)

                    elif property_name in ['geo_enabled', 'contributors_enabled', 'is_translator',
                                         'is_translation_enabled', 'profile_background_tile',
                                         'profile_use_background_image', 'has_extended_profile',
                                         'default_profile']:
                        # Try to get from user data, default to 0 if not found
                        value = user_data.get(property_name, False)
                        if isinstance(value, str):
                            prop.append(1 if value.lower() in ['true', '1', 'yes'] else 0)
                        elif isinstance(value, bool):
                            prop.append(1 if value else 0)
                        else:
                            prop.append(0)

                    else:
                        # Default case
                        prop.append(0)
            else:
                # No data available, use defaults
                prop = [0] * len(properties)
                prop[-1] = 1  # Assume default profile image if no data

            categorical_properties.append(prop)

        # Convert to tensor
        cat_properties_tensor = torch.tensor(categorical_properties, dtype=torch.float32)
        print(f"Categorical features shape: {cat_properties_tensor.shape}")

    # Load edges
    edge_types = set(edge["relation"])

    # Create heterogeneous graph
    graph = HeteroData()

    # Process different edge types
    for edge_type in edge_types:
        src = list(edge[edge["relation"] == edge_type]["source_id"])
        dst = list(edge[edge["relation"] == edge_type]["target_id"])

        if edge_type == "post":
            # User posts tweet: user -> tweet
            new_src = []
            new_dst = []

            for s, t in zip(src, dst):
                if s in uid_to_user_index and t in tid_to_tweet_index:
                    new_src.append(uid_to_user_index[s])
                    new_dst.append(tid_to_tweet_index[t])

            if new_src:  # Only add if we have valid edges
                graph["user", "posts", "tweet"].edge_index = torch.LongTensor([new_src, new_dst])

        else:
            # User-to-user relationships (follow, mention, etc.)
            new_src = []
            new_dst = []

            for s, t in zip(src, dst):
                if s in uid_to_user_index and t in uid_to_user_index:
                    new_src.append(uid_to_user_index[s])
                    new_dst.append(uid_to_user_index[t])

            if new_src:  # Only add if we have valid edges
                graph["user", edge_type, "user"].edge_index = torch.LongTensor([new_src, new_dst])

    # Create train/val/test splits
    train_uid_with_label = user[user.split == "train"][["id", "split", "label"]]
    valid_uid_with_label = user[user.split == "val"][["id", "split", "label"]]
    test_uid_with_label = user[user.split == "test"][["id", "split", "label"]]

    if include_node_feature:
        if "hetero_graph_info.pt" not in os.listdir(dataset_dir):
            print('Loading RoBERTa tokenizer & model')
            tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
            model = RobertaModel.from_pretrained("roberta-base")

            # Process user text features
            user_text_feats = []
            for t in tqdm(user_text, desc="Processing user descriptions"):
                if t is None:
                    user_text_feats.append(torch.zeros(768))
                    continue
                encoded_input = tokenizer(t, return_tensors="pt")
                user_text_feats.append(model(**encoded_input)["pooler_output"][0])

            # Process tweet text features
            tweet_text_feats = []
            for text in tqdm(tweet_text, desc="Processing tweet content"):
                if text is None:
                    tweet_text_feats.append(torch.zeros(768))
                    continue
                if type(text) == float:  # Handle NaN values
                    tweet_text_feats.append(torch.zeros(768))
                    continue
                encoded_input = tokenizer(text, return_tensors="pt")
                tweet_text_feats.append(model(**encoded_input)["pooler_output"][0])

            # Add features to graph
            graph["user"].x = torch.stack(user_text_feats, dim=0)
            graph["tweet"].x = torch.stack(tweet_text_feats, dim=0)

            # Save processed data
            hetero_info = {
                "graph": graph,
                "uid_to_user_index": uid_to_user_index,
                "tid_to_tweet_index": tid_to_tweet_index,
                "train_uid_with_label": train_uid_with_label,
                "valid_uid_with_label": valid_uid_with_label,
                "test_uid_with_label": test_uid_with_label,
                "num_properties_tensor": num_properties_tensor,
                "cat_properties_tensor": cat_properties_tensor,
            }

            torch.save(hetero_info, dataset_dir / "hetero_graph_info.pt")

            return (graph, uid_to_user_index, tid_to_tweet_index,
                   train_uid_with_label, valid_uid_with_label, test_uid_with_label,
                   num_properties_tensor, cat_properties_tensor)

        else:
            print("Loading cached heterogeneous graph data...")
            cached_data = torch.load(dataset_dir / "hetero_graph_info.pt")
            return (cached_data["graph"], cached_data["uid_to_user_index"],
                   cached_data["tid_to_tweet_index"], cached_data["train_uid_with_label"],
                   cached_data["valid_uid_with_label"], cached_data["test_uid_with_label"],
                   cached_data["num_properties_tensor"], cached_data["cat_properties_tensor"])

    else:
        return (graph, uid_to_user_index, tid_to_tweet_index,
               train_uid_with_label, valid_uid_with_label, test_uid_with_label,
               num_properties_tensor, cat_properties_tensor)

def df_to_mask(uid_with_label, uid_to_user_index, phase="train"):
    user_list = list(uid_with_label[uid_with_label.split == phase].id)
    phase_index = list(map(lambda x: uid_to_user_index[x], user_list))
    return torch.LongTensor(phase_index)

def create_labels_and_masks(user, uid_to_user_index):
    """
    Create labels tensor and train/val/test masks
    """
    # Create labels (human=0, bot=1)
    labels = list(user.label)
    labels = list(map(lambda x: 0 if x == "human" else 1, labels))
    labels = torch.LongTensor(labels)

    # Create train/val/test splits
    train_uid_with_label = user[user.split == "train"][["id", "split", "label"]]
    valid_uid_with_label = user[user.split == "val"][["id", "split", "label"]]
    test_uid_with_label = user[user.split == "test"][["id", "split", "label"]]

    # Create masks
    train_mask = df_to_mask(train_uid_with_label, uid_to_user_index, "train")
    valid_mask = df_to_mask(valid_uid_with_label, uid_to_user_index, "val")
    test_mask = df_to_mask(test_uid_with_label, uid_to_user_index, "test")

    return labels, train_mask, valid_mask, test_mask

def load_preprocessed_data():
    print("Loading existing preprocessed data...")

    processed_dir = Path("processed_data")

    # Check if all required files exist
    required_files = [
        "des_tensor.pt", "tweets_tensor.pt", "num_properties_tensor.pt",
        "cat_properties_tensor.pt", "edge_index.pt", "edge_type.pt",
        "label.pt", "train_idx.pt", "val_idx.pt", "test_idx.pt"
    ]

    missing_files = []
    for file in required_files:
        if not (processed_dir / file).exists():
            missing_files.append(file)

    if missing_files:
        print(f"Missing preprocessed files: {missing_files}")
        print("Please run preprocessing first")
        return None

    # Load all preprocessed data
    print("Loading preprocessed tensors...")
    des_tensor = torch.load(processed_dir / "des_tensor.pt")
    tweets_tensor = torch.load(processed_dir / "tweets_tensor.pt")
    num_properties_tensor = torch.load(processed_dir / "num_properties_tensor.pt")
    cat_properties_tensor = torch.load(processed_dir / "cat_properties_tensor.pt")
    edge_index = torch.load(processed_dir / "edge_index.pt")
    edge_type = torch.load(processed_dir / "edge_type.pt")
    labels = torch.load(processed_dir / "label.pt")
    train_idx = torch.load(processed_dir / "train_idx.pt")
    val_idx = torch.load(processed_dir / "val_idx.pt")
    test_idx = torch.load(processed_dir / "test_idx.pt")

    print("All preprocessed data loaded successfully!")
    print(f"Dataset: {labels.shape[0]} users, {edge_index.shape[1]} edges")
    print(f"Bots: {(labels == 1).sum().item()},  Humans: {(labels == 0).sum().item()}")
    print(f"Features: des({des_tensor.shape}), tweets({tweets_tensor.shape})")
    print(f"Numerical({num_properties_tensor.shape}), Categorical({cat_properties_tensor.shape})")

    return (des_tensor, tweets_tensor, num_properties_tensor, cat_properties_tensor,
            edge_index, edge_type, labels, train_idx, val_idx, test_idx)

def complete_bot_detection_preprocessing(include_all_features=True):

    # Try to load existing preprocessed data first
    result = load_preprocessed_data()
    if result is not None:
        return result

    # Fallback to processing if no preprocessed data exists
    print("No preprocessed data found, starting fresh preprocessing...")

    # Get all data with features
    (graph, uid_to_user_index, tid_to_tweet_index,
     train_uid_with_label, valid_uid_with_label, test_uid_with_label,
     num_properties_tensor, cat_properties_tensor) = hetero_graph_vectorize_users_and_tweets(
        include_node_feature=include_all_features,
        include_numerical_features=include_all_features,
        include_categorical_features=include_all_features
    )

    # Get user data for labels
    user, _ = fast_merge()

    # Create labels and masks
    labels, train_idx, val_idx, test_idx = create_labels_and_masks(user, uid_to_user_index)

    # Extract features from graph
    des_tensor = graph["user"].x if include_all_features else None
    tweets_tensor = graph["tweet"].x if include_all_features else None

    # Create simple edge structure for compatibility
    edge_index_list = []
    edge_type_list = []

    # Add user-user edges
    for edge_key, edge_data in graph.edge_index_dict.items():
        if edge_key[0] == "user" and edge_key[2] == "user":
            edge_index_list.append(edge_data)
            edge_type_list.extend([0] * edge_data.shape[1])  # User-user edges = type 0

    if edge_index_list:
        edge_index = torch.cat(edge_index_list, dim=1)
        edge_type = torch.tensor(edge_type_list, dtype=torch.long)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_type = torch.empty(0, dtype=torch.long)

    print("Preprocessing completed!")
    print(f"Dataset: {labels.shape[0]} users, {edge_index.shape[1]} edges")
    print(f"Bots: {(labels == 1).sum().item()}, Humans: {(labels == 0).sum().item()}")

    return (des_tensor, tweets_tensor, num_properties_tensor, cat_properties_tensor,
            edge_index, edge_type, labels, train_idx, val_idx, test_idx)

if __name__ == "__main__":
    # Test complete preprocessing
    print("Testing complete bot detection preprocessing:")
    try:
        result = complete_bot_detection_preprocessing(include_all_features=True)
        print("Complete preprocessing test successful!")
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        import traceback
        traceback.print_exc()