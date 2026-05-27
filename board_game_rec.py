########################################################################################
# Board Game Recommender
# Created by David Hatchett
# Created on: 2026-02-14
#
# Description:
# This script is used to train a board game recommender model.
# It uses a neural network to recommend board games to users based on their ratings
# of other board games. If this code is called normally it will run a training pass
# of the model. If the code is loaded as a module it will not run a training pass,
# but other functions and classes will be usable.
#
# This is based on the model created here:
#   https://pureai.substack.com/p/recommender-systems-with-pytorch
# and here:
#   https://www.youtube.com/watch?v=cqnrFrF3nJ8
# It uses an NCF model originally described here:
#   https://arxiv.org/abs/1708.05031
#
# AI was used to help fix errors in some of the functions, however most of the code
# was written by me.
########################################################################################

import contextlib
import torch
import pandas as pd
import sys
import matplotlib.pyplot as plt
import numpy as np
import ast
import json
import os
import joblib

import torchmetrics.functional as tmf

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler

# Detect device once at module level so it is not recomputed throughout the code.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BoardGameRecommender(nn.Module):
    '''
    Defines the neural network model used for board game recommendations.

    Parameters:
    - num_users: the number of users in the dataset
    - num_games: the number of games in the dataset
    - num_categories: the number of categories in the dataset
    - num_mechanics: the number of mechanics in the dataset
    - dropout_rate: the dropout rate for the neural network
    - embedding_user_dim: the dimension of the user embedding
    - embedding_game_dim: the dimension of the game embedding
    - embedding_category_dim: the dimension of the category embedding
    - embedding_mechanic_dim: the dimension of the mechanic embedding
    - hidden_dim: the dimension of the hidden layer

    Returns:
    - a tensor of the predicted ratings
    '''
    def __init__(
        self,
        num_users,
        num_games,
        num_categories,
        num_mechanics,
        dropout_rate=0.2,
        embedding_user_dim=512,
        embedding_game_dim=128,
        embedding_category_dim=32,
        embedding_mechanic_dim=64,
        hidden_dim=256,
    ):
        super(BoardGameRecommender, self).__init__()

        # Persist hyperparams so checkpoints can be self-describing.
        # load_assets() in recommend.py reads these back to reconstruct the
        # model without relying on hardcoded defaults.
        self.hparams = {
            "num_users":              num_users,
            "num_games":              num_games,
            "num_categories":         num_categories,
            "num_mechanics":          num_mechanics,
            "dropout_rate":           dropout_rate,
            "embedding_user_dim":     embedding_user_dim,
            "embedding_game_dim":     embedding_game_dim,
            "embedding_category_dim": embedding_category_dim,
            "embedding_mechanic_dim": embedding_mechanic_dim,
            "hidden_dim":             hidden_dim,
        }

        # Number of scaled numeric features passed through the forward method:
        # avg_usr_rating, avg_usr_weight, bayes_average, age, game_owners
        self.num_numeric_features = 5

        # Embedding layers
        self.user_embedding = nn.Embedding(num_users, embedding_user_dim)
        self.game_embedding = nn.Embedding(num_games, embedding_game_dim)
        self.category_embedding = nn.EmbeddingBag(num_categories, embedding_category_dim, mode="mean")
        self.mechanic_embedding = nn.EmbeddingBag(num_mechanics, embedding_mechanic_dim, mode="mean")

        self.embedding_dim = (
            embedding_user_dim
            + embedding_game_dim
            + embedding_category_dim
            + embedding_mechanic_dim
            + self.num_numeric_features
        )

        self.dropout = nn.Dropout(dropout_rate)
        # Pyramid MLP: embedding_dim → embedding_dim → hidden_dim → hidden_dim//2 → 1
        # The first layer is same-size (no compression) to let the model mix
        # embedding signals freely before the pyramid begins.
        self.fc1 = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.fc2 = nn.Linear(self.embedding_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc4 = nn.Linear(hidden_dim // 2, 1)
        self.relu = nn.ReLU()

    def forward(
        self,
        user_id,
        game_id,
        avg_usr_rating,
        avg_usr_weight,
        bayes_average,
        age,
        game_owners,
        category_indices,
        category_offsets,
        mechanic_indices,
        mechanic_offsets,
    ):
        """
        Forward pass for the BoardGameRecommender model.

        Parameters:
        - user_id: the id of the user
        - game_id: the id of the game
        - avg_usr_rating: the average user rating of the game
        - avg_usr_weight: the average user weight of the game
        - bayes_average: the bayes average of the game
        - age: the age of the game
        - game_owners: the number of owners of the game
        - category_indices: the indices of the categories of the game
        - category_offsets: the offsets of the categories of the game
        - mechanic_indices: the indices of the mechanics of the game
        - mechanic_offsets: the offsets of the mechanics of the game
        """
        x = torch.cat([
            self.user_embedding(user_id),
            self.game_embedding(game_id),
            self.category_embedding(category_indices, category_offsets),
            self.mechanic_embedding(mechanic_indices, mechanic_offsets),
            avg_usr_rating.unsqueeze(1),
            avg_usr_weight.unsqueeze(1),
            bayes_average.unsqueeze(1),
            age.unsqueeze(1),
            game_owners.unsqueeze(1),
        ], dim=1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        return self.fc4(x)


class UserGameDataSet(Dataset):
    '''
    Dataset class for game data and user ratings.
    Used to batch the data for the DataLoader.
    Must be used alongside the collate_fn function to correctly batch
    variable-length category and mechanic index lists.
    '''
    def __init__(self, data: pd.DataFrame):
        self.users_id = data["user_id"]
        self.game_id = data["game_id_encoded"]
        self.user_rating = data["user_rating"]
        self.avg_usr_rating = data["avg_usr_rating_scaled"]
        self.avg_usr_weight = data["avg_usr_weight_scaled"]
        self.bayes_average = data["bayes_average_scaled"]
        self.age = data["age_scaled"]
        self.game_owners = data["game_owners_scaled"]
        self.category_indices = data["category_indices"]
        self.mechanic_indices = data["mechanic_indices"]

    def __len__(self):
        return len(self.users_id)

    def __getitem__(self, idx):
        return {
            "users_id":      torch.tensor(self.users_id.iloc[idx],      dtype=torch.long),
            "game_id":       torch.tensor(self.game_id.iloc[idx],       dtype=torch.long),
            "user_rating":   torch.tensor(self.user_rating.iloc[idx],   dtype=torch.float32),
            "avg_usr_rating":torch.tensor(self.avg_usr_rating.iloc[idx],dtype=torch.float32),
            "avg_usr_weight":torch.tensor(self.avg_usr_weight.iloc[idx],dtype=torch.float32),
            "bayes_average": torch.tensor(self.bayes_average.iloc[idx], dtype=torch.float32),
            "age":           torch.tensor(self.age.iloc[idx],           dtype=torch.float32),  # scaled float
            "game_owners":   torch.tensor(self.game_owners.iloc[idx],   dtype=torch.float32),  # scaled float
            "category_indices": torch.tensor(self.category_indices.iloc[idx], dtype=torch.long),
            "mechanic_indices": torch.tensor(self.mechanic_indices.iloc[idx], dtype=torch.long),
        }


def collate_fn(batch):
    '''
    Collate function that prepares a batch for the neural network.
    Builds the flat index tensors and offset tensors required by EmbeddingBag.
    '''
    category_indices, category_offsets = get_embedding_bag(batch, 'category_indices')
    mechanic_indices, mechanic_offsets = get_embedding_bag(batch, 'mechanic_indices')

    return {
        "users_id":        torch.stack([b["users_id"]        for b in batch]).to(DEVICE),
        "game_id":         torch.stack([b["game_id"]         for b in batch]).to(DEVICE),
        "user_rating":     torch.stack([b["user_rating"]     for b in batch]).to(DEVICE),
        "avg_usr_rating":  torch.stack([b["avg_usr_rating"]  for b in batch]).to(DEVICE),
        "avg_usr_weight":  torch.stack([b["avg_usr_weight"]  for b in batch]).to(DEVICE),
        "bayes_average":   torch.stack([b["bayes_average"]   for b in batch]).to(DEVICE),
        "age":             torch.stack([b["age"]             for b in batch]).to(DEVICE),
        "game_owners":     torch.stack([b["game_owners"]     for b in batch]).to(DEVICE),
        "category_indices": category_indices.to(DEVICE),
        "category_offsets": category_offsets.to(DEVICE),
        "mechanic_indices": mechanic_indices.to(DEVICE),
        "mechanic_offsets": mechanic_offsets.to(DEVICE),
    }


def get_embedding_bag(batch, field: str) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Creates the flat index tensor and offset tensor needed by EmbeddingBag.
    Takes a batch of data and a field name; returns (indices, offsets).
    """
    indices = []
    offsets = []
    offset = 0

    for row in batch:
        offsets.append(offset)
        tokens = row[field].tolist()
        indices.extend(tokens)
        offset += len(tokens)

    return (
        torch.tensor(indices, dtype=torch.long),
        torch.tensor(offsets, dtype=torch.long),
    )


def json_out(file_name: str, data: dict):
    '''
    Saves a dictionary to a JSON file.
    '''
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=True, indent=4)


def get_vocab(series: pd.Series) -> dict:
    '''
    Creates a vocabulary mapping for a Series of lists.
    Strips whitespace, removes duplicates via a set, then maps each token to an integer.
    '''
    out_set = set()
    for record in series:
        for token in record:
            out_set.add(token.strip())
    return {token: i for i, token in enumerate(sorted(out_set))}


def encode_text(series: pd.Series) -> dict:
    '''
    Encodes a text Series into a dictionary mapping each unique value to an integer.
    Keys are always stored as strings so they survive a JSON round-trip without
    type changes (json.dump silently converts integer keys to strings).
    '''
    return {str(item): i for i, item in enumerate(sorted(set(series)))}


def create_encoder(file_name: str, data: pd.Series = None, field: str = None) -> dict:
    '''
    Creates or loads an encoder for a given field.
    If the encoder file already exists it will be loaded; otherwise it will be
    created from `data` and saved to `file_name`.

    added an else statement and raise for if no data and no field vars where passed.
    '''
    if os.path.exists(file_name):
        with open(file_name, "r", encoding="utf-8") as f:
            return json.load(f)

    if data is not None and field is not None:
        if field in ("category", "mechanic"):
            vocab = get_vocab(data)
        else:
            vocab = encode_text(data)
        with open(file_name, "w", encoding="utf-8") as f:
            json.dump(vocab, f, ensure_ascii=True)
        return vocab
    else:
        raise ValueError("data and field are required to create a new encoder") 


def process_game_data(file_name: str) -> pd.DataFrame:
    '''
    Loads and cleans the raw game CSV file.
    Returns a DataFrame ready for encoding and scaling.
    Raises FileNotFoundError if the file does not exist.
    '''
    if not os.path.exists(file_name):
        raise FileNotFoundError(f"Game data file not found: {file_name}")

    df = pd.read_csv(file_name)
    df = df[[
        'id', 'name', 'yearpublished', 'boardgamecategory',
        'boardgamemechanic', 'average', 'bayesaverage',
        'owned', 'averageweight',
    ]].copy()

    df.rename(columns={
        'id':                'game_id',
        'name':              'game_name',
        'yearpublished':     'year_published',
        'boardgamecategory': 'category',
        'boardgamemechanic': 'mechanic',
        'average':           'avg_usr_rating',
        'owned':             'game_owners',
        'averageweight':     'avg_usr_weight',
        'bayesaverage':      'bayes_average',
    }, inplace=True)
    df.dropna(inplace=True)

    df['game_id'] = df['game_id'].astype(int)
    df['age'] = df['year_published'].apply(lambda x: 2025 - x if x >= 0 else x * -1)
    df['avg_usr_weight'] = df['avg_usr_weight'].replace(0, np.nan)
    df['avg_usr_weight'] = df['avg_usr_weight'].fillna(df['avg_usr_weight'].mean())
    df['category'] = df['category'].apply(ast.literal_eval)
    df['mechanic'] = df['mechanic'].apply(ast.literal_eval)

    return df


def prep_game_data(
    game_data: pd.DataFrame,
    game_encoder: dict,
    category_encoder: dict,
    mechanic_encoder: dict,
) -> tuple[pd.DataFrame, dict]:
    '''
    Applies encoding and StandardScaler normalization to the game DataFrame.

    Fits a separate StandardScaler for each numeric column and returns both the
    processed DataFrame and a dictionary of fitted scalers.  The scalers must be
    saved alongside the model so the same transformation can be applied at
    inference time.
    '''
    game_data = game_data.copy()
    # game_encoder keys are strings (JSON round-trip converts int keys to str),
    # so cast game_id to str before mapping to avoid silent NaN from type mismatch.
    game_data["game_id_encoded"] = game_data["game_id"].astype(str).map(game_encoder)

    game_data["category_indices"] = game_data["category"].apply(
        lambda x: [category_encoder[item] for item in x]
    )
    game_data["mechanic_indices"] = game_data["mechanic"].apply(
        lambda x: [mechanic_encoder[item] for item in x]
    )

    numeric_cols = ['age', 'avg_usr_weight', 'avg_usr_rating', 'bayes_average', 'game_owners']
    scalers = {}
    for col in numeric_cols:
        scaler = StandardScaler()
        game_data[f'{col}_scaled'] = scaler.fit_transform(game_data[[col]]).ravel()
        scalers[col] = scaler

    return game_data, scalers


def get_game_data(config: dict) -> pd.DataFrame:
    '''
    Orchestrates game data loading, encoding, and scaling.
    Caches the processed DataFrame and fitted scalers to disk so subsequent runs
    skip expensive reprocessing.
    '''
    game_data_file       = config["data"]["games"]
    game_data_model_path = config["data_model"]["game_data_model"]
    game_encoder_path    = config["encoders"]["game_name"]
    category_encoder_path= config["encoders"]["category"]
    mechanic_encoder_path= config["encoders"]["mechanic"]
    scalers_path         = config["models"]["scalers"]

    print("Process game data")
    cache_is_valid = False
    if os.path.exists(game_data_model_path):
        game_data = pd.read_csv(game_data_model_path)
        # Guard: if game_id_encoded is entirely NaN the cache is stale
        # (written before the str-key encoder fix). Wipe it and start fresh.
        if "game_id_encoded" not in game_data.columns or game_data["game_id_encoded"].isna().all():
            print("WARNING: cached game_data_model.csv has invalid game_id_encoded — regenerating.")
            os.remove(game_data_model_path)
            game_data = process_game_data(game_data_file)
        else:
            cache_is_valid = True
    else:
        game_data = process_game_data(game_data_file)

    print("Create encoders")
    game_encoder     = create_encoder(game_encoder_path,     game_data['game_id'],  "game_id")
    category_encoder = create_encoder(category_encoder_path, game_data['category'], "category")
    mechanic_encoder = create_encoder(mechanic_encoder_path, game_data['mechanic'], "mechanic")

    print("Prep game data for training")
    if not cache_is_valid:
        game_data, scalers = prep_game_data(
            game_data, game_encoder, category_encoder, mechanic_encoder
        )
        game_data.to_csv(game_data_model_path, index=False)
        joblib.dump(scalers, scalers_path)
        print(f"Scalers saved to {scalers_path}")

    return game_data


def process_user_data(user_data_file: str) -> pd.DataFrame:
    '''
    Loads and cleans the raw user-ratings CSV file.
    Raises FileNotFoundError if the file does not exist.
    '''
    if not os.path.exists(user_data_file):
        raise FileNotFoundError(f"User data file not found: {user_data_file}")

    user_data = pd.read_csv(user_data_file, usecols=['ID', 'user', 'rating'])
    user_data.rename(columns={'ID': 'game_id', 'rating': 'user_rating'}, inplace=True)
    user_data['game_id'] = user_data['game_id'].astype(int)
    return user_data


def prep_user_data(user_data: pd.DataFrame, user_encoder: dict) -> pd.DataFrame:
    '''
    Maps raw user name strings to encoded integer IDs and returns the DataFrame.
    '''
    user_data = user_data.copy()
    user_data["user_id"] = user_data["user"].map(user_encoder)
    return user_data


def get_user_data(config: dict) -> pd.DataFrame:
    '''
    Orchestrates user data loading, encoding, and caching.
    '''
    user_data_file      = config["data"]["users"]
    user_data_model_path= config["data_model"]["user_data_model"]
    user_encoder_path   = config["encoders"]["user_id"]

    if not os.path.exists(user_data_model_path):
        user_data = process_user_data(user_data_file)
    else:
        user_data = pd.read_csv(user_data_model_path)

    user_encoder = create_encoder(user_encoder_path, user_data['user'].unique(), "user")

    print("Prep user data for training")
    if not os.path.exists(user_data_model_path):
        user_data = prep_user_data(user_data, user_encoder)
        user_data.dropna(inplace=True)
        user_data.to_csv(user_data_model_path, index=False)

    return user_data


def setup_config(config_file: str) -> dict:
    '''
    Loads and returns the JSON config dictionary.
    '''
    with open(config_file, "r", encoding="utf-8") as f:
        return json.load(f)


def create_train_data(
    game_data: pd.DataFrame,
    user_data: pd.DataFrame,
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    '''
    Merges game and user data then splits into train, validation, and test sets.
    Saves each split to the paths defined in the config.
    '''
    print("Merge game and user data")
    game_data_model = pd.merge(user_data, game_data, on='game_id', how='inner')
    game_data_model = game_data_model[[
        'user_id',
        'game_id_encoded',
        'user_rating',
        'avg_usr_rating_scaled',
        'avg_usr_weight_scaled',
        'bayes_average_scaled',
        'age_scaled',
        'game_owners_scaled',
        'category_indices',
        'mechanic_indices',
    ]]
    game_data_model.dropna(inplace=True)
    print(game_data_model.shape)

    # List columns are strings when loaded from CSV but already lists when passed
    # directly from prep_game_data; handle both cases.
    def ensure_list(val):
        return val if isinstance(val, list) else ast.literal_eval(val)

    game_data_model['category_indices'] = game_data_model['category_indices'].apply(ensure_list)
    game_data_model['mechanic_indices'] = game_data_model['mechanic_indices'].apply(ensure_list)

    print("Split into train, validation, and test sets")
    train_data, test_data       = model_selection.train_test_split(game_data_model, test_size=0.2,  random_state=42)
    train_data, validation_data = model_selection.train_test_split(train_data,      test_size=0.2,  random_state=42)

    print("Save train, validation, and test sets")
    train_data.to_csv(     config["data_model"]["train_data_path"],      index=False)
    validation_data.to_csv(config["data_model"]["validation_data_path"], index=False)
    test_data.to_csv(      config["data_model"]["test_data_path"],       index=False)

    return train_data, validation_data, test_data


def get_data_loaders(
    train_data: pd.DataFrame,
    validation_data: pd.DataFrame,
    test_data: pd.DataFrame,
    batch_size: int = 100,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    '''
    Creates and returns DataLoaders for the train, validation, and test sets.
    '''
    print("Create datasets")
    train_dataset      = UserGameDataSet(train_data)
    validation_dataset = UserGameDataSet(validation_data)
    test_dataset       = UserGameDataSet(test_data)

    print("Create data loaders")
    train_loader      = DataLoader(train_dataset,      batch_size=batch_size, shuffle=True,  num_workers=num_workers, collate_fn=collate_fn)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
    test_loader       = DataLoader(test_dataset,       batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
    return train_loader, validation_loader, test_loader


def get_encoders(config: dict) -> tuple[dict, dict, dict, dict]:
    '''
    Loads and returns the four encoder dictionaries from disk.
    '''
    with open(config["encoders"]["user_id"],   "r", encoding="utf-8") as f:
        user_encoder = json.load(f)
    with open(config["encoders"]["game_name"], "r", encoding="utf-8") as f:
        game_encoder = json.load(f)
    with open(config["encoders"]["category"],  "r", encoding="utf-8") as f:
        category_encoder = json.load(f)
    with open(config["encoders"]["mechanic"],  "r", encoding="utf-8") as f:
        mechanic_encoder = json.load(f)
    return user_encoder, game_encoder, category_encoder, mechanic_encoder


def log_progress(epoch, epochs, step_count, avg_loss, avg_nrmse, avg_mae, avg_r2, data_size):
    '''
    Writes a single-line progress update to stderr using a carriage return so it
    overwrites the previous line in the terminal.
    Accepts pre-computed running averages so this function is always O(1).
    '''
    sys.stderr.write(
        f"\r{epoch+1:02d}/{epochs:02d} | Step: {step_count}/{data_size}"
        f" | Loss: {avg_loss:<10.6f}"
        f" | NRMSE: {avg_nrmse:<8.6f}"
        f" | MAE: {avg_mae:<8.6f}"
        f" | R²: {avg_r2:<8.6f}"
    )
    sys.stderr.flush()


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    epoch: int,
    epochs: int,
    log_progress_step: int,
    optimizer: torch.optim.Optimizer = None,
) -> tuple[float, float, float, float]:
    '''
    Runs one full pass over `loader` and returns epoch-average metrics.

    If `optimizer` is provided the model is set to train mode and weights are
    updated after each batch.  If `optimizer` is None the model runs in eval
    mode under torch.no_grad() — suitable for validation and test passes.

    Maintains O(1) running sums so neither logging nor the final averages
    require iterating over accumulated lists.

    Returns:
        (avg_loss, avg_nrmse, avg_mae, avg_r2) — epoch averages as floats.
    '''
    is_training = optimizer is not None
    model.train() if is_training else model.eval()

    total_loss  = 0.0
    total_nrmse = 0.0
    total_mae   = 0.0
    total_r2    = 0.0
    step_count  = 0
    data_size   = len(loader)

    # nullcontext() lets us write the loop once regardless of train vs eval.
    grad_ctx = contextlib.nullcontext() if is_training else torch.no_grad()
    with grad_ctx:
        for batch in loader:
            if is_training:
                optimizer.zero_grad()

            x = model(
                user_id          = batch["users_id"],
                game_id          = batch["game_id"],
                avg_usr_rating   = batch["avg_usr_rating"],
                avg_usr_weight   = batch["avg_usr_weight"],
                bayes_average    = batch["bayes_average"],
                age              = batch["age"],
                game_owners      = batch["game_owners"],
                category_indices = batch["category_indices"],
                category_offsets = batch["category_offsets"],
                mechanic_indices = batch["mechanic_indices"],
                mechanic_offsets = batch["mechanic_offsets"],
            ).squeeze()

            out_true = batch["user_rating"].to(torch.float32)
            loss = criterion(x, out_true)
            total_loss += loss.item()

            if is_training:
                loss.backward()
                optimizer.step()

            total_nrmse += tmf.normalized_root_mean_squared_error(x, out_true, normalization='range').item()
            total_mae   += tmf.mean_absolute_error(x, out_true).item()
            total_r2    += tmf.r2_score(x, out_true).item()

            # Log every log_progress_step batches using the running average
            # up to this point.  step_count+1 is the number of batches seen.
            if step_count % log_progress_step == 0:
                n = step_count + 1
                log_progress(
                    epoch, epochs, step_count,
                    total_loss / n, total_nrmse / n, total_mae / n, total_r2 / n,
                    data_size,
                )
            step_count += 1

    return total_loss / data_size, total_nrmse / data_size, total_mae / data_size, total_r2 / data_size


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    config: dict,
    epochs: int = 10,
    learning_rate: float = 0.001,
    weight_decay: float = 0.0001,
) -> tuple[nn.Module, dict]:
    '''
    Trains the model and saves a checkpoint after each epoch.
    Returns the trained model and a history dictionary of per-epoch metrics.
    Tracks NRMSE, MAE, and R² to match the evaluation metrics used in the
    results notebook.

    Uses ReduceLROnPlateau to halve the learning rate when validation loss
    has not improved for `lr_patience` consecutive epochs, helping the model
    escape the early plateau seen in longer training runs.
    '''
    optimizer  = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode      = 'min',   # reduce when monitored metric stops decreasing
        factor    = 0.5,     # halve the LR on each trigger
        patience  = 2,       # wait 2 epochs of no improvement before reducing
        min_lr    = 1e-6,    # floor so LR never reaches zero
    )
    criterion         = nn.MSELoss()
    log_progress_step = 50

    history = {
        "train_loss":      [],
        "validation_loss": [],
        "train_nrmse":     [],
        "train_mae":       [],
        "train_r2":        [],
        "validation_nrmse":[],
        "validation_mae":  [],
        "validation_r2":   [],
        "learning_rate":   [],
    }

    print(f"Training model for {epochs} epochs")
    for epoch in range(epochs):
        avg_train_loss, avg_train_nrmse, avg_train_mae, avg_train_r2 = _run_epoch(
            model, train_loader, criterion,
            epoch, epochs, log_progress_step, optimizer,
        )
        avg_val_loss, avg_val_nrmse, avg_val_mae, avg_val_r2 = _run_epoch(
            model, validation_loader, criterion,
            epoch, epochs, log_progress_step,
        )

        # Step the scheduler on validation loss; must come before reading the LR
        # so the history records the LR that will be used next epoch.
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch+1}/{epochs} — Train Loss: {avg_train_loss:.6f}  Val Loss: {avg_val_loss:.6f}"
              f"  Val NRMSE: {avg_val_nrmse:.4f}  Val MAE: {avg_val_mae:.4f}"
              f"  Val R²: {avg_val_r2:.4f}  LR: {current_lr:.2e}")

        history["train_loss"].append(avg_train_loss)
        history["validation_loss"].append(avg_val_loss)
        history["train_nrmse"].append(avg_train_nrmse)
        history["train_mae"].append(avg_train_mae)
        history["train_r2"].append(avg_train_r2)
        history["validation_nrmse"].append(avg_val_nrmse)
        history["validation_mae"].append(avg_val_mae)
        history["validation_r2"].append(avg_val_r2)
        history["learning_rate"].append(current_lr)

        torch.save(
            {"state_dict": model.state_dict(), "hparams": model.hparams},
            config["models"]["recommender"].format(epoch + 1),
        )

    return model, history


def main():

    print("Setup config")
    config = setup_config("config/config.json")

    # ── Data preparation ─────────────────────────────────────────────────────
    print("Check if train, validation, and test sets exist")
    if (
        os.path.exists(config["data_model"]["train_data_path"])
        and os.path.exists(config["data_model"]["validation_data_path"])
        and os.path.exists(config["data_model"]["test_data_path"])
    ):
        print("Loading cached train / validation / test sets")
        train_data      = pd.read_csv(config["data_model"]["train_data_path"])
        validation_data = pd.read_csv(config["data_model"]["validation_data_path"])
        test_data       = pd.read_csv(config["data_model"]["test_data_path"])

        for df in (train_data, validation_data, test_data):
            df['category_indices'] = df['category_indices'].apply(ast.literal_eval)
            df['mechanic_indices'] = df['mechanic_indices'].apply(ast.literal_eval)
    else:
        print("Get Game Data")
        game_data = get_game_data(config)

        print("Get User Data")
        user_data = get_user_data(config)

        print("Create Train Data")
        train_data, validation_data, test_data = create_train_data(game_data, user_data, config)
        del game_data, user_data

    # ── DataLoaders ──────────────────────────────────────────────────────────
    print("Get Data Loaders")
    train_loader, validation_loader, test_loader = get_data_loaders(
        train_data, validation_data, test_data, batch_size=1000
    )
    del train_data, validation_data, test_data

    # ── Encoders & model ─────────────────────────────────────────────────────
    print("Get Encoders")
    user_encoder, game_encoder, category_encoder, mechanic_encoder = get_encoders(config)

    print(f"Instantiate model (device: {DEVICE})")
    model = BoardGameRecommender(
        num_users       = len(user_encoder),
        num_games       = len(game_encoder),
        num_categories  = len(category_encoder),
        num_mechanics   = len(mechanic_encoder),
        dropout_rate           = 0.2,
        embedding_user_dim     = 512,
        embedding_game_dim     = 128,
        embedding_category_dim = 32,
        embedding_mechanic_dim = 64,
        hidden_dim             = 256,
    ).to(DEVICE)

    # ── Training ─────────────────────────────────────────────────────────────
    print("Train model")
    model, history = train_model(
        model             = model,
        train_loader      = train_loader,
        validation_loader = validation_loader,
        config            = config,
        epochs            = 10,
        learning_rate     = 0.001,
        weight_decay      = 0.0001,
    )

    print("Save history")
    with open(config["models"]["history"], "w", encoding="utf-8") as f:
        json.dump(history, f)


if __name__ == "__main__":
    main()
