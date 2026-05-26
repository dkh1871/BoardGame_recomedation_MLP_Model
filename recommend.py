########################################################################################
# Board Game Recommender — Inference Script
# Created by David Hatchett
#
# Usage:
#   python recommend.py <username>
#   python recommend.py <username> --top-k 20
#   python recommend.py <username> --epoch 5 --include-rated
#   python recommend.py --list-users          (print sample usernames and exit)
########################################################################################

import argparse
import ast
import sys
import torch
import pandas as pd

from torch.utils.data import DataLoader

import board_game_rec as bgr


# ── Asset loading ─────────────────────────────────────────────────────────────

def load_assets(config: dict, epoch: int, device: torch.device) -> tuple:
    """
    Loads encoders, game catalogue, user ratings, and the trained model.
    Returns (user_encoder, game_data, user_data, model).
    """
    user_encoder, game_encoder, category_encoder, mechanic_encoder = bgr.get_encoders(config)

    # Game catalogue — already encoded and scaled; parse list columns from CSV strings
    game_data = pd.read_csv(config["data_model"]["game_data_model"])
    game_data["category_indices"] = game_data["category_indices"].apply(ast.literal_eval)
    game_data["mechanic_indices"] = game_data["mechanic_indices"].apply(ast.literal_eval)

    # User ratings — used to exclude already-rated games
    user_data = pd.read_csv(config["data_model"]["user_data_model"])

    # Model
    checkpoint_path = config["models"]["recommender"].format(epoch)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "hparams" in checkpoint:
        # New-style checkpoint: self-describing with saved hyperparams
        hparams    = checkpoint["hparams"]
        state_dict = checkpoint["state_dict"]
    else:
        # Backward-compatible fallback for old bare state_dict checkpoints
        hparams = dict(
            num_users      = len(user_encoder),
            num_games      = len(game_encoder),
            num_categories = len(category_encoder),
            num_mechanics  = len(mechanic_encoder),
        )
        state_dict = checkpoint

    model = bgr.BoardGameRecommender(**hparams)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return user_encoder, game_data, user_data, model


# ── Core functions ────────────────────────────────────────────────────────────

def recommend(
    username: str,
    user_encoder: dict,
    game_data: pd.DataFrame,
    user_data: pd.DataFrame,
    model: torch.nn.Module,
    top_k: int = 10,
    exclude_rated: bool = True,
    batch_size: int = 1000,
) -> pd.DataFrame:
    """
    Returns the top_k board games predicted for a given BGG username.

    Parameters
    ----------
    username      : BGG username — must have ratings in the training data.
    user_encoder  : Dict mapping username strings to encoded integer IDs.
    game_data     : Pre-processed game catalogue DataFrame.
    user_data     : Pre-processed user ratings DataFrame.
    model         : Trained BoardGameRecommender model.
    top_k         : Number of recommendations to return.
    exclude_rated : If True, games the user has already rated are excluded.
    batch_size    : Inference batch size.

    Returns
    -------
    DataFrame ranked by predicted_rating descending.
    """
    if username not in user_encoder:
        raise ValueError(
            f"Unknown user '{username}'. "
            "The username must appear in the training data."
        )

    user_id_enc = user_encoder[username]

    # Build inference DataFrame — one row per game in the catalogue
    infer_df = game_data[[
        "game_id", "game_name", "year_published",
        "game_id_encoded",
        "avg_usr_rating_scaled", "avg_usr_weight_scaled",
        "bayes_average_scaled", "age_scaled", "game_owners_scaled",
        "category_indices", "mechanic_indices",
        "avg_usr_rating",
    ]].copy()

    # Columns required by UserGameDataSet
    infer_df["user_id"]     = user_id_enc
    infer_df["user_rating"] = 0.0  # placeholder — not used in the forward pass

    # Optionally drop games the user has already rated
    if exclude_rated:
        rated_ids = set(user_data.loc[user_data["user_id"] == user_id_enc, "game_id"])
        infer_df  = infer_df[~infer_df["game_id"].isin(rated_ids)]

    infer_df = infer_df.reset_index(drop=True)

    # Run inference using the same Dataset / collate_fn as training
    dataset = bgr.UserGameDataSet(infer_df)
    loader  = DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = 0,
        collate_fn  = bgr.collate_fn,
    )

    all_preds = []
    with torch.no_grad():
        for batch in loader:
            preds = model(
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
            ).cpu().numpy().flatten()
            all_preds.extend(preds.tolist())

    infer_df["predicted_rating"] = all_preds

    results = (
        infer_df
        .nlargest(top_k, "predicted_rating")
        [["game_name", "predicted_rating", "avg_usr_rating", "year_published"]]
        .rename(columns={"avg_usr_rating": "community_avg"})
        .reset_index(drop=True)
    )
    results.index      += 1
    results.index.name  = "rank"
    results["predicted_rating"] = results["predicted_rating"].round(2)
    results["community_avg"]    = results["community_avg"].round(2)
    results["year_published"]   = results["year_published"].astype(int)

    return results


def get_user_ratings(
    username: str,
    user_encoder: dict,
    game_data: pd.DataFrame,
    user_data: pd.DataFrame,
    top_k: int = 10,
    min_rating: float = 7.0,
) -> pd.DataFrame:
    """
    Returns the top-rated games for a user — useful for sanity-checking recommendations.
    """
    if username not in user_encoder:
        raise ValueError(f"Unknown user '{username}'.")

    uid   = user_encoder[username]
    rated = user_data[user_data["user_id"] == uid][["game_id", "user_rating"]].copy()
    rated = rated[rated["user_rating"] >= min_rating]

    rated = rated.merge(
        game_data[["game_id", "game_name", "avg_usr_rating", "year_published"]],
        on="game_id",
        how="left",
    )
    rated = (
        rated
        .sort_values("user_rating", ascending=False)
        .head(top_k)
        .rename(columns={"avg_usr_rating": "community_avg"})
        .reset_index(drop=True)
    )
    rated.index      += 1
    rated.index.name  = "rank"
    rated["community_avg"]  = rated["community_avg"].round(2)
    rated["year_published"] = rated["year_published"].astype(int)

    return rated[["game_name", "user_rating", "community_avg", "year_published"]]


# ── Display helpers ───────────────────────────────────────────────────────────

def print_table(df: pd.DataFrame, title: str):
    """Prints a DataFrame with a header banner."""
    sep = "─" * 70
    print(f"\n{sep}")
    print(f"  {title}")
    print(sep)
    print(df.to_string())
    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate board game recommendations for a BGG user.",
    )
    p.add_argument(
        "username",
        nargs="?",
        help="BGG username to generate recommendations for.",
    )
    p.add_argument(
        "--top-k", type=int, default=10,
        help="Number of recommendations to return (default: 10).",
    )
    p.add_argument(
        "--epoch", type=int, default=10,
        help="Model checkpoint epoch to load (default: 10).",
    )
    p.add_argument(
        "--include-rated", action="store_true",
        help="Include games the user has already rated (excluded by default).",
    )
    p.add_argument(
        "--show-ratings", action="store_true",
        help="Also print the user's highest-rated games for comparison.",
    )
    p.add_argument(
        "--list-users", action="store_true",
        help="Print 20 sample usernames from the encoder and exit.",
    )
    return p


def main():
    parser = build_parser()
    args   = parser.parse_args()

    # ── Device ────────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # ── Load assets ───────────────────────────────────────────────────────────
    print("Loading assets…")
    config = bgr.setup_config("config/config.json")

    try:
        user_encoder, game_data, user_data, model = load_assets(config, args.epoch, device)
    except FileNotFoundError as e:
        sys.exit(f"ERROR: {e}\nRun board_game_rec.py first to generate the required files.")

    print(f"Device        : {device}")
    print(f"Users         : {len(user_encoder):,}")
    print(f"Games         : {len(game_data):,}")
    print(f"Checkpoint    : epoch {args.epoch}")

    # ── --list-users ──────────────────────────────────────────────────────────
    if args.list_users:
        print("\nSample usernames (first 20):")
        for u in list(user_encoder.keys())[:20]:
            print(f"  {u}")
        return

    # ── Username required from here ───────────────────────────────────────────
    if not args.username:
        parser.print_help()
        sys.exit("\nERROR: username is required (or use --list-users to browse).")

    username = args.username

    # ── Optionally show what the user has rated ───────────────────────────────
    if args.show_ratings:
        try:
            ratings = get_user_ratings(username, user_encoder, game_data, user_data, top_k=args.top_k)
            print_table(ratings, f"Top-rated games by '{username}' (rated >= 7)")
        except ValueError as e:
            sys.exit(f"ERROR: {e}")

    # ── Recommendations ───────────────────────────────────────────────────────
    print(f"\nGenerating recommendations for '{username}'…")
    try:
        recs = recommend(
            username      = username,
            user_encoder  = user_encoder,
            game_data     = game_data,
            user_data     = user_data,
            model         = model,
            top_k         = args.top_k,
            exclude_rated = not args.include_rated,
        )
    except ValueError as e:
        sys.exit(f"ERROR: {e}")

    label = f"Top {args.top_k} recommendations for '{username}'"
    if not args.include_rated:
        label += "  (already-rated games excluded)"
    print_table(recs, label)


if __name__ == "__main__":
    main()
