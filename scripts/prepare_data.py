"""
Télécharge ArSarcasm via Hugging Face datasets (iabufarha/ar_sarcasm),
nettoie les tweets, crée les splits train/val/test.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.preprocess import clean_arabic_tweet

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

DATA_DIR = "data"

LABEL2ID = {'positive': 0, 'negative': 1, 'neutral': 2}
ID2LABEL  = {v: k for k, v in LABEL2ID.items()}
ID2LABEL_HF = {0: 'positive', 1: 'negative', 2: 'neutral'}


def download_dataset():
    """Télécharge ArSarcasm depuis Hugging Face."""
    os.makedirs(DATA_DIR, exist_ok=True)

    train_path = os.path.join(DATA_DIR, "arsarcasm_train.csv")
    test_path  = os.path.join(DATA_DIR, "arsarcasm_test.csv")

    if os.path.exists(train_path) and os.path.exists(test_path):
        print("Dataset déjà téléchargé.")
        return

    print("Téléchargement via Hugging Face datasets...")
    try:
        from datasets import load_dataset
    except ImportError:
        print("Installation de 'datasets'...")
        os.system(f"{sys.executable} -m pip install datasets -q")
        from datasets import load_dataset

    ds = load_dataset("iabufarha/ar_sarcasm")

    train_df = ds["train"].to_pandas()
    test_df  = ds["test"].to_pandas()

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"Sauvegardé: {train_path} ({len(train_df)} lignes)")
    print(f"Sauvegardé: {test_path} ({len(test_df)} lignes)")


def load_and_clean():
    train_raw = pd.read_csv(os.path.join(DATA_DIR, "arsarcasm_train.csv"))
    test_raw  = pd.read_csv(os.path.join(DATA_DIR, "arsarcasm_test.csv"))
    df = pd.concat([train_raw, test_raw], ignore_index=True)

    print(f"Échantillons bruts: {len(df)}")
    print(f"Colonnes: {df.columns.tolist()}")
    print(f"Aperçu colonne 'sentiment': {df['sentiment'].head(5).tolist()}")

    # Renommer 'tweet' → 'text' si nécessaire
    if 'tweet' in df.columns:
        df = df.rename(columns={'tweet': 'text'})

    # Mapping sentiment → label (robuste : gère int, int64, str "0"/"1"/"2", ou texte)
    if 'sentiment' in df.columns:
        try:
            df['label'] = df['sentiment'].astype(int).map(ID2LABEL_HF)
        except (ValueError, TypeError):
            # Déjà du texte ('positive', 'negative', 'neutral')
            df['label'] = df['sentiment'].astype(str).str.strip().str.lower()
    else:
        raise ValueError("Colonne 'sentiment' introuvable dans le dataset.")

    # Vérification avant filtrage
    print(f"Labels uniques avant filtrage: {df['label'].unique().tolist()}")

    df = df[df['label'].isin(LABEL2ID.keys())].copy()

    if len(df) == 0:
        raise ValueError(
            "Aucun échantillon valide après mapping des labels. "
            "Vérifie les valeurs de la colonne 'sentiment'."
        )

    df['label_id'] = df['label'].map(LABEL2ID)

    # Garder les colonnes utiles
    cols = ['text', 'label', 'label_id']
    for c in ['dialect', 'sarcasm']:
        if c in df.columns:
            cols.append(c)
    df = df[cols].copy()

    df = df.dropna(subset=['text', 'label_id']).reset_index(drop=True)

    print("Nettoyage des tweets...")
    df['text_clean'] = df['text'].apply(clean_arabic_tweet)

    before = len(df)
    df = df[df['text_clean'].str.len() > 5].reset_index(drop=True)
    print(f"Supprimés après nettoyage: {before - len(df)}")
    print(f"Échantillons propres: {len(df)}")
    return df


def split_and_save(df):
    os.makedirs(DATA_DIR, exist_ok=True)

    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df['label_id'], random_state=SEED
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df['label_id'], random_state=SEED
    )

    for name, split in [('train', train_df), ('val', val_df), ('test', test_df)]:
        path = os.path.join(DATA_DIR, f"{name}.csv")
        split.to_csv(path, index=False)
        print(f"✅ Sauvegardé {path}  ({len(split)} lignes)")

    return train_df, val_df, test_df


if __name__ == "__main__":
    print("=== prepare_data.py ===\n")
    download_dataset()
    df = load_and_clean()

    print("\nDistribution des labels:")
    print(df['label'].value_counts().to_string())
    if 'dialect' in df.columns:
        print("\nDistribution des dialectes:")
        print(df['dialect'].value_counts().to_string())

    print("\nSplit 70/15/15 (stratifié)...")
    train_df, val_df, test_df = split_and_save(df)

    print("\nFichiers dans data/:")
    for f in sorted(os.listdir(DATA_DIR)):
        size = os.path.getsize(os.path.join(DATA_DIR, f))
        print(f"  {f}  ({size/1024:.1f} KB)")

    print("\nÉtape suivante: ouvre notebooks/training_colab.ipynb sur Google Colab.")
