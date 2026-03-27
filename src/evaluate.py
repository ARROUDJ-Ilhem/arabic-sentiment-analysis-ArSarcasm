"""
Metrics, confusion matrix, and error analysis helpers.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, accuracy_score
)
from .dataset import LABEL2ID, ID2LABEL


def compute_metrics_from_arrays(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return macro-F1 and accuracy as a dict."""
    return {
        'macro_f1': f1_score(y_true, y_pred, average='macro'),
        'accuracy': accuracy_score(y_true, y_pred),
    }


def print_classification_report(y_true, y_pred):
    print(classification_report(
        y_true, y_pred,
        target_names=list(LABEL2ID.keys())
    ))


def plot_confusion_matrix(y_true, y_pred, save_path: str = 'confusion_matrix.png'):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=list(LABEL2ID.keys()),
        yticklabels=list(LABEL2ID.keys()),
        ax=ax
    )
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved confusion matrix to {save_path}")
    plt.show()
    return cm


def show_error_examples(test_df: pd.DataFrame,
                         y_pred: np.ndarray,
                         true_label: str = 'Negative',
                         pred_label: str = 'Neutral',
                         n: int = 5):
    """Print n misclassified examples for a given (true, predicted) pair."""
    df = test_df.copy().reset_index(drop=True)
    df['pred_label'] = [ID2LABEL[p] for p in y_pred]
    errors = df[
        (df['label'] == true_label) &
        (df['pred_label'] == pred_label)
    ][['text_clean', 'label', 'pred_label']].head(n)

    print(f"── {true_label} predicted as {pred_label} ──")
    for i, row in errors.iterrows():
        print(f"[{i}] True={row['label']} | Pred={row['pred_label']}")
        print(f"     {row['text_clean'][:100]}")
        print()


if __name__ == "__main__":
    # Smoke test with dummy predictions
    # Run with:  python -m src.evaluate
    y_true = np.array([0, 1, 2, 3, 1, 0, 2])
    y_pred = np.array([0, 2, 2, 3, 1, 0, 1])
    metrics = compute_metrics_from_arrays(y_true, y_pred)
    print("Smoke test metrics:", json.dumps(metrics, indent=2))
    print_classification_report(y_true, y_pred)
