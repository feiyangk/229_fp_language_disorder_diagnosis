#!/usr/bin/env python3
"""
Logistic-regression classifier for dysarthria detection.

Uses the processed TORGO features (MFCC or wav2vec + Frenchay + optional Sentence-BERT prompt
embeddings) to train a linear classifier and reports test metrics.
"""

from __future__ import annotations

import argparse
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    try:
        from tensorboardX import SummaryWriter
    except ImportError:
        SummaryWriter = None

DEFAULT_DATA_DIR = Path("torgo_processed_data")


def load_array(path: Path, description: str) -> np.ndarray:
    arr = np.load(path)
    return arr


def load_labels(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing labels: {path}")
    labels = np.load(path)
    if labels.ndim != 1:
        raise ValueError(f"Labels must be 1D, got shape {labels.shape}")
    return labels.astype(int)


def maybe_load_prompt_embeddings(root: Path) -> Optional[np.ndarray]:
    prompt_path = root / "X_prompt_sbert.npy"
    if not prompt_path.exists():
        print("Note: Prompt embeddings not found; proceeding without them.")
        return None
    arr = np.load(prompt_path)
    if arr.ndim != 2:
        raise ValueError(f"Prompt embeddings must be 2D, got {arr.shape}")
    return arr


def build_feature_matrix(root: Path, audio_feature_type: str = "mfcc") -> np.ndarray:
    blocks: List[np.ndarray] = []
    names: List[str] = []

    # Load audio features (MFCC or wav2vec)
    if audio_feature_type.lower() == "mfcc":
        audio_path = root / "X_mfcc.npy"
        audio_name = "MFCC"
    elif audio_feature_type.lower() == "wav2vec":
        audio_path = root / "X_wav2vec.npy"
        audio_name = "wav2vec"
    else:
        raise ValueError(f"Unknown audio feature type: {audio_feature_type}. Must be 'mfcc' or 'wav2vec'")
    
    if not audio_path.exists():
        available = []
        if (root / "X_mfcc.npy").exists():
            available.append("mfcc")
        if (root / "X_wav2vec.npy").exists():
            available.append("wav2vec")
        
        error_msg = f"Audio features file not found: {audio_path}"
        if available:
            error_msg += f"\nAvailable audio feature types: {', '.join(available)}"
            error_msg += f"\nTry using: --audio-features {available[0]}"
        else:
            error_msg += f"\nNo audio feature files found in {root}"
        raise FileNotFoundError(error_msg)
    
    X_audio = load_array(audio_path, f"{audio_name} features")
    blocks.append(X_audio)
    names.append(audio_name)

    X_frenchay = load_array(root / "X_frenchay.npy", "Frenchay features")
    blocks.append(X_frenchay)
    names.append("Frenchay")

    prompt_embeddings = maybe_load_prompt_embeddings(root)
    if prompt_embeddings is not None:
        blocks.append(prompt_embeddings)
        names.append("Sentence-BERT prompts")

    lengths = {block.shape[0] for block in blocks}
    if len(lengths) != 1:
        raise ValueError(f"Feature blocks have mismatched row counts: {lengths}")

    X = np.hstack(blocks)
    print(f"Combined feature matrix shape: {X.shape} ({' + '.join(names)})")
    return X


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Logistic regression dysarthria classifier.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing processed TORGO arrays (default: %(default)s)",
    )
    parser.add_argument(
        "--audio-features",
        type=str,
        choices=["mfcc", "wav2vec"],
        default="mfcc",
        help="Audio feature type to use: 'mfcc' or 'wav2vec' (default: %(default)s)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of samples to hold out for testing (default: %(default)s)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for train/test split and classifier (default: %(default)s)",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=2000,
        help="Maximum iterations for LogisticRegression (default: %(default)s)",
    )
    parser.add_argument(
        "--model-out",
        type=Path,
        default=None,
        help="Optional path to pickle the fitted model pipeline.",
    )
    parser.add_argument(
        "--tensorboard-dir",
        type=Path,
        default=None,
        help="Directory for TensorBoard logs (default: runs/logreg_<timestamp>)",
    )
    parser.add_argument(
        "--no-tensorboard",
        action="store_true",
        help="Disable TensorBoard logging",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir: Path = args.data_dir

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    print(f"Loading processed data from: {data_dir}")
    print(f"Using audio features: {args.audio_features}")
    X = build_feature_matrix(data_dir, audio_feature_type=args.audio_features)
    y = load_labels(data_dir / "Y.npy")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )
    print(f"Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

    clf = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "logreg",
                LogisticRegression(
                    penalty="l2",
                    class_weight="balanced",
                    solver="lbfgs",
                    max_iter=args.max_iter,
                    random_state=args.random_state,
                ),
            ),
        ]
    )

    print("\nTraining logistic regression...")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest accuracy: {acc:.4f}")
    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=4))

    if args.model_out:
        args.model_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.model_out, "wb") as f:
            pickle.dump(clf, f)
        print(f"\nSaved fitted model to: {args.model_out}")


if __name__ == "__main__":
    main()

