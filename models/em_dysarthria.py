#!/usr/bin/env python3
"""
Gaussian Mixture (EM) dysarthria detector for the TORGO dataset.

This script fits a Gaussian Mixture Model (trained via Expectation-Maximization)
on the combined MFCC, Frenchay, and optional Sentence-BERT prompt features, then
maps mixture components to dysarthria labels with majority voting for reporting.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss
from sklearn.preprocessing import StandardScaler

DEFAULT_DATA_DIR = Path("torgo_processed_data")


def load_array(path: Path, description: str) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")
    arr = np.load(path)
    if arr.ndim != 2:
        raise ValueError(f"{description} must be 2D, got shape {arr.shape}")
    return arr


def load_prompt_embeddings(root: Path) -> Optional[np.ndarray]:
    prompt_path = root / "X_prompt_sbert.npy"
    if not prompt_path.exists():
        print("Note: Prompt embeddings not found; continuing without them.")
        return None
    arr = np.load(prompt_path)
    if arr.ndim != 2:
        raise ValueError(f"Prompt embeddings must be 2D, got shape {arr.shape}")
    return arr


def load_labels(root: Path) -> np.ndarray:
    label_path = root / "Y.npy"
    if not label_path.exists():
        raise FileNotFoundError(f"Missing labels: {label_path}")
    y = np.load(label_path)
    if y.ndim != 1:
        raise ValueError(f"Labels must be 1D, got shape {y.shape}")
    return y.astype(int)


def build_feature_matrix(root: Path) -> np.ndarray:
    blocks = []
    names = []

    X_mfcc = load_array(root / "X_mfcc.npy", "MFCC features")
    blocks.append(X_mfcc)
    names.append("MFCC")

    X_frenchay = load_array(root / "X_frenchay.npy", "Frenchay features")
    blocks.append(X_frenchay)
    names.append("Frenchay")

    prompt_embeddings = load_prompt_embeddings(root)
    if prompt_embeddings is not None:
        blocks.append(prompt_embeddings)
        names.append("Sentence-BERT prompts")

    if len({block.shape[0] for block in blocks}) != 1:
        shapes = {block.shape for block in blocks}
        raise ValueError(f"Feature blocks have inconsistent shapes: {shapes}")

    X = np.hstack(blocks)
    print(f"Combined feature matrix shape: {X.shape} ({' + '.join(names)})")
    return X


def majority_vote_mapping(component_labels: np.ndarray, y_true: np.ndarray) -> Dict[int, int]:
    mapping: Dict[int, int] = {}
    for comp in np.unique(component_labels):
        mask = component_labels == comp
        if mask.sum() == 0:
            mapping[comp] = 0
            continue
        comp_labels = y_true[mask]
        mapping[comp] = int(comp_labels.mean() >= 0.5)
    return mapping


def evaluate_components(component_labels: np.ndarray, y_true: np.ndarray) -> None:
    mapping = majority_vote_mapping(component_labels, y_true)
    y_pred = np.vectorize(mapping.get)(component_labels)

    acc = accuracy_score(y_true, y_pred)
    print(f"\nComponent-to-label mapping: {mapping}")
    print(f"Accuracy after mapping: {acc:.4f}")
    print("\nConfusion matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, digits=4))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gaussian Mixture (EM) clustering for dysarthria detection.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing processed TORGO features (default: %(default)s)",
    )
    parser.add_argument(
        "--components",
        type=int,
        default=2,
        help="Number of Gaussian components (default: %(default)s)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducible initialization (default: %(default)s)",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=500,
        help="Maximum EM iterations (default: %(default)s)",
    )
    parser.add_argument(
        "--covariance-type",
        choices=["full", "tied", "diag", "spherical"],
        default="full",
        help="Covariance type for GaussianMixture (default: %(default)s)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir: Path = args.data_dir

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    print(f"Loading processed data from: {data_dir}")
    X = build_feature_matrix(data_dir)
    y = load_labels(data_dir)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(
        f"\nFitting GaussianMixture with {args.components} components, "
        f"covariance='{args.covariance_type}', random_state={args.random_state}..."
    )
    gmm = GaussianMixture(
        n_components=args.components,
        covariance_type=args.covariance_type,
        max_iter=args.max_iter,
        random_state=args.random_state,
        init_params="kmeans",
        verbose=1,
    )
    gmm.fit(X_scaled)
    component_labels = gmm.predict(X_scaled)

    evaluate_components(component_labels, y)

    if args.components == 2:
        probs = gmm.predict_proba(X_scaled)
        # align component order with mapping heuristic
        component_mapping = majority_vote_mapping(component_labels, y)
        prob_dys = np.zeros_like(probs[:, 0])
        for comp_idx, label in component_mapping.items():
            prob_dys += probs[:, comp_idx] * label
        print(f"\nAverage predicted dysarthria probability: {prob_dys.mean():.4f}")
        print(f"Log loss (relative to true labels): {log_loss(y, prob_dys):.4f}")


if __name__ == "__main__":
    main()

