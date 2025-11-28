#!/usr/bin/env python3
"""
K-Means dysarthria detector for the TORGO dataset.

This script clusters the combined feature vectors (MFCC, Frenchay, and optional
Sentence-BERT prompt embeddings) into two clusters and evaluates how well those
clusters align with the ground-truth dysarthria labels.

Usage:
    python kmeans_dysarthria.py

Optional arguments:
    --data-dir /path/to/torgo_processed_data
    --clusters 2
    --random-state 42
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

DEFAULT_DATA_DIR = Path("/Users/kuangfy/Desktop/229 FP/torgo_processed_data")


def load_array(path: Path, description: str) -> np.ndarray:
    return np.load(path)


def load_prompt_embeddings(root: Path) -> Optional[np.ndarray]:
    prompt_path = root / "X_prompt_sbert.npy"
    arr = np.load(prompt_path)
    return arr


def load_labels(root: Path) -> np.ndarray:
    label_path = root / "Y.npy"
    y = np.load(label_path)
    return y.astype(int)


def build_feature_matrix(root: Path) -> np.ndarray:
    feature_blocks = []
    block_names = []

    X_mfcc = load_array(root / "X_mfcc.npy", "MFCC features")
    feature_blocks.append(X_mfcc)
    block_names.append("MFCC")

    X_frenchay = load_array(root / "X_frenchay.npy", "Frenchay features")
    feature_blocks.append(X_frenchay)
    block_names.append("Frenchay")

    prompt_embeddings = load_prompt_embeddings(root)
    if prompt_embeddings is not None:
        feature_blocks.append(prompt_embeddings)
        block_names.append("Sentence-BERT prompts")    
    return np.hstack(feature_blocks)


def majority_vote_mapping(cluster_labels: np.ndarray, y_true: np.ndarray) -> Dict[int, int]:
    mapping: Dict[int, int] = {}
    for cluster_id in np.unique(cluster_labels):
        mask = cluster_labels == cluster_id
        if mask.sum() == 0:
            mapping[cluster_id] = 0
            continue
        cluster_labels_true = y_true[mask]
        majority_label = int(cluster_labels_true.mean() >= 0.5)
        mapping[cluster_id] = majority_label
    return mapping


def evaluate_clusters(cluster_labels: np.ndarray, y_true: np.ndarray) -> None:
    mapping = majority_vote_mapping(cluster_labels, y_true)
    y_pred = np.vectorize(mapping.get)(cluster_labels)

    acc = accuracy_score(y_true, y_pred)
    print(f"\nCluster-to-label mapping: {mapping}")
    print(f"Accuracy after mapping: {acc:.4f}")
    print("\nConfusion matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, digits=4))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="K-Means clustering for dysarthria detection.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing the processed TORGO dataset (default: %(default)s)",
    )
    parser.add_argument(
        "--clusters",
        type=int,
        default=2,
        help="#clusters for K-Means (default: %(default)s)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for K-Means initialization (default: %(default)s)",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=500,
        help="Maximum iterations for K-Means (default: %(default)s)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir: Path = args.data_dir
    print(f"Loading processed data from: {data_dir}")
    X = build_feature_matrix(data_dir)
    y = load_labels(data_dir)

    scaler = StandardScaler() # normalize features
    X_scaled = scaler.fit_transform(X)

    print(f"\nRunning K-Means with k={args.clusters} with random_state={args.random_state}...")
    kmeans = KMeans(
        n_clusters=args.clusters,
        n_init="auto",
        random_state=args.random_state,
        max_iter=args.max_iter,
        verbose=0,
    )
    cluster_labels = kmeans.fit_predict(X_scaled)
    evaluate_clusters(cluster_labels, y)


if __name__ == "__main__":
    main()

