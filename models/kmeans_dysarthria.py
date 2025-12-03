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
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
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


def build_feature_matrix(root: Path) -> Tuple[np.ndarray, List[str]]:
    feature_blocks: List[np.ndarray] = []
    feature_names: List[str] = []

    def add_block(block: np.ndarray, prefix: str) -> None:
        feature_blocks.append(block)
        feature_names.extend([f"{prefix}_{i}" for i in range(block.shape[1])])

    X_mfcc = load_array(root / "X_mfcc.npy", "MFCC features")
    add_block(X_mfcc, "MFCC")

    X_frenchay = load_array(root / "X_frenchay.npy", "Frenchay features")
    add_block(X_frenchay, "Frenchay")

    prompt_embeddings = load_prompt_embeddings(root)
    if prompt_embeddings is not None:
        add_block(prompt_embeddings, "SentenceBERT")

    if len({block.shape[0] for block in feature_blocks}) != 1:
        shapes = {block.shape for block in feature_blocks}
        raise ValueError(f"Feature blocks have inconsistent sample counts: {shapes}")

    X = np.hstack(feature_blocks)
    return X, feature_names


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
    parser.add_argument(
        "--cluster-shap",
        action="store_true",
        help="Enable ClusterSHAP feature attribution using a surrogate random forest.",
    )
    parser.add_argument(
        "--cluster-shap-samples",
        type=int,
        default=800,
        help="Max #examples to use for ClusterSHAP (0 uses entire dataset).",
    )
    parser.add_argument(
        "--cluster-shap-top-k",
        type=int,
        default=10,
        help="How many top features to display per cluster for ClusterSHAP.",
    )
    parser.add_argument(
        "--cluster-shap-trees",
        type=int,
        default=400,
        help="Number of trees for the surrogate random forest used by ClusterSHAP.",
    )
    return parser.parse_args()


def run_cluster_shap(
    X: np.ndarray,
    cluster_labels: np.ndarray,
    feature_names: List[str],
    *,
    sample_size: int,
    top_k: int,
    n_estimators: int,
    random_state: int,
) -> None:
    try:
        import shap  # type: ignore
    except ImportError:
        print(
            "\nClusterSHAP requested, but the 'shap' package is not installed. "
            "Install it via `pip install shap` and rerun to see attributions."
        )
        return

    rng = np.random.default_rng(random_state)
    n_samples = X.shape[0]
    if sample_size and 0 < sample_size < n_samples:
        subset_idx = rng.choice(n_samples, size=sample_size, replace=False)
    else:
        subset_idx = np.arange(n_samples)

    X_subset = X[subset_idx]
    y_subset = cluster_labels[subset_idx]

    surrogate = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    surrogate.fit(X_subset, y_subset)

    explainer = shap.TreeExplainer(surrogate)
    raw_shap = explainer.shap_values(X_subset)

    if isinstance(raw_shap, list):
        shap_by_class = {cls: raw_shap[idx] for idx, cls in enumerate(surrogate.classes_)}
    else:
        # Fallback for binary classification; SHAP returns a single matrix
        if len(surrogate.classes_) != 2:
            raise RuntimeError("Unexpected SHAP output shape for multi-class surrogate.")
        shap_by_class = {
            surrogate.classes_[1]: raw_shap,
            surrogate.classes_[0]: -raw_shap,
        }

    print(
        f"\nClusterSHAP: explaining {X_subset.shape[0]} samples "
        f"({len(feature_names)} features) using RandomForest surrogate."
    )
    print("Values shown are mean |SHAP| scores; higher values indicate stronger influence.\n")

    for cluster_id in surrogate.classes_:
        mask = y_subset == cluster_id
        if not mask.any():
            continue
        cluster_shap = shap_by_class[cluster_id][mask]
        mean_abs = np.mean(np.abs(cluster_shap), axis=0)
        top_indices = np.argsort(mean_abs)[::-1][:top_k]
        print(f"Cluster {cluster_id}:")
        for rank, feat_idx in enumerate(top_indices, start=1):
            # Convert numpy-derived indices of arbitrary shape into native ints
            feat_arr = np.asarray(feat_idx).reshape(-1)
            feature_idx = int(feat_arr[0])
            shap_score = float(np.asarray(mean_abs[feature_idx]).reshape(-1)[0])
            print(
                f"  {rank:2d}. {feature_names[feature_idx]} (|SHAP|={shap_score:.4f})"
            )
        print()


def main() -> None:
    args = parse_args()
    data_dir: Path = args.data_dir
    print(f"Loading processed data from: {data_dir}")
    X, feature_names = build_feature_matrix(data_dir)
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

    if args.cluster_shap:
        run_cluster_shap(
            X_scaled,
            cluster_labels,
            feature_names,
            sample_size=args.cluster_shap_samples,
            top_k=args.cluster_shap_top_k,
            n_estimators=args.cluster_shap_trees,
            random_state=args.random_state,
        )


if __name__ == "__main__":
    main()

