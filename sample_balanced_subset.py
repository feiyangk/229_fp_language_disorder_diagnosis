#!/usr/bin/env python3
"""
Sample a balanced TORGO subset with equal gender/label representation.

This script reads the processed dataset produced by `generate_torgo_dataset.py`,
randomly selects the requested number of female/male samples for each class
label (Y=0, Y=1), and writes the subset to a separate folder.

Default selection: 500 female + 500 male for Y=0 and the same for Y=1
(2000 rows total).
"""

from __future__ import annotations

import argparse
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

DEFAULT_DATA_DIR = Path("/Users/kuangfy/Desktop/229 FP/torgo_processed_data")
DEFAULT_OUTPUT_DIR = Path("/Users/kuangfy/Desktop/229 FP/torgo_balanced_subset")


def infer_gender(speaker_name: str) -> str:
    speaker_upper = speaker_name.strip().upper()
    if speaker_upper.startswith("F"):
        return "female"
    if speaker_upper.startswith("M"):
        return "male"
    raise ValueError(f"Unable to infer gender from speaker name: {speaker_name}")


def load_pickle(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing required pickle file: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def load_numpy(path: Path, allow_missing: bool = False) -> np.ndarray | None:
    if not path.exists():
        if allow_missing:
            return None
        raise FileNotFoundError(f"Missing required array file: {path}")
    return np.load(path)


def sample_indices(
    entries: List[Dict],
    samples_per_group: int,
    rng: np.random.Generator,
) -> np.ndarray:
    indices_by_group: Dict[Tuple[str, int], List[int]] = defaultdict(list)
    for idx, entry in enumerate(entries):
        gender = infer_gender(entry["speaker"])
        label = int(entry["y"])
        indices_by_group[(gender, label)].append(idx)

    selected_indices: List[int] = []
    for gender in ("female", "male"):
        for label in (0, 1):
            key = (gender, label)
            group_indices = indices_by_group.get(key, [])
            count = len(group_indices)
            if count < samples_per_group:
                raise ValueError(
                    f"Not enough samples for group {key}: "
                    f"requested {samples_per_group}, available {count}"
                )
            chosen = rng.choice(group_indices, size=samples_per_group, replace=False)
            selected_indices.extend(chosen.tolist())
            print(
                f"Selected {samples_per_group} samples for "
                f"{gender}, y={label} (from {count} available)"
            )

    selected_array = np.array(selected_indices, dtype=int)
    rng.shuffle(selected_array)
    print(f"Total selected samples: {selected_array.size}")
    return selected_array


def save_subset(
    indices: np.ndarray,
    dataset: List[Dict],
    data_dir: Path,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    subset_entries = [dataset[i] for i in indices]
    with open(output_dir / "torgo_dataset.pkl", "wb") as f:
        pickle.dump(subset_entries, f)

    X_mfcc = load_numpy(data_dir / "X_mfcc.npy")[indices]
    X_frenchay = load_numpy(data_dir / "X_frenchay.npy")[indices]
    Y = load_numpy(data_dir / "Y.npy")[indices]
    prompts = load_pickle(data_dir / "prompts.pkl")
    subset_prompts = [prompts[i] for i in indices]

    np.save(output_dir / "X_mfcc.npy", X_mfcc)
    np.save(output_dir / "X_frenchay.npy", X_frenchay)
    np.save(output_dir / "Y.npy", Y)

    with open(output_dir / "prompts.pkl", "wb") as f:
        pickle.dump(subset_prompts, f)

    prompt_embeddings = load_numpy(data_dir / "X_prompt_sbert.npy", allow_missing=True)
    if prompt_embeddings is not None:
        np.save(output_dir / "X_prompt_sbert.npy", prompt_embeddings[indices])
    else:
        print("Note: prompt embeddings not found; skipping X_prompt_sbert.npy.")

    metadata = load_pickle(data_dir / "metadata.pkl")
    metadata_subset = {
        **metadata,
        "total_entries": int(indices.size),
        "notes": (
            f"Balanced subset with {indices.size} samples "
            f"(female/male x y=0/1 = {indices.size // 4} each)."
        ),
    }
    with open(output_dir / "metadata.pkl", "wb") as f:
        pickle.dump(metadata_subset, f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample a balanced subset of the TORGO processed dataset."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing the processed TORGO dataset (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Destination directory for the balanced subset (default: %(default)s)",
    )
    parser.add_argument(
        "--samples-per-group",
        type=int,
        default=500,
        help="Number of samples to draw for each (gender, label) group (default: %(default)s)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducible sampling (default: %(default)s)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir: Path = args.data_dir
    output_dir: Path = args.output_dir

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    print(f"Loading dataset from: {data_dir}")
    dataset = load_pickle(data_dir / "torgo_dataset.pkl")

    rng = np.random.default_rng(args.random_state)
    selected_indices = sample_indices(dataset, args.samples_per_group, rng)

    print(f"\nSaving balanced subset to: {output_dir}")
    save_subset(selected_indices, dataset, data_dir, output_dir)
    print("Balanced subset creation complete.")


if __name__ == "__main__":
    main()

