#!/usr/bin/env python3
"""
Regenerate wav2vec features from an existing TORGO dataset.

This script loads an existing processed dataset (torgo_dataset.pkl) and regenerates
wav2vec features for all entries. Useful when:
- You want to add wav2vec features to an existing dataset
- Your audio files are stored in S3 (supports s3:// paths)
- You've already processed MFCC features but want to add wav2vec

Usage:
    python generate_wav2vec_from_dataset.py --dataset torgo_processed_data/torgo_dataset.pkl --output-dir torgo_processed_data
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import List, Dict, Any

import numpy as np

# Import the extraction function from generate_torgo_dataset
import sys
sys.path.insert(0, str(Path(__file__).parent))
from generate_torgo_dataset import extract_wav2vec_features, load_audio_from_path


def load_dataset(dataset_path: Path) -> List[Dict[str, Any]]:
    """Load the processed dataset."""
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    
    print(f"Loading dataset from: {dataset_path}")
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    
    print(f"Loaded {len(dataset)} entries")
    return dataset


def regenerate_wav2vec_features(dataset: List[Dict[str, Any]], output_dir: Path) -> None:
    """Regenerate wav2vec features for all entries in the dataset."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nRegenerating wav2vec features...")
    print("=" * 60)
    
    wav2vec_features_list = []
    successful = 0
    failed = 0
    
    for idx, entry in enumerate(dataset):
        wav_file = entry.get('wav_file', '')
        
        if not wav_file:
            print(f"  Entry {idx}: No wav_file path found, skipping")
            failed += 1
            wav2vec_features_list.append(None)
            continue
        
        # Extract wav2vec features
        print(f"  Processing {idx+1}/{len(dataset)}: {wav_file}")
        wav2vec_features = extract_wav2vec_features(wav_file)
        
        if wav2vec_features is not None:
            wav2vec_features_list.append(wav2vec_features)
            entry['wav2vec_features'] = wav2vec_features
            successful += 1
        else:
            print(f"    Warning: Failed to extract wav2vec features")
            wav2vec_features_list.append(None)
            failed += 1
    
    print("=" * 60)
    print(f"\nSummary:")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    
    # Check if all features were extracted successfully
    if all(f is not None for f in wav2vec_features_list):
        X_wav2vec = np.array(wav2vec_features_list)
        
        # Save wav2vec features
        wav2vec_path = output_dir / "X_wav2vec.npy"
        np.save(wav2vec_path, X_wav2vec)
        print(f"\nSaved wav2vec features to: {wav2vec_path}")
        print(f"  Shape: {X_wav2vec.shape}")
        
        # Update dataset file
        dataset_path = output_dir / "torgo_dataset.pkl"
        if dataset_path.exists():
            print(f"\nUpdating dataset file: {dataset_path}")
            with open(dataset_path, 'wb') as f:
                pickle.dump(dataset, f)
        
        # Update metadata if it exists
        metadata_path = output_dir / "metadata.pkl"
        if metadata_path.exists():
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            
            metadata['wav2vec_features'] = X_wav2vec.shape[1]
            
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
            print(f"Updated metadata: wav2vec_features = {X_wav2vec.shape[1]}")
    else:
        print("\nWarning: Not all wav2vec features were extracted successfully.")
        print("X_wav2vec.npy was not saved. Please check the errors above.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate wav2vec features from existing TORGO dataset."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("torgo_processed_data/torgo_dataset.pkl"),
        help="Path to the existing dataset pickle file (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("torgo_processed_data"),
        help="Output directory for wav2vec features (default: %(default)s)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    # Load dataset
    dataset = load_dataset(args.dataset)
    
    # Regenerate wav2vec features
    regenerate_wav2vec_features(dataset, args.output_dir)
    
    print("\nDone!")


if __name__ == "__main__":
    main()

