"""
TORGO Dataset Generator

This script processes TORGO speech data and generates a dataset for binary classification
(dystharthia vs. no dystharthia).

Required packages:
    - numpy
    - librosa
    - sentence-transformers (installs PyTorch automatically)
    - pickle (built-in)
    - pathlib (built-in)
    - csv (built-in)

Install dependencies:
    pip install numpy librosa sentence-transformers

Usage:
    python generate_torgo_dataset.py

Output:
    The script creates a folder 'torgo_processed_data' containing:
    - torgo_dataset.pkl: Full dataset with all information
    - X_mfcc.npy: MFCC features (128 features per sample)
    - X_frenchay.npy: Frenchay notes (28 features per sample)
    - X_prompt_sbert.npy: Sentence-BERT prompt embeddings
    - Y.npy: Binary labels (1 = dystharthia, 0 = no dystharthia)
    - prompts.pkl: Prompt texts for each sample
    - metadata.pkl: Dataset metadata
"""

import os
import csv
import numpy as np
import librosa
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Any
import re

# Mapping for Frenchay scores
FRENCHAY_MAPPING = {
    'a': 8,
    'b': 6,
    'c': 4,
    'd': 2,
    'e': 0
}

# Fields to extract from CSV (in order)
FRENCHAY_FIELDS = [
    'Cough', 'Swallow', 'Dribble/Drool',
    'At Rest',  # Resp
    'In Speech',  # Resp
    'At Rest',  # Lips
    'Spread', 'Seal', 'Alternate',  # Lips
    'In Speech',  # Lips
    'At Rest',  # Jaw
    'In Speech',  # Jaw
    'Fluids', 'Maintenance', 'In Speech',  # Palate
    'Time', 'Pitch', 'Volume', 'In Speech',  # Laryngeal
    'At Rest',  # Tongue
    'Protrusion', 'Elevation', 'Lateral', 'Alternate',  # Tongue
    'In Speech',  # Tongue
    'Words', 'Sentences', 'Conversation'  # Intel
]

DEFAULT_S_BERT_MODEL = "all-MiniLM-L6-v2"
PROMPT_EMBED_BATCH_SIZE = 64
PROMPT_EMBED_NORMALIZE = False

def parse_frenchay_value(value: str) -> float:
    """Convert Frenchay letter score to numeric value."""
    if not value or value.strip() == '':
        return 0.0
    
    value = value.strip().lower()
    
    # Handle single letter
    if value in FRENCHAY_MAPPING:
        return float(FRENCHAY_MAPPING[value])
    
    # Handle combinations like "c/d" or "d/e" - take mean
    if '/' in value:
        parts = value.split('/')
        if len(parts) == 2:
            val1 = FRENCHAY_MAPPING.get(parts[0].strip(), 0)
            val2 = FRENCHAY_MAPPING.get(parts[1].strip(), 0)
            return (val1 + val2) / 2.0
    
    return 0.0


def extract_frenchay_notes(csv_path: str) -> List[float]:
    """Extract Frenchay notes from CSV file. Returns zeros if CSV doesn't contain Frenchay data."""
    frenchay_values = []
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)
        
        # Check if this CSV has Frenchay data by looking for key indicators
        has_frenchay_data = False
        for row in rows:
            if len(row) > 0:
                category = row[0].strip()
                if category in ['Reflex', 'Resp.', 'Lips', 'Jaw', 'Palate', 'Laryngeal', 'Tongue', 'Intel.']:
                    has_frenchay_data = True
                    break
        
        # If no Frenchay data found (e.g., MC folders), return zeros
        if not has_frenchay_data:
            return [0.0] * 28
        
        # Build a mapping of field names to values
        field_map = {}
        current_category = None
        
        # Categories to skip
        skip_categories = ['Assessment', 'Date', 'Influencing factors', 
                          'Session', 'System', 'Head Correction', 'Notes', 'Summary',
                          'Hearing', 'Sight', 'Teeth', 'Language', 'Mood', 'Posture', 
                          'Rate', 'Sensation', 'Gender', 'DOB', 'Control']
        
        for row in rows:
            if len(row) < 2:
                continue
            
            category = row[0].strip() if len(row) > 0 else ''
            field = row[1].strip() if len(row) > 1 else ''
            value = row[2].strip() if len(row) > 2 else ''
            
            # Stop processing if we hit certain sections
            if category in ['Influencing factors', 'Session']:
                break
            
            # Track current category (categories that aren't in skip list)
            if category and category not in skip_categories:
                current_category = category
            
            # Skip rows without a field name
            if not field:
                continue
            
            # Map fields based on category and field name
            if current_category == 'Reflex':
                if field == 'Cough':
                    field_map['Cough'] = value
                elif field == 'Swallow':
                    field_map['Swallow'] = value
                elif field == 'Dribble/Drool':
                    field_map['Dribble/Drool'] = value
            elif current_category == 'Resp.':
                if field == 'At Rest':
                    field_map['Resp_At_Rest'] = value
                elif field == 'In Speech':
                    field_map['Resp_In_Speech'] = value
            elif current_category == 'Lips':
                if field == 'At Rest':
                    field_map['Lips_At_Rest'] = value
                elif field == 'Spread':
                    field_map['Spread'] = value
                elif field == 'Seal':
                    field_map['Seal'] = value
                elif field == 'Alternate':
                    field_map['Lips_Alternate'] = value
                elif field == 'In Speech':
                    field_map['Lips_In_Speech'] = value
            elif current_category == 'Jaw':
                if field == 'At Rest':
                    field_map['Jaw_At_Rest'] = value
                elif field == 'In Speech':
                    field_map['Jaw_In_Speech'] = value
            elif current_category == 'Palate':
                if field == 'Fluids':
                    field_map['Fluids'] = value
                elif field == 'Maintenance':
                    field_map['Maintenance'] = value
                elif field == 'In Speech':
                    field_map['Palate_In_Speech'] = value
            elif current_category == 'Laryngeal':
                if field == 'Time':
                    field_map['Time'] = value
                elif field == 'Pitch':
                    field_map['Pitch'] = value
                elif field == 'Volume':
                    field_map['Volume'] = value
                elif field == 'In Speech':
                    field_map['Laryngeal_In_Speech'] = value
            elif current_category == 'Tongue':
                if field == 'At Rest':
                    field_map['Tongue_At_Rest'] = value
                elif field == 'Protrusion':
                    field_map['Protrusion'] = value
                elif field == 'Elevation':
                    field_map['Elevation'] = value
                elif field == 'Lateral':
                    field_map['Lateral'] = value
                elif field == 'Alternate':
                    field_map['Tongue_Alternate'] = value
                elif field == 'In Speech':
                    field_map['Tongue_In_Speech'] = value
            elif current_category == 'Intel.':
                if field == 'Words':
                    field_map['Words'] = value
                elif field == 'Sentences':
                    field_map['Sentences'] = value
                elif field == 'Conversation':
                    field_map['Conversation'] = value
        
        # Extract values in the specified order
        field_order = [
            'Cough', 'Swallow', 'Dribble/Drool',
            'Resp_At_Rest', 'Resp_In_Speech',
            'Lips_At_Rest', 'Spread', 'Seal', 'Lips_Alternate', 'Lips_In_Speech',
            'Jaw_At_Rest', 'Jaw_In_Speech',
            'Fluids', 'Maintenance', 'Palate_In_Speech',
            'Time', 'Pitch', 'Volume', 'Laryngeal_In_Speech',
            'Tongue_At_Rest', 'Protrusion', 'Elevation', 'Lateral', 'Tongue_Alternate', 'Tongue_In_Speech',
            'Words', 'Sentences', 'Conversation'
        ]
        
        for field in field_order:
            value = field_map.get(field, '')
            frenchay_values.append(parse_frenchay_value(value))
        
    except Exception as e:
        print(f"Error reading CSV {csv_path}: {e}")
        import traceback
        traceback.print_exc()
        # Return zeros if CSV can't be read
        frenchay_values = [0.0] * 28
    
    return frenchay_values


def extract_mfcc_features(wav_path: str, n_mfcc: int = 128) -> np.ndarray:
    """Extract MFCC features from WAV file."""
    try:
        # Load audio file
        y, sr = librosa.load(wav_path, sr=None)
        
        # Extract MFCC features
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        
        # Take mean across time dimension to get a fixed-size feature vector
        mfcc_mean = np.mean(mfccs, axis=1)
        
        return mfcc_mean
    except Exception as e:
        print(f"Error processing WAV file {wav_path}: {e}")
        return np.zeros(n_mfcc)


def read_prompt(prompt_path: str) -> str:
    """Read prompt text from file."""
    try:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        print(f"Error reading prompt {prompt_path}: {e}")
        return ""


def generate_prompt_embeddings(
    prompts: List[str],
    model_name: str = DEFAULT_S_BERT_MODEL,
    batch_size: int = PROMPT_EMBED_BATCH_SIZE,
    normalize: bool = PROMPT_EMBED_NORMALIZE
) -> np.ndarray | None:
    """Encode prompts with a Sentence-BERT model."""
    if not prompts:
        return np.zeros((0, 0), dtype=np.float32)
    
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print(
            "Warning: sentence-transformers not installed. "
            "Run `pip install sentence-transformers` to generate prompt embeddings."
        )
        return None
    
    print(
        f"\nEncoding {len(prompts)} prompts with Sentence-BERT "
        f"('{model_name}', batch_size={batch_size})..."
    )
    
    try:
        model = SentenceTransformer(model_name)
        embeddings = model.encode(
            prompts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=normalize
        )
        return embeddings.astype(np.float32)
    except Exception as exc:
        print(f"Warning: Failed to encode prompts with Sentence-BERT: {exc}")
        return None


def process_speaker_folder(speaker_folder: Path, base_path: Path) -> List[Dict[str, Any]]:
    """Process all sessions for a speaker and return data entries."""
    speaker_name = speaker_folder.name
    print(f"Processing speaker: {speaker_name}")
    
    # Determine Y value: 1 if no 'C' in folder name, 0 if 'C' present
    has_dystharthia = 'C' not in speaker_name
    y_value = 1 if has_dystharthia else 0
    
    # Read Frenchay notes from CSV (if available)
    csv_path = speaker_folder / "Notes" / f"{speaker_name}.csv"
    if not csv_path.exists():
        print(f"Warning: CSV file not found for {speaker_name}: {csv_path}")
        # Use zeros for Frenchay features if CSV doesn't exist
        frenchay_notes = [0.0] * 28
        print(f"  Using zero Frenchay features (no CSV available)")
    else:
        frenchay_notes = extract_frenchay_notes(str(csv_path))
        expected_frenchay_count = 28  # Should match the number of fields
        if len(frenchay_notes) != expected_frenchay_count:
            print(f"  Warning: Expected {expected_frenchay_count} Frenchay features, got {len(frenchay_notes)}")
            # Pad or truncate to expected length
            if len(frenchay_notes) < expected_frenchay_count:
                frenchay_notes.extend([0.0] * (expected_frenchay_count - len(frenchay_notes)))
            else:
                frenchay_notes = frenchay_notes[:expected_frenchay_count]
        print(f"  Extracted {len(frenchay_notes)} Frenchay features")
    
    data_entries = []
    
    # Process all sessions
    for session_dir in sorted(speaker_folder.iterdir()):
        if not session_dir.is_dir() or session_dir.name == "Notes":
            continue
        
        session_name = session_dir.name
        wav_dir = session_dir / "wav_arrayMic"
        prompts_dir = session_dir / "prompts"
        
        if not wav_dir.exists():
            print(f"  Warning: wav_arrayMic not found in {session_name}")
            continue
        
        # Get all WAV files
        wav_files = sorted(wav_dir.glob("*.wav"))
        print(f"  Processing {session_name}: {len(wav_files)} WAV files")
        
        for wav_file in wav_files:
            # Get corresponding prompt file
            wav_stem = wav_file.stem
            prompt_file = prompts_dir / f"{wav_stem}.txt"
            
            if not prompt_file.exists():
                # Try alternative naming (some might have different extensions or naming)
                # Skip if no prompt found
                print(f"    Warning: Prompt not found for {wav_file.name}, skipping")
                continue
            
            # Extract features
            mfcc_features = extract_mfcc_features(str(wav_file))
            
            # Check if MFCC extraction was successful
            if mfcc_features is None or len(mfcc_features) != 128:
                print(f"    Warning: Failed to extract MFCC features for {wav_file.name}, skipping")
                continue
            
            prompt_text = read_prompt(str(prompt_file))
            
            # Create data entry
            # Store full path for wav_file to avoid issues with multiple base paths
            entry = {
                'speaker': speaker_name,
                'session': session_name,
                'wav_file': str(wav_file),
                'mfcc_features': mfcc_features,
                'prompt': prompt_text,
                'frenchay_notes': np.array(frenchay_notes),
                'y': y_value
            }
            
            data_entries.append(entry)
    
    print(f"  Total entries for {speaker_name}: {len(data_entries)}")
    return data_entries


def main():
    """Main function to process all TORGO data."""
    # Base paths containing speaker folders
    data_root = Path("/Users/kuangfy/Desktop/229 FP/data")
    base_paths = [
        data_root / "F",
        data_root / "MC",
        data_root / "FC",
        data_root / "M",
    ]
    
    # Filter to only existing paths
    existing_paths = [p for p in base_paths if p.exists()]
    
    if not existing_paths:
        print("Error: No base paths found!")
        return
    
    # Output directory
    output_dir = Path("/Users/kuangfy/Desktop/229 FP/torgo_processed_data")
    output_dir.mkdir(exist_ok=True)
    
    print(f"Processing TORGO data from: {len(existing_paths)} directory(ies)")
    for path in existing_paths:
        print(f"  - {path}")
    print(f"Output directory: {output_dir}")
    print("-" * 60)
    
    all_data_entries = []
    
    # Process each base path
    for base_path in existing_paths:
        print(f"\nProcessing directory: {base_path.name}")
        # Process each speaker folder
        for speaker_folder in sorted(base_path.iterdir()):
            if not speaker_folder.is_dir() or speaker_folder.name.startswith('.'):
                continue
            
            entries = process_speaker_folder(speaker_folder, base_path)
            all_data_entries.extend(entries)
    
    print("-" * 60)
    print(f"Total data entries: {len(all_data_entries)}")
    
    # Count by class
    y_values = [entry['y'] for entry in all_data_entries]
    dystharthia_count = sum(y_values)
    no_dystharthia_count = len(y_values) - dystharthia_count
    print(f"  Dystharthia (Y=1): {dystharthia_count}")
    print(f"  No Dystharthia (Y=0): {no_dystharthia_count}")
    
    # Save data
    output_file = output_dir / "torgo_dataset.pkl"
    print(f"\nSaving dataset to: {output_file}")
    
    with open(output_file, 'wb') as f:
        pickle.dump(all_data_entries, f)
    
    # Also save as numpy arrays for easier loading
    X_mfcc = np.array([entry['mfcc_features'] for entry in all_data_entries])
    X_frenchay = np.array([entry['frenchay_notes'] for entry in all_data_entries])
    Y = np.array([entry['y'] for entry in all_data_entries])
    
    # Save prompts separately (as list since they're strings)
    prompts = [entry['prompt'] for entry in all_data_entries]
    prompt_embeddings = generate_prompt_embeddings(prompts)
    prompt_embeddings_shape = (
        prompt_embeddings.shape if prompt_embeddings is not None and prompt_embeddings.size else None
    )
    
    np.save(output_dir / "X_mfcc.npy", X_mfcc)
    np.save(output_dir / "X_frenchay.npy", X_frenchay)
    np.save(output_dir / "Y.npy", Y)
    
    with open(output_dir / "prompts.pkl", 'wb') as f:
        pickle.dump(prompts, f)
    
    if prompt_embeddings_shape:
        np.save(output_dir / "X_prompt_sbert.npy", prompt_embeddings)
    
    # Save metadata
    metadata = {
        'total_entries': len(all_data_entries),
        'mfcc_features': 128,
        'frenchay_features': len(FRENCHAY_FIELDS),
        'dystharthia_count': dystharthia_count,
        'no_dystharthia_count': no_dystharthia_count,
        'frenchay_field_names': FRENCHAY_FIELDS,
        'prompt_embedding_model': DEFAULT_S_BERT_MODEL if prompt_embeddings_shape else None,
        'prompt_embedding_dim': prompt_embeddings_shape[1] if prompt_embeddings_shape else 0
    }
    
    with open(output_dir / "metadata.pkl", 'wb') as f:
        pickle.dump(metadata, f)
    
    print("\nDataset generation complete!")
    print(f"Files saved in: {output_dir}")
    print(f"  - torgo_dataset.pkl: Full dataset with all information")
    print(f"  - X_mfcc.npy: MFCC features (shape: {X_mfcc.shape})")
    print(f"  - X_frenchay.npy: Frenchay notes (shape: {X_frenchay.shape})")
    print(f"  - Y.npy: Labels (shape: {Y.shape})")
    if prompt_embeddings_shape:
        print(f"  - X_prompt_sbert.npy: Sentence-BERT embeddings (shape: {prompt_embeddings_shape})")
    else:
        print("  - X_prompt_sbert.npy: skipped (install sentence-transformers to enable)")
    print(f"  - prompts.pkl: Prompt texts")
    print(f"  - metadata.pkl: Dataset metadata")


if __name__ == "__main__":
    main()

