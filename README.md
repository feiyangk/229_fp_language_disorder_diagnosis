# 229_fp_language_disorder_diagnosis

## Setup

### Create Virtual Environment

Create and activate a Python virtual environment:

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate
```

### Install Dependencies

Install required packages:

```bash
pip install numpy scikit-learn

# Optional: Install SHAP for ClusterSHAP analysis (used in K-Means model)
pip install shap
```

## Training Models

This project includes three models for dysarthria detection. All models use processed TORGO features (MFCC/wav2vec, Frenchay, and optional Sentence-BERT prompt embeddings).

**Note:** All data directory paths are relative to the project root. If your data is in a different location (e.g., on a remote server), use the `--data-dir` option to specify the full path.

### Logistic Regression

Train a logistic regression classifier:

```bash
python models/logreg_dysarthria.py
```

**Options:**
- `--data-dir`: Directory containing processed TORGO arrays (default: `torgo_processed_data`)
- `--audio-features`: Audio feature type - `mfcc` or `wav2vec` (default: `mfcc`)
- `--test-size`: Fraction of samples for testing (default: `0.2`)
- `--random-state`: Random seed (default: `42`)
- `--max-iter`: Maximum iterations for LogisticRegression (default: `2000`)
- `--model-out`: Optional path to save the trained model (pickle format)

**Examples:**
```bash
# Basic training with default settings
python models/logreg_dysarthria.py

# Train with wav2vec features and save model
python models/logreg_dysarthria.py --audio-features wav2vec --test-size 0.3 --model-out models/logreg_wav2vec.pkl

# Train with custom data directory
python models/logreg_dysarthria.py --data-dir torgo_balanced_subset --max-iter 3000
```

### Gaussian Mixture Model (EM)

Train a Gaussian Mixture Model using Expectation-Maximization:

```bash
python models/em_dysarthria.py
```

**Options:**
- `--data-dir`: Directory containing processed TORGO features (default: `torgo_processed_data`)
- `--components`: Number of Gaussian components (default: `2`)
- `--random-state`: Random seed (default: `42`)
- `--max-iter`: Maximum EM iterations (default: `500`)
- `--covariance-type`: Covariance type - `full`, `tied`, `diag`, or `spherical` (default: `full`)

**Examples:**
```bash
# Basic training with default settings
python models/em_dysarthria.py

# Train with 3 components and diagonal covariance
python models/em_dysarthria.py --components 3 --covariance-type diag --max-iter 1000

# Train with custom data directory
python models/em_dysarthria.py --data-dir torgo_balanced_subset --covariance-type tied
```

### K-Means Clustering

Train a K-Means clustering model:

```bash
python models/kmeans_dysarthria.py
```

**Options:**
- `--data-dir`: Directory containing processed TORGO features (default: `torgo_processed_data`)
- `--clusters`: Number of clusters (default: `2`)
- `--random-state`: Random seed (default: `42`)
- `--max-iter`: Maximum iterations (default: `500`)
- `--cluster-shap`: Enable ClusterSHAP feature attribution analysis
- `--cluster-shap-samples`: Max examples for ClusterSHAP (default: `800`, `0` uses entire dataset)
- `--cluster-shap-top-k`: Top features to display per cluster (default: `10`)
- `--cluster-shap-trees`: Number of trees for surrogate random forest (default: `400`)

**Examples:**
```bash
# Basic training with default settings
python models/kmeans_dysarthria.py

# Train with ClusterSHAP analysis
python models/kmeans_dysarthria.py --clusters 2 --cluster-shap --cluster-shap-top-k 15

# Train with custom data directory and more iterations
python models/kmeans_dysarthria.py --data-dir torgo_balanced_subset --max-iter 1000

# Train with ClusterSHAP using entire dataset
python models/kmeans_dysarthria.py --cluster-shap --cluster-shap-samples 0
```