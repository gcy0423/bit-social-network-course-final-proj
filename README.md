# Weibo Engagement Prediction for Social Network Analysis

Final project for the **Social Network Analysis 2026** course at **Beijing Institute of Technology**.

## Team Members

Gu Changyu, Liu Bingzhen, Ren Zhenni, An Yi and Shahid Muhammad Behzad

## Project Overview

This project aims to predict the engagement of Weibo posts, including:

- `forward_count`
- `comment_count`
- `like_count`

Given historical post data (`uid`, `mid`, `time`, `content`, and labels), the model predicts the engagement counts of future posts one week after publication.

The current best version is based on:

- user historical behavior features
- recent temporal statistics
- pre-trained sentence embeddings
- a two-stage modeling strategy:
  - stage 1: classify whether the target is zero or non-zero
  - stage 2: regress the positive value

This design matches the competition metric better than plain regression.

## Environment Setup

### Recommended Environment

- Python 3.9 or 3.10
- Linux
- CUDA is optional but recommended for sentence embedding generation
- GPU is helpful, but the code can also run on CPU

### Create Environment

Using `conda`:

```bash
conda create -n weibo python=3.10 -y
conda activate weibo
````

Install dependencies:

```bash
pip install numpy pandas scipy scikit-learn lightgbm joblib torch sentence-transformers
```

## Data Preparation

Place the dataset files in a local directory, for example:

```text
data/
├── weibo_train_data.txt
└── weibo_predict_data.txt
```

### Training Data Format

Each line in the training file should contain:

```text
uid    mid    time    forward_count    comment_count    like_count    content
```

### Test Data Format

Each line in the prediction file should contain:

```text
uid    mid    time    content
```

The file should be tab-separated.

## Main Script

The main script is:

```text
weibo_lgbm_baseline_v5.py
```

This version is the recommended one.

## Configuration

This project does **not** rely on command-line arguments.
Instead, you should edit the `CONFIG` dictionary at the bottom of the Python file.

Example:

```python
CONFIG = {
    "mode": "validate",
    "train_path": "data/weibo_train_data.txt",
    "test_path": "data/weibo_predict_data.txt",
    "valid_start": "2015-07-01",
    "cv_valid_starts": ["2015-06-01", "2015-07-01"],
    "output_dir": "./outputs/weibo_baseline_v5",
    "submission_name": "submission.txt",

    "use_sentence_embeddings": True,
    "sentence_model_name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "embedding_device": "auto",
    "embedding_batch_size": 128,
    "embedding_normalize": True,
    "embedding_max_chars": 192,
    "embedding_max_seq_length": 192,
    "embedding_text_prefix": "",
    "embedding_cache_dir": "./outputs/weibo_baseline_v5/embedding_cache",

    "recent_ks": [3, 5, 10],
    "recent_days": [7, 14, 30],

    "seed": 42,
    "num_threads": 8,
    "validate_config": None,
}
```

## How to Use

### 1. Single Validation

Use this mode to run one validation split.

```python
"mode": "validate"
```

Then run:

```bash
python weibo_lgbm_baseline_v5.py
```

This is useful for quick experiments and local tuning.

### 2. Strict Rolling CV

Use this mode to evaluate generalization across different months.

```python
"mode": "cv"
```

Then run:

```bash
python weibo_lgbm_baseline_v5.py
```

This performs month-based non-overlapping validation, for example:

* train on February–May, validate on June
* train on February–June, validate on July

This mode is more reliable for estimating online performance.

### 3. Full Training and Submission Generation

Use this mode to train on the full training set and generate predictions for the test set.

```python
"mode": "full"
```

Set `validate_config` to a configuration file obtained from `validate` or `cv`, for example:

```python
"validate_config": "./outputs/weibo_baseline_v5/fold_2_2015-07-01/fold_summary.json"
```

Then run:

```bash
python weibo_lgbm_baseline_v5.py
```

## Output Files

The script saves results to `output_dir`.

Typical outputs include:

* `validate_config.json`
* `fold_summary.json`
* `cv_summary.csv`
* `cv_summary.json`
* `submission.txt`

### Submission Format

The final submission file follows this format:

```text
uid<TAB>mid<TAB>forward_count,comment_count,like_count
```

All predicted counts must be integers.

## Recommended Workflow

1. Run `validate` for quick testing
2. Run `cv` for more reliable evaluation
3. Select the best configuration
4. Run `full` to generate the final submission

## Notes

* The first run may download the pre-trained sentence embedding model from Hugging Face.
* Sentence embeddings are cached locally, so repeated runs are faster.
* GPU mainly helps with sentence embedding generation. The LightGBM part mainly uses CPU.
* If memory is limited, reduce `embedding_batch_size`.

## Acknowledgment

This repository was developed as the final project for the **Social Network Analysis 2026** course at **Beijing Institute of Technology**.
