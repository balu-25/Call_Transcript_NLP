"""
prepare_dataset.py
-------------------
Builds the training dataset used by the Purpose Classifier.

This project is designed around the Kaggle dataset:
    "Call Center Conversation / Customer Support Dialogues"
    (e.g. https://www.kaggle.com/datasets/veeralakrishna/call-centre-dataset
     or  https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter )

Because Kaggle requires authenticated API access (kaggle.json credentials),
this script supports TWO modes:

1. KAGGLE MODE (preferred for real deployment):
   - Place your kaggle.json in ~/.kaggle/kaggle.json
   - Run:  python prepare_dataset.py --kaggle "veeralakrishna/call-centre-dataset"
   - It downloads + merges the dataset into data/calls_dataset.csv

2. OFFLINE / BOOTSTRAP MODE (default, no internet needed):
   - Uses a curated, hand-labelled seed corpus (data/seed_labeled_calls.csv)
     covering all 5 purpose classes (Family, Friend, Complaint, Business, Fraud)
     plus tone labels, shipped with this project so the app works out of the box.
   - Run:  python prepare_dataset.py

Either way, the output is data/calls_dataset.csv with columns:
    text, purpose, tone
which model_training.py consumes.
"""

import argparse
import os
import subprocess
import sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SEED_PATH = os.path.join(HERE, "seed_labeled_calls.csv")
OUTPUT_PATH = os.path.join(HERE, "calls_dataset.csv")


def download_from_kaggle(dataset_slug: str, dest_dir: str):
    """Downloads a dataset from Kaggle using the kaggle CLI (requires kaggle.json)."""
    try:
        import kaggle  # noqa: F401
    except ImportError:
        print("Installing kaggle package...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle", "--break-system-packages"])

    os.makedirs(dest_dir, exist_ok=True)
    cmd = ["kaggle", "datasets", "download", "-d", dataset_slug, "-p", dest_dir, "--unzip"]
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)
    print(f"Downloaded to {dest_dir}. Inspect the CSV(s) and map columns to text/purpose/tone,")
    print("then merge them into calls_dataset.csv manually or extend this script's merge logic.")


def build_offline_dataset():
    """Uses the bundled seed corpus as the training dataset."""
    if not os.path.exists(SEED_PATH):
        raise FileNotFoundError(
            f"Seed file not found at {SEED_PATH}. It should ship with this project."
        )
    df = pd.read_csv(SEED_PATH)
    df = df.dropna(subset=["text", "purpose", "tone"])
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Offline dataset built: {OUTPUT_PATH} ({len(df)} rows)")
    print(df["purpose"].value_counts())
    return df


def main():
    parser = argparse.ArgumentParser(description="Prepare call transcript dataset")
    parser.add_argument("--kaggle", type=str, default=None,
                         help="Kaggle dataset slug, e.g. 'owner/dataset-name'")
    args = parser.parse_args()

    if args.kaggle:
        dest = os.path.join(HERE, "kaggle_raw")
        download_from_kaggle(args.kaggle, dest)
        print("NOTE: after downloading, update this script's merge step to map the")
        print("Kaggle columns into text/purpose/tone format, or use the offline seed set.")
    else:
        build_offline_dataset()


if __name__ == "__main__":
    main()
