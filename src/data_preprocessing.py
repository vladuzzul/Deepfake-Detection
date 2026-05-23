'''
Title: The file that download the dataset
Author: Vlad Cozma
Creation Date: 19.05.2026
'''

import os
from pathlib import Path
import requests
from tqdm import tqdm
import numpy as np
import pandas as pd
from PIL import Image
import warnings 
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

DATASET_PATH = DATA_DIR / "FINAL_DATASET.csv"
DOWNLOADED_IMAGES_PATH = DATA_DIR / "downloaded_images.csv"
IMAGE_FOLDER = DATA_DIR / "working_images"

TRAIN_PATH = DATA_DIR / "train_df.csv"
VAL_PATH = DATA_DIR / "val_df.csv"
TEST_PATH = DATA_DIR / "test_df.csv"

df = None

def load_dataframe():
    global df
    print("\nLoading dataframe...")
    df = pd.read_csv(DATASET_PATH)
    print("Dataframe loaded.")

    print("\nDataset keys: ")
    print(df.keys())

    print("\nDataset Information:")
    print(df.describe())

    print("Initial Images:", df.shape)


def download_images():
    global df
    os.makedirs(IMAGE_FOLDER, exist_ok=True)

    # Download images only if they aren't downloaded
    if os.path.exists(DOWNLOADED_IMAGES_PATH):
        print("Loading cached CSV...")
        df = pd.read_csv(DOWNLOADED_IMAGES_PATH)
    else:
        print("Downloading images...")

        image_paths = []

        for idx, row in tqdm(df.iterrows(), total=len(df)):
            url = row['image_url']

            try:
                response = requests.get(url, timeout=10)

                file_path = IMAGE_FOLDER / f"{idx}.jpg"

                with open(file_path, "wb") as f:
                    f.write(response.content)

                image_paths.append(str(file_path))

            except:
                image_paths.append(None)

        df["image_path"] = image_paths

        # Check if the image downloaded successfully
        valid_images = []

        for path in df["image_path"]:
            try:
                if pd.isna(path):
                    raise Exception("Missing path")

                img = Image.open(path)
                img.verify()

                valid_images.append(True)

            except:
                valid_images.append(False)

        df = df[valid_images].reset_index(drop=True)

    df.dropna()
    df.to_csv(DOWNLOADED_IMAGES_PATH, index=False)

    print(f"Saved CSV to: {DOWNLOADED_IMAGES_PATH}")

    if "image_path" not in df.columns:
        df = pd.read_csv(DOWNLOADED_IMAGES_PATH)

    print("Remaining Images:", df.shape)


def split_dataset():
    global df
    # Split the dataset
    train_df = df[df['dataset_split'] == 'train']
    val_df   = df[df['dataset_split'] == 'val']
    test_df  = df[df['dataset_split'] == 'test']

    train_df.to_csv(TRAIN_PATH)
    print(f"Saved train CSV to: {TRAIN_PATH}")

    val_df.to_csv(VAL_PATH)
    print(f"Saved validation CSV to: {VAL_PATH}")

    test_df.to_csv(TEST_PATH)
    print(f"Saved testing CSV to: {TEST_PATH}")
    

    print("\nTraining dataframe:", train_df.shape)
    print("Validation dataframe:", val_df.shape)
    print("Test dataframe:", test_df.shape)


def main():
    load_dataframe()
    download_images()
    split_dataset()

if __name__ == "__main__":
    main()
