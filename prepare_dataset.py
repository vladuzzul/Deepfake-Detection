'''
Title: The file that download the dataset and trains the model
Author: Vlad Cozma
Creation Date: 19.05.2026
'''

import os
import requests
from tqdm import tqdm
import numpy as np
import pandas as pd
from PIL import Image
import warnings 
warnings.filterwarnings('ignore')

DATASET_PATH = "dataset/FINAL_DATASET.csv"
DOWNLOADED_IMAGES_PATH = "dataset/downloaded_images.csv"
IMAGE_FOLDER = "dataset/working_images"

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

                file_path = f"{IMAGE_FOLDER}/{idx}.jpg"

                with open(file_path, "wb") as f:
                    f.write(response.content)

                image_paths.append(file_path)

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

    print("\nTraining dataframe:", train_df.shape)
    print("Validation dataframe:", val_df.shape)
    print("Test dataframe:", test_df.shape)


def main():
    load_dataframe()
    download_images()
    split_dataset()

if __name__ == "__main__":
    main()
