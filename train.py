'''
Title: The file that trains the model
Author: Vlad Cozma
Creation Date: 19.05.2026
'''

import os
import requests
from tqdm import tqdm
import cv2
import random
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import warnings 
warnings.filterwarnings('ignore')

DATASET_PATH = "dataset/FINAL_DATASET.csv"

print("\nLoading dataframe...")
df = pd.read_csv(DATASET_PATH)
print("Dataframe loaded.")

print("\nDataset keys: ")
print(df.keys())

print("\nDataset Information:")
print(df.describe())



DOWNLOADED_IMAGES_PATH = "dataset/downloaded_images.csv"
IMAGE_FOLDER = "dataset/working_images"

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

    df.to_csv(DOWNLOADED_IMAGES_PATH, index=False)

    print(f"Saved CSV to: {DOWNLOADED_IMAGES_PATH}")

if "image_path" not in df.columns:
    df = pd.read_csv(DOWNLOADED_IMAGES_PATH)

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

print("Remaining Images:", df.shape)

def show_images(label_name, n):

    sample_df = df[df['label'] == label_name].sample(n)

    fig, axes = plt.subplots(3,4, figsize=(15,10))

    axes = axes.flatten()

    for ax, (_, row) in zip(axes, sample_df.iterrows()):

        img_path = row['image_path']

        try:
            image = np.asarray(Image.open(img_path))

            ax.imshow(image)
            ax.axis('off')

        except:
            ax.text(0.5,0.5,'Image Error',
                    ha='center', va='center')

    plt.suptitle(f"{label_name} Samples", fontsize=22)

    plt.tight_layout()
    plt.show()

show_images("FAKE", 12)
