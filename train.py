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

print("Initial Images:", df.shape)

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

df.dropna()
df.to_csv(DOWNLOADED_IMAGES_PATH, index=False)

print(f"Saved CSV to: {DOWNLOADED_IMAGES_PATH}")


