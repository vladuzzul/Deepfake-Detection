"""
Title: The file that trains and creates the model
Author: Vlad Cozma
Creation Date: 21.05.2026
"""

import pandas as pd
from pathlib import Path

from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau
)

from data_preprocessing import main

import mlflow

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

DATASET_PATH = DATA_DIR / "FINAL_DATASET.csv"
DOWNLOADED_IMAGES_PATH = DATA_DIR / "downloaded_images.csv"
IMAGE_FOLDER = DATA_DIR / "working_images"

TRAIN_PATH = DATA_DIR / "train_df.csv"
VAL_PATH = DATA_DIR / "val_df.csv"
TEST_PATH = DATA_DIR / "test_df.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "deepfake_detector.h5"
HISTORY_PATH = DATA_DIR / "model_history.csv"

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 15
EXPERIMENT_NAME = "deepfake-detection"


# Downloads the dataset
main()


df = pd.read_csv(DOWNLOADED_IMAGES_PATH)
train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VAL_PATH)
test_df = pd.read_csv(TEST_PATH)

mlflow.set_experiment(EXPERIMENT_NAME)
mlflow.start_run(run_name="training")
mlflow.log_params({
    "img_size": IMG_SIZE,
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "base_model": "EfficientNetB0",
    "base_weights": "imagenet",
    "base_trainable": False,
    "dropout": 0.3,
    "optimizer": "adam",
    "loss": "binary_crossentropy",
    "train_samples": len(train_df),
    "val_samples": len(val_df),
    "test_samples": len(test_df)
})

# Prepare the training and validation images
train_datagen = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator()

# Dataloader for training data
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='image_path',
    y_col='label',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# Dataloader for validation data
val_generator = test_datagen.flow_from_dataframe(
    dataframe=val_df,
    x_col='image_path',
    y_col='label',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# Dataloader for testing data
test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col='image_path',
    y_col='label',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

# Use EfficientNet as a base model
base_model = EfficientNetB0(
    weights='imagenet',
    include_top=False,
    input_shape=(224,224,3)
)
base_model.trainable = False

# Make a custom model based on the loaded one
model = Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(
        1,
        activation='sigmoid'
    )
])
model.summary()

# Configure the model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Create callbacks to not waste time on training
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    patience=2,
    factor=0.2
)

# Train model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=[
        early_stop,
        reduce_lr
    ]
)

for epoch in range(len(history.history["loss"])):
    mlflow.log_metrics({
        "train_loss": history.history["loss"][epoch],
        "train_accuracy": history.history["accuracy"][epoch],
        "val_loss": history.history["val_loss"][epoch],
        "val_accuracy": history.history["val_accuracy"][epoch]
    }, step=epoch + 1)

# Save the history of the model for evaluation
history_df = pd.DataFrame(history.history)
history_df.to_csv(HISTORY_PATH)
mlflow.log_artifact(str(HISTORY_PATH))

# Save model
model.save(
    MODEL_PATH
)
mlflow.log_artifact(str(MODEL_PATH))
mlflow.end_run()
print("Model Saved Successfully!")   
