"""
Title: The file evaluates the performance of the model
Author: Vlad Cozma
Creation Date: 23.05.2026
"""

import pandas as pd
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_auc_score
)

from tensorflow.keras.preprocessing.image import ImageDataGenerator


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

VAL_PATH = DATA_DIR / "val_df.csv"
TEST_PATH = DATA_DIR / "test_df.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "deepfake_detector.h5"
HISTORY_PATH = DATA_DIR / "model_history.csv"

IMG_SIZE = 224
BATCH_SIZE = 32

history_df = None
val_df = None
test_df = None
model = None
true_labels = None
preds = None
test_datagen = None
test_generator = None
pred_probs = None


def load_dataframes():
    global history_df
    global test_df

    history_df = pd.read_csv(HISTORY_PATH)
    test_df = pd.read_csv(TEST_PATH)

def load_model():
    global model

    model = tf.keras.models.load_model(MODEL_PATH)

def init_dataloaders():
    global test_datagen
    global test_generator

    test_datagen = ImageDataGenerator()

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

def show_accuracy_curve():
    global history_df

    plt.figure(figsize=(10,5))

    plt.plot(
        history_df['accuracy'],
        label='Train Accuracy'
    )

    plt.plot(
        history_df['val_accuracy'],
        label='Validation Accuracy'
    )

    plt.legend()

    plt.title("Accuracy Curve")

    plt.show()

def show_confusion_matrix():
    global true_labels
    global preds

    cm = confusion_matrix(
    true_labels,
    preds
    )

    plt.figure(figsize=(6,5))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues'
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.title("Confusion Matrix")

    plt.show()

def get_accuracy():
    global model
    global true_labels
    global preds
    global pred_probs

    # Predict on test set
    pred_probs = model.predict(
        test_generator
    )

    pred_probs = pred_probs.ravel()
    preds = (pred_probs > 0.5).astype(int)

    true_labels = test_generator.classes

    # Show accuracy
    acc = accuracy_score(
        true_labels,
        preds
    )
    print("Accuracy:", acc)


def get_report():
    global true_labels
    global preds

    # Show model report
    print(
        classification_report(
            true_labels,
            preds
        )
    )

def get_auc_score():
    global true_labels
    global pred_probs

    # Show ROC AUC score
    auc = roc_auc_score(
        true_labels,
        pred_probs
    )
    print("ROC AUC Score:", auc)


def main():
    load_dataframes()
    load_model()
    init_dataloaders()

    get_accuracy()
    get_report()
    get_auc_score()
    show_accuracy_curve()
    show_confusion_matrix()


if __name__ == "__main__":
    main()