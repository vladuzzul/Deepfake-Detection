"""
Title: The file evaluates the performance of the model
Author: Vlad Cozma
Creation Date: 23.05.2026
"""

import json
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

import mlflow


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

VAL_PATH = DATA_DIR / "val_df.csv"
TEST_PATH = DATA_DIR / "test_df.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "deepfake_detector.h5"
HISTORY_PATH = DATA_DIR / "model_history.csv"

IMG_SIZE = 224
BATCH_SIZE = 32
EXPERIMENT_NAME = "deepfake-detection"

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

def show_accuracy_curve(save_path=None):
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
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path)
    plt.show()

def show_confusion_matrix(save_path=None):
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
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path)
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
    return acc


def get_report():
    global true_labels
    global preds

    report_text = classification_report(
        true_labels,
        preds,
        zero_division=0
    )
    report_dict = classification_report(
        true_labels,
        preds,
        zero_division=0,
        output_dict=True
    )
    print(report_text)
    return report_text, report_dict

def get_auc_score():
    global true_labels
    global pred_probs

    # Show ROC AUC score
    auc = roc_auc_score(
        true_labels,
        pred_probs
    )
    print("ROC AUC Score:", auc)
    return auc

def test_model():
    user_df = pd.read_csv(DATA_DIR / "user_data" / "user.csv")

    # Dataloader for user testing data
    user_generator = test_datagen.flow_from_dataframe(
        dataframe=user_df,
        x_col="image_path",
        y_col="label",
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=1,
        class_mode='binary',
        shuffle=False
    )

    pred_probs = model.predict(
        user_generator
    )

    pred_probs = pred_probs.ravel()
    preds = (pred_probs > 0.5).astype(int)

    print(preds)

def main():
    mlflow.set_experiment(EXPERIMENT_NAME)
    mlflow.start_run(run_name="evaluation")
    mlflow.log_params({
        "img_size": IMG_SIZE,
        "batch_size": BATCH_SIZE,
        "threshold": 0.5
    })

    load_dataframes()
    load_model()
    init_dataloaders()

    acc = get_accuracy()
    report_text, report_dict = get_report()
    auc = get_auc_score()

    artifacts_dir = DATA_DIR / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    accuracy_curve_path = artifacts_dir / "accuracy_curve.png"
    confusion_matrix_path = artifacts_dir / "confusion_matrix.png"
    report_path = artifacts_dir / "classification_report.txt"
    report_json_path = artifacts_dir / "classification_report.json"

    show_accuracy_curve(save_path=accuracy_curve_path)
    show_confusion_matrix(save_path=confusion_matrix_path)

    report_path.write_text(report_text)
    report_json_path.write_text(json.dumps(report_dict, indent=2))
    mlflow.log_metrics({
        "accuracy": acc,
        "roc_auc": auc
    })
    mlflow.log_artifact(str(accuracy_curve_path))
    mlflow.log_artifact(str(confusion_matrix_path))
    mlflow.log_artifact(str(report_path))
    mlflow.log_artifact(str(report_json_path))
    mlflow.end_run()

if __name__ == "__main__":
    main()