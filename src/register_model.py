from pathlib import Path

import mlflow
import tensorflow as tf
from mlflow.tracking import MlflowClient

def register_model():
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    MODEL_PATH = PROJECT_ROOT / "models" / "deepfake_detector.h5"
    TRACKING_DIR = PROJECT_ROOT / "mlruns"
    mlflow.set_tracking_uri(str(TRACKING_DIR))

    registered_name = "deepfake_detector_model"
    registered_alias = "current"

    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model file not found at {MODEL_PATH}. Train the model first.")

    keras_model = tf.keras.models.load_model(MODEL_PATH)

    mlflow.set_experiment("model-registration")

    with mlflow.start_run(run_name="register-model") as registration_run:
        mlflow.keras.log_model(keras_model, artifact_path="model")
        model_uri = f"runs:/{registration_run.info.run_id}/model"

    result = mlflow.register_model(model_uri=model_uri, name=registered_name)
    MlflowClient().set_registered_model_alias(registered_name, registered_alias, result.version)

    print(f"Registered model: {result.name} (version {result.version})")
    print(f"Serve with: mlflow models serve -m \"models:/{result.name}@{registered_alias}\" -p 1234 --env-manager local")

if __name__ == "__main__":
    register_model()