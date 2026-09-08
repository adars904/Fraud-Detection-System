import joblib
import os


def save_model(model, threshold, model_path="artifacts/model.pkl", threshold_path="artifacts/threshold.pkl"):
    """
    Save a fitted pipeline and its selected classification threshold to disk.
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    joblib.dump(model, model_path)
    joblib.dump(threshold, threshold_path)

    print(f"Model saved to: {model_path}")
    print(f"Threshold saved to: {threshold_path}")


def load_model(model_path="artifacts/model.pkl", threshold_path="artifacts/threshold.pkl"):
    """
    Load a previously saved fitted pipeline and its threshold.
    """
    model = joblib.load(model_path)
    threshold = joblib.load(threshold_path)

    return model, threshold