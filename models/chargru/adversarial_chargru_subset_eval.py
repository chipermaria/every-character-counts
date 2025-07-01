import pandas as pd
import numpy as np
import json

from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score


from data import Data
from metrics import compute_print_f1

if __name__ == "__main__":

    with open("config.json", encoding="utf-8") as f:
        config = json.load(f)

    alphabet = config["data"]["alphabet"]
    input_size = config["data"]["input_size"]

    csv_path = "../../data/subset_1000.csv" 
    df = pd.read_csv(csv_path)

    df = df.dropna(subset=["body", "label"])
    df["body"] = df["body"].astype(str).str.slice(0, input_size)
    df["label"] = df["label"].str.lower().map({
        "phishing": 1,
        "clean": 0,
    })
    df = df.dropna(subset=["label"])

    data = Data(list(zip(df["body"], df["label"])), alphabet, input_size)
    X, y_onehot = data.convert_data()
    y_true = np.argmax(y_onehot, axis=1)

    model = load_model("adv_train_test_model.h5")

    y_probs = model.predict(X, batch_size=128, verbose=1)
    y_pred = np.argmax(y_probs, axis=1)

    print("\n=== Evaluation Metrics ===")
    acc = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {acc:.5f}")
    compute_print_f1(y_pred, y_true, average="macro")
    compute_print_f1(y_pred, y_true, average="weighted")

    df["predicted"] = y_pred
    output_path = csv_path.replace(".csv", "_chargru_predictions.csv")
    df.to_csv(output_path, index=False)
    print(f"Predictions saved to: {output_path}")
