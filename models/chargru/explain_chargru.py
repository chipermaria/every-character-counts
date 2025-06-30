import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
import json
import os

from data import Data
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split


parser = argparse.ArgumentParser(description="Run Grad-CAM explainability for CharGRU model.")
parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
parser.add_argument("--num_samples", type=int, default=300, help="Number of test samples to run Grad-CAM on")
args = parser.parse_args()

with open("config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

alphabet = config["data"]["alphabet"]
input_size = config["data"]["input_size"]

model = load_model(args.model_path)
model_basename = os.path.splitext(os.path.basename(args.model_path))[0]

df = pd.read_csv("../../data/custom_email_dataset.csv")
df = df.dropna(subset=["body", "label"])
df["labels"] = df["label"].map({"phishing": 1, "clean": 0})
df["contents"] = df["body"].astype(str).str.replace("\x00", "", regex=False)

X_train, X_test_val, y_train, y_test_val = train_test_split(
    df["contents"], df["labels"], test_size=0.3, random_state=42
)
X_test, X_val, y_test, y_val = train_test_split(
    X_test_val, y_test_val, test_size=0.5, random_state=42
)

df_test = pd.DataFrame({"contents": X_test, "labels": y_test}).reset_index(drop=True)


def gradcam_chargru(model, input_tensor, layer_name="gru", class_idx=None):
    """
    Compute Grad-CAM over a GRU layer for a given input tensor.
    Returns: normalized CAM array and model confidence for the given class.
    """
    grad_model = tf.keras.models.Model([model.input], [model.get_layer(layer_name).output, model.output])
    with tf.GradientTape() as tape:
        gru_output, predictions = grad_model(input_tensor)
        if class_idx is None:
            class_idx = tf.argmax(predictions[0])
        class_output = predictions[:, class_idx]
    grads = tape.gradient(class_output, gru_output)[0]
    weights = tf.reduce_mean(grads, axis=0)
    cam = tf.reduce_sum(tf.multiply(weights, gru_output[0]), axis=-1)
    cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-8)
    return cam.numpy(), predictions.numpy()[0][class_idx]


def highlight_text_with_cam(text, cam, input_size, color_scheme="red"):
    """
    Generate an HTML-formatted string that visualizes importance scores (CAM)
    overlaid on input characters using color transparency.
    """
    tokens = list(text.strip())[-input_size:]
    cam = cam[-len(tokens):]
    cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-8)
    cam = np.power(cam, 2.5)

    rgb = (255, 0, 0) if color_scheme == "red" else (0, 0, 139)
    html = ""
    for char, score in zip(tokens, cam):
        opacity = min(max(score, 0.05), 1.0)
        color = f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {opacity:.2f})"
        html += f"<span style='background-color:{color}; font-size:38px;'>{char}</span>"
    return html


def export_classified_html(results_df, input_size, filename, condition, color_scheme):
    """
    Export HTML visualization for a filtered subset of results (e.g. only true positives).
    """
    subset = results_df[condition]
    html_blocks = []
    for _, row in subset.iterrows():
        text = row["text"]
        cam = np.array(row["cam"])
        cam_html = highlight_text_with_cam(text, cam, input_size, color_scheme=color_scheme)
        html_blocks.append(
            f"<div style='margin:10px;padding:10px;border:1px solid #ccc;'>"
            f"<strong>Label: {row['label']} | Prediction: {row['prediction']}</strong><br>"
            f"{cam_html}</div>"
        )
    html_content = "<html><body>" + "\n".join(html_blocks) + "</body></html>"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Saved: {filename}")


converter = Data([], alphabet, input_size)  
results = []

df_test_sample = df_test.head(args.num_samples)

for i, row in df_test_sample.iterrows():
    text = row["contents"]
    label = row["labels"]
    try:
        input_tensor = np.expand_dims(converter.str_to_indexes(text), axis=0)
        cam, confidence = gradcam_chargru(model, input_tensor, class_idx=label)
        prediction = int(confidence >= 0.5)
        results.append({
            "text": text,
            "label": int(label),
            "prediction": prediction,
            "confidence": float(confidence),
            "cam": cam.tolist()
        })
    except Exception as e:
        print(f"Error at index {i}: {e}")

df_results = pd.DataFrame(results)
df_results["cam"] = df_results["cam"].apply(np.array)


export_classified_html(
    df_results, input_size, f"{model_basename}_tp.html",
    (df_results.label == 1) & (df_results.prediction == 1), color_scheme="red"
)

export_classified_html(
    df_results, input_size, f"{model_basename}_fp.html",
    (df_results.label == 0) & (df_results.prediction == 1), color_scheme="red"
)

export_classified_html(
    df_results, input_size, f"{model_basename}_tn.html",
    (df_results.label == 0) & (df_results.prediction == 0), color_scheme="blue"
)

export_classified_html(
    df_results, input_size, f"{model_basename}_fn.html",
    (df_results.label == 1) & (df_results.prediction == 0), color_scheme="blue"
)

