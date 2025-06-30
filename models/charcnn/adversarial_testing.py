import pandas as pd
import random
import json
from sklearn.model_selection import train_test_split
from data import Data
from model import CharCNN


def deepwordbug_attack(text, perturbation_rate, alphabet):
    """
    Applies character-level adversarial perturbations (swap, delete, insert, substitute)
    inspired by DeepWordBug from Ji Gao et al., 2018 (https://arxiv.org/abs/1801.04354)
    to simulate adversarial atacks.
    """
    chars = list(text)
    num_perturb = max(1, int(len(chars) * perturbation_rate))
    indices = random.sample(range(len(chars)), num_perturb)

    for idx in indices:
        operation = random.choice(["swap", "delete", "insert", "substitute"])
        if operation == "swap" and idx < len(chars) - 1:
            chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
        elif operation == "delete":
            chars[idx] = ''
        elif operation == "insert":
            insert_char = random.choice(alphabet)
            chars.insert(idx, insert_char)
        elif operation == "substitute":
            substitute_char = random.choice(alphabet)
            chars[idx] = substitute_char
    return ''.join(chars)


if __name__ == "__main__":

    csv_path = "../../data/custom_email_dataset.csv" 
    df = pd.read_csv(csv_path)

    config_path = "./config.json"
    config = json.load(open(config_path, encoding="utf-8", errors="ignore"))

    df = df.dropna(subset=["body", "label"])
    df["labels"] = df["label"].map({"phishing": 1, "clean": 0})
    df["contents"] = df["body"].astype(str).str.replace("\x00", "", regex=False)

    test_size = 0.3
    val_size = 0.5

    X_train, X_test_val, y_train, y_test_val = train_test_split(
        df["contents"], df["labels"], test_size=test_size, random_state=42
    )
    X_test, X_val, y_test, y_val = train_test_split(
        X_test_val, y_test_val, test_size=val_size, random_state=42
    )

    df_train = pd.DataFrame({"contents": X_train, "labels": y_train}).reset_index(drop=True)
    df_val = pd.DataFrame({"contents": X_val, "labels": y_val}).reset_index(drop=True)
    df_test = pd.DataFrame({"contents": X_test, "labels": y_test}).reset_index(drop=True)

    print(f"Training samples: {len(df_train)}")
    print(f"Validation samples: {len(df_val)}")
    print(f"Test samples: {len(df_test)}")

    attack_alphabet = config["data"]["alphabet"]

    # Apply DeepWordBug attack to phishing samples
    df_val.loc[df_val["labels"] == 1, "contents"] = df_val[df_val["labels"] == 1]["contents"].apply(
        lambda x: deepwordbug_attack(x, perturbation_rate=0.1, alphabet=attack_alphabet)
    )
    df_test.loc[df_test["labels"] == 1, "contents"] = df_test[df_test["labels"] == 1]["contents"].apply(
        lambda x: deepwordbug_attack(x, perturbation_rate=0.1, alphabet=attack_alphabet)
    )

    # Extract data
    contents_train = df_train.contents.values
    labels_train = df_train.labels.values

    contents_val = df_val.contents.values
    labels_val = df_val.labels.values

    contents_test = df_test.contents.values
    labels_test = df_test.labels.values


    alphabet = config["data"]["alphabet"]
    input_size = config["data"]["input_size"]

    # Convert data
    dataTrain = Data(list(zip(contents_train, labels_train)), alphabet, input_size)
    train_data, train_labels = dataTrain.convert_data()

    dataVal = Data(list(zip(contents_val, labels_val)), alphabet, input_size)
    val_data, val_labels = dataVal.convert_data()

    dataTest = Data(list(zip(contents_test, labels_test)), alphabet, input_size)
    test_data, test_labels = dataTest.convert_data()

    # Initialize the model
    model = CharCNN(
        input_sz=config["data"]["input_size"],
        alphabet_sz=config["data"]["alphabet_size"],
        emb_sz=config["char_cnn_zhang"]["embedding_size"],
        conv_layers=config["char_cnn_zhang"]["conv_layers"],
        fc_layers=[],
        threshold=config["char_cnn_zhang"]["threshold"],
        dropout_p=config["char_cnn_zhang"]["dropout_p"],
        optimizer=config["char_cnn_zhang"]["optimizer"],
        loss=config["char_cnn_zhang"]["loss"]
    )

    # Train the model
    model.train(
        train_inputs=train_data,
        train_labels=train_labels,
        val_inputs=val_data,
        val_labels=val_labels,
        epochs=config["training"]["epochs"],
        bs=config["training"]["batch_size"]
    )

    # Save the trained model
    model.save("adv_test_model.h5")
