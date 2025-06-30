import pandas as pd
import json
from sklearn.model_selection import train_test_split
from data import Data
from model import CharBiLSTM

if __name__ == "__main__":

    csv_path = "../../data/custom_email_dataset.csv" 
    df = pd.read_csv(csv_path)
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

    # Extract data
    contents_train = df_train.contents.values
    labels_train = df_train.labels.values

    contents_val = df_val.contents.values
    labels_val = df_val.labels.values

    contents_test = df_test.contents.values
    labels_test = df_test.labels.values

    config_path = "./config.json"
    config = json.load(open(config_path, encoding="utf-8", errors="ignore"))

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
    model = CharBiLSTM(
        input_sz=config["data"]["input_size"],
        alphabet_sz=config["data"]["alphabet_size"],
        emb_sz=config["charbilstm"]["embedding_size"],
        lstm_units=config["charbilstm"]["lstm_units"],
        dropout_p=config["charbilstm"]["dropout_p"],
        optimizer=config["charbilstm"]["optimizer"],
        loss=config["charbilstm"]["loss"]
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

    # Evaluate the model
    results = model.test(test_data, test_labels, bs=128)
    model.test_model(test_data, test_labels, bs=128)

    # Save the trained model
    model.save("standard_model.h5")
