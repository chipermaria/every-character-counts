from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dense, Dropout

import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
from metrics import Metrics, compute_print_f1


class CharBiLSTM(object):
    """
    Class that implements the Character-Level Bidirectional LSTM Network for Text Classification.
    This architecture captures both forward and backward character-level context using BiLSTM layers.
    """

    def __init__(self, input_sz, alphabet_sz, emb_sz, lstm_units, dropout_p,
                 optimizer='adam', loss='categorical_crossentropy'):
        """
        Initialize the Character-Level BiLSTM model.
        :param input_sz: Length of the input character sequence.
        :param alphabet_sz: Size of the character alphabet.
        :param emb_sz: Size of the character embedding vectors.
        :param lstm_units: Number of LSTM units.
        :param dropout_p: Dropout rate.
        :param optimizer: Optimization algorithm.
        :param loss: Loss function.
        """
        self.input_sz = input_sz
        self.alphabet_sz = alphabet_sz
        self.emb_sz = emb_sz
        self.lstm_units = lstm_units
        self.dropout_p = dropout_p
        self.optimizer = optimizer
        self.loss = loss
        self.num_of_classes = 2 

        self._build_model()

    def _build_model(self):
        """
        Build and compile the Character-Level BiLSTM model.
        """
        inputs = Input(shape=(self.input_sz,), dtype='int16')
        x = Embedding(self.alphabet_sz + 1, self.emb_sz, input_length=self.input_sz)(inputs)
        x = Bidirectional(LSTM(self.lstm_units, return_sequences=False))(x)
        x = Dropout(self.dropout_p)(x)
        predictions = Dense(self.num_of_classes, activation="softmax")(x)

        model = Model(inputs=inputs, outputs=predictions)
        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=['accuracy'])
        self.model = model
        self.model.summary()

    def train(self, train_inputs, train_labels, val_inputs, val_labels, epochs, bs):
        """
        Train the model.
        :param train_inputs: Training data.
        :param train_labels: Training labels.
        :param val_inputs: Validation data.
        :param val_labels: Validation labels.
        :param epochs: Number of training epochs.
        :param bs: Batch size.
        """
        checkpoint = ModelCheckpoint("checkpoints/charbilstm-best.hdf5", monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
        es = EarlyStopping(monitor='val_accuracy', patience=5, mode='max', verbose=1)
        metrics = Metrics((val_inputs, val_labels))
        callbacks = [checkpoint, metrics, es]

        self.model.fit(train_inputs, train_labels,
                       validation_data=(val_inputs, val_labels),
                       epochs=epochs, batch_size=bs, verbose=1, callbacks=callbacks)

    def test(self, test_inputs, test_labels, bs):
        """
        Evaluate the model on test data.
        :param test_inputs: Test data.
        :param test_labels: Test labels.
        :param bs: Batch size.
        :return: Evaluation metrics (loss and accuracy).
        """
        return self.model.evaluate(test_inputs, test_labels, batch_size=bs, verbose=1)

    def test_model(self, test_inputs, test_labels, bs):
        """
        Evaluate the model and print detailed F1 scores.
        :param test_inputs: Test data.
        :param test_labels: One-hot encoded test labels.
        :param bs: Batch size.
        """
        metrics = Metrics((test_inputs, test_labels))
        self.model.evaluate(test_inputs, test_labels, batch_size=bs, callbacks=[metrics])

        pred_probs = self.model.predict(test_inputs, batch_size=bs, verbose=1)
        pred_labels = np.argmax(pred_probs, axis=1)
        true_labels = [np.argmax(label) for label in test_labels]

        compute_print_f1(pred_labels, np.array(true_labels), "weighted")
        compute_print_f1(pred_labels, np.array(true_labels), "macro")

    def save(self, path):
        """
        Save the model to the given path.
        :param path: Path to store the trained model.
        """
        self.model.save(path)

