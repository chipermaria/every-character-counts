from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Dropout, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, GRU, Dense, Dropout

import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
from metrics import Metrics, compute_print_f1


class CharGRU(object):
    """
    Class that implements a Character-Level GRU Neural Network for Text Classification,
    using character embeddings.
    """

    def __init__(self, input_sz, alphabet_sz, emb_sz, gru_units,
                 dropout_p, optimizer='adam', loss='categorical_crossentropy'):
        """
        Initialize the Character-Level GRU model.
        :param input_sz: Length of the input character sequence.
        :param alphabet_sz: Size of the character alphabet.
        :param emb_sz: Size of the character embedding vectors.
        :param gru_units: Number of GRU units.
        :param dropout_p: Dropout rate.
        :param optimizer: Optimization algorithm.
        :param loss: Loss function.
        """
        self.input_sz = input_sz
        self.alphabet_sz = alphabet_sz
        self.emb_sz = emb_sz
        self.gru_units = gru_units
        self.dropout_p = dropout_p
        self.optimizer = optimizer
        self.loss = loss
        self.num_of_classes = 2 

        self._build_model()

    def _build_model(self):
        """
        Build and compile the Character-Level GRU model.
        """
        inputs = Input(shape=(self.input_sz,), dtype='int16')
        x = Embedding(self.alphabet_sz + 1, self.emb_sz, input_length=self.input_sz)(inputs)

        x = GRU(self.gru_units, return_sequences=True, name="gru")(x)

        x = GlobalAveragePooling1D()(x)

        # Dropout and output
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
        :return: None
        """
        checkpoint = ModelCheckpoint("checkpoints/chargru-best.hdf5", monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
        es = EarlyStopping(monitor='val_accuracy', patience=5, mode='max', verbose=1)
        metrics = Metrics((val_inputs, val_labels))
        callbacks = [checkpoint, metrics, es]

        self.model.fit(train_inputs, train_labels,
                       validation_data=(val_inputs, val_labels),
                       epochs=epochs, batch_size=bs, verbose=1, callbacks=callbacks)

    def test(self, test_inputs, test_labels, bs):
        """
        Evaluate the model.
        :param test_inputs: Test data.
        :param test_labels: Test labels.
        :param bs: Batch size.
        :return: Loss and accuracy.
        """
        return self.model.evaluate(test_inputs, test_labels, batch_size=bs, verbose=1)

    def test_model(self, test_inputs, test_labels, bs):
        """
        Evaluate the model and display F1 scores.
        :param test_inputs: Test data.
        :param test_labels: One-hot encoded test labels.
        :param bs: Batch size.
        :return: None
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
        Save the model to a file.
        :param path: Path to save the model.
        :return: None
        """
        self.model.save(path)

