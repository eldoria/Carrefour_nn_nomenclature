import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout


config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8))
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

def plot_log(all_logs):
    for logs in all_logs:
        losses = logs.history['loss']
        name = logs.history['name'] + " - training"
        plt.plot(list(range(len(losses))), losses, label=name)
        losses = logs.history['val_loss']
        name = logs.history['name'] + " - testing"
        plt.plot(list(range(len(losses))), losses, label=name)
    plt.xlabel("number of epochs")
    plt.ylabel("error")
    plt.title("error on training/testing")
    plt.legend()
    plt.show()

    for logs in all_logs:
        metric = logs.history['categorical_accuracy']
        name = logs.history['name'] + " - training"
        plt.plot(list(range(len(metric))), metric, label=name)
        metric = logs.history['val_categorical_accuracy']
        name = logs.history['name'] + " - testing"
        plt.plot(list(range(len(metric))), metric, label=name)
    plt.xlabel("number of epochs")
    plt.ylabel("accuracy")
    plt.legend()
    plt.title("prediction accuracy on training/testing")
    plt.show()


def auto_encoder_x(products_name):
    tokenizer = Tokenizer(num_words=20000, oov_token="<OOV>")

    tokenizer.fit_on_texts(products_name)

    sequences = tokenizer.texts_to_sequences(products_name)

    padded = pad_sequences(sequences, padding='post', truncating='pre')

    return padded


def number_to_binary(number, size):
    return [0 if i != number else 1 for i in range(size)]


def auto_encoder_y(label_name):
    possibilities = label_name.unique()
    size = possibilities.shape[0]

    result = []
    words = {possibilities[i]: i for i in range(size)}

    for l in label_name:
        nb = words[l]
        val = number_to_binary(nb, size)
        result.append(val)

    return result


def delete_duplicate(x, y):
    df = pd.concat([x, y], axis=1)

    df.drop_duplicates(subset="product_name", keep=False, inplace=True)

    return df["product_name"], df["hypDepartmentDesc"]


def get_data(f, repartition):
    products = pd.read_csv(f, sep=';')

    products = products[~products['hypSectorDesc'].isnull()]

    x = products['product_name']
    y = products['hypDepartmentDesc']

    print("NUMBER OF DATA BEFORE ELIMINATION OF duplicate : " + str(len(x)) + "/" + str(len(y)))

    x, y = delete_duplicate(x, y)

    print("NUMBER OF DATA AFTER ELIMINATION OF duplicate : " + str(len(x)) + "/" + str(len(y)))

    size_y = y.nunique()

    x = auto_encoder_x(x)
    y = np.array(auto_encoder_y(y))

    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=repartition)

    return x_tr, x_te, y_tr, y_te, size_y


def neural_network(size_y, batch_size):
    model = keras.Sequential([
        Embedding(15000, 128, input_length=25),
        LSTM(64, return_sequences=True),
        LSTM(32, return_sequences=True),
        LSTM(16),
        Dense(64, kernel_regularizer=keras.regularizers.l2(0.01)),
        Dense(32, kernel_regularizer=keras.regularizers.l2(0.01)),
        Dense(size_y, activation=keras.activations.softmax)
    ])

    model.summary()

    model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.categorical_crossentropy,
                  metrics=keras.metrics.categorical_accuracy)

    logs = model.fit(x_train, y_train, batch_size=batch_size, epochs=150, validation_data=(x_test, y_test)
                     , callbacks=keras.callbacks.EarlyStopping(monitor='val_loss', patience=10), verbose=2)

    return logs


if __name__ == "__main__":
    name_file = "produits_carrefour_nomenclatures.csv"
    batch_size = 128

    all_logs = []

    x_train, x_test, y_train, y_test, size_y = get_data(name_file, 0.2)

    logs = neural_network(size_y, batch_size)

    logs.history['name'] = "LSTM - " + str(batch_size)

    all_logs.append(logs)

    plot_log(all_logs)

