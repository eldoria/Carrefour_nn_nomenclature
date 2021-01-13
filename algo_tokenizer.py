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


def auto_encoder_x(training_data, testing_data):
    # 20 000 mots maximum à garder et remplace les mots inconnus avec le token out-of-value
    tokenizer = Tokenizer(num_words=20000, oov_token="<OOV>")

    # Crée le dictionnaire
    tokenizer.fit_on_texts(training_data)

    # Les mots sont remplacées par leurs numéros associés
    sequences_training = tokenizer.texts_to_sequences(training_data)
    sequences_testing = tokenizer.texts_to_sequences(testing_data)

    # Ajoute du padding pour que chaque ligne ait la même taille
    padded_training = pad_sequences(sequences_training, padding='post', truncating='pre', maxlen=20)
    padded_testing = pad_sequences(sequences_testing, padding='post', truncating='pre', maxlen=20)

    return padded_training, padded_testing


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

    print("nombre de lignes avant de supprimer celles sans rayon : " + str(products.shape[0]))
    # élimine les preoduits sans nom de rayon associé
    products = products[~products['hypSectorDesc'].isnull()]
    print("nombre de lignes après la supression de celles sans rayon : " + str(products.shape[0]))

    x = products['product_name']  # récupérer le nom des produits
    y = products['hypDepartmentDesc']  # récupérer le nom des rayons

    print("apercu des données : " + x.head())
    print("apercu des labels : " + y.head())

    print("nombre de données avant la supression des doublons : " + str(len(x)))

    x, y = delete_duplicate(x, y)

    print("nombre de données après la supression de doublons : " + str(len(x)))

    size_y = y.nunique()

    # Associe des numéros pour chaque rayon et transforme les numéros en un vecteur binaire
    # ex : 28 rayons, le rayon ayant pour num 0 aura comme vecteur : (1, 0, 0, 0 , ..., 0) le vecteur ayant 28 places car
    # il y a 28 valeurs différentes de y
    y = np.array(auto_encoder_y(y))

    # Sépare en données d'entrainement et de test
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=repartition)

    # Convertit les mots en numéro pour analyse par nlp
    x_tr, x_te = auto_encoder_x(x_tr, x_te)

    return x_tr, x_te, y_tr, y_te, size_y


def neural_network(size_y, batch_size):
    model = keras.Sequential([
        Embedding(15000, 128, input_length=20),
        LSTM(48, return_sequences=True),
        LSTM(48, return_sequences=True),
        LSTM(48),
        Dense(64),
        Dropout(0.4),
        Dense(32),
        Dropout(0.4),
        Dense(size_y, activation=keras.activations.softmax)
    ])

    model.summary()

    model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.categorical_crossentropy,
                  metrics=keras.metrics.categorical_accuracy)

    logs = model.fit(x_train, y_train, batch_size=batch_size, epochs=150, validation_data=(x_test, y_test)
                     , callbacks=keras.callbacks.EarlyStopping(monitor='val_loss', patience=20), verbose=2)

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

