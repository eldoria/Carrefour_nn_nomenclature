import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from sklearn.model_selection import train_test_split

from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import LSTM, Dense, Embedding, LayerNormalization, GRU, Dropout

name_products = "articleShortTitle"
dep_products = "hypDepartmentDesc"

ref_batch_size = 4096.0
ref_lr = 0.0009
batch_size = [2048]
dropout = 0.25


nb_size_min = 2
nb_size_max = 20


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
    tokenizer = Tokenizer(num_words=28200, oov_token="<OOV>")

    # Crée le dictionnaire
    tokenizer.fit_on_texts(training_data)

    # Les mots sont remplacées par leurs numéros associés
    sequences_training = tokenizer.texts_to_sequences(training_data)
    sequences_testing = tokenizer.texts_to_sequences(testing_data)

    # Ajoute du padding pour que chaque ligne ait la même taille
    padded_training = pad_sequences(sequences_training, padding='post', truncating='pre', maxlen=nb_size_max)
    padded_testing = pad_sequences(sequences_testing, padding='post', truncating='pre', maxlen=nb_size_max)

    print("Nombre de mots : " + str(len(tokenizer.word_index)))

    return padded_training, padded_testing, len(tokenizer.word_index)


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

    df.drop_duplicates(subset=name_products, keep="first", inplace=True)

    return df[name_products], df[dep_products]


def get_max_size_words(values):
    expr = re.compile("\W+", re.U)
    max_size_word = 0
    for value in values:
        l = expr.split(value)
        if len(l) > max_size_word:
            max_size_word = len(l)
            print(str(max_size_word) + str(l))


def eliminate_too_short_names_of_products(data_x, data_y, n):
    if n == 0:
        return data_x, data_y

    result_x = []
    result_y = []

    for line_x, line_y in zip(data_x, data_y):
        line_x = line_x.split(" ")
        size = len(line_x)
        if size > n:
            line_x = " ".join(line_x)
            result_x.append(line_x)
            result_y.append(line_y)

    result_x = pd.Series(result_x)
    result_y = pd.Series(result_y)

    return result_x, result_y


def get_data(f, repartition):
    products = pd.read_csv(f, sep='$')
    products = products.astype(str)

    x = products[name_products]  # récupérer le nom des produits
    y = products[dep_products]  # récupérer le nom des rayons

    print("nombre de données avant la supression des doublons : " + str(len(x)))

    x, y = delete_duplicate(x, y)

    print("nombre de données après la supression de doublons : " + str(len(x)))

    print("avant supression des mots de taille " + str(nb_size_min) + " : " + str(len(x)))

    x, y = eliminate_too_short_names_of_products(x, y, nb_size_min)

    print("après supression des mots de taille " + str(nb_size_min) + " : " + str(len(x)))

    get_max_size_words(x)


    size_y = y.nunique()
    print("nombre de catégories : " + str(size_y))

    # Associe des numéros pour chaque rayon et transforme les numéros en un vecteur binaire
    # ex : 28 rayons, le rayon ayant pour num 0 aura comme vecteur : (1, 0, 0, 0 , ..., 0) le vecteur ayant 28 places
    # car il y a 28 valeurs différentes de y
    y = np.array(auto_encoder_y(y))

    # Sépare en données d'entrainement et de test
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=repartition)

    # Convertit les mots en numéro pour analyse par nlp
    x_tr, x_te, nb_words = auto_encoder_x(x_tr, x_te)

    return x_tr, x_te, y_tr, y_te, size_y, nb_words


def neural_network(size_y, batch_size, nb_words):
    model = keras.Sequential([
        Embedding(nb_words + 1, 20, input_length=nb_size_max),
        LayerNormalization(),
        Dropout(0.40),
        GRU(64, dropout=dropout, return_sequences=True),
        LayerNormalization(),
        GRU(32, dropout=dropout, return_sequences=True),
        LayerNormalization(),
        GRU(16, dropout=dropout),
        LayerNormalization(),
        Dense(64),
        LayerNormalization(),
        Dense(48),
        Dense(size_y, activation=keras.activations.softmax)
    ])

    model.summary()

    model.compile(optimizer=keras.optimizers.Adam(lr=ref_lr / ref_batch_size * bt_s),
                  loss=keras.losses.categorical_crossentropy,
                  metrics=[keras.metrics.categorical_accuracy])

    # model_saver = tf.keras.callbacks.ModelCheckpoint(filepath="training/weigths.ckpt", save_weights_only=True,
    #                                                  save_best_only=True, monitor="val_categorical_accuracy", verbose=1)

    logs = model.fit(x_train, y_train, batch_size=batch_size, epochs=150, validation_data=(x_test, y_test)
                     , callbacks=[keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy', patience=15)],
                     verbose=2)

    return logs


if __name__ == "__main__":
    name_file = "data/carrefour_products_cleaned_maj.csv"
    x_train, x_test, y_train, y_test, size_y, nb_words = get_data(name_file, 0.2)

    for bt_s in batch_size:
        lr = ref_lr / ref_batch_size * bt_s

        all_logs = []

        logs = neural_network(size_y, bt_s, nb_words)

        logs.history['name'] = "LSTM - size_min_mot : " + str(nb_size_min) + " - batch_size : " + str(bt_s)

        all_logs.append(logs)

        plot_log(all_logs)