from define_parameters import *

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import mpu
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Embedding, LayerNormalization, GRU, Dropout

name_products = training_columnData
cat_products = training_columnToPredict

ref_batch_size = 4096.0
ref_lr = 0.0009
batch_size = 2048
dropout = 0.25

nb_size_min = nb_size_min
nb_size_max = nb_size_max

file_for_training = name_folder_data + training_folder + "/" + training_cleanedFile

# 28 200 mots maximum à garder et remplace les mots inconnus avec le token out-of-value
tokenizer = Tokenizer(num_words=20000, oov_token="<OOV>")


def plot_log(all_logs):
    for logs in all_logs:
        losses = logs.history['loss']
        name = logs.history['name'] + " - model"
        plt.plot(list(range(len(losses))), losses, label=name)
        losses = logs.history['val_loss']
        name = logs.history['name'] + " - testing"
        plt.plot(list(range(len(losses))), losses, label=name)
    plt.xlabel("number of epochs")
    plt.ylabel("error")
    plt.title("error on model/testing")
    plt.legend()
    plt.show()

    for logs in all_logs:
        metric = logs.history['sparse_categorical_accuracy']
        name = logs.history['name'] + " - model"
        plt.plot(list(range(len(metric))), metric, label=name)
        metric = logs.history['val_sparse_categorical_accuracy']
        name = logs.history['name'] + " - testing"
        plt.plot(list(range(len(metric))), metric, label=name)
    plt.xlabel("number of epochs")
    plt.ylabel("accuracy")
    plt.legend()
    plt.title("prediction accuracy on model/testing")
    plt.show()


def auto_encoder_x(training_data, testing_data):
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


def auto_encoder_y(label_name):
    possibilities = label_name.unique()
    size = possibilities.shape[0]

    result = []
    categories = {possibilities[i]: i for i in range(size)}
    print(categories)
    mpu.io.write('model/Carrefour/dict_categories.pickle', categories)

    for l in label_name:
        result.append(categories[l])

    return result


def delete_duplicate(x, y):
    df = pd.concat([x, y], axis=1)

    df.drop_duplicates(subset=name_products, keep="first", inplace=True)

    return df[name_products], df[cat_products]


def get_max_size_words(values):
    expr = re.compile("\W+", re.U)
    max_size_word = 0
    print("TEST")
    for value in values:
        l = expr.split(value)
        if len(l) > max_size_word:
            max_size_word = len(l)
            print(str(max_size_word) + str(l))


def eliminate_too_short_names_of_products(data_x, data_y, n):
    result_x = []
    result_y = []

    for line_x, line_y in zip(data_x, data_y):
        line_x = line_x.split(" ")
        size = len(line_x)
        if size == 0:
            print("test")
        if size > n:
            line_x = " ".join(line_x)
            result_x.append(line_x)
            result_y.append(line_y)

    result_x = pd.Series(result_x)
    result_y = pd.Series(result_y)

    return result_x, result_y


def get_nb_categories(y):
    return y.nunique()


def get_data(f, repartition):
    products = pd.read_csv(f, sep='$')
    products = products.astype(str)

    x = products[name_products]  # récupérer le nom des produits
    y = products[cat_products]  # récupérer le nom des rayons

    print("nombre de données avant la supression des doublons : " + str(len(x)))

    x, y = delete_duplicate(x, y)

    print("nombre de données après la supression de doublons : " + str(len(x)))

    print("avant supression des mots de taille " + str(nb_size_min - 1) + " : " + str(len(x)))

    x, y = eliminate_too_short_names_of_products(x, y, nb_size_min - 1)

    print("après supression des mots de taille " + str(nb_size_min - 1) + " : " + str(len(x)))

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


def create_model(size_y, nb_words):
    model = keras.Sequential([
        Embedding(input_dim=nb_words+1, output_dim=200, input_length=nb_size_max,
                  name='embeddings'),
        LayerNormalization(),
        Dropout(0.4),
        GRU(64, dropout=dropout, return_sequences=True, activation=keras.activations.relu),
        LayerNormalization(),
        GRU(32, dropout=dropout, return_sequences=True, activation=keras.activations.relu),
        LayerNormalization(),
        GRU(16, dropout=dropout, activation=keras.activations.relu),
        LayerNormalization(),
        Dense(64),
        LayerNormalization(),
        Dense(48),
        Dense(size_y, activation=keras.activations.softmax)
    ])
    return model


def neural_network(size_y, nb_words, x_train, x_test, y_train, y_test):
    model = create_model(size_y, nb_words)

    model.summary()

    model.compile(optimizer=keras.optimizers.Adam(lr=ref_lr / ref_batch_size * batch_size),
                  loss=keras.losses.sparse_categorical_crossentropy,
                  metrics=[keras.metrics.sparse_categorical_accuracy])

    # model.save("./model/model.h5") -> ne pas faire sinon pb shape embeddings

    model_saver = tf.keras.callbacks.ModelCheckpoint(filepath="model/Carrefour/weigths.ckpt", save_weights_only=True,
                                                     save_best_only=True, monitor="val_sparse_categorical_accuracy",
                                                     verbose=1)

    logs = model.fit(x_train, y_train, batch_size=batch_size, epochs=100, validation_data=(x_test, y_test),
                     callbacks=[keras.callbacks.EarlyStopping(monitor='val_sparse_categorical_accuracy', patience=15),
                                model_saver], verbose=2)

    mpu.io.write('model/Carrefour/word_index.pickle', tokenizer.word_index)

    score_test = model.evaluate(x_test, y_test, verbose=0)
    print(score_test)
    score_train = model.evaluate(x_train, y_train, verbose=0)
    print(score_train)
    score_total = score_train[1] * 0.8 + score_test[1] * 0.2
    print("/////////////////")
    print(score_total)

    return logs


def train_model():
    x_train, x_test, y_train, y_test, size_y, nb_words = get_data(file_for_training, 0.2)

    exit()

    all_logs = []

    logs = neural_network(size_y, nb_words, x_train, x_test, y_train, y_test)

    logs.history['name'] = "LSTM - size_min_mot : " + str(nb_size_min) + " - nb_max : " + str(nb_size_max)

    all_logs.append(logs)

    plot_log(all_logs)


if __name__ == "__main__":
    train_model()
