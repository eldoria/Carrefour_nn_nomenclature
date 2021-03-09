from cleaning_prediction_data import file_cleaned_3
from define_parameters import *

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import mpu
import pandas as pd

file_data = file_cleaned_3
file_with_predictions = folder_data + prediction_name_folder + "/" + prediction_subName_folder + "/" + prediction_folder + "/" + prediction_file
file_dict_categories = folder_model + prediction_name_folder + "/" + training_subName_folder + "/dict_categories.pickle"
file_dict_words = folder_model + prediction_name_folder + "/" + training_subName_folder + "/word_index.pickle"
file_weights = folder_model + training_name_folder + "/" + training_subName_folder + "/" + "weigths.ckpt"

separator = new_separator

value_predicted = columnName_prediction
percentage_prediction = columnName_percentage
str_for_no_prediction = str_for_no_prediction

name_products = prediction_columnData


def encode_x(x, dict_words):
    token = Tokenizer(num_words=len(dict_words), oov_token="<OOV>")
    token.word_index = dict_words

    x = token.texts_to_sequences(x)

    x = pad_sequences(x, padding='post', truncating='pre', maxlen=nb_size_max)

    return x


def write_prediction(x, dict_categories, model):
    prediction = model.predict(x)
    tab_percentage, tab_categories = return_max_prediction(prediction, dict_categories)

    file = open(file_data, 'r')
    file2 = open(file_with_predictions, 'w', encoding='utf-8')
    i = -1
    for line in file:
        if i == -1:
            file2.write(line[:-1] + separator + value_predicted +
                        separator + percentage_prediction + '\n')
            i += 1
        elif count_nb_words(line) >= nb_size_min:

            file2.write(line[:-1] + separator + tab_categories[i] +
                        separator + str(tab_percentage[i]) + '\n')
            i += 1
        else:
            file2.write(line[:-1] + separator + str_for_no_prediction + separator + str_for_no_prediction + '\n')

    file.close()
    file2.close()


def return_line(data, index):
    print(data[index])
    print(data[index:index])
    exit()


def count_nb_words(line):
    return len(line.split('$')[1].split(' '))


def return_max_prediction(values, dict_cat):
    val_percentage = []
    categories_predicted = []

    for value in values:
        max_value = np.amax(value)
        index = np.where(value == np.amax(value))[0][0]
        cat = list(dict_cat.keys())[list(dict_cat.values()).index(index)]

        val_percentage.append(max_value)
        categories_predicted.append(cat)

    return val_percentage, categories_predicted


def from_data_to_values(var1, dict_words):
    data = pd.read_csv(file_data, sep=separator)

    x = data[var1]
    # x, y = delete_duplicate(x, y)
    # x, y = eliminate_too_short_names_of_products(x, y, nb_size_min - 1)

    x = encode_x(x, dict_words)

    return x


def neural_network_feed_forward(var1):
    dict_categories = mpu.io.read(file_dict_categories)
    dict_words = mpu.io.read(file_dict_words)

    print(dict_categories)

    model = return_model(len(dict_categories), len(dict_words))

    model.load_weights(file_weights).expect_partial()
    model.summary()

    x = from_data_to_values(var1, dict_words)

    write_prediction(x, dict_categories, model)


def use_model_prediction():
    neural_network_feed_forward(name_products)


if __name__ == "__main__":
    use_model_prediction()
