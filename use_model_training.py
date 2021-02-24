from train_model import *
from cleaning_training_data import file_cleaned_3

file_data = file_cleaned_3
file_with_predictions = name_folder_data + training_folder + "/" + training_folder + "_prediction.csv"
separator = '$'

value_predicted = training_columnToPredict
percentage_prediction = columnName_percentage
str_for_no_prediction = str_for_no_prediction

name_products = training_columnData


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
            file2.write(line[:-1] + separator + columnName_prediction +
                        separator + percentage_prediction + '\n')
        elif count_nb_words(line) >= nb_size_min:

            file2.write(line[:-1] + separator + tab_categories[i] +
                        separator + str(tab_percentage[i]) + '\n')
        else:
            file2.write(line[:-1] + separator + str_for_no_prediction + separator + str_for_no_prediction + '\n')
        i += 1

    file.close()
    file2.close()


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


def from_data_to_values(var1, dict_words, dict_categories):
    # data = data.astype(str)
    data = pd.read_csv(file_data, sep=separator)
    x = data[var1]
    # x, y = delete_duplicate(x, y)
    # x, y = eliminate_too_short_names_of_products(x, y, nb_size_min - 1)

    x = encode_x(x, dict_words)

    return x


def neural_network_feed_forward(var1):
    dict_categories = mpu.io.read('model/Carrefour/dict_categories.pickle')
    dict_words = mpu.io.read('model/Carrefour/word_index.pickle')

    print(dict_categories)

    model = create_model(len(dict_categories), len(dict_words))
    # model = create_model(len(dict_categories), len(embeddings))
    model.load_weights("model/Carrefour/weigths.ckpt").expect_partial()
    model.summary()

    x = from_data_to_values(var1, dict_words, dict_categories)

    write_prediction(x, dict_categories, model)


def evaluate_score():
    data = pd.read_csv(file_with_predictions, sep=separator)
    data = data[data[percentage_prediction] != str_for_no_prediction]

    values = [0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]

    for val in values:
        data_tmp = data[pd.to_numeric(data[percentage_prediction]) >= val]
        y_solution = data_tmp[columnName_prediction]
        y_predicted = data_tmp[value_predicted]
        nb_rows = len(data_tmp)
        nb_errors = len(y_solution.compare(y_predicted, keep_equal=True))
        errors = y_solution.compare(y_predicted, keep_equal=True)
        print("On garde les prédictions supérieures ou égales à : " + str(val))
        print("Détail des erreurs : ")
        print(errors['self'].value_counts())
        print("nombre de lignes : " + str(nb_rows))
        print("accuracy : " + str(100 - (nb_errors / nb_rows) * 100) + "%")
        print("//////////////////////////////////")


def use_model_prediction():
    neural_network_feed_forward(name_products)
    evaluate_score()
    # evaluate_score works only for data with the solution provided as the training datas


if __name__ == "__main__":
    use_model_prediction()
    evaluate_score()
