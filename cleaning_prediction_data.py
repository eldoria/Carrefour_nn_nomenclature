from define_parameters import *
import nltk
import pandas as pd
import re

bar_code = prediction_columnKey
name_products = prediction_columnData
original_name_products = name_products + "_original"

type_folder = training_data_folder

base_folder = folder_data + prediction_name_folder + "/"

file_source = base_folder + training_data_folder + "/" + training_file

file_stop_words = folder_data + stop_words_file

path_to_cleaned_data = prediction_subName_folder + "/" + prediction_cleaned_folder + "/" + prediction_subName_folder

file_cleaned_1 = base_folder + path_to_cleaned_data + "_cleaned_1.csv"
file_cleaned_2 = base_folder + path_to_cleaned_data + "_cleaned_2.csv"
file_cleaned_3 = base_folder + path_to_cleaned_data + "_cleaned_3.csv"

separator = new_separator


def clean_csv_carrefour():
    print(file_source)
    data = pd.read_csv(file_source, usecols=[bar_code, name_products], sep=prediction_separator, dtype=str)

    file = open(file_cleaned_1, 'w', encoding="utf-8")
    file.write(bar_code + separator + name_products + separator + original_name_products + "\n")
    for var1, var2, in zip(data[bar_code], data[name_products]):
        file.write(str(var1) + separator + str(var2) + separator + str(var2) + "\n")
    file.close()

    data = pd.read_csv(file_cleaned_1, sep=separator, dtype=str)

    file2 = open(file_cleaned_2, 'w', encoding='utf-8')
    file2.write(bar_code + separator + name_products + separator + original_name_products + "\n")
    for barcode, name, or_name in zip(data[bar_code], data[name_products], data[original_name_products]):
        name = str(name)
        or_name = str(or_name)
        barcode = str(barcode)

        name = name.split(".")
        name = " ".join(name)

        '''
        name = name.replace(',', '.')
        name = name.replace(';', ' ')
        name = name.replace('-', ' ')
        name = name.replace('/', ' ')
        '''

        name = re.sub("[-;/]", " ", name)
        name = re.sub(",", ".", name)
        name = re.sub("[éèêë]", "e", name)
        name = re.sub("[àâä]", "e", name)
        name = re.sub("[ôö]", "o", name)
        name = re.sub("[ùûü]", "u", name)

        file2.write(barcode + separator + name + separator + or_name + "\n")
    file.close()


def create_new_csv(f_stop_words, file_csv, var1, var2, var3):
    words = read_lines(f_stop_words)
    results = []

    data = delete_rows_with_missing_values(file_csv)
    print(data.shape)
    # data = pd.read_csv(file_csv, sep=separator)

    bar_code = data[var1]
    name_products = data[var2]
    or_name_products = data[var3]

    for data in name_products:
        text_tokens = nltk.tokenize.word_tokenize(data)
        # print(text_tokens)
        name_product = [contains_car(word) for word in text_tokens if not word.lower() in words
                        and contains_car(word) is not None]
        # print(name_product)
        results.append(" ".join(name_product))

    file_2 = open(file_cleaned_3, 'w', encoding='utf-8')
    file_2.write(var1 + separator + var2 + separator + var3 + "\n")

    for bar_code, words, or_words in zip(bar_code, results, or_name_products):
        if len(words) != 0:
            val = str(bar_code) + separator + str(words.lower()) + separator + str(or_words) + "\n"
            file_2.write(val)
    file_2.close()


def contains_car(word):
    test_grammes = re.search("^[0-9]+.?[0-9]*[mk]?gr?", word.lower())
    test_grammes_2 = re.search("^(mg|g|gr)$", word.lower())
    test_litres = re.search("^[0-9]+.?[0-9]*[mcd]?l", word.lower())
    test_litres_2 = re.search("^(ml|cl|dl|l)$", word.lower())
    test_mesure = re.search("[cdm]²$", word.lower())
    test_mesure_2 = re.search("gr/m", word.lower())
    test_number = re.search("[0-9]+", word.lower())
    test_car = re.search("^.$", word.lower())
    test_car_2 = re.search("^(l'|d')", word.lower())
    test_car_3 = re.search("^-*$", word.lower())
    test_absence_carre = re.search("²", word.lower())
    test_alpha_num = re.search("^[a-z Ü-ü]+$", word.lower())

    if (test_grammes or test_grammes_2) and test_absence_carre is None:
        return "grammes"
    elif (test_litres or test_litres_2) and test_absence_carre is None:
        return "litres"
    elif test_mesure or test_mesure_2:
        return "/mesure/"
    elif test_car_2:
        return word[2:]
    elif test_alpha_num and test_car is None:
        return word


def read_lines(file):
    file = open(file, 'r', encoding='ISO-8859-1')
    result = []
    for line in file:
        # Get rid of the "\n"
        line = line[:-1]
        result.append(line)
    file.close()

    return result


def delete_rows_with_missing_values(file):
    products = pd.read_csv(file, sep=separator, dtype=str)
    print(products.shape)
    products = products[~products[name_products].isnull()]
    print(products.shape)

    return products


def clean_data_training():
    clean_csv_carrefour()
    create_new_csv(file_stop_words, file_cleaned_2, bar_code, name_products, original_name_products)


if __name__ == "__main__":
    clean_data_training()
