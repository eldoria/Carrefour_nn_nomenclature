from define_parameters import *
import nltk
import pandas as pd
import re


name_products = training_columnData
bar_code = training_columnKey
name_cat = training_columnToPredict

type_folder = training_folder

base_folder = name_folder_data + type_folder + "/"

file_source = base_folder + training_file

file_stop_words = name_folder_data + stop_words_file

file_cleaned_1 = base_folder + type_folder + "_cleaned_1.csv"
file_cleaned_2 = base_folder + type_folder + "_cleaned_2.csv"
file_cleaned_3 = base_folder + type_folder + "_cleaned_3.csv"

separator = new_separator


def clean_csv_carrefour():
    data = pd.read_csv(file_source, usecols=[bar_code, name_products, name_cat], sep=training_separator)

    file = open(file_cleaned_1, 'w', encoding="utf-8")
    file.write(bar_code + separator + name_products + separator + name_cat + "\n")
    for var1, var2, var3 in zip(data[bar_code], data[name_products], data[name_cat]):
        file.write(str(var1) + separator + str(var2) + separator + str(var3) + "\n")
    file.close()

    data = pd.read_csv(file_cleaned_1, sep=separator)

    file2 = open(file_cleaned_2, 'w', encoding='utf-8')
    file2.write(bar_code + separator + name_products + separator + name_cat + "\n")
    for name, barcode, cat in zip(data[name_products], data[bar_code], data[name_cat]):
        name = str(name)
        barcode = str(barcode)
        cat = str(cat)

        name = name.split(".")
        name = " ".join(name)

        name = name.replace(',', '.')
        name = name.replace(';', ' ')
        name = name.replace('-', ' ')

        file2.write(barcode + separator + name + separator + cat + "\n")
    file.close()


def create_new_csv(f_stop_words, file_csv, var1, var2, var3):
    words = read_lines(f_stop_words)
    results = []

    data = delete_rows_with_missing_values(file_csv)
    # data = pd.read_csv(file_csv, sep=separator)

    name_products = data[var1]
    bar_code = data[var2]
    name_cat = data[var3]

    for data in name_products:
        text_tokens = nltk.tokenize.word_tokenize(data)
        # print(text_tokens)
        name_product = [contains_car(word) for word in text_tokens if not word.lower() in words
                        and contains_car(word) is not None]
        # print(name_product)
        results.append(" ".join(name_product))

    file_2 = open(file_cleaned_3, 'w', encoding='utf-8')
    file_2.write(var2 + separator + var1 + separator + var3 + "\n")
    for bar_code, words, cat in zip(bar_code, results, name_cat):
        if len(words) != 0:
            val = str(bar_code) + separator + str(words) + separator + str(cat) + "\n"
            file_2.write(val.lower())
    file_2.close()


def contains_car(word):
    test_grammes = re.search("[0-9]+[mk]?gr?", word.lower())
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
    test_alpha_num = re.search("^[a-z]+$", word.lower())

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
    file = open(file, 'r')
    result = []
    for line in file:
        # Get rid of the "\n"
        line = line[:-1]
        result.append(line)
    file.close()

    return result


def delete_rows_with_missing_values(file):
    products = pd.read_csv(file, sep=separator)
    print(products.shape)
    products = products[~products[name_products].isnull()]
    products = products[~products[name_cat].isnull()]
    print(products.shape)

    return products


def clean_data_training():
    clean_csv_carrefour()
    create_new_csv(file_stop_words, file_cleaned_2, name_products, bar_code, name_cat)


if __name__ == "__main__":
    clean_data_training()
