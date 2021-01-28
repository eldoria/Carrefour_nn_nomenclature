import nltk
from nltk.corpus import stopwords
import pandas as pd
from spacy.lang.fr.stop_words import STOP_WORDS as fr_stop
import re


name_products = "articleShortTitle"
dep_products = "hypDepartmentDesc"

base_folder = "./data"

file_stop_words = f"{base_folder}/stop_words_nltk.txt"

file_source = f"{base_folder}/carrefour_products.csv"
file_destination = f"{base_folder}/carrefour_products_cleaned.csv"


file_source_cleaned_1 = f"{base_folder}/produits_carrefour_nomenclatures_cleaned_1.csv"
file_source_cleaned_2 = f"{base_folder}/produits_carrefour_nomenclatures_cleaned_2.csv"
file_dest = f"{base_folder}/produits_clean_nltk.csv"

separator = "$"


def clean_csv_carrefour():
    data = pd.read_csv(file_source, usecols=[name_products, dep_products])

    file = open(file_source_cleaned_1, 'w', encoding="utf-8")
    file.write(name_products + separator + dep_products + "\n")
    for var1, var2 in zip(data[name_products], data[dep_products]):
        file.write(str(var1) + separator + str(var2) + "\n")
    file.close()

    data = pd.read_csv(file_source_cleaned_1, sep=separator)

    file2 = open(file_source_cleaned_2, 'w', encoding='utf-8')
    file2.write(name_products + separator + dep_products + "\n")
    for name, dep in zip(data[name_products], data[dep_products]):
        name = str(name)
        dep = str(dep)

        name = name.split(".")
        name = " ".join(name)

        name = name.replace(',', '.')
        name = name.replace(';', ' ')

        file2.write(name.lower() + separator + dep + "\n")
    file.close()


def create_new_csv(f_stop_words, file_csv, var1, var2, new_file_name):
    words = read_lines(f_stop_words)
    results = []

    data = delete_rows_with_missing_values(file_csv)

    data_1 = data[var1]
    data_2 = data[var2]

    for data in data_1:
        text_tokens = nltk.tokenize.word_tokenize(data)
        # print(text_tokens)
        name_product = [contains_car(word) for word in text_tokens if not word in words
                        and contains_car(word) is not None]
        # print(name_product)
        results.append(" ".join(name_product))

    file = open(new_file_name, 'w', encoding="utf-8")
    file.write(name_products + separator + dep_products + "\n")
    for words, labels in zip(results, data_2):
        val = str(words) + separator + str(labels) + "\n"
        file.write(val)
    file.close()


def contains_car(words):
    test_grammes = re.search("[0-9]*k?gr?$", words)
    test_litres = re.search("^[0-9]+c?l$", words)
    test_temps = re.search("^[0-9]+[m|n]+$", words)
    test_mesure = re.search("[0-9]+[c|d|m²]+$", words)
    test_mesure_2 = re.search("gr/m", words)
    test_number = re.search("[0-9]+", words)
    test_car = re.search("^[a-z Ü-ü]$", words)
    test_car_2 = re.search("^[l'][d']", words)
    test_car_3 = re.search("^[a-z Ü-ü -]*$", words)
    test_car_4 = re.search("^-*$", words)

    if test_grammes:
        return "grammes"
    elif test_litres:
        return "litres"
    elif test_temps:
        return "temps"
    elif test_mesure or test_mesure_2:
        return "mesure"
    elif test_car_2:
        return words[2:]
    elif test_number is None and test_car is None and test_car_3 is not None and test_car_4 is None:
        return words


def write_words_nltk():
    nltk.download('stopwords')
    nltk.download('punkt')

    file = open(file_stop_words, 'w')
    for words in stopwords.words('french'):
        file.write(words + "\n")
    file.close()


def write_words_spacy():
    file = open("data/stop_words_spacy.txt", 'w')
    for words in fr_stop:
        file.write(words + "\n")
    file.close()


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
    products = products[~products[name_products].isnull()]
    products = products[~products[dep_products].isnull()]

    return products


if __name__ == "__main__":
    clean_csv_carrefour()
    create_new_csv(file_stop_words, file_source_cleaned_2, name_products, dep_products, file_destination)
