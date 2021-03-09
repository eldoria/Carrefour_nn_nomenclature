from define_parameters import *
import pandas as pd

file_source = folder_data + training_name_folder + "/" + training_data_folder + "/" + "MarketPlace_data.csv"
file_destination_1 = folder_data + training_name_folder + "/" + training_data_folder + "/" + "MarketPlace_products.csv"
file_destination_2 = folder_data + training_name_folder + "/" + training_data_folder + "/" + "MarketPlace_products_to_predict.csv"

def select_only_marketplace_data():
    data = pd.read_csv(file_source, sep=",", encoding="utf-8", header=0, dtype=str)
    print(data.shape)
    data = data[data["sourceCode"] == "MKP_FOOD"]
    print(data.shape)
    file = open(file_destination_1, "w", encoding="utf-8")
    file.write(training_columnKey + new_separator + training_columnData
               + new_separator + training_columnToPredict + '\n')
    for primary_key, name_products, name_cat in zip(data[training_columnKey], data[training_columnData],
                                                    data[training_columnToPredict]):
        file.write(str(primary_key) + new_separator + str(name_products)
                   + new_separator + str(name_cat) + "\n")
    file.close()


def select_data_without_classification_code():
    data = pd.read_csv(file_source, sep=',', encoding='utf8', header=0, dtype=str)
    print(data.shape)
    data = data[data["sourceCode"] == 'MKP_FOOD']
    print(data.shape)
    data = data[data[training_columnToPredict].isnull()]
    print(data.shape)
    file = open(file_destination_2, "w", encoding="utf-8")
    file.write(training_columnKey + new_separator + training_columnData + '\n')
    for primary_key, name_products in zip(data[training_columnKey], data[training_columnData]):
        file.write(str(primary_key) + new_separator + str(name_products) + "\n")
    file.close()


if __name__ == "__main__":
    select_data_without_classification_code()
