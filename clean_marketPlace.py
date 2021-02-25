from define_parameters import *
import pandas as pd

file_source = name_folder_data + training_folder + "/" + "MarketPlace_data.csv"
file_destination= name_folder_data + training_folder + "/" + "MarketPlace_products.csv"


if __name__ == "__main__":
    data = pd.read_csv(file_source, sep=',', encoding='utf8', header=0, dtype=str)
    print(data.shape)
    data = data[data["sourceCode"] == 'MKP_FOOD']
    print(data.shape)
    file = open(file_destination, 'w', encoding='utf-8')
    file.write(training_columnKey + new_separator + training_columnData
               + new_separator + training_columnToPredict + '\n')
    for primary_key, name_products, name_cat in zip(data[training_columnKey], data[training_columnData], data[training_columnToPredict]):
        file.write(str(primary_key) + new_separator + str(name_products)
                   + new_separator + str(name_cat) + "\n")
    file.close()
