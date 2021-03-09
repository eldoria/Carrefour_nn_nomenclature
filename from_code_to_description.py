from define_parameters import *

import pandas as pd

file_with_predictions = folder_data + prediction_name_folder + "/" + prediction_subName_folder \
                        + "/" + prediction_folder + "/" + prediction_file
file_destination = folder_data + prediction_name_folder + "/" + prediction_subName_folder \
                        + "/" + prediction_folder + "/" + "MarketPlace_prediction_2.csv"
file_for_conversion = folder_data + prediction_name_folder + "/" +  training_data_folder + "/" + "classification_code_to_description.csv"


def write_classification():
    data = pd.read_csv(file_with_predictions, sep=new_separator)
    print(data.shape)
    # data.drop_duplicates(subset='tradeItemKey', inplace=True)
    data = data[data.duplicated(subset='tradeItemKey')]
    print(data)
    print(data.shape)

    columns = data.columns
    print(columns)
    print(len(columns))
    if len(columns) > 5:
        exit()

    dict_code_description = return_dict_code_classification()

    file = open(file_with_predictions, "r")
    file2 = open(file_destination, "w")

    file2.write("tradeItemKey$tradeItemMarketingDescription$classificationCode_predicted$classificationDescription$percentageOfPredictionEstimate" + "\n")
    file.readline()

    for line in file:
        print(line)
        bar_code = line.split('$')[0]
        or_name_products = line.split('$')[2]
        code = line.split('$')[3]
        percentage = str(line.split('$')[4])
        description = dict_code_description[int(code)]

        file2.write(bar_code + new_separator + or_name_products + new_separator + code +
                    new_separator + description + new_separator + percentage)
    file.close()
    file2.close()


def return_dict_code_classification():
    data = pd.read_csv(file_for_conversion)
    dict = {}
    for key, value in zip(data["classificationCode"], data["classificationDescription"]):
        dict[key] = value
    print(dict)
    print(len(dict))
    return dict


def check_things():
    data = pd.read_csv("data/MarketPlace/MarketPlace_training/MarketPlace_training_dataCleaned/MarketPlace_training_cleaned_3.csv", sep=new_separator)
    print(data.shape)
    data.drop_duplicates(keep="first", subset="tradeItemKey", inplace=True)
    print(data)
    print(data.shape)


if __name__ == "__main__":
    write_classification()
    # check_things()
