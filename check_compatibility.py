from define_parameters import *
import pandas as pd
import numpy as np

training_cleanedFile = name_folder_data + training_folder + "/" + training_cleanedFile
prediction_cleanedFile = name_folder_data + prediction_folder + "/" + prediction_cleanedFile
training_y = training_columnToPredict
prediction_y = prediction_columnToPredict


def create_tab_diff():
    data1 = pd.read_csv(training_cleanedFile, sep=new_separator)
    data2 = pd.read_csv(prediction_cleanedFile, sep=new_separator)
    tab_diff = (list(list(set(data1[training_y]) - set(data2[prediction_y])) +
                      list(set(data2[prediction_y]) - set(data1[training_y]))))
    return tab_diff


def check_compatibility(return_type):
    data2 = pd.read_csv(prediction_cleanedFile, sep=new_separator)
    tab_diff = create_tab_diff()
    if return_type == "prediction":
        print("Avant supression des colonnes non représentées dans les données d'entrainement : " + str(data2.shape))
        # data2 = data2[prediction_y].replace(dict_diff, np.nan).dropna().values
        data2 = data2[~data2[prediction_y].isin(tab_diff)]
        print("Après supression des colonnes non représentées dans les données d'entrainement : " + str(data2.shape))
        return data2


if __name__ == "__main__":
    check_compatibility(return_type="prediction")
