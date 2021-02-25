name_folder_data = "data/"
name_folder_model_saved = "/model/"

# parameters for training data
'''
training_folder = "Carrefour"
training_file = "Carrefour_products.csv"
training_cleanedFile = "Carrefour_cleaned_3.csv"
training_columnKey = "BARCODE"
training_columnData = "articleShortTitle"
# training_columnToPredict = "hypDepartmentDesc"
training_columnToPredict = "hypGrpClassDesc"
'''
training_folder = "MarketPlace"
training_file = "MarketPlace_products.csv"
training_cleanedFile = "MarketPlace_cleaned_3.csv"
training_columnKey = "tradeItemKey"
training_columnData = "tradeItemMarketingDescription"
training_columnToPredict = "classificationCode"
training_columns = [training_columnKey, training_columnData, training_columnToPredict, ""]
training_separator = "$"
nb_size_min = 1
nb_size_max = 13


# parameters for data to predict
prediction_folder = "Monoprix"
prediction_file = "Monoprix_products.csv"
prediction_cleanedFile = "Monoprix_cleaned_3.csv"
prediction_separator = ";"
prediction_columnKey = "ean"
prediction_columnData = "product_name"




# other paramaters
stop_words_file = "stop_words_spacy.txt"
columnName_prediction = training_columnToPredict + "_predicted "
columnName_percentage = "percentageOfPredictionEstimate"
str_for_no_prediction = "#no_prediction#"
new_separator = "$"
