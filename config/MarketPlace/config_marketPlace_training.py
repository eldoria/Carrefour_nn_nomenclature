from tensorflow import keras
from tensorflow.keras.layers import Dense, GRU, LayerNormalization, Dropout, Embedding

# folders
training_name_folder = "MarketPlace"
prediction_name_folder = "MarketPlace"

training_subName_folder = "MarketPlace_training"
prediction_subName_folder = "MarketPlace_training"

training_data_folder = "MarketPlace_data"
training_cleaned_folder = "MarketPlace_training_dataCleaned"
prediction_cleaned_folder = "MarketPlace_training_dataCleaned"
prediction_folder = "MarketPlace_training_prediction"

# files
training_file = "MarketPlace_products.csv"
training_cleaned_file = "MarketPlace_training_cleaned_3.csv"
prediction_file = "MarketPlace_training_prediction.csv"
# folders and files training/prediction are equal if the csv for training is the same that the csv for prediction


# variables training
training_columnKey = "tradeItemKey"
training_columnData = "tradeItemMarketingDescription"
training_columnToPredict = "classificationCode"

# variables prediction
prediction_columnKey = "tradeItemKey"
prediction_columnData = "tradeItemMarketingDescription"
columnName_prediction = training_columnToPredict + "_predicted"
# variables of training/prediction are equal if the csv for training is the same that the csv for prediction


# ...
# columns_data =
# training_request =


# separators
training_separator = "$"
prediction_separator = "$"
new_separator = "$"

# variables model
nb_size_min = 1
nb_size_max = 16
dropout = 0.25


def return_model(size_y, nb_words):
    model = keras.Sequential([
        Embedding(input_dim=nb_words+1, output_dim=200, input_length=nb_size_max,
                  name='embeddings'),
        LayerNormalization(),
        Dropout(0.4),

        GRU(64, dropout=dropout, return_sequences=True),
        LayerNormalization(),

        GRU(32, dropout=dropout, return_sequences=True),
        LayerNormalization(),

        GRU(16, dropout=dropout),
        LayerNormalization(),

        Dense(64),
        LayerNormalization(),
        Dropout(dropout),

        Dense(48),
        LayerNormalization(),
        Dropout(dropout),

        Dense(size_y, activation=keras.activations.softmax)
    ])
    return model
