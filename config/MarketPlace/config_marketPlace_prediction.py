from tensorflow import keras
from tensorflow.keras.layers import Dense, GRU, LayerNormalization, Dropout, Embedding

# folders
training_name_folder = "MarketPlace"
prediction_name_folder = "MarketPlace"

training_subName_folder = "MarketPlace_training"
prediction_subName_folder = "MarketPlace_prediction"

training_data_folder = "MarketPlace_data"
training_cleaned_folder = "MarketPlace_prediction_dataCleaned"
prediction_cleaned_folder = "MarketPlace_prediction_dataCleaned"
prediction_folder = "MarketPlace_prediction_prediction"

# files
training_file = "MarketPlace_products_to_predict.csv"
training_cleaned_file = "MarketPlace_prediction_cleaned_3.csv"
prediction_file = "MarketPlace_prediction_1.csv"
# folders and files training/prediction are equal if the csv for training is the same that the csv for prediction


# variables training
training_columnKey = "tradeItemKey"
training_columnData = "tradeItemMarketingDescription"


# variables prediction
prediction_columnKey = "tradeItemKey"
prediction_columnData = "tradeItemMarketingDescription"
prediction_columnData_original = prediction_columnData + "_original"
columnName_prediction = "classificationCode_predicted"
# variables of training/prediction are equal if the csv for training is the same that the csv for prediction


# separators
training_separator = "$"
prediction_separator = ","
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

        Dense(2048),
        LayerNormalization(),
        Dropout(dropout),

        Dense(1524),
        LayerNormalization(),
        Dropout(dropout),

        Dense(size_y, activation=keras.activations.softmax)
    ])
    return model


'''
On garde les prédictions supérieures ou égales à : 0
Détail des erreurs : 
28599    4618
28095    3596
28078    3309
25972    2985
29198    2983
         ... 
26073       1
26457       1
26682       1
26542       1
26611       1
Name: self, Length: 1077, dtype: int64
nombre de lignes : 469518
accuracy : 76.54871591717463%
//////////////////////////////////
On garde les prédictions supérieures ou égales à : 0.5
Détail des erreurs : 
28599    4441
28095    3134
28078    2957
25972    2465
28087    1452
         ... 
26810       1
26797       1
26682       1
26093       1
26611       1
Name: self, Length: 954, dtype: int64
nombre de lignes : 405336
accuracy : 83.16113051887817%
//////////////////////////////////
On garde les prédictions supérieures ou égales à : 0.6
Détail des erreurs : 
28599    3935
28078    2363
28095    1836
25972    1553
28087    1268
         ... 
27025       1
26407       1
26657       1
26425       1
28602       1
Name: self, Length: 844, dtype: int64
nombre de lignes : 354690
accuracy : 88.17051509769094%
//////////////////////////////////
On garde les prédictions supérieures ou égales à : 0.7
Détail des erreurs : 
28599    3051
28078    1404
28577    1098
28087     947
28095     940
         ... 
27002       1
2268        1
26896       1
26890       1
26579       1
Name: self, Length: 722, dtype: int64
nombre de lignes : 314982
accuracy : 92.61767339086043%
//////////////////////////////////
On garde les prédictions supérieures ou égales à : 0.8
Détail des erreurs : 
28599    1750
28577     849
28087     599
28078     448
2270      296
         ... 
26776       1
26346       1
26654       1
26768       1
26775       1
Name: self, Length: 608, dtype: int64
nombre de lignes : 284498
accuracy : 96.05972625466612%
//////////////////////////////////
On garde les prédictions supérieures ou égales à : 0.9
Détail des erreurs : 
28599    671
28577    401
28087    161
28585    138
28078    104
        ... 
28315      1
26322      1
26326      1
27140      1
26591      1
Name: self, Length: 432, dtype: int64
nombre de lignes : 256894
accuracy : 98.29968780897958%
//////////////////////////////////
On garde les prédictions supérieures ou égales à : 0.95
Détail des erreurs : 
28599    249
28577    193
28628     84
28087     61
28818     49
        ... 
26795      1
27154      1
26287      1
26280      1
26849      1
Name: self, Length: 313, dtype: int64
nombre de lignes : 235673
accuracy : 99.19591977019005%
'''
