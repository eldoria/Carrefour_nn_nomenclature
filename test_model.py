from tensorflow.keras.models import load_model
import pandas as pd

file_data = "./data/carrefour_products_cleaned_min.csv"
nb_size_min = 2
nb_size_max = 20


def from_data_to_values():
    data = pd.read_csv(file_data)


def neural_network_feed_forward():
    model = load_model("./training/model.h5")

