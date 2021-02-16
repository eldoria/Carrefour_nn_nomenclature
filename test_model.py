from train_model import *

file_data = "./data/carrefour_products_cleaned_min.csv"
separator = '$'
name_products = "articleShortTitle"
dep_products = "hypDepartmentDesc"

nb_size_min = 2
nb_size_max = 20


def get_model(len_words, emb_weights, nb_categories):
    embedding = Embedding(len_words+1, 200, input_length=nb_size_max, trainable=False, name='embedding')
    embedding.build(input_shape=(1,)),
    embedding.set_weights([emb_weights]),

    model = keras.Sequential([
            embedding,
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
            Dense(48),
            Dense(nb_categories, activation=keras.activations.softmax)
        ])

    model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.sparse_categorical_crossentropy,
                  metrics=[keras.metrics.sparse_categorical_accuracy])

    return model


def encode_x(x, dict_words):
    token = Tokenizer(num_words=len(dict_words), oov_token="<OOV>")
    token.word_index = dict_words
    x = token.texts_to_sequences(x)

    x = pad_sequences(x, padding='post', truncating='pre', maxlen=nb_size_max)

    return x


def encode_y(y, dict_categories):
    result = []
    for cat in y:
        result.append(dict_categories[cat])

    return np.asarray(result)


def from_data_to_values(var1, var2, dict_words, dict_categories):
    data = pd.read_csv(file_data, sep=separator)
    data = data.astype(str)

    x = data[var1]
    y = data[var2]

    x, y = delete_duplicate(x, y)
    x, y = eliminate_too_short_names_of_products(x, y, nb_size_min - 1)

    x = encode_x(x, dict_words)
    y = encode_y(y, dict_categories)

    return x, y


def neural_network_feed_forward(var1, var2):
    embeddings = mpu.io.read('model/embeddings.pickle')
    dict_categories = mpu.io.read('model/dict_categories.pickle')
    dict_words = mpu.io.read('model/word_index.pickle')

    model = get_model(len(dict_words), embeddings, len(dict_categories))
    # model = create_model(len(dict_categories), len(embeddings))
    model.load_weights("model/weigths.ckpt")
    model.summary()

    x, y = from_data_to_values(var1, var2, dict_words, dict_categories)
    print(x.shape)

    # score = model.evaluate(x, y, verbose=0)
    # print(score)

    str = "yaourt aux fruits"

    str = encode_x(str, dict_words)

    print(str)

    print(model.predict(str))


if __name__ == "__main__":
    neural_network_feed_forward(name_products, dep_products)
