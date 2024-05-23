import tensorflow as tf
from tensorflow.keras import layers, Model


class FFNModel:
    def __init__(self, max_len, vocab_size, embed_dim=16, feed_forward_dim=64, number_of_sentiment=3):
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.feed_forward_dim = feed_forward_dim
        self.number_of_sentiment = number_of_sentiment

    def build_model(self):
        inputs_tokens = layers.Input(shape=(self.max_len,), dtype=tf.int32)
        embedding_layer = layers.Embedding(input_dim=self.vocab_size,
                                           output_dim=self.embed_dim,
                                           input_length=self.max_len)
        x = embedding_layer(inputs_tokens)
        x = layers.Flatten()(x)
        dense_layer = layers.Dense(self.feed_forward_dim, activation='relu')
        x = dense_layer(x)
        x = layers.Dropout(.5)(x)
        outputs = layers.Dense(self.number_of_sentiment)(x)

        model = Model(inputs=inputs_tokens, outputs=outputs)
        return model


class Conv1DModel:
    def __init__(self, max_len, vocab_size, embed_dim=300, num_filters=128, kernel_sizes=[3, 4, 5],
                 number_of_sentiment=3):
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_filters = num_filters
        self.kernel_sizes = kernel_sizes
        self.number_of_sentiment = number_of_sentiment

    def build_model(self):
        inputs = tf.keras.Input(shape=(self.max_len,), dtype="int32")
        embedding_layer = layers.Embedding(self.vocab_size, self.embed_dim, input_length=self.max_len)
        x = embedding_layer(inputs)
        x = layers.SpatialDropout1D(0.2)(x)

        pooled_outputs = []
        for kernel_size in self.kernel_sizes:
            conv = layers.Conv1D(filters=self.num_filters, kernel_size=kernel_size, padding='valid', activation='relu')(
                x)
            pool = layers.GlobalMaxPooling1D()(conv)
            pooled_outputs.append(pool)

        x = layers.concatenate(pooled_outputs)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(self.number_of_sentiment, activation='softmax')(x)

        model = Model(inputs, outputs)
        return model


def main():
    MAX_LEN = 256
    VOCAB_SIZE = 10000  # Adjust this as necessary
    EMBED_DIM = 300
    NUM_CLASSES = 2

    ffn_model = FFNModel(MAX_LEN, VOCAB_SIZE, EMBED_DIM, 64, NUM_CLASSES).build_model()
    ffn_model.summary()

    conv_model = Conv1DModel(MAX_LEN, VOCAB_SIZE, EMBED_DIM, 128, [3, 4, 5], NUM_CLASSES).build_model()
    conv_model.summary()


if __name__ == "__main__":
    main()
