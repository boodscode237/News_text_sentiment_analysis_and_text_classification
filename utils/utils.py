import tensorflow as tf
import numpy as np


def create_text_vectorization_layer(vocab_size, max_len):
    return tf.keras.layers.TextVectorization(
        max_tokens=vocab_size,
        output_mode='int',
        output_sequence_length=max_len
    )


def prepare_data_for_model(text_vectorizer, train_features, val_features, test_features, batch_size):
    train_ds = tf.data.Dataset.from_tensor_slices(train_features).batch(batch_size).map(lambda x: text_vectorizer(x))
    val_ds = tf.data.Dataset.from_tensor_slices(val_features).batch(batch_size).map(lambda x: text_vectorizer(x))
    test_ds = tf.data.Dataset.from_tensor_slices(test_features).batch(batch_size).map(lambda x: text_vectorizer(x))

    return train_ds, val_ds, test_ds
