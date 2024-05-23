import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

import data_processing


def train_model(model, train_ds, val_ds, epochs=20, patience=3):
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
                  metrics=['accuracy'])

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=[early_stopping])

    return history


def evaluate_model(model, test_ds):
    results = model.evaluate(test_ds)
    print(f"Test Loss, Test Accuracy: {results}")
    return results


def plot_metrics(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(len(acc))

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


def main():
    from data_processing.data_processing import process_sst2 as dp
    import models.models as mdl

    MAX_LEN = 256
    BATCH_SIZE = 64
    EPOCHS = 20

    train_features, train_target, val_features, val_target, test_features, test_target, vocab_size = dp.process_sst2(
        "/path/to/data/directory")

    train_ds = tf.data.Dataset.from_tensor_slices((train_features, train_target)).batch(BATCH_SIZE).shuffle(
        buffer_size=1000)
    val_ds = tf.data.Dataset.from_tensor_slices((val_features, val_target)).batch(BATCH_SIZE)
    test_ds = tf.data.Dataset.from_tensor_slices((test_features, test_target)).batch(BATCH_SIZE)

    ffn_model = mdl.FFNModel(MAX_LEN, vocab_size).build_model()
    history_ffn = train_model(ffn_model, train_ds, val_ds, EPOCHS)
    plot_metrics(history_ffn)

    test_loss, test_acc = evaluate_model(ffn_model, test_ds)

    predictions = np.argmax(ffn_model.predict(test_ds), axis=1)
    report = classification_report(test_target, predictions)
    print(report)
    plot_confusion_matrix(test_target, predictions, class_names=["negative", "positive"])


if __name__ == "__main__":
    main()
