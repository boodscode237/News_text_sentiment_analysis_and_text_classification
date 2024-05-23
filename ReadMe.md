# My Solution

This repository contains code for various tasks related to natural language processing (NLP) using TensorFlow and Python. Below is an overview of the code structure and how to use it.

## Code Structure

The code is organized into several modules:

1. `data_processing.py`: Contains functions for processing and preparing data for model training and evaluation.
2. `models.py`: Defines different neural network models for NLP tasks.
3. `train_eval.py`: Contains functions for training and evaluating the models.
4. `utils.py`: Utility functions used across modules.

## Usage

### 1. Data Preparation (`data_processing.py`)

- `process_sst2(data_dir)`: Processes the SST-2 dataset for sentiment analysis. This function reads the data from the specified directory, preprocesses it, and splits it into train, validation, and test sets.

### 2. Model Definition (`models.py`)

- `FFNModel`: Defines a FeedForward Neural Network model for text classification tasks.
- `Conv1DModel`: Defines a 1D Convolutional Neural Network model for text classification tasks.

### 3. Training and Evaluation (`train_eval.py`)

- `train_model(model, train_ds, val_ds, epochs=20, patience=3)`: Trains the specified model using the training dataset and validates it using the validation dataset. The function supports early stopping based on validation loss.
- `evaluate_model(model, test_ds)`: Evaluates the trained model on the test dataset and returns the loss and accuracy.
- `plot_metrics(history)`: Plots the training and validation metrics (accuracy and loss) over epochs.
- `plot_confusion_matrix(y_true, y_pred, class_names)`: Plots the confusion matrix for the model's predictions on the test dataset.

### 4. Utility Functions (`utils.py`)

- Various utility functions are defined here, including functions for creating a text vectorization layer and preparing data for the model.

## Example Usage

```python
import data_processing as dp
import models as mdl
import train_eval as te

# Process data
train_features, train_target, val_features, val_target, test_features, test_target, vocab_size = dp.process_sst2("/path/to/data/directory")

# Build and train the model
ffn_model = mdl.FFNModel(MAX_LEN, vocab_size).build_model()
history_ffn = te.train_model(ffn_model, train_ds, val_ds, EPOCHS)
te.plot_metrics(history_ffn)

# Evaluate the model
test_loss, test_acc = te.evaluate_model(ffn_model, test_ds)

# Get predictions and plot confusion matrix
predictions = np.argmax(ffn_model.predict(test_ds), axis=1)
te.plot_confusion_matrix(test_target, predictions, class_names=["negative", "positive"])
```

Please replace `/path/to/data/directory` with the actual path to your data directory.

## Requirements

- Python 3.x
- TensorFlow 2.x
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- nltk (for data preprocessing)