import pandas as pd
import os
import nltk
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt


def read_tsv_file(path):
    return pd.read_csv(path, sep="\t")


def convert_to_csv(df, output_path):
    df.to_csv(output_path, index=False)


def process_sst2(data_dir):
    sst2_train_df = pd.read_csv(f"{data_dir}/SST-2/train.csv")
    sst2_test_df = pd.read_csv(f"{data_dir}/SST-2/test.csv")
    sst2_dev_df = pd.read_csv(f"{data_dir}/SST-2/dev.csv")

    df_text = sst2_train_df[['sentence', 'label']]
    test_text = sst2_test_df.copy()
    valid_text = sst2_dev_df.copy()

    df_text['words'] = df_text['sentence'].apply(lambda x: len(word_tokenize(x)))

    sns.set_style("whitegrid")
    sns.set(font_scale=2)
    count_plot = df_text.label.value_counts().plot.bar(x="Label", y="Texts", figsize=(32, 8),
                                                       color=['green', 'blue', 'red'])
    plt.title('Label Counts', fontsize=24, fontweight='bold', color='#333333')
    plt.xlabel('Label', fontsize=20, fontweight='bold', color='#333333')
    plt.ylabel('Counts', fontsize=20, fontweight='bold', color='#333333')
    plt.xticks(fontsize=16, fontweight='bold', color='#333333')
    plt.yticks(fontsize=16, fontweight='bold', color='#333333')
    plt.grid(linestyle='--', linewidth=1.5, color='#cccccc')
    plt.show()

    data = df_text[df_text['words'] >= 1]
    vocab = {word.lower() for sentence in [x.split() for x in data['sentence'].tolist()] for word in sentence}
    print("Number of distinct words in raw data:", len(vocab))

    data = data.sample(frac=1)
    data['sentiment'] = np.where(data['label'] == 0, 'negative', 'positive')
    data["sentiment"] = data["sentiment"].astype('category')
    data["label"] = data["label"].astype('int')
    data["sentiment_id"] = data["sentiment"].cat.codes

    train_text, test_text = train_test_split(data, test_size=0.2, random_state=23052024)

    train_features = train_text['sentence']
    train_target = train_text['label']
    val_features = valid_text['sentence']
    val_target = valid_text['label']
    test_features = test_text['sentence']
    test_target = test_text['label']

    return train_features, train_target, val_features, val_target, test_features, test_target, len(vocab)


def main():
    DATASET_ROOT = "/path/to/data/directory"
    TASKS = ["SST-2", "CoLA"]
    SPLIT_NAMES = ["train", "test", "dev"]
    FILE_EXTENSION = ".tsv"

    for task in TASKS:
        task_root = f"{DATASET_ROOT}/{task}"
        for split in SPLIT_NAMES:
            input_path = f"{task_root}/{split}{FILE_EXTENSION}"
            output_path = f"{task_root}/{split}.csv"
            df = read_tsv_file(input_path)
            convert_to_csv(df, output_path)

    train_features, train_target, val_features, val_target, test_features, test_target, vocab_size = process_sst2(
        DATASET_ROOT)
    print("Preprocessing completed.")


if __name__ == "__main__":
    main()
