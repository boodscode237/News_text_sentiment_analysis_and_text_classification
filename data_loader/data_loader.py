import os
import urllib.request
import zipfile
import io

TASK2PATH = {
    "CoLA": 'https://dl.fbaipublicfiles.com/glue/data/CoLA.zip',
    "SST": 'https://dl.fbaipublicfiles.com/glue/data/SST-2.zip',
    "QQP": 'https://dl.fbaipublicfiles.com/glue/data/QQP-clean.zip',
    "STS": 'https://dl.fbaipublicfiles.com/glue/data/STS-B.zip',
    "MNLI": 'https://dl.fbaipublicfiles.com/glue/data/MNLI.zip',
    "QNLI": 'https://dl.fbaipublicfiles.com/glue/data/QNLIv2.zip',
    "RTE": 'https://dl.fbaipublicfiles.com/glue/data/RTE.zip',
    "WNLI": 'https://dl.fbaipublicfiles.com/glue/data/WNLI.zip',
    "diagnostic": 'https://dl.fbaipublicfiles.com/glue/data/AX.tsv'
}

MRPC_TRAIN = 'https://dl.fbaipublicfiles.com/senteval/senteval_data/msr_paraphrase_train.txt'
MRPC_TEST = 'https://dl.fbaipublicfiles.com/senteval/senteval_data/msr_paraphrase_test.txt'


def download_and_extract(task, data_dir):
    print(f"Downloading and extracting {task}...")
    data_file = f"{task}.zip"
    urllib.request.urlretrieve(TASK2PATH[task], data_file)
    with zipfile.ZipFile(data_file, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
    os.remove(data_file)
    print(f"\tCompleted {task}!")


def format_mrpc(data_dir):
    print("Processing MRPC...")
    mrpc_dir = os.path.join(data_dir, "MRPC")
    if not os.path.exists(mrpc_dir):
        os.makedirs(mrpc_dir)

    mrpc_train_file = os.path.join(mrpc_dir, "msr_paraphrase_train.txt")
    mrpc_test_file = os.path.join(mrpc_dir, "msr_paraphrase_test.txt")
    urllib.request.urlretrieve(MRPC_TRAIN, mrpc_train_file)
    urllib.request.urlretrieve(MRPC_TEST, mrpc_test_file)

    with io.open(mrpc_test_file, encoding='utf-8') as data_fh, \
            io.open(os.path.join(mrpc_dir, "test.tsv"), 'w', encoding='utf-8') as test_fh:
        header = data_fh.readline()
        test_fh.write("index\t#1 ID\t#2 ID\t#1 String\t#2 String\n")
        for idx, row in enumerate(data_fh):
            label, id1, id2, s1, s2 = row.strip().split('\t')
            test_fh.write(f"{idx}\t{id1}\t{id2}\t{s1}\t{s2}\n")

    print("\tCompleted MRPC!")


def download_diagnostic(data_dir):
    print("Downloading and extracting diagnostic...")
    diagnostic_dir = os.path.join(data_dir, "diagnostic")
    if not os.path.exists(diagnostic_dir):
        os.makedirs(diagnostic_dir)
    data_file = os.path.join(diagnostic_dir, "diagnostic.tsv")
    urllib.request.urlretrieve(TASK2PATH["diagnostic"], data_file)
    print("\tCompleted diagnostic!")


def main():
    data_dir = '/path/to/data/directory'

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    tasks = ["CoLA", "SST", "QQP", "STS", "MNLI", "QNLI", "RTE", "WNLI", "MRPC", "diagnostic"]

    for task in tasks:
        if task == 'MRPC':
            format_mrpc(data_dir)
        elif task == 'diagnostic':
            download_diagnostic(data_dir)
        else:
            download_and_extract(task, data_dir)

    print("All tasks downloaded and extracted.")


if __name__ == "__main__":
    main()
