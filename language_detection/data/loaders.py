import pathlib

from language_detection.data.data import RawDataset


def load_wili_2018_dataset(data_path: str) -> RawDataset:

    required_files = ["labels.csv", "urls.txt", "x_test.txt", "x_train.txt", "y_test.txt", "y_train.txt"]
    expected_languages = 235

    dataset_directory: pathlib.Path = pathlib.Path(data_path)
    if not dataset_directory.is_dir():
        raise NotADirectoryError(f"dataset directory not found: {dataset_directory}")
    for required_file in required_files:
        filepath: pathlib.Path = pathlib.Path(data_path, required_file)
        if not filepath.is_file():
            raise FileNotFoundError(f"missing required dataset file {filepath}!")

    x_train: list[str] = open(pathlib.Path(data_path, "x_train.txt")).read().splitlines()
    y_train: list[str] = open(pathlib.Path(data_path, "y_train.txt")).read().splitlines()
    if len(x_train) != len(y_train):
        raise ValueError(f"x_train {len(x_train)} lines != y_train {len(y_train)} lines!")

    x_test: list[str] = open(pathlib.Path(data_path, "x_test.txt")).read().splitlines()
    y_test: list[str] = open(pathlib.Path(data_path, "y_test.txt")).read().splitlines()
    if len(x_test) != len(y_test):
        raise ValueError(f"x_test {len(x_test)} lines != y_test {len(y_test)} lines!")

    label_lines: list[str] = open(pathlib.Path(data_path, "labels.csv")).read().splitlines()
    labels: dict[str, str] = dict([line.split(";")[:2] for line in label_lines[1:]])
    if len(labels) != expected_languages:
        raise ValueError(f"expected {expected_languages} languages, found {len(labels)}!")

    for lbl in y_train + y_test:
        if lbl not in labels:
            raise ValueError(f"y data label '{lbl}' not in label keys!")
        
    dataset = RawDataset(
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        labels=labels
    )

    return dataset
