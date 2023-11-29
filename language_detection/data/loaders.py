import pathlib

from loguru import logger

from language_detection.data.data import RawDataset


def load_wili_2018_dataset(data_path: str, drop_duplicates: bool = True) -> RawDataset:
    """
    load the Wikipedia Language Identification database (WiLI-2018) from its extracted zip contents

    Parameters
    ----------
    data_path : str
        path to the directory where dataset zip contents were extracted
    drop_duplicates : bool, optional
        drop samples from training data if samples also in test data, by default True

    Returns
    -------
    RawDataset
        object with training and test splits
    """

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

    # map of language codes to full language names
    label_lines: list[str] = open(pathlib.Path(data_path, "labels.csv")).read().splitlines()
    code_lang_labels: dict[str, str] = dict([line.split(";")[:2] for line in label_lines[1:]])
    if len(code_lang_labels) != expected_languages:
        raise ValueError(f"expected {expected_languages} languages, found {len(code_lang_labels)}!")

    # map of integer index to language code label (sorted)
    lang_codes = sorted(code_lang_labels.keys())
    idx2lang = dict([(idx, code) for idx, code in enumerate(lang_codes)])
    lang2idx = dict([(code, idx) for idx, code in enumerate(lang_codes)])

    for lbl in y_train + y_test:
        if lbl not in code_lang_labels:
            raise ValueError(f"y data label '{lbl}' not in label keys!")

    # optionally, remove duplicate data
    dropped_samples: list[str] = []
    if drop_duplicates:
        test_set = set(x_test)
        duplicate_indices: set[int] = set()

        logger.info(f"'drop_duplicates' is true, dropping duplicates from *training* set...")
        for idx, train_sample in enumerate(x_train):
            if train_sample in test_set:
                duplicate_indices.add(idx)
                dropped_samples.append(train_sample)
        x_train = [sample for idx, sample in enumerate(x_train) if idx not in duplicate_indices]
        y_train = [label for idx, label in enumerate(y_train) if idx not in duplicate_indices]
        logger.info(f"dropped {len(dropped_samples)} samples from training data that also appeared in the test data")

    dataset = RawDataset(
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        idx2lang=idx2lang,
        lang2idx=lang2idx,
        labels=code_lang_labels,
        dropped=dropped_samples,
    )

    return dataset
