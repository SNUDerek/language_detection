import datetime

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score  # type: ignore

from language_detection.data.data import BytesDataset, RawDataset


def create_datasets(
    raw_data: RawDataset, max_seq_len: int = 1024, dev_pct=0.10
) -> tuple[BytesDataset, BytesDataset, BytesDataset]:
    """generate train, dev and test pytorch datasets."""

    if dev_pct < 0 or dev_pct >= 1.0:
        raise ValueError(f"dev pct must be between 0.0 and 1.0, recommend 0.1 to 0.2")
    if max_seq_len < 1:
        raise ValueError(f"max_seq_len must be greater than 0, recommend 512 or 1024")

    randomized_train_indices = list(range(len(raw_data.x_train)))
    np.random.shuffle(randomized_train_indices)
    dev_cutoff = int(len(raw_data.x_train) * (dev_pct))
    dev_indices = sorted(randomized_train_indices[:dev_cutoff])
    train_indices = sorted(randomized_train_indices[dev_cutoff:])

    train_dataset = BytesDataset(
        texts=[raw_data.x_train[i] for i in train_indices],
        languages=[raw_data.y_train[i] for i in train_indices],
        mapping=raw_data.lang2idx,
        max_length=max_seq_len,
        is_training=True,
    )
    dev_dataset = BytesDataset(
        texts=[raw_data.x_train[i] for i in dev_indices],
        languages=[raw_data.y_train[i] for i in dev_indices],
        mapping=raw_data.lang2idx,
        max_length=max_seq_len,
        is_training=True,
    )
    test_dataset = BytesDataset(
        texts=raw_data.x_test,
        languages=raw_data.y_test,
        mapping=raw_data.lang2idx,
        max_length=max_seq_len,
        is_training=False,
    )
    return train_dataset, dev_dataset, test_dataset


def evaluate_model(
    set_name: str, targets: np.ndarray | list[int], predictions: np.ndarray | list[int], print_values: bool = True
) -> dict:
    """evaluate precision, recall and f1 and optionally print results"""
    micro_prc = precision_score(targets, predictions, average="micro", zero_division=0)
    micro_rcl = recall_score(targets, predictions, average="micro", zero_division=0)
    micro_f1b = f1_score(targets, predictions, average="micro", zero_division=0)
    macro_prc = precision_score(targets, predictions, average="macro", zero_division=0)
    macro_rcl = recall_score(targets, predictions, average="macro", zero_division=0)
    macro_f1b = f1_score(targets, predictions, average="macro", zero_division=0)
    if print_values:
        print(f"[{datetime.datetime.now().isoformat()}] {set_name} micro prc: {micro_prc:.5f},\tmacro {macro_prc:.5f}")
        print(f"[{datetime.datetime.now().isoformat()}] {set_name} micro rcl: {micro_rcl:.5f},\tmacro {macro_rcl:.5f}")
        print(f"[{datetime.datetime.now().isoformat()}] {set_name} micro f1b: {micro_f1b:.5f},\tmacro {macro_f1b:.5f}")
    return {
        "micro_prc": micro_prc,
        "micro_rcl": micro_rcl,
        "micro_f1b": micro_f1b,
        "macro_prc": macro_prc,
        "macro_rcl": macro_rcl,
        "macro_f1b": macro_f1b,
    }
