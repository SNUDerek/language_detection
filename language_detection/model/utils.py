import datetime

import numpy as np
import pandas as pd
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


def export_results_tsv(
    samples: list[str],
    targets: list[int],
    predictions: list[int],
    idx_mapping: dict,
    output_filename: str,
    desc_mapping: dict | None = None,
):
    """save tsv of results"""
    if not (len(samples) == len(targets) == len(predictions)):
        raise ValueError(f"samples, targets, predictions lengths not equal!")
    true_codes = [idx_mapping[p] for p in targets]
    pred_codes = [idx_mapping[p] for p in predictions]
    col_ord = ["true_label", "pred_label", "sample"]
    df_data = {"true_label": true_codes, "pred_label": pred_codes, "sample": samples}
    if desc_mapping:
        true_names = [desc_mapping[p] for p in true_codes]
        pred_names = [desc_mapping[p] for p in pred_codes]
        col_ord = ["true_label", "true_name", "pred_label", "pred_name", "sample"]
        df_data["true_name"] = true_names
        df_data["pred_name"] = pred_names
    df = pd.DataFrame.from_dict(df_data, orient="index").transpose()
    df = df[col_ord]
    df.to_csv(output_filename, sep="\t")
