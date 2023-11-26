import numpy as np
from language_detection.data.data import BytesDataset, RawDataset


def create_datasets(
    raw_data: RawDataset, max_seq_len: int = 1024, dev_pct=0.10
) -> tuple[BytesDataset, BytesDataset, BytesDataset]:
    """generate train, dev and test pytorch datasets"""

    if dev_pct >= 1.0:
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
        texts=raw_data.y_test,
        languages=raw_data.y_test,
        mapping=raw_data.lang2idx,
        max_length=max_seq_len,
        is_training=False,
    )
    return train_dataset, dev_dataset, test_dataset
