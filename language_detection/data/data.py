from dataclasses import dataclass

@dataclass
class RawDataset:
    x_train: list[str]
    x_test: list[str]
    y_train: list[str]
    y_test: list[str]
    labels: dict[str | int, str]
