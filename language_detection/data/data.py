import numpy as np
import torch

from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader


PAD_ID = 0
CLS_ID = 1
EOS_ID = 2
MSK_ID = 3


def transform_text(
    text: str, do_masking: bool, mask_pct: float = 0.15, run_tests: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """transform text to bytes sequence"""

    # convert to integer sequence of bytes
    byte_sequence: np.ndarray = np.array(memoryview(bytes(text, encoding="utf8")).tolist())
    # offset by 4 (for pad=0, cls=1, eos=2, and mask=3)
    byte_sequence += 4
    # add cls (1) and eos (2)
    target_sequence = np.concatenate((np.array([CLS_ID]), byte_sequence, np.array([EOS_ID])))
    input_sequence = np.copy(target_sequence)
    modded_idxs = np.array([])

    if do_masking:
        # BERT mask inputs at mask_pct rate.
        mod_count = int(np.round(len(byte_sequence) * mask_pct))
        to_mod_idxs = np.random.choice(np.arange(1, len(byte_sequence) - 2), replace=False, size=mod_count)
        # of those, mask 80%, random value 10%, keep 10%
        split_indices = np.cumsum(
            [int(len(to_mod_idxs) * 0.80), int(len(to_mod_idxs) * 0.10), int(len(to_mod_idxs) * 0.10)]
        )
        split_indices[-1] = len(to_mod_idxs)
        mask_idxs, repl_idxs, keep_idxs, _ = np.split(to_mod_idxs, split_indices)
        target_sequence[mask_idxs] = MSK_ID
        target_sequence[repl_idxs] = np.random.randint(4, 256 + 4, size=(len(repl_idxs),))
        modded_idxs = np.sort(to_mod_idxs)

        if run_tests:
            # check that all differing idxs are within the modded idxs
            diff_idx = np.where(np.invert(np.isclose(input_sequence, target_sequence)))
            okay = np.isin(diff_idx, modded_idxs).all(1)[0]
            assert len(diff_idx) > 0
            assert okay

            # check that initial, final tokens not changed
            assert len(input_sequence) == len(target_sequence)
            assert input_sequence[0] == target_sequence[0] == CLS_ID
            assert input_sequence[-1] == target_sequence[-1] == EOS_ID

    return input_sequence, target_sequence, modded_idxs


@dataclass
class RawDataset:
    x_train: list[str]
    x_test: list[str]
    y_train: list[str]
    y_test: list[str]
    idx2lang: dict[int, str]
    lang2idx: dict[str, int]
    labels: dict[str, str] | None
    dropped: list[str] | None = None


class BytesDataset(Dataset):
    def __init__(self, texts: list[str], languages: list[str], mapping: dict[str, int], do_masking: bool):
        """
        Arguments:
            texts (list[str]): input texts
            languages (list[str]): target languages
            mapping (dict[str, int]): mapping of language to index
            do_masking (bool): mask data for training
        """
        if len(texts) != len(languages):
            raise ValueError(f"length of texts {len(texts)} != length of labels {len(languages)}")
        missing_langs = set(languages) - set(mapping.keys())
        if len(missing_langs) > 0:
            raise ValueError(f"following languages not found in mapping: {missing_langs}")
        self.texts = texts
        self.languages = languages
        self.mapping = mapping
        self.lang_idxs = [self.mapping[lang] for lang in self.languages]
        self.do_masking = do_masking

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx) -> tuple[torch.LongTensor, torch.LongTensor, np.ndarray, int]:
        idx = int(idx)
        text = self.texts[idx]
        lang_id = self.lang_idxs[idx]
        input_sequence, target_sequence, modded_idxs = transform_text(text=text, do_masking=self.do_masking)

        return torch.LongTensor(input_sequence), torch.LongTensor(target_sequence), modded_idxs, lang_id
