import numpy as np
import torch

from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader


PAD_ID = 0
CLS_ID = 1
EOS_ID = 2
MSK_ID = 3


def alt_char_based_byte_sequence(text: str, max_length: int) -> np.ndarray:
    """(not used currently) create byte sequence by converting char by char and fitting to window"""
    # truncate by character if needed
    char_byte_sequences: list[list[int]] = [memoryview(bytes(c, encoding="utf8")).tolist() for c in text]
    char_seq_lens: list[int] = [len(c) for c in char_byte_sequences]
    if sum(char_seq_lens) > max_length:
        for idx in reversed(range(len(char_seq_lens))):
            subtotal = sum(char_seq_lens[:idx])
            if subtotal <= max_length - 2:
                char_byte_sequences = char_byte_sequences[:idx]
                break
    byte_sequence: np.ndarray = np.array([byte_val for char in char_byte_sequences for byte_val in char])
    return byte_sequence


def pad_sequence(seq: torch.Tensor, max_length: int) -> torch.Tensor:
    """pad and truncate a sequence"""
    return torch.nn.functional.pad(seq, pad=(0, max(PAD_ID, max_length - len(seq))))[:max_length].to(torch.long)


def get_mask_from_lengths(seq_lengths: torch.Tensor, max_seq_len: int, device) -> torch.Tensor:
    """convert an array of minibatch sequence lengths into 2D boolean masks"""
    seq_lengths = seq_lengths.to(device)
    mask = torch.arange(max_seq_len).expand(len(seq_lengths), max_seq_len).to(device) >= seq_lengths.unsqueeze(1).to(
        device
    )
    return mask


def transform_text(
    text: str, is_training: bool, max_length: int = 1024, mask_pct: float = 0.15, run_tests: bool = False
) -> tuple[torch.Tensor, torch.Tensor, int, np.ndarray]:
    """transform text to bytes sequence"""
    # convert to integer sequence of bytes
    byte_sequence: np.ndarray = np.array(memoryview(bytes(text, encoding="utf8")).tolist())
    # byte_sequence: np.ndarray = alt_char_based_byte_sequence(text=text, max_length=max_length)
    # offset by 4 (for pad=0, cls=1, eos=2, and mask=3) and truncate
    byte_sequence += 4
    byte_sequence = byte_sequence[: max_length - 2]  # will add tokens
    # add cls (1) and eos (2)
    target_sequence = np.concatenate((np.array([CLS_ID]), byte_sequence, np.array([EOS_ID])))
    input_sequence = np.copy(target_sequence)
    modded_idxs = np.array([])

    # if training dataset, then do masking
    if is_training:
        # BERT mask inputs at mask_pct rate.
        mod_count = int(np.round(len(byte_sequence) * mask_pct))
        to_mod_idxs = np.random.choice(np.arange(1, len(byte_sequence) - 2), replace=False, size=mod_count)
        # of those, mask 80%, random value 10%, keep 10%
        split_indices = np.cumsum(
            [int(len(to_mod_idxs) * 0.80), int(len(to_mod_idxs) * 0.10), int(len(to_mod_idxs) * 0.10)]
        )
        split_indices[-1] = len(to_mod_idxs)
        mask_idxs, repl_idxs, keep_idxs, _ = np.split(to_mod_idxs, split_indices)
        input_sequence[mask_idxs] = MSK_ID
        input_sequence[repl_idxs] = np.random.randint(4, 256 + 4, size=(len(repl_idxs),))
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

    length = len(input_sequence)
    input_sequence_padded = pad_sequence(torch.from_numpy(input_sequence), max_length)
    target_sequence_padded = pad_sequence(torch.from_numpy(target_sequence), max_length)

    return input_sequence_padded, target_sequence_padded, length, modded_idxs


def batch_collate_function(
    samples,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor], torch.Tensor]:
    """collate batch data from Dataloader"""
    xs = [sample[0] for sample in samples]
    ys = [sample[1] for sample in samples]
    lens = [sample[2] for sample in samples]
    idxs = [torch.Tensor(sample[3]) for sample in samples]
    lbls = [sample[4] for sample in samples]
    return torch.stack(xs), torch.stack(ys), torch.Tensor(lens).long(), idxs, torch.Tensor(lbls).long()


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
    def __init__(
        self, texts: list[str], languages: list[str], mapping: dict[str, int], max_length: int, is_training: bool
    ):
        """
        Arguments:
            texts (list[str]): input texts
            languages (list[str]): target languages
            mapping (dict[str, int]): mapping of language to index
            max_length (int): max sequence length of model
            is_training (bool): mask data for training
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
        self.is_training = is_training

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor, int, np.ndarray, int]:
        idx = int(idx)
        text = self.texts[idx]
        lang_id = self.lang_idxs[idx]
        input_sequence, target_sequence, length, modded_idxs = transform_text(text=text, is_training=self.is_training)

        return input_sequence, target_sequence, length, modded_idxs, lang_id
