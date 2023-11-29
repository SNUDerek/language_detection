import pathlib
import random

import numpy as np
import torch

from language_detection.data import transform_text, get_mask_from_lengths
from language_detection.model import TrainingConfig, TransformerClassifier


class LanguageDetector:
    def __init__(self, checkpoint_filepath: str):
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        self.checkpoint = self.load_checkpoint(checkpoint_filepath)
        if "output_mapping" not in self.checkpoint:
            raise ValueError("checkpoint file is missing 'output_mapping'!")
        self.config = self.load_config()
        self.model = self.load_model()
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)

    def load_checkpoint(self, checkpoint_path: str):
        if not pathlib.Path(checkpoint_path).is_file():
            raise ValueError(f"checkpoint file '{checkpoint_path}' does not exist!")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        return checkpoint

    def load_config(self) -> TrainingConfig:
        if not self.checkpoint:
            raise RuntimeError("no checkpoint loaded, load checkpoint first!")
        config = TrainingConfig(**self.checkpoint["config"])
        return config

    def load_model(self) -> TransformerClassifier:
        if not self.config:
            raise RuntimeError("no config loaded, load config first!")
        model = TransformerClassifier(num_classes=self.checkpoint["num_classes"])
        model.load_state_dict(self.checkpoint["model_state_dict"])
        _ = model.to(self.device)
        model.eval()
        return model

    def create_input_tensor(self, test_text: str) -> tuple[torch.Tensor, torch.Tensor]:
        """create model inputs for single sample"""
        x_input, y_output, seq_len, _idxs = transform_text(
            text=test_text, is_training=False, max_length=self.config.max_length
        )
        x_input = x_input.reshape(1, self.config.max_length)
        seq_lens = torch.tensor([seq_len])
        x_input = x_input.to(self.device)
        pad_mask = get_mask_from_lengths(seq_lens, self.config.max_length, self.device)
        return x_input, pad_mask

    def predict(self, test_text: str) -> str:
        x_input, x_mask = self.create_input_tensor(test_text)
        clf_logits, _mlm_logits = self.model.forward(x_input, x_mask)
        preds = clf_logits.max(1).indices.detach().cpu().numpy()
        lang_code = self.checkpoint["output_mapping"][preds[0]]
        if "extended_labels" in self.checkpoint:
            lang_name = self.checkpoint["extended_labels"][lang_code]
            return lang_name
        return lang_code
