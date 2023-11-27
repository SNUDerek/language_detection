from dataclasses import dataclass

import torch


@dataclass
class TrainingConfig:
    accumulate_steps: int
    batch_size: int
    clip_grad_norm: float
    data_path: str
    dev_pct: float
    disp_loss_win: int
    init_lr: int
    max_length: int
    max_lr: int
    save_base: str
    seed: int
    total_epochs: int
    trial_name: str
    warmup_pct: float


class TransformerClassifier(torch.nn.Module):
    def __init__(
        self,
        num_classes: int,
        num_features: int = 256 + 4,
        transformer_layers: int = 6,
        ffn_dims: int = 1024,
        output_dims: int = 512,
        attn_heads: int = 4,
        activation: str = "gelu",
    ):
        """
        joint classification and masked language model transformer encoder

        Arguments
        - num_classes (int) : number of output classes
        - num_features (int) : number of input features, default 260
        - transformer_layers (int) : number of transformer layers, default 6
        - output_dims (int) : output dimensions of embedding and transformer, default 512
        - ffn_dims (int) : transformer ffn dimensions, default 1024
        - attn_heads (int) : transformer attention heads, default 4
        - activation (str) : transformer activation, choose 'relu' or 'gelu', default "gelu"
        """
        super().__init__()

        self.num_features = num_features
        self.num_classes = num_classes
        self.transformer_layer_count = transformer_layers
        self.ffn_dims = ffn_dims
        self.output_dims = output_dims
        self.attn_heads = attn_heads

        self.embedding = torch.nn.Embedding(num_embeddings=self.num_features, embedding_dim=self.output_dims)
        self.transformer_layers = torch.nn.ModuleList(
            [
                torch.nn.TransformerEncoderLayer(
                    d_model=self.output_dims,
                    nhead=self.attn_heads,
                    dim_feedforward=self.ffn_dims,
                    activation=activation,
                    batch_first=True,
                )
                for _ in range(self.transformer_layer_count)
            ]
        )
        self.clf_layer = torch.nn.Linear(in_features=self.output_dims, out_features=self.num_classes)
        self.mlm_layer = torch.nn.Linear(in_features=self.output_dims, out_features=self.num_features)

    def forward(self, x, pad_mask):
        x = self.embedding(x)
        for lyr in self.transformer_layers:
            x = lyr(x, src_key_padding_mask=pad_mask)
        mlm_preds = self.mlm_layer(x)
        clf_preds = self.clf_layer(x[:, 0, :])
        return clf_preds, mlm_preds
