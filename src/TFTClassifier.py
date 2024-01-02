import math
import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset

import math

class TimeSeriesValueEmbedding(nn.Module):
    def __init__(self, feature_size, d_model):
        super().__init__()
        self.value_projection = nn.Linear(
            in_features=feature_size, out_features=d_model, bias=False
        )

    def forward(self, x):
        return self.value_projection(x)


class PositionalEncoder(nn.Module):
    def __init__(
        self,
        dropout: float = 0.1,
        max_seq_len: int = 5000,
        d_model: int = 512,
        batch_first: bool = False,
    ):
        """
        Parameters:
            dropout: the dropout rate
            max_seq_len: the maximum length of the input sequences
            d_model: The dimension of the output of sub-layers in the model
                     (Vaswani et al, 2017)
        """
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        self.batch_first = batch_first
        self.x_dim = 1 if batch_first else 0
        # copy pasted from PyTorch tutorial
        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_seq_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, enc_seq_len, dim_val] or
               [enc_seq_len, batch_size, dim_val]
        """
        x = x + self.pe[: x.size(self.x_dim)]
        return self.dropout(x)



class TransformerClassifier(nn.Module):

    def __init__(self, numerical_ft: list,
                 categorical_ft: list,
                 output_dim: int, max_seq_len: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.d_model = d_model
        self.pos_encoder = PositionalEncoder(
            dropout=dropout,
            d_model=d_model)
        self.embedding = nn.Linear(in_features=len(categorical_ft), out_features=d_model, bias=False)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.flatten = nn.Flatten()
        self.dense_block1 = self.dense_block(d_model*max_seq_len, 256, dropout=dropout)
        self.dense_block2 = self.dense_block(256, 128, dropout=dropout)
        self.dense_block3 = self.dense_block(128, 64, dropout=dropout)
        self.output_linear = nn.Linear(64, out_features=output_dim)
        self.init_weights()

    def dense_block(self, dense_input, dense_output, dropout, n_channels=None):
        if n_channels is None:
            n_channels = dense_output
        linear_layer = nn.Linear(dense_input, dense_output)
        nn.init.xavier_uniform_(linear_layer.weight)
        nn.init.zeros_(linear_layer.bias)
        return nn.Sequential(
            linear_layer,
            nn.BatchNorm1d(n_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.output_linear.bias.data.zero_()
        self.output_linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor = None) -> Tensor:
        """
        Arguments:
            src: Tensor, shape ``[seq_len, batch_size]``
            src_mask: Tensor, shape ``[seq_len, seq_len]``

        Returns:
            output Tensor of shape ``[seq_len, batch_size, ntoken]``
        """
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.flatten(output)
        output = self.dense_block1(output)
        output = self.dense_block2(output)
        output = self.dense_block3(output)
        output = self.output_linear(output)
        return nn.Sigmoid()(output)
