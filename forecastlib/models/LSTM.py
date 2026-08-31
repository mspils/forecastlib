import torch
import torch.nn.functional as F
from torch import nn


class Model(nn.Module):
    """A highly adaptable LSTM model that takes multivariate input and returns univariate output.

    Features:
    - Configurable LSTM layers
    - Batch normalization
    - Dropout
    - Residual connections
    - Layer normalization
    - Configurable activation functions
    - Dense output layers

    Args:
        in_features (int): Number of input features (multivariate input size)
        seq_length (int): Length of input sequence
        out_size (int): Length of output sequence
        hidden_size (int): Size of LSTM hidden state
        num_layers (int): Number of LSTM layers
        dropout (float): Dropout probability
        bidirectional (bool): Whether to use bidirectional LSTM
        use_layernorm (bool): Whether to use layer normalization
        use_residual (bool): Whether to use residual connections
        activation (str): Activation function to use ('relu', 'tanh', 'leakyrelu', 'gelu')
        output_activation (str or None): Final activation function (None for linear output)

    """

    def __init__(self, configs):
        super().__init__()
        # def __init__(self, feature_count, in_size=144, out_size=48, hidden_size_lstm=128, hidden_size=64,num_layers_lstm=2, dropout=0.2, num_layers=2):
        self.configs = configs

        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len  # unused
        self.pred_len = configs.pred_len

        self.enc_in = configs.enc_in
        self.hidden_size = configs.hidden_size
        self.num_layers = configs.num_layers
        self.bidirectional = configs.bidirectional
        self.use_layernorm = configs.use_layernorm
        self.use_residual = configs.use_residual
        self.dropout_prob = configs.dropout

        assert not (self.use_residual and self.bidirectional)
        # Directional factor (2 if bidirectional, else 1)
        self.d_factor = 2 if self.bidirectional else 1

        # LSTM Layer
        self.lstm = nn.LSTM(
            input_size=self.enc_in,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout_prob if self.num_layers > 1 else 0,
            bidirectional=self.bidirectional,
        )

        # Dropout layer
        self.dropout = nn.Dropout(self.dropout_prob)

        # Dense layers for output projection
        self.dense1 = nn.Linear(self.hidden_size * self.d_factor, self.hidden_size)

        # Layer Normalization for LSTM output
        if self.use_layernorm:
            self.ln_lstm = nn.LayerNorm([self.hidden_size * self.d_factor])
        if self.use_layernorm:
            self.ln_dense = nn.LayerNorm([self.hidden_size])

        # Final output layer
        if self.configs.features == "M":
            self.output_layer = nn.Linear(self.hidden_size, self.pred_len * self.configs.enc_in)
        else:
            self.output_layer = nn.Linear(self.hidden_size, self.pred_len)

        # Set activation function
        if configs.activation == "relu":
            self.activation = F.relu
        elif configs.activation == "tanh":
            self.activation = torch.tanh
        elif configs.activation == "leakyrelu":
            self.activation = F.leaky_relu
        elif configs.activation == "gelu":
            self.activation = F.gelu
        else:
            self.activation = F.relu  # Default to ReLU

        # Set output activation function
        if configs.output_activation == "sigmoid":
            self.output_activation = torch.sigmoid
        elif configs.output_activation == "tanh":
            self.output_activation = torch.tanh
        elif configs.output_activation == "softmax":
            self.output_activation = lambda x: F.softmax(x, dim=-1)
        else:
            self.output_activation = None  # Linear output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        x = x_enc
        batch_size = x.size(0)

        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers * self.d_factor, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * self.d_factor, batch_size, self.hidden_size).to(x.device)

        # LSTM forward pass
        lstm_out, _ = self.lstm(x, (h0, c0))
        # (batch_size, seq_len, 2 (if bi_directionel) * hidden_size)
        lstm_out = lstm_out[:, -1, :]

        # Store original output for residual connection if needed
        original_lstm_out = lstm_out

        # Apply layer normalization if specified
        if self.use_layernorm:
            lstm_out = self.ln_lstm(lstm_out)

        # Apply dropout
        lstm_out = self.dropout(lstm_out)

        # First dense layer
        dense_out = self.dense1(lstm_out)
        dense_out = self.activation(dense_out)

        # Apply layer normalization on dense output if specified
        if self.use_layernorm:
            dense_out = self.ln_dense(dense_out)

        # Apply residual connection if specified
        if self.use_residual:  # and self.hidden_size == self.hidden_size * self.d_factor:
            dense_out = dense_out + original_lstm_out

        # Apply dropout again
        dense_out = self.dropout(dense_out)

        # Final output layer
        if self.configs.features == "M":
            output = self.output_layer(dense_out).view(batch_size, self.pred_len, self.configs.enc_in)
        else:
            output = self.output_layer(dense_out)
            # output = output[:,:,None]
            output = output[
                :, :, None
            ].expand(
                -1, -1, x.shape[-1]
            )  # THe wrappers assume that we predict all features, so we just make something up for the irrelevant values.

        if self.output_activation is not None:
            output = self.output_activation(output)

        # [B, L, D]
        return output
