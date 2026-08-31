import torch
import torch.nn.functional as F
from torch import nn


class Model(nn.Module):
    """A highly adaptable LSTM model that takes multivariate input and returns univariate output.

    Features:
    - Configurable LSTM layers
    - Batch normalization
    - Dropout
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
        activation (str): Activation function to use ('relu', 'tanh', 'leakyrelu', 'gelu')

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
        self.dropout_prob = configs.dropout

        assert self.configs.features == "MS", "Currently only MS implemented for LSTM_uncertain"
        # assert not (self.use_residual and self.bidirectional)
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

        self.mean_predictor = nn.Sequential(
            nn.Linear(self.hidden_size * self.d_factor, self.hidden_size),
            self.dropout,
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.pred_len),
        )
        self.uncertainty_predictor = nn.Sequential(
            nn.Linear(self.hidden_size * self.d_factor, self.hidden_size),
            self.dropout,
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.pred_len),
        )

        # Dense layers for output projection
        # self.dense1 = nn.Linear(self.hidden_size * self.d_factor, self.hidden_size)

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

        # Apply layer normalization if specified
        if self.use_layernorm:
            lstm_out = self.ln_lstm(lstm_out)

        # Apply dropout
        lstm_out = self.dropout(lstm_out)

        mean_prediction = self.mean_predictor(lstm_out)
        log_variance = self.uncertainty_predictor(lstm_out)
        uncertainty = torch.exp(0.5 * log_variance)

        mean_prediction = mean_prediction[:, :, None].expand(-1, -1, x.shape[-1])
        uncertainty = uncertainty[:, :, None].expand(-1, -1, x.shape[-1])

        return mean_prediction, uncertainty
