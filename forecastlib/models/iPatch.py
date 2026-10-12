# From NoChamps (https://github.com/AIHNlab/NoChamps, models/iPatch.py), MIT License.
# Changes for forecastlib, marked "forecastlib:" below:
# - seq_len must be a multiple of main_cycle, checked when the model is built. Upstream padded the input at the
#   end by repeating the last value, and computed the padded length and the number of cycles wrongly, so any other
#   seq_len crashed with an unclear shape error.
# - The trend decomposition uses an odd moving-average kernel; an even main_cycle (e.g. 24) shortened the trend by
#   one step and model_trend crashed.
# - The number of variates and of time-mark columns is taken from the input instead of c_out / x_mark_size, so the
#   model works for features M, MS and S without extra settings.
# - A duplicated embedding initialisation is removed.
import torch
import torch.nn as nn
import torch.nn.functional as F

from forecastlib.layers.Autoformer_EncDec import series_decomp
from forecastlib.layers.Embed import DataEmbedding_inverted
from forecastlib.layers.iPatch_EncDec import iPatchEncoderLayer, PeriodicityReshape, PositionalEncoding
from forecastlib.layers.SelfAttention_Family import AttentionLayer, FullAttention
from forecastlib.layers.Transformer_EncDec import Encoder


class Model(nn.Module):
    """
    iPatch is a hybrid transformer architecture that integrates temporal patching
    from PatchTST with variate-centric attention from iTransformer.

    The input is cut into cycles of ``main_cycle`` steps (e.g. 24 for a daily cycle in hourly data), so seq_len must
    be a multiple of main_cycle; every cycle of every variate is a token. Attention runs across the variates within a
    cycle, then across the cycles of a variate.

    Settings: main_cycle (required), d_model, n_heads, e_layers, d_ff, dropout, factor, activation, use_norm
    (default True), model_trend (default False: a separate linear trend branch like DLinear) and x_mark_size
    (default: all time-mark columns as extra variate tokens, like iTransformer; 0: no marks).
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        if getattr(configs, "main_cycle", None) is None:
            raise ValueError("iPatch needs main_cycle, the number of steps of one cycle, e.g. 24 for hourly data")
        self.main_cycle = configs.main_cycle
        # forecastlib: no padding, the input must consist of whole cycles
        if configs.seq_len % self.main_cycle:
            raise ValueError(
                f"iPatch needs seq_len to be a multiple of main_cycle, got seq_len={configs.seq_len} and "
                f"main_cycle={self.main_cycle}; e.g. seq_len={configs.seq_len // self.main_cycle * self.main_cycle} "
                f"or {(configs.seq_len // self.main_cycle + 1) * self.main_cycle}"
            )
        self.seq_len = configs.seq_len
        self.n_cycles = self.seq_len // self.main_cycle
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model
        self.model_trend = getattr(configs, "model_trend", False)
        self.use_norm = getattr(configs, "use_norm", True)
        # forecastlib: None uses every mark column, 0 none
        self.x_mark_size = getattr(configs, "x_mark_size", None)
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(
            self.main_cycle, configs.d_model, configs.embed, configs.freq, configs.dropout
        )
        self.positional_encoding = PositionalEncoding(configs.d_model)
        self.periodicity_reshape = PeriodicityReshape(self.main_cycle)
        self.trend_projection = nn.Sequential(
            nn.Linear(self.seq_len, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.pred_len),
        )
        # forecastlib: an odd kernel keeps the length of the moving average
        self.decomposition = series_decomp(self.main_cycle + 1 - self.main_cycle % 2)
        # Encoder
        self.encoder = Encoder(
            [
                iPatchEncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    configs.d_model,
                    self.n_cycles,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )
        # Decoder
        if self.task_name == "long_term_forecast" or self.task_name == "short_term_forecast":
            self.projection = nn.Linear(configs.d_model * self.n_cycles, configs.pred_len, bias=True)
        else:
            raise NotImplementedError("Only forecasting task is implemented in iPatch model.")

    def _apply_positional_encoding(self, enc_out, n_tokens):
        enc_out_parts = torch.chunk(enc_out, n_tokens, dim=1)  # Split along the second dimension
        encoded_parts = [self.positional_encoding(part) for part in enc_out_parts]  # Positional encoding per part
        enc_out = torch.cat(encoded_parts, dim=1)  # Combine the parts
        return enc_out

    def _marks(self, x_mark_enc):
        """forecastlib: the mark columns used as tokens, reshaped by periodicity, and their number."""
        if x_mark_enc is None or self.x_mark_size == 0:
            return None, 0
        if self.x_mark_size is not None:
            if x_mark_enc.shape[-1] < self.x_mark_size:
                raise ValueError(f"x_mark_size={self.x_mark_size}, but the marks have {x_mark_enc.shape[-1]} columns")
            x_mark_enc = x_mark_enc[:, :, : self.x_mark_size]
        n_marks = x_mark_enc.shape[-1]
        return self.periodicity_reshape(x_mark_enc, n_marks, "apply"), n_marks

    def _encode(self, x_enc, x_mark_enc, n_features, n_marks):
        """Embed the cycles of every variate (and mark), encode them and project to pred_len per variate."""
        x_enc = self.periodicity_reshape(x_enc, n_features, "apply")
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out = self._apply_positional_encoding(enc_out, n_features + n_marks)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        enc_out = enc_out.reshape(enc_out.shape[0], n_features + n_marks, self.n_cycles * self.d_model)
        return self.projection(enc_out).permute(0, 2, 1)[:, :, :n_features]

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        n_features = x_enc.shape[-1]
        x_mark_enc, n_marks = self._marks(x_mark_enc)
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        dec_out = self._encode(x_enc, x_mark_enc, n_features, n_marks)

        # De-Normalization from Non-stationary Transformer
        if self.use_norm:
            dec_out = dec_out * (stdev[:, -1, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, -1, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out

    def forecast_with_trend(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        n_features = x_enc.shape[-1]
        x_mark_enc, n_marks = self._marks(x_mark_enc)
        # Add decomposition from DLinear
        seasonal_init, trend_init = self.decomposition(x_enc)

        if self.use_norm:
            # Normalize seasonal component
            means_seasonal = seasonal_init.mean(1, keepdim=True).detach()
            seasonal_init = seasonal_init - means_seasonal
            stdev_seasonal = torch.sqrt(torch.var(seasonal_init, dim=1, keepdim=True, unbiased=False) + 1e-5)
            seasonal_init /= stdev_seasonal

            # Normalize trend component separately
            means_trend = trend_init.mean(1, keepdim=True).detach()
            trend_init = trend_init - means_trend
            stdev_trend = torch.sqrt(torch.var(trend_init, dim=1, keepdim=True, unbiased=False) + 1e-5)
            trend_init /= stdev_trend

        # Process seasonal component
        seasonal_out = self._encode(seasonal_init, x_mark_enc, n_features, n_marks)

        # Process trend with simple linear projection like DLinear
        trend_out = self.trend_projection(trend_init.permute(0, 2, 1)).permute(0, 2, 1)

        if self.use_norm:
            # De-normalize seasonal
            seasonal_out = seasonal_out * (stdev_seasonal[:, -1, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            seasonal_out = seasonal_out + (means_seasonal[:, -1, :].unsqueeze(1).repeat(1, self.pred_len, 1))

            # De-normalize trend
            trend_out = trend_out * (stdev_trend[:, -1, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            trend_out = trend_out + (means_trend[:, -1, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        # Combine components
        dec_out = seasonal_out + trend_out

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == "long_term_forecast" or self.task_name == "short_term_forecast":
            if self.model_trend:
                dec_out = self.forecast_with_trend(x_enc, x_mark_enc, x_dec, x_mark_dec)
            else:
                dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len :, :]  # [B, L, D]

        else:
            raise NotImplementedError("Only forecasting task is implemented in iPatch model.")
