import io
import pickle
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import nn
from torch.optim.lr_scheduler import ConstantLR, CosineAnnealingLR, ExponentialLR, LambdaLR, SequentialLR

from forecastlib.models import (
    LSTM,
    MICN,
    Autoformer,
    Crossformer,
    DLinear,
    ETSformer,
    FEDformer,
    FiLM,
    FreTS,
    Informer,
    LightTS,
    LSTM_uncertain,
    MambaSimple,
    MultiPatchFormer,
    Nonstationary_Transformer,
    PatchTST,
    PAttn,
    Pyraformer,
    Reformer,
    SCINet,
    SegRNN,
    TemporalFusionTransformer,
    TiDE,
    TimeMixer,
    TimesNet,
    TimeXer,
    Transformer,
    TSMixer,
    WPMixer,
    iTransformer,
    DishTS
)
from forecastlib.utils.quantile import QuantileLoss, median_index, pinball_loss, resolve_quantiles
from forecastlib.utils.timefeatures import FREQ_MAP, time_features_from_frequency_str
from forecastlib.utils.tools import ConfigTracker, load_model_settings


class CustomLightningModule(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model_dict = {
            "TimesNet": TimesNet,
            "Autoformer": Autoformer,
            "Transformer": Transformer,
            "Nonstationary_Transformer": Nonstationary_Transformer,
            "DLinear": DLinear,
            "FEDformer": FEDformer,
            "Informer": Informer,
            "LightTS": LightTS,
            "Reformer": Reformer,
            "ETSformer": ETSformer,
            "PatchTST": PatchTST,
            "Pyraformer": Pyraformer,
            "MICN": MICN,
            "Crossformer": Crossformer,
            "FiLM": FiLM,
            "iTransformer": iTransformer,
            "TiDE": TiDE,
            "FreTS": FreTS,
            "MambaSimple": MambaSimple,
            "TimeMixer": TimeMixer,
            "TSMixer": TSMixer,
            "SegRNN": SegRNN,
            "TemporalFusionTransformer": TemporalFusionTransformer,
            "SCINet": SCINet,
            "PAttn": PAttn,
            "TimeXer": TimeXer,
            "WPMixer": WPMixer,
            "MultiPatchFormer": MultiPatchFormer,
            "LSTM": LSTM,
        }

        # Loss registry: name -> callable(pred, true) -> scalar. "pinball" needs
        # its quantile levels bound before it fits that shape; see below.
        self.loss_dict = {"MSE": F.mse_loss, "pinball": pinball_loss, "quantile": pinball_loss}

        self.args = ConfigTracker(args)
        self.features = self.args.features
        self.model_id = self.args.model_id
        self.scaler = args.scaler  # This one we don't want to track, i think. Because it's in a pickled represantation in the data_moduls hparams
        self.Model_class = self.model_dict[self.args.model].Model
        self.model = self.Model_class(self.args).float()
        self.criterion = self.loss_dict[self.args.loss]

        # Quantile forecasting is gated on the loss, not the model, so any model
        # that grows a quantile head can opt in later by setting n_quantiles.
        # resolve_quantiles() reads the same config the model did, so the two
        # cannot disagree - the check below is against a model that ignored it.
        self.quantiles = resolve_quantiles(self.args)
        self.median_idx = None
        if self.quantiles is not None:
            model_n_quantiles = getattr(self.model, "n_quantiles", None)
            if model_n_quantiles != len(self.quantiles):
                raise ValueError(
                    f"loss={self.args.loss!r} asks for {len(self.quantiles)} quantiles {self.quantiles}, but "
                    f"{self.args.model} exposes n_quantiles={model_n_quantiles}. That model has no quantile "
                    f"head; use a point loss (e.g. MSE) or a model that supports quantiles."
                )
            # Bind the levels so the call stays criterion(outputs, batch_y).
            self.criterion = QuantileLoss(self.quantiles)
            self.median_idx = median_index(self.quantiles)
        # Opt-in like EnsembleModule.set_return_subresults: off by default so
        # predict_step keeps its 3-tuple contract for existing consumers.
        self.return_quantiles = False
        # self.f_dim = -1 if self.args.features == 'MS' else 0
        self.lradj = self.args.lradj
        self.learning_rate = self.args.learning_rate
        self.num_epochs = self.args.train_epochs

        num_time_features = (
            FREQ_MAP[args.freq] if self.args.embed == "timeF" else len(time_features_from_frequency_str(args.freq))
        )
        # Datasets that pack extra known covariates onto x_mark (e.g.
        # Dataset_GEMS' forecast_in_mark) advertise the extra width via
        # ``args.known_len_extra`` so the example_input_array - and the
        # sanity forward pass it drives below - get the right mark shape.
        mark_features = num_time_features + int(getattr(self.args, "known_len_extra", 0))

        # HACK All models beside the LSTM return all dimensions, this old LSTM always just returns one, so this hack is necessary.
        # if self.args.model == "LSTM":
        #    self.args.target_idx = -1
        self.target_idx = self.args.target_idx

        self.diff = self.args.diff
        if self.diff:
            # self.base_col = self.args.base_col
            self.base_idx = self.args.base_idx
        else:
            self.base_idx = self.target_idx
        self.diff_comb = getattr(self.args, "diff_comb", False)

        #DISH-TS
        if "norm_model" in self.args:
            if self.args.norm_model == "DishTS":
                self.norm_model = DishTS.Model(self.args)
            elif self.args.norm_model == '':
                self.norm_model = None
            else:
                raise ValueError("Only DishTS implemented at the moment")
        else:
            self.norm_model = None

        if self.norm_model is not None and self.quantiles is not None:
            raise NotImplementedError(
                "norm_model (DishTS) does not handle the quantile axis; use it with a point loss for now"
            )

        self.example_input_array = (
            torch.randn(2, self.args.seq_len, self.args.enc_in),
            torch.randn(2, self.args.pred_len + self.args.label_len, self.args.enc_in),
            torch.randn(2, self.args.seq_len, mark_features),
            torch.randn(2, self.args.pred_len + self.args.label_len, mark_features),
        )

        # WPMixer Bullshit
        try:
            self.to(self.args.device)
            self.example_input_array = tuple([part.to(self.args.device) for part in self.example_input_array])
        except (KeyError, AttributeError):
            pass

        _ = self(
            *self.example_input_array
        )  # This assures that parameters that aren't accessed in the __init__ are still available as a hparam

        temp_hparams = {k: self.args[k] for k in self.args.accessed_attrs}

        try:
            temp_hparams = temp_hparams | {"patience": args.patience, "device": pickle.dumps(args.device)}
        except (KeyError, AttributeError):
            temp_hparams = temp_hparams | {"patience": args.patience}

        # temp_hparams = temp_hparams | {'patience' : args.patience,
        #                                "device": pickle.dumps(args.device)}

        self.save_hyperparameters(temp_hparams, ignore=["scaler", "device"])

    @classmethod
    def from_disk(cls, model_dir, device=None):
        if isinstance(model_dir, str):
            model_dir = Path(model_dir)
        args = load_model_settings(model_dir, device)
        checkpoint_path = next((model_dir / "checkpoints").iterdir())
        # model = CustomLightningModule.load_from_checkpoint(checkpoint_path,args=args)
        #model = cls.load_from_checkpoint(checkpoint_path, args=args)
        model = cls.load_from_checkpoint(checkpoint_path, map_location=device, args=args)

        return model

    @classmethod
    def from_db(cls, yaml_clob, model_blob, device=None):
        yaml_data = yaml.load(yaml_clob, Loader=yaml.FullLoader)
        yaml_data["scaler"] = pickle.loads(yaml_data["scaler"])

        if "device" in yaml_data:
            yaml_data["device"] = pickle.loads(yaml_data["device"])
            print(
                "Warning: This will break if it doesn't use the same torch.device as when trained."
            )  # TODO Fix, manual override?

        yaml_data = ConfigTracker(yaml_data)
        with io.BytesIO(model_blob) as checkpoint_stream:
            model = cls.load_from_checkpoint(checkpoint_stream, args=yaml_data)

        return model

    def forward(self, seq_x, seq_y, seq_x_mark, seq_y_mark):
        # Careful, changing order of inputs
        dec_inp = torch.zeros_like(seq_y[:, -self.args.pred_len :, :])  # .float()
        dec_inp = torch.cat([seq_y[:, : self.args.label_len, :], dec_inp], dim=1)  # .float().to(self.device)

        return self.model(seq_x, seq_x_mark, dec_inp, seq_y_mark)

    def _common_step(self, batch, batch_idx):
        batch_x, batch_y, batch_x_mark, batch_y_mark = batch
        batch_x = batch_x.float()
        batch_y = batch_y.float()
        batch_x_mark = batch_x_mark.float()
        batch_y_mark = batch_y_mark.float()
        # decoder input
        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len :, :])
        dec_inp = torch.cat([batch_y[:, : self.args.label_len, :], dec_inp], dim=1)

        if self.norm_model is not None:
            batch_x, dec_inp = self.norm_model(batch_x, 'forward', dec_inp)

        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)


        if self.norm_model is not None:
            outputs = self.norm_model(outputs, 'inverse')

        return outputs, batch_x, batch_y

    def set_return_quantiles(self, return_quantiles):
        """Make predict_step append the full quantile tensor as a 4th return value.

        Off by default so predict_step keeps returning ``(x, pred, true)`` for
        every existing consumer; ``pred`` is then the median quantile. Only
        meaningful for a quantile model.
        """
        if return_quantiles and self.quantiles is None:
            raise ValueError("set_return_quantiles(True) needs a quantile loss; this is a point model")
        self.return_quantiles = return_quantiles

    def training_step(self, batch, batch_idx):
        outputs, batch_x, batch_y = self._common_step(batch, batch_idx)

        # For a quantile model outputs carries a trailing quantile axis, which
        # this slicing preserves: target_idx indexes the channel axis, so the
        # result is [B,pred_len,n_targets(,n_q)] and the criterion (pinball,
        # which expects exactly that extra axis) takes it from there.
        outputs = outputs[:, -self.args.pred_len :, self.target_idx]
        batch_y = batch_y[:, -self.args.pred_len :, self.target_idx]
        loss = self.criterion(outputs, batch_y)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        outputs, batch_x, batch_y = self._common_step(batch, batch_idx)

        outputs = outputs[:, -self.args.pred_len :, self.target_idx]
        batch_y = batch_y[:, -self.args.pred_len :, self.target_idx]
        # Scalar either way, so EarlyStopping/ModelCheckpoint on val_loss are
        # unaffected by the quantile head.
        loss = self.criterion(outputs, batch_y)

        self.log("val_loss", loss)

    def _inverse_transform_quantiles(self, outputs):
        """Undo the scaler on [B,pred_len,C,n_q] predictions.

        The scaler is fitted on C features, so the quantile axis has to be
        folded into the sample axis instead of the feature axis - reshaping
        straight to (-1, C*n_q) is what silently corrupts the numbers.
        """
        b, p, c, n_q = outputs.shape
        flat = outputs.permute(0, 1, 3, 2).reshape(b * p * n_q, c).cpu()
        flat = self.scaler.inverse_transform(flat)
        return torch.Tensor(flat).reshape(b, p, n_q, c).permute(0, 1, 3, 2)

    def _predict_step_quantile(self, x, outputs, batch_y):
        """predict_step's quantile path: outputs is [B,pred_len,C,n_q]."""
        shape = batch_y.shape
        if outputs.shape[2] != shape[-1]:
            raise ValueError(
                f"quantile predictions cover {outputs.shape[2]} channels but the targets have {shape[-1]}; "
                "the point path's tiling fallback is not defined for quantiles"
            )

        outputs = self._inverse_transform_quantiles(outputs)
        batch_y = self.scaler.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1).cpu()).reshape(shape)

        outputs = outputs[:, :, self.target_idx]  # [B,pred_len,n_targets,n_q]
        batch_y = torch.Tensor(batch_y[:, :, self.base_idx])

        if self.diff and not self.diff_comb:
            # Same undifferencing as the point path, applied per quantile.
            # Caveat: cumulatively summing a per-step quantile is only the
            # quantile of the cumulative sum if the step errors are perfectly
            # rank-correlated, so these bands are an upper bound on the true
            # width. Fine for ranking/threshold work, not a calibrated interval.
            x_base = x[:, -1, self.base_idx][:, None]
            outputs = x_base.unsqueeze(-1) + outputs.cumsum(axis=1)

        point = outputs[..., self.median_idx]
        if self.return_quantiles:
            return x, point, batch_y, outputs
        return x, point, batch_y

    def predict_step(self, batch, batch_idx, dataloader_idx=0):

        outputs, batch_x, batch_y = self._common_step(batch, batch_idx)

        x_shape = batch_x.shape
        x = self.scaler.inverse_transform(batch_x.cpu().reshape(x_shape[0] * x_shape[1], -1)).reshape(x_shape)
        x = torch.Tensor(x)

        # I think the following parts works for M/S and MS?
        # outputs = outputs[:, -self.args.pred_len:, self.target_idx]
        # batch_y = batch_y[:, -self.args.pred_len:, self.target_idx]
        outputs = outputs[:, -self.args.pred_len :, :]
        batch_y = batch_y[:, -self.args.pred_len :, :]

        if self.quantiles is not None:
            # Everything below assumes outputs and batch_y have the same rank,
            # which stops holding once outputs carries a quantile axis.
            return self._predict_step_quantile(x, outputs, batch_y)

        shape = batch_y.shape
        if outputs.shape[-1] != batch_y.shape[-1]:
            print("Check mystery line 178 in LightningWrapper")
            outputs = np.tile(
                outputs, [1, 1, int(batch_y.shape[-1] / outputs.shape[-1])]
            )  # I honestly have no idea what this is good for.

        outputs = self.scaler.inverse_transform(outputs.reshape(shape[0] * shape[1], -1).cpu()).reshape(shape)
        batch_y = self.scaler.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1).cpu()).reshape(shape)
        outputs = torch.Tensor(outputs[:, :, self.target_idx])
        # batch_y = torch.Tensor(batch_y[:, :, self.target_idx])
        batch_y = torch.Tensor(batch_y[:, :, self.base_idx])

        if self.diff and not self.diff_comb:
            x_base = x[:, -1, self.base_idx][:, None]
            outputs = x_base + outputs.cumsum(axis=1)

        return x, outputs, batch_y

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)  # TODO optimizer params
        if self.lradj == "type1":
            scheduler = ExponentialLR(optimizer, gamma=0.5)
        elif self.lradj == "type2":
            lr_adjust = {0: self.learning_rate, 2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6, 10: 5e-7, 15: 1e-7, 20: 5e-8}
            scheduler = LambdaLR(
                optimizer, lr_lambda=lambda epoch: lr_adjust[max([x for x in lr_adjust if x <= epoch])]
            )
        elif self.lradj == "type3":
            scheduler1 = ConstantLR(optimizer, factor=1, total_iters=4)
            scheduler2 = ExponentialLR(optimizer, gamma=0.90)
            scheduler = SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[2])
        elif self.lradj == "cosine":
            scheduler = CosineAnnealingLR(optimizer, self.num_epochs)
        elif self.lradj is None or self.lradj == "":  # TODO this is untested
            return optimizer
        else:
            raise ValueError(f"Unknown lradj value {self.lradj}, possible values are type1 - type3 and cosine")

        return {"optimizer": optimizer, "lr_scheduler": scheduler}


class UncertaintyLightningModule(pl.LightningModule):  # TODO an Timeserieslibrary anpassen
    def __init__(self, args):
        super().__init__()
        self.model_dict = {
            "LSTM_uncertain": LSTM_uncertain,
        }
        self.loss_dict = {"nll": F.gaussian_nll_loss}

        self.args = ConfigTracker(args)

        ##
        self.features = self.args.features
        self.model_id = self.args.model_id
        self.scaler = args.scaler  # This one we don't want to track, i think. Because it's in a pickled represantation in the data_moduls hparams
        self.Model_class = self.model_dict[self.args.model].Model
        self.model = self.Model_class(self.args).float()
        self.criterion = self.loss_dict[self.args.loss]
        # self.f_dim = -1 if self.args.features == 'MS' else 0
        self.lradj = self.args.lradj
        self.learning_rate = self.args.learning_rate
        self.num_epochs = self.args.train_epochs

        num_time_features = (
            FREQ_MAP[self.args.freq]
            if self.args.embed == "timeF"
            else len(time_features_from_frequency_str(self.args.freq))
        )

        # HACK All models beside the LSTM return all dimensions, this old LSTM always just returns one, so this hack is necessary.
        # if self.args.model == "LSTM":
        #    self.args.target_idx = -1
        self.target_idx = self.args.target_idx

        self.diff = self.args.diff
        if self.diff:
            # self.base_col = self.args.base_col
            self.base_idx = self.args.base_idx
        else:
            self.base_idx = self.target_idx
        self.diff_comb = self.args.diff_comb

        self.example_input_array = (
            torch.randn(2, args.seq_len, args.enc_in),
            torch.randn(2, args.pred_len + args.label_len, args.enc_in),
            torch.randn(2, args.seq_len, num_time_features),
            torch.randn(2, args.pred_len + args.label_len, num_time_features),
        )
        # seq_x, seq_y, seq_x_mark, seq_y_mark

        _ = self(
            *self.example_input_array
        )  # This assures that parameters that aren't accessed in the __init__ are still available as a hparam

        temp_hparams = {k: self.args[k] for k in self.args.accessed_attrs}

        temp_hparams = temp_hparams | {"patience": args.patience}

        self.save_hyperparameters(temp_hparams, ignore="scaler")

    @classmethod
    def from_disk(cls, model_dir):
        if isinstance(model_dir, str):
            model_dir = Path(model_dir)
        args = load_model_settings(model_dir)
        checkpoint_path = next((model_dir / "checkpoints").iterdir())
        # model = CustomLightningModule.load_from_checkpoint(checkpoint_path,args=args)
        model = cls.load_from_checkpoint(checkpoint_path, args=args)
        return model

    @classmethod
    def from_db(cls, yaml_clob, model_blob):
        yaml_data = yaml.load(yaml_clob, Loader=yaml.FullLoader)
        yaml_data["scaler"] = pickle.loads(yaml_data["scaler"])
        yaml_data = ConfigTracker(yaml_data)
        with io.BytesIO(model_blob) as checkpoint_stream:
            model = cls.load_from_checkpoint(checkpoint_stream, args=yaml_data)

        return model

    def forward(self, seq_x, seq_y, seq_x_mark, seq_y_mark):
        dec_inp = torch.zeros_like(seq_y[:, -self.args.pred_len :, :])  # .float()
        dec_inp = torch.cat([seq_y[:, : self.args.label_len, :], dec_inp], dim=1)  # .float().to(self.device)

        return self.model(seq_x, seq_x_mark, dec_inp, seq_y_mark)

    def _common_step(self, batch, batch_idx):
        batch_x, batch_y, batch_x_mark, batch_y_mark = batch
        batch_x = batch_x.float()
        batch_y = batch_y.float()
        batch_x_mark = batch_x_mark.float()
        batch_y_mark = batch_y_mark.float()
        # decoder input
        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len :, :])
        dec_inp = torch.cat([batch_y[:, : self.args.label_len, :], dec_inp], dim=1)

        mean_prediction, uncertainty = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

        return mean_prediction, uncertainty, batch_x, batch_y

    def training_step(self, batch, batch_idx):
        mean_prediction, uncertainty, batch_x, batch_y = self._common_step(batch, batch_idx)
        mean_prediction = mean_prediction[:, -self.args.pred_len :, self.target_idx]
        uncertainty = uncertainty[:, -self.args.pred_len :, self.target_idx]
        batch_y = batch_y[:, -self.args.pred_len :, self.target_idx]

        loss = self.criterion(mean_prediction, batch_y, uncertainty)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        mean_prediction, uncertainty, batch_x, batch_y = self._common_step(batch, batch_idx)
        mean_prediction = mean_prediction[:, -self.args.pred_len :, self.target_idx]
        uncertainty = uncertainty[:, -self.args.pred_len :, self.target_idx]
        batch_y = batch_y[:, -self.args.pred_len :, self.target_idx]

        loss = self.criterion(mean_prediction, batch_y, uncertainty)

        self.log("val_loss", loss)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # TODO this could all be much more efficient, if we only use MS
        outputs, std, batch_x, batch_y = self._common_step(batch, batch_idx)
        # outputs = outputs[:, -self.args.pred_len:, self.target_idx]
        # uncertainty = uncertainty[:, -self.args.pred_len:, self.target_idx]
        # batch_y = batch_y[:, -self.args.pred_len:, self.target_idx]

        #############################################
        # loss = self.criterion(mean_prediction,batch_y,uncertainty)
        # self.log("val_loss", loss)
        x_shape = batch_x.shape
        x = self.scaler.inverse_transform(batch_x.cpu().reshape(x_shape[0] * x_shape[1], -1)).reshape(x_shape)
        x = torch.Tensor(x)

        # I think the following parts works for M/S and MS?
        # outputs = outputs[:, -self.args.pred_len:, self.target_idx]
        # batch_y = batch_y[:, -self.args.pred_len:, self.target_idx]
        outputs = outputs[:, -self.args.pred_len :, :]
        std = std[:, -self.args.pred_len :, :]
        batch_y = batch_y[:, -self.args.pred_len :, :]

        shape = batch_y.shape
        if outputs.shape[-1] != batch_y.shape[-1]:
            print("Check mystery line 386 in LightningWrapper")
            outputs = np.tile(
                outputs, [1, 1, int(batch_y.shape[-1] / outputs.shape[-1])]
            )  # I honestly have no idea what this is good for.

        outputs = self.scaler.inverse_transform(outputs.reshape(shape[0] * shape[1], -1).cpu()).reshape(shape)
        batch_y = self.scaler.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1).cpu()).reshape(shape)
        outputs = torch.Tensor(outputs[:, :, self.target_idx])
        # batch_y = torch.Tensor(batch_y[:, :, self.target_idx])
        batch_y = torch.Tensor(batch_y[:, :, self.base_idx])

        # std = std * abs(scale)
        std = torch.Tensor((std.cpu() * self.scaler.scale_)[:, :, self.target_idx])
        # cumulative_stds_naiv = torch.sqrt(torch.cumsum(std ** 2, dim=1))

        if self.diff and not self.diff_comb:
            x_base = x[:, -1, self.base_idx][:, None]
            outputs = x_base + outputs.cumsum(axis=1)

        return x, outputs, std, batch_y

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)  # TODO optimizer params
        if self.lradj == "type1":
            scheduler = ExponentialLR(optimizer, gamma=0.5)
        elif self.lradj == "type2":
            lr_adjust = {0: self.learning_rate, 2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6, 10: 5e-7, 15: 1e-7, 20: 5e-8}
            scheduler = LambdaLR(
                optimizer, lr_lambda=lambda epoch: lr_adjust[max([x for x in lr_adjust if x <= epoch])]
            )
        elif self.lradj == "type3":
            scheduler1 = ConstantLR(optimizer, factor=1, total_iters=4)
            scheduler2 = ExponentialLR(optimizer, gamma=0.90)
            scheduler = SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[2])
        elif self.lradj == "cosine":
            scheduler = CosineAnnealingLR(optimizer, self.num_epochs)
        elif self.lradj is None or self.lradj == "":  # TODO this is untested
            return optimizer
        else:
            raise ValueError(f"Unknown lradj value {self.lradj}, possible values are type1 - type3 and cosine")

        return {"optimizer": optimizer, "lr_scheduler": scheduler}


class WeightingModel(nn.Module):
    """A PyTorch module that computes the weights for each model in the ensemble.

    Args:
        args

    """

    def __init__(self, args):
        super().__init__()
        # feature_count,in_size,model_count,args.pred_len=48,hidden_size=256,num_layers=2,dropout=0.25,norm_func='softmax'
        # self.feature_count,
        # self.max_in_size,
        # len(model_list),
        # self.args.pred_len,
        # hidden_size=hidden_size,
        # num_layers=num_layers,
        # dropout=dropout,
        # norm_func=norm_func

        self.model_count = args.model_count
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.activation = nn.ReLU()
        self.norm_func_name = args.norm_func
        self.enc_in = args.enc_in
        # self.hidden_size = args.hidden_size

        layers = [
            nn.Linear(self.seq_len * self.enc_in, args.hidden_size),
            nn.Dropout(args.dropout),
            self.activation,
        ]
        for _ in range(args.num_layers - 2):
            layers += [
                nn.Linear(in_features=args.hidden_size, out_features=args.hidden_size),
                self.activation,
                nn.Dropout(args.dropout),
            ]
        layers += [nn.Linear(args.hidden_size, args.model_count * args.pred_len)]
        self.mlp = nn.Sequential(*layers)

        if args.norm_func == "softmax":
            self.norm_func = lambda x: nn.functional.softmax(x, dim=1)
        elif args.norm_func == "minmax":
            self.norm_func = lambda x: (
                (x - x.min(dim=1, keepdim=True).values.repeat(1, self.model_count, 1))
                / (x - x.min(dim=1, keepdim=True).values.repeat(1, self.model_count, 1)).sum(dim=1, keepdim=True)
            )

        # self.l1 = nn.Linear(in_size*feature_count,hidden_size)
        # self.l2 = nn.Linear(hidden_size,model_count*args.pred_len)

    def forward(self, x):
        x = torch.reshape(x, (x.size(0), -1))
        x = self.mlp(x)
        # x = self.l1(x)
        # x = self.activation(x)
        # x = self.l2(x)
        x = x.view(x.size(0), self.model_count, self.pred_len)
        x = self.norm_func(x)
        return x


class EnsembleModule(pl.LightningModule):
    """Since the data for this is normalized using the scaler from the first model in model_list all models should be trained on the same or at least similar data i think?"""

    # pylint: disable-next=unused-argument
    def __init__(self, args):
        super().__init__()
        self.loss_dict = {"MSE": F.mse_loss}
        assert args.features == "MS", "feature must be MS"
        assert args.diff, "Diff must be true"

        self.args = ConfigTracker(args)

        self.model_list = self.args.model_list
        try:
            self.model_path_list = self.args.model_path_list
        except KeyError:
            self.model_path_list = []

        self.args.model_count = len(self.args.model_list)
        self.base_idx = self.args.base_idx
        self.features = self.args.features
        self.model_id = self.args.model_id
        self.scaler = args.scaler  # This one we don't want to track, i think. Because it's in a pickled represantation in the data_moduls hparams
        self.criterion = self.loss_dict[self.args.loss]
        self.lradj = self.args.lradj
        self.learning_rate = self.args.learning_rate
        self.num_epochs = self.args.train_epochs

        num_time_features = (
            FREQ_MAP[args.freq] if self.args.embed == "timeF" else len(time_features_from_frequency_str(args.freq))
        )

        self.weighting = WeightingModel(self.args)

        # TODO these params need to be the same for all submodels:
        # timeenc
        # freq
        # embed
        # seq_len
        # pred_len
        # label_len

        self.example_input_array = (
            torch.randn(2, args.seq_len, args.enc_in),
            torch.randn(2, args.pred_len + args.label_len, args.enc_in),
            torch.randn(2, args.seq_len, num_time_features),
            torch.randn(2, args.pred_len + args.label_len, num_time_features),
        )
        # seq_x, seq_y, seq_x_mark, seq_y_mark

        _ = self(
            *self.example_input_array
        )  # This assures that parameters that aren't accessed in the __init__ are still available as a hparam

        temp_hparams = {k: self.args[k] for k in self.args.accessed_attrs}

        temp_hparams.pop("model_list")
        temp_hparams = temp_hparams | {"patience": args.patience, "model_path_list": self.model_path_list}
        self.save_hyperparameters(temp_hparams, ignore=["scaler", "model_list"])

        self.return_subresults = False
        for model in self.model_list:
            model.freeze()
            model.eval()

    @classmethod
    def from_disk(cls, model_dir, model_list):
        if isinstance(model_dir, str):
            model_dir = Path(model_dir)

        if not isinstance(model_list[0], CustomLightningModule):
            model_list = [CustomLightningModule.from_disk(model_folder) for model_folder in model_list]

        args = load_model_settings(model_dir)
        checkpoint_path = next((model_dir / "checkpoints").iterdir())
        args.model_list = model_list
        # model = CustomLightningModule.load_from_checkpoint(checkpoint_path,args=args)
        model = cls.load_from_checkpoint(checkpoint_path, args=args)
        return model

    @classmethod
    def from_db(cls, yaml_clob, model_blob, model_list):
        yaml_data = yaml.load(yaml_clob, Loader=yaml.FullLoader)
        yaml_data["scaler"] = pickle.loads(yaml_data["scaler"])
        yaml_data = ConfigTracker(yaml_data)

        yaml_data.model_list = model_list
        with io.BytesIO(model_blob) as checkpoint_stream:
            model = cls.load_from_checkpoint(checkpoint_stream, args=yaml_data)
        return model

    def set_return_subresults(self, return_subresults):
        self.return_subresults = return_subresults

    def _common_step(self, batch, batch_idx, dataloader_idx=0):
        """Computes the weighted average of the predictions of the models in the ensemble.

        Args:
            batch (tuple): A tuple containing the input data and the target data.
            batch_idx (int): The index of the batch.
            dataloader_idx (int, optional): The index of the dataloader. Defaults to 0.

        Returns:
            torch.Tensor: The weighted average of the predictions of the models in the ensemble.

        """
        # x  =
        w = self.weighting(batch[0])
        y_hat_list = []

        seq_x, seq_y, seq_x_mark, seq_y_mark = batch
        nodiff_batch = (seq_x[:, :, :-1], seq_y[:, :, :-1], seq_x_mark, seq_y_mark)

        # TODO hier aus batch die difference spalten rausschneiden
        for model in self.model_list:
            if model.diff:
                x, outputs, batch_y = model.predict_step(batch, batch_idx, dataloader_idx)
            else:
                x, outputs, batch_y = model.predict_step(nodiff_batch, batch_idx, dataloader_idx)

            y_hat_list.append(outputs)

        batch_y = batch_y.squeeze().to(w.device)

        # TODO what about scaling?! before/after?
        # y_hats = torch.stack(y_hat_list,axis=1)

        y_hats = torch.stack(y_hat_list, axis=1).squeeze().to(w.device)
        y_hat = torch.sum((y_hats * w), axis=1)
        if self.return_subresults:
            return x, y_hat, batch_y, y_hats
        return x, y_hat, batch_y

    def forward(self, seq_x, seq_y, seq_x_mark, seq_y_mark):
        # Careful, changing order of inputs
        dec_inp = torch.zeros_like(seq_y[:, -self.args.pred_len :, :])  # .float()
        dec_inp = torch.cat([seq_y[:, : self.args.label_len, :], dec_inp], dim=1)  # .float().to(self.device)

        return self.weighting(seq_x)

    # pylint: disable-next=arguments-differ
    def training_step(self, batch, batch_idx):
        """Computes the training loss for the ensemble.

        Args:
            batch (tuple): A tuple containing the input data and the target data.
            batch_idx (int): The index of the batch.

        Returns:
            torch.Tensor: The training loss for the ensemble.

        """
        _, y_hat, y_inv = self._common_step(batch, batch_idx)

        loss = self.criterion(y_hat, y_inv)
        self.log("train_loss", loss)
        return loss

    # pylint: disable-next=arguments-differ
    def validation_step(self, batch, batch_idx):
        """Computes the validation loss for the ensemble.

        Args:
            batch (tuple): A tuple containing the input data and the target data.
            batch_idx (int): The index of the batch.

        """
        _, y_hat, y_inv = self._common_step(batch, batch_idx)

        loss = self.criterion(y_hat, y_inv)
        self.log("val_loss", loss)

    # pylint: disable-next=arguments-differ
    def test_step(self, batch, batch_idx):
        """Computes the test loss for the ensemble.

        Args:
            batch (tuple): A tuple containing the input data and the target data.
            batch_idx (int): The index of the batch.

        """
        _, y_hat, y_inv = self._common_step(batch, batch_idx)

        loss = self.criterion(y_hat, y_inv)
        self.log("test_loss", loss)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # Could be one line, but I find this a little clearer
        if self.return_subresults:
            x, y_hat, batch_y, y_hats = self._common_step(batch, batch_idx, dataloader_idx)
            return x, y_hat, batch_y, y_hats
        x, y_hat, batch_y = self._common_step(batch, batch_idx, dataloader_idx)
        return x, y_hat, batch_y

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)  # TODO optimizer params
        if self.lradj == "type1":
            scheduler = ExponentialLR(optimizer, gamma=0.5)
        elif self.lradj == "type2":
            lr_adjust = {0: self.learning_rate, 2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6, 10: 5e-7, 15: 1e-7, 20: 5e-8}
            scheduler = LambdaLR(
                optimizer, lr_lambda=lambda epoch: lr_adjust[max([x for x in lr_adjust if x <= epoch])]
            )
        elif self.lradj == "type3":
            scheduler1 = ConstantLR(optimizer, factor=1, total_iters=4)
            scheduler2 = ExponentialLR(optimizer, gamma=0.90)
            scheduler = SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[2])
        elif self.lradj == "cosine":
            scheduler = CosineAnnealingLR(optimizer, self.num_epochs)
        elif self.lradj is None or self.lradj == "":  # TODO this is untested
            return optimizer
        else:
            raise ValueError(f"Unknown lradj value {self.lradj}, possible values are type1 - type3 and cosine")

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
