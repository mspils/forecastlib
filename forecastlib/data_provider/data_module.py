import inspect
import pickle
from functools import partial

import lightning.pytorch as pl
from torch.utils.data import DataLoader

from forecastlib.data_provider.data_loader import (
    Dataset_Custom,
    Dataset_MW,
)
from forecastlib.utils.tools import ConfigTracker, load_model_settings

data_dict = {
    "custom": Dataset_Custom,
    "MW": Dataset_MW,
}


class CustomDataModule(pl.LightningDataModule):
    @staticmethod
    def _get_dataset_name(dataset_arg):
        if isinstance(dataset_arg, str):
            return dataset_arg

        if inspect.isclass(dataset_arg):
            for name, cls in data_dict.items():
                if cls is dataset_arg:
                    return name
            return dataset_arg.__name__

        raise TypeError("args.data must be either a dataset name string or a dataset class")

    @staticmethod
    def _resolve_dataset_class(dataset_arg):
        if isinstance(dataset_arg, str):
            return data_dict[dataset_arg]
        if inspect.isclass(dataset_arg):
            return dataset_arg
        raise TypeError("args.data must be either a dataset name string or a dataset class")

    def __init__(self, args) -> None:
        super().__init__()
        self.args = ConfigTracker(args)

        self.Dataset_class = self._resolve_dataset_class(self.args.data)
        self.data_name = self._get_dataset_name(self.args.data)
        self.args.timeenc = 0 if self.args.embed != "timeF" else 1

        self.num_workers = self.args.num_workers

        self.train_set = None
        self.val_set = None
        self.test_set = None
        self.setup("fit")

        temp_hparams = {}
        for k in self.args.accessed_attrs:
            temp_hparams[k] = self.data_name if k == "data" else self.args[k]
        temp_hparams |= {
            "batch_size": self.args.batch_size,
            "scaler": pickle.dumps(self.scaler),
        }
        self.save_hyperparameters(temp_hparams, logger=False)
        self.save_hyperparameters(
            {k: temp_hparams[k] for k in ["batch_size", "data_path", "target"]}
        )  # These are also logged to tensorboard
        # TODO Differencing

    @classmethod
    def from_disk(cls, model_dir, root_path=None, data_path=None, dataset_class=None, device=None):
        args = load_model_settings(model_dir, device=device)
        args.root_path = root_path or args.root_path
        args.data_path = data_path or args.data_path
        args.data = dataset_class or args.data

        checkpoint_path = next((model_dir / "checkpoints").iterdir())
        return cls.load_from_checkpoint(checkpoint_path, map_location=device, args=args)

    def setup(self, stage: str) -> None:

        # self.save_hyperparameters({"batch_size": self.hparams.batch_size})

        # I don't really do setup i guess?
        # That train_set objects etc don't have a state, so no need to reload them or anything.
        # Additionally, i need all of them in my custom callback so i'll just load them once in the beginning and then ignore the setup method
        if self.train_set is None:
            args = self.args
            data_set_template = partial(
                self.Dataset_class,
                root_path=args.root_path,
                data_path=args.data_path,
                size=[args.seq_len, args.label_len, args.pred_len],
                features=args.features,
                target=args.target,
                timeenc=args.timeenc,
                freq=args.freq,
            )

            self.train_set = data_set_template(flag="train", args=args)
            # if (dynamic_scaler := getattr(self.train_set, "dynamic_scaler", None)) is not None: this is now done in the dataset
            #    args.dynamic_scaler = dynamic_scaler
            self.val_set = data_set_template(flag="val", args=args)
            self.test_set = data_set_template(flag="test", args=args)
            self.scaler = self.train_set.scaler

    def train_dataloader(self, num_workers=None, **kwargs) -> DataLoader:
        num_workers = num_workers or self.num_workers
        return DataLoader(
            self.train_set, batch_size=self.args.batch_size, shuffle=True, num_workers=num_workers, **kwargs
        )

    def val_dataloader(self, num_workers=None, **kwargs) -> DataLoader:
        num_workers = num_workers or self.num_workers
        return DataLoader(
            self.val_set, batch_size=self.args.batch_size, shuffle=False, num_workers=num_workers, **kwargs
        )

    def test_dataloader(self, num_workers=None, **kwargs) -> DataLoader:
        num_workers = num_workers or self.num_workers
        return DataLoader(
            self.test_set, batch_size=self.args.batch_size, shuffle=False, num_workers=num_workers, **kwargs
        )

    def predict_dataloader(self, num_workers=None, **kwargs) -> DataLoader:
        num_workers = num_workers or self.num_workers
        # return the trainset, but sorted
        return DataLoader(
            self.train_set, batch_size=self.args.batch_size, shuffle=False, num_workers=num_workers, **kwargs
        )

    @property
    def n_cols(self) -> int:
        """Number of input feature columns, read from the scaler fitted during setup."""
        return self.scaler.mean_.shape[0]

    @property
    def feature_names(self) -> list[str] | None:
        """Ordered column names from the training set, or None if unavailable."""
        raw = getattr(self.train_set, "data_x_raw", None)
        return list(raw) if raw is not None else None

    def infer_args(self) -> ConfigTracker:
        """Populate self.args with dataset characteristics (sizes, scaler, feature indices)."""
        args = self.args
        num_cols = self.n_cols
        args.enc_in = num_cols
        args.dec_in = num_cols
        args.c_out = num_cols
        args.scaler = self.scaler

        args.diff = getattr(args, "diff", False)
        args.diff_comb = getattr(args, "diff_comb", False)

        if args.diff_comb:
            if not args.diff:
                raise ValueError("If combining normal and differenced prediction, diff must be true")
            if args.features != "M":
                raise ValueError("Only implemented for features == M")
            args.base_idx = list(range(num_cols // 2))
            args.target_idx = list(range(num_cols))

        elif args.diff and args.features in {"MS", "S"}:
            names = self.feature_names
            if names is None:
                raise TypeError("Differencing on data level not implemented for this dataset type")
            args.base_idx = [names.index(args.target)]
            args.target_idx = [names.index(f"{args.target}_d1")]

        elif args.diff and args.features == "M":
            args.base_idx = list(range(num_cols // 2))
            args.target_idx = list(range(num_cols // 2, num_cols))

        elif args.features in {"MS", "S"}:
            names = self.feature_names
            if names is None:
                raise TypeError("Target index inference requires a dataset that exposes column names")
            args.target_idx = [names.index(args.target)]

        elif args.features == "M":
            args.target_idx = list(range(num_cols))

        else:
            raise ValueError(f"Invalid value {args.features!r} for features, must be M, MS, or S")

        return args
