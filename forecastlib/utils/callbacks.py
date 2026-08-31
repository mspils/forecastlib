"""Contains callbacks for LightningModules."""

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import Callback, TQDMProgressBar
from scipy import stats
from sklearn.preprocessing import StandardScaler

import forecastlib.utils.metrics as mt
from forecastlib.models.LightningWrapper import CustomLightningModule, EnsembleModule, UncertaintyLightningModule
from forecastlib.data_provider.data_loader import Dataset_Diff
from forecastlib.utils.quantile import (
    coverage_series,
    interval_width_series,
    pinball_series,
    quantile_crossing_series,
)


def combine_repeat(pred, pred_metric, combined_metrics=False):
    """Find local minima in pred_metric along axis 0 (forecast horizon length) and adjust pred values.
    This function just repeats the previous forecast step.

    Args:
        pred: torch.Tensor of shape [batch_size, pred_len, num_features]
        pred_metric: torch.Tensor of shape [pred_len, num_features]
        combined_features: bool, if True use mean across features for minima detection

    Returns:
        pred_adjusted: Modified pred tensor

    """
    # Clone pred to avoid modifying the original
    pred_adjusted = pred.clone()
    pred_len = pred.shape[1]
    num_features = pred.shape[2]

    if combined_metrics:
        # Use mean across features for minima detection
        metric_combined = pred_metric.mean(dim=1)

        # Find local minima along axis 0
        local_minima_1d = torch.zeros(pred_len, dtype=torch.bool)
        local_minima_1d[1:-1] = (metric_combined[1:-1] < metric_combined[:-2]) & (
            metric_combined[1:-1] < metric_combined[2:]
        )
        local_minima_1d[0] = metric_combined[0] < metric_combined[1]
        local_minima_1d[-1] = metric_combined[-1] < metric_combined[-2]

        # Get indices of local minima (same for all features)
        minima_indices = torch.where(local_minima_1d)[0]

        if len(minima_indices) == 0:
            return pred_adjusted

        # Process all features with the same minima indices
        for j in range(num_features):
            # Process segments between consecutive minima
            for idx, current_min_idx in enumerate(minima_indices):
                current_min_idx = current_min_idx.item()

                # Determine the end of the segment
                if idx < len(minima_indices) - 1:
                    next_min_idx = minima_indices[idx + 1].item()
                    end_idx = next_min_idx
                else:
                    end_idx = pred_len  # Go to the end

                # For positions after the current minimum up to (but not including) the next
                if current_min_idx + 1 < end_idx:
                    # Get the value at the local minimum
                    base_value = pred[:, current_min_idx, j]
                    # pred_adjusted[:,current_min_idx + 1:end_idx,j] = base_value

                    # Replace values in the segment
                    for k in range(current_min_idx + 1, end_idx):
                        # Cumulative sum of pred_diff from current_min to position k
                        # cumsum_diff = pred_diff[:, current_min_idx:k, j].sum(dim=1)
                        pred_adjusted[:, k, j] = base_value  # + cumsum_diff
    else:
        # Find local minima along axis 0 of pred_metric
        # A point is a local minimum if it's smaller than both neighbors
        local_minima = torch.zeros(pred_len, num_features, dtype=torch.bool)

        # Interior points: compare with both neighbors
        local_minima[1:-1] = (pred_metric[1:-1] < pred_metric[:-2]) & (pred_metric[1:-1] < pred_metric[2:])

        # Boundary conditions (optional - treat boundaries as potential minima)
        local_minima[0] = pred_metric[0] < pred_metric[1]
        local_minima[-1] = pred_metric[-1] < pred_metric[-2]

        # Process each position in the num_features dimension
        for j in range(num_features):
            # Get indices of local minima for this j
            minima_indices = torch.where(local_minima[:, j])[0]

            if len(minima_indices) == 0:
                continue

            # Process segments between consecutive minima
            for idx, current_min_idx in enumerate(minima_indices):
                current_min_idx = current_min_idx.item()

                # Determine the end of the segment
                if idx < len(minima_indices) - 1:
                    next_min_idx = minima_indices[idx + 1].item()
                    end_idx = next_min_idx
                else:
                    end_idx = pred_len  # Go to the end

                # For positions after the current minimum up to (but not including) the next
                if current_min_idx + 1 < end_idx:
                    # Get the value at the local minimum
                    base_value = pred[:, current_min_idx, j]
                    # pred_adjusted[:,current_min_idx + 1:end_idx,j] = base_value

                    # Replace values in the segment
                    for k in range(current_min_idx + 1, end_idx):
                        #    # Cumulative sum of pred_diff from current_min to position k
                        #    cumsum_diff = pred_diff[:, current_min_idx:k, j].sum(dim=1)
                        pred_adjusted[:, k, j] = base_value  # + cumsum_diff

    return pred_adjusted


def combine_pred(pred, pred_diff, pred_metric, combined_metrics=False):
    """Find local minima in pred_metric along axis 0 (forecast horizon length) and adjust pred values.
    This is done by adding the values from pred_diff to the values from pred until we reach another local minima in pred_metric.


    Args:
        pred: torch.Tensor of shape [batch_size, pred_len, num_features]
        pred_diff: torch.Tensor of shape [batch_size, pred_len, num_features]
        pred_metric: torch.Tensor of shape [pred_len, num_features]
        combined_features: bool, if True use mean across features for minima detection

    Returns:
        pred_adjusted: Modified pred tensor

    """
    # Clone pred to avoid modifying the original
    pred_adjusted = pred.clone()
    pred_len = pred.shape[1]
    num_features = pred.shape[2]

    # TODO this seems inefficient with the loop over the features with combined_metrics
    if combined_metrics:
        # Use mean across features for minima detection
        metric_combined = pred_metric.mean(dim=1)

        # Find local minima along axis 0
        local_minima_1d = torch.zeros(pred_len, dtype=torch.bool)
        local_minima_1d[1:-1] = (metric_combined[1:-1] < metric_combined[:-2]) & (
            metric_combined[1:-1] < metric_combined[2:]
        )
        local_minima_1d[0] = metric_combined[0] < metric_combined[1]
        local_minima_1d[-1] = metric_combined[-1] < metric_combined[-2]

        # Get indices of local minima (same for all features)
        minima_indices = torch.where(local_minima_1d)[0]

        if len(minima_indices) == 0:
            return pred_adjusted

        # Process all features with the same minima indices
        for j in range(num_features):
            # Process segments between consecutive minima
            for idx, current_min_idx in enumerate(minima_indices):
                current_min_idx = current_min_idx.item()

                # Determine the end of the segment
                if idx < len(minima_indices) - 1:
                    next_min_idx = minima_indices[idx + 1].item()
                    end_idx = next_min_idx
                else:
                    end_idx = pred_len  # Go to the end

                # For positions after the current minimum up to (but not including) the next
                if current_min_idx + 1 < end_idx:
                    # Get the value at the local minimum
                    base_value = pred[:, current_min_idx, j]

                    # Replace values in the segment
                    for k in range(current_min_idx + 1, end_idx):
                        # Cumulative sum of pred_diff from current_min to position k
                        cumsum_diff = pred_diff[:, current_min_idx:k, j].sum(dim=1)
                        pred_adjusted[:, k, j] = base_value + cumsum_diff
    else:
        # Find local minima along axis 0 of pred_metric
        # A point is a local minimum if it's smaller than both neighbors
        local_minima = torch.zeros(pred_len, num_features, dtype=torch.bool)

        # Interior points: compare with both neighbors
        local_minima[1:-1] = (pred_metric[1:-1] < pred_metric[:-2]) & (pred_metric[1:-1] < pred_metric[2:])

        # Boundary conditions (optional - treat boundaries as potential minima)
        local_minima[0] = pred_metric[0] < pred_metric[1]
        local_minima[-1] = pred_metric[-1] < pred_metric[-2]

        # Process each position in the num_features dimension
        for j in range(num_features):
            # Get indices of local minima for this j
            minima_indices = torch.where(local_minima[:, j])[0]

            if len(minima_indices) == 0:
                continue

            # Process segments between consecutive minima
            for idx, current_min_idx in enumerate(minima_indices):
                current_min_idx = current_min_idx.item()

                # Determine the end of the segment
                if idx < len(minima_indices) - 1:
                    next_min_idx = minima_indices[idx + 1].item()
                    end_idx = next_min_idx
                else:
                    end_idx = pred_len  # Go to the end

                # For positions after the current minimum up to (but not including) the next
                if current_min_idx + 1 < end_idx:
                    # Get the value at the local minimum
                    base_value = pred[:, current_min_idx, j]

                    # Replace values in the segment
                    for k in range(current_min_idx + 1, end_idx):
                        # Cumulative sum of pred_diff from current_min to position k
                        cumsum_diff = pred_diff[:, current_min_idx:k, j].sum(dim=1)
                        pred_adjusted[:, k, j] = base_value + cumsum_diff

    return pred_adjusted


def combine_pred_conditional(pred, pred_diff, pred_metric, pred_diff_metric, combined_metrics=False):
    """Find local minima in pred_metric along axis 0 (forecast horizon length) and adjust pred values.
    This is done by adding the values from pred_diff to the values from pred until the pred_metric value is higher than accumulated pred_diff_metric..

    Args:
        pred: torch.Tensor of shape [batch_size, pred_len, num_features]
        pred_diff: torch.Tensor of shape [batch_size, pred_len, num_features]
        pred_metric: torch.Tensor of shape [pred_len, num_features]
        pred_diff_metric: torch.Tensor of shape [pred_len, num_features] (non-differenced metrics)
        combined_features: bool, if True use mean across features for minima detection

    Returns:
        pred_adjusted: Modified pred tensor

    """
    # Clone pred to avoid modifying the original
    pred_adjusted = pred.clone()
    pred_len = pred.shape[1]
    num_features = pred.shape[2]

    # Calculate differenced metrics from pred_diff_metric
    pred_diff_metric_diff = torch.zeros_like(pred_diff_metric)
    pred_diff_metric_diff[1:] = pred_diff_metric[1:] - pred_diff_metric[:-1]
    pred_diff_metric_diff[0] = pred_diff_metric[0]  # First value stays as is

    if combined_metrics:
        # Use mean across features for minima detection
        metric_combined = pred_metric.mean(dim=1)
        pred_diff_metric_diff_combined = pred_diff_metric_diff.mean(dim=1)

        # Find local minima along axis 0
        local_minima_1d = torch.zeros(pred_len, dtype=torch.bool)
        local_minima_1d[1:-1] = (metric_combined[1:-1] < metric_combined[:-2]) & (
            metric_combined[1:-1] < metric_combined[2:]
        )
        local_minima_1d[0] = metric_combined[0] < metric_combined[1]
        local_minima_1d[-1] = metric_combined[-1] < metric_combined[-2]

        # Get indices of local minima (same for all features)
        minima_indices = torch.where(local_minima_1d)[0]

        if len(minima_indices) == 0:
            return pred_adjusted

        # Process all features with the same minima indices
        for j in range(num_features):
            # Process segments between consecutive minima
            for idx, current_min_idx in enumerate(minima_indices):
                current_min_idx = current_min_idx.item()

                # Determine the end of the segment
                if idx < len(minima_indices) - 1:
                    next_min_idx = minima_indices[idx + 1].item()
                    end_idx = next_min_idx
                else:
                    end_idx = pred_len  # Go to the end

                # For positions after the current minimum up to (but not including) the next
                if current_min_idx + 1 < end_idx:
                    # Get the value at the local minimum
                    base_value = pred[:, current_min_idx, j]

                    # Replace values in the segment conditionally
                    for k in range(current_min_idx + 1, end_idx):
                        # Cumulative sum of differenced pred_diff_metric from current_min_idx to k
                        cumsum_metric_diff = pred_diff_metric_diff_combined[current_min_idx:k].sum()

                        # Only replace if pred_metric at k is higher than base metric + accumulated change
                        # Use the combined metric for comparison
                        if metric_combined[k] > metric_combined[current_min_idx] + cumsum_metric_diff:
                            # Cumulative sum of pred_diff from current_min to position k
                            cumsum_diff = pred_diff[:, current_min_idx:k, j].sum(dim=1)
                            pred_adjusted[:, k, j] = base_value + cumsum_diff
    else:
        # Find local minima along axis 0 of pred_metric
        # A point is a local minimum if it's smaller than both neighbors
        local_minima = torch.zeros(pred_len, num_features, dtype=torch.bool)

        # Interior points: compare with both neighbors
        local_minima[1:-1] = (pred_metric[1:-1] < pred_metric[:-2]) & (pred_metric[1:-1] < pred_metric[2:])

        # Boundary conditions (optional - treat boundaries as potential minima)
        local_minima[0] = pred_metric[0] < pred_metric[1]
        local_minima[-1] = pred_metric[-1] < pred_metric[-2]

        # Process each position in the num_features dimension
        for j in range(num_features):
            # Get indices of local minima for this j
            minima_indices = torch.where(local_minima[:, j])[0]

            if len(minima_indices) == 0:
                continue

            # Process segments between consecutive minima
            for idx, current_min_idx in enumerate(minima_indices):
                current_min_idx = current_min_idx.item()

                # Determine the end of the segment
                if idx < len(minima_indices) - 1:
                    next_min_idx = minima_indices[idx + 1].item()
                    end_idx = next_min_idx
                else:
                    end_idx = pred_len  # Go to the end

                # For positions after the current minimum up to (but not including) the next
                if current_min_idx + 1 < end_idx:
                    # Get the value at the local minimum
                    base_value = pred[:, current_min_idx, j]

                    # Replace values in the segment conditionally
                    for k in range(current_min_idx + 1, end_idx):
                        # Cumulative sum of differenced pred_diff_metric from current_min_idx to k
                        cumsum_metric_diff = pred_diff_metric_diff[current_min_idx:k, j].sum()

                        # Only replace if pred_metric at k is higher than base metric + accumulated change
                        if pred_metric[k, j] > pred_metric[current_min_idx, j] + cumsum_metric_diff:
                            # Cumulative sum of pred_diff from current_min to position k
                            cumsum_diff = pred_diff[:, current_min_idx:k, j].sum(dim=1)
                            pred_adjusted[:, k, j] = base_value + cumsum_diff

    return pred_adjusted


class StepWiseMetricsCallbackWaterlevel(Callback):
    """Callback with 2 tasks
    First initialize all metrics and losses so that they are displayed in the tensorboard hyperparameter tab.
    After fitting the model metrics are calculated for each forecast horizon on the predict, val and test dataloaders
    provided by the used trainers DataModule.
    Each metric is also for the subset of samples that are defined as flood samples.

    METRICS ARE CALCULATED FOR THE FIRST MODEL CHECKPOINT FOUND!

    SHOULD ALWAYS BE THE LAST CALLBACK IN THE CALLBACK LIST. Uses predict, which overwrites some values in the trainer.

    Args:
        chosen_metrics (list, optional): list of metric names and functions.
            Defaults to ['mse', 'mae', 'kge', 'rmse', 'r2', 'nse', 'p10', 'p20','wape','conf50','conf10','conf05'].
        filter_dict (dict, optional): dict with names of filter to apply to the x,true,pred and a function taking these values and returning a mask.
            By default this is 'all', so no filter and rain_20' which checks if the sum of the second last column of x is >= 20, so heavyish rain.

    """

    # pylint: disable-next=dangerous-default-value
    def __init__(self, chosen_metrics=None, filter_dict=None, col_wise=False, filter=True) -> None:

        super().__init__()
        default_metrics = ["mse", "mae", "kge", "rmse", "r2", "nse", "p10", "p20", "wape", "conf50", "conf10", "conf05"]
        # default_metrics = ['mse', 'mae', 'mape','r2']
        chosen_metrics = chosen_metrics or default_metrics
        self.chosen_metrics = mt.get_metric_dict(chosen_metrics)
        # TODO reimplement _flood metrics
        default_filters = {
            "": lambda x, true, pred: torch.ones((x.shape[0]), dtype=bool)
        }  # ,'_rain20' : lambda x,true,pred: x[:,:,-2].sum(axis=1) >= 20}
        self.filter_dict = filter_dict or default_filters
        self.col_wise = col_wise
        self.filter = filter

        self.confidence_levels = [0.95, 0.90, 0.50]
        self.z_scores = {conf: stats.norm.ppf(1 - (1 - conf) / 2) for conf in self.confidence_levels}

    def on_train_start(self, trainer, pl_module) -> None:

        # initialize metrics to log in the hyperparameter tab (necessary for the val_loss, others would still work without this)
        # TODO update syntax when 3.9
        # metric_placeholders = {s: ut.get_objective_metric(s) for s in self.chosen_metrics}
        # metric_placeholders = metric_placeholders | {f'{k}_flood': v for k, v in metric_placeholders.items()}
        # metric_placeholders = [{f'hp/{cur_set}_{k}': v for k, v in metric_placeholders.items()} for cur_set in ['train', 'val', 'test']] + [{'hp/val_loss': 0, 'hp/train_loss': 0}]
        # metric_placeholders = reduce(lambda a, b: {**a, **b}, metric_placeholders)

        # pl_module.logger.log_hyperparams( pl_module.hparams, metric_placeholders)
        pl_module.logger.log_hyperparams(pl_module.hparams)

    def on_fit_end(self, trainer, pl_module) -> None:
        # This is necessary, because the trainer clears the callback_metrics after training
        # Technically i could look before, but logging them at the end is easier
        # metric_placeholder = trainer.callback_metrics.copy()

        train_loader = trainer.datamodule.predict_dataloader()
        val_loader = trainer.datamodule.val_dataloader()
        test_loader = trainer.datamodule.test_dataloader()

        trainer2 = pl.Trainer(
            # default_root_dir=self.log_dir,
            logger=False,
            accelerator=trainer.accelerator,
            devices=trainer.device_ids,
            callbacks=[TQDMProgressBar(refresh_rate=20)],
        )

        if isinstance(pl_module, CustomLightningModule):
            model = CustomLightningModule.from_disk(trainer.log_dir)
        elif isinstance(pl_module, EnsembleModule):
            model = EnsembleModule.from_disk(trainer.log_dir, pl_module.model_list)
        elif isinstance(pl_module, UncertaintyLightningModule):
            model = UncertaintyLightningModule.from_disk(trainer.log_dir)
        else:
            msg = f"model is neither CustomLightningModule nor EnsemleModule, but {type(pl_module)}"
            raise ValueError(msg)
        model.eval()

        metric_dict = {}
        dataloaders = {"train": train_loader, "val": val_loader, "test": test_loader}

        # A quantile model still returns the median as `pred`, so every point
        # metric below is computed exactly as for a point model; the quantiles
        # come along as a 4th value for the interval metrics.
        quantiles = getattr(model, "quantiles", None)
        if quantiles is not None:
            model.set_return_quantiles(True)

        for subset_name, dataloader in dataloaders.items():
            if isinstance(pl_module, UncertaintyLightningModule):
                x, pred, std, true = zip(*trainer2.predict(model=model, dataloaders=dataloader), strict=True)

                std = torch.concat(std).squeeze()
                std = torch.sqrt(torch.cumsum(std**2, dim=1))

            elif quantiles is not None:
                x, pred, true, pred_q = zip(*trainer2.predict(model=model, dataloaders=dataloader), strict=True)
                pred_q = torch.concat(pred_q).squeeze()
            else:
                x, pred, true = zip(*trainer2.predict(model=model, dataloaders=dataloader), strict=True)

            x = torch.concat(x).squeeze()
            pred = torch.concat(pred).squeeze()
            true = torch.concat(true).squeeze()

            for f_name, f_function in self.filter_dict.items():
                mask = f_function(x, true, pred)
                for m_name, m_function in self.chosen_metrics.items():
                    full_m_name = f"{subset_name}_{m_name}{f_name}"
                    metric_dict[full_m_name] = m_function(true[mask], pred[mask])

                if quantiles is not None:
                    metric_dict[f"{subset_name}_pinball{f_name}"] = pinball_series(
                        true[mask], pred_q[mask], quantiles
                    )
                    metric_dict[f"{subset_name}_crossing{f_name}"] = quantile_crossing_series(pred_q[mask])
                    # Symmetric bands, widest first: (q10,q90) for the default levels.
                    for i in range(len(quantiles) // 2):
                        lower, upper = pred_q[mask][..., i], pred_q[mask][..., -(i + 1)]
                        nominal = int(round((quantiles[-(i + 1)] - quantiles[i]) * 100))
                        metric_dict[f"{subset_name}_coverage_{nominal}{f_name}"] = coverage_series(
                            true[mask], lower, upper
                        )
                        metric_dict[f"{subset_name}_interval_{nominal}{f_name}"] = interval_width_series(lower, upper)

                if isinstance(pl_module, UncertaintyLightningModule):
                    for conf_level, z_score in self.z_scores.items():
                        conf_size = z_score * std
                        lower_bound = pred - conf_size
                        upper_bound = pred + conf_size

                        coverage = ((lower_bound <= true) & (true <= upper_bound))[mask].float().mean(dim=0)

                        metric_dict[f"{subset_name}_coverage_{int(conf_level * 100)}{f_name}"] = coverage
                        metric_dict[f"{subset_name}_interval_{int(conf_level * 100)}{f_name}"] = conf_size[mask].mean(
                            axis=0
                        )


        if trainer.datamodule.hparams.get("features") == "M":
            print("Warning, pretty much untested")
            #if isinstance(trainer.datamodule.train_set,Dataset_Diff):
            if getattr(trainer.datamodule.train_set, "_supports_multicol", False):

                cols = trainer.datamodule.train_set.data_x_raw.columns
                if trainer.datamodule.hparams.get("diff"):
                    cols = cols[:len(cols) // 2]


                for i in range(pl_module.hparams.pred_len):
                    tmp_metrics = {f"{k}": v[i].mean().item() for k, v in metric_dict.items()}

                    if self.col_wise:
                        tmp_metrics = tmp_metrics | {f"{k}_{cols[j]}": v[i,j].item() for k, v in metric_dict.items() for j in range(len(cols))}

                    trainer.logger.log_metrics(tmp_metrics,step=i+1)
            else:
                for i in range(pl_module.hparams.pred_len):
                    mean_metrics = {f"{k}": v[i].mean().item() for k, v in metric_dict.items()}
                    trainer.logger.log_metrics(mean_metrics,step=i+1)

        else:

            for i in range(pl_module.hparams.pred_len):
                trainer.logger.log_metrics({k: v[i].item()
                                            for k, v in metric_dict.items()}, step=i+1)


        self.metric_dict = metric_dict


class GemsGerCallback(Callback):
    """Callback with 2 tasks
    First initialize all metrics and losses so that they are displayed in the tensorboard hyperparameter tab.
    After fitting the model metrics are calculated for each forecast horizon on the predict, val and test dataloaders
    provided by the used trainers DataModule.
    Each metric is also for the subset of samples that are defined as flood samples.

    METRICS ARE CALCULATED FOR THE FIRST MODEL CHECKPOINT FOUND!

    SHOULD ALWAYS BE THE LAST CALLBACK IN THE CALLBACK LIST. Uses predict, which overwrites some values in the trainer.

    Args:
        chosen_metrics (list, optional): list of metric names and functions.
            Defaults to ['mse', 'mae', 'kge', 'rmse', 'r2', 'nse', 'p10', 'p20','wape','conf50','conf10','conf05'].
        filter_dict (dict, optional): dict with names of filter to apply to the x,true,pred and a function taking these values and returning a mask.
            By default this is 'all', so no filter and rain_20' which checks if the sum of the second last column of x is >= 20, so heavyish rain.

    """

    # pylint: disable-next=dangerous-default-value
    def __init__(self, chosen_metrics=None) -> None:

        super().__init__()
        default_metrics = ["mae", "rmse", "r2", "nse", "bias"]
        # default_metrics = ['mse', 'mae', 'mape','r2']
        chosen_metrics = chosen_metrics or default_metrics
        self.chosen_metrics = mt.get_metric_dict(chosen_metrics)

    def on_train_start(self, trainer, pl_module) -> None:
        pl_module.logger.log_hyperparams(pl_module.hparams)

    def on_fit_end(self, trainer, pl_module) -> None:
        # This is necessary, because the trainer clears the callback_metrics after training
        # Technically i could look before, but logging them at the end is easier
        # metric_placeholder = trainer.callback_metrics.copy()

        train_loader = trainer.datamodule.predict_dataloader()
        val_loader = trainer.datamodule.val_dataloader()
        test_loader = trainer.datamodule.test_dataloader()

        trainer2 = pl.Trainer(
            # default_root_dir=self.log_dir,
            logger=False,
            accelerator=trainer.accelerator,
            devices=trainer.device_ids,
            callbacks=[TQDMProgressBar(refresh_rate=20)],
        )

        if isinstance(pl_module, CustomLightningModule):
            model = CustomLightningModule.from_disk(trainer.log_dir)
        elif isinstance(pl_module, EnsembleModule):
            model = EnsembleModule.from_disk(trainer.log_dir, pl_module.model_list)
        elif isinstance(pl_module, UncertaintyLightningModule):
            model = UncertaintyLightningModule.from_disk(trainer.log_dir)
        else:
            msg = f"model is neither CustomLightningModule nor EnsemleModule, but {type(pl_module)}"
            raise ValueError(msg)
        model.eval()

        metric_dict = {}
        dataloaders = {"train": train_loader, "val": val_loader, "test": test_loader}

        for subset_name, dataloader in dataloaders.items():
            if isinstance(pl_module, UncertaintyLightningModule):
                x, pred, std, true = zip(*trainer2.predict(model=model, dataloaders=dataloader), strict=True)

                std = torch.concat(std).squeeze()
                std = torch.sqrt(torch.cumsum(std**2, dim=1))
            else:
                _x, pred, true = zip(*trainer2.predict(model=model, dataloaders=dataloader), strict=True)

            # x = torch.concat(x).squeeze()
            pred = torch.concat(pred).squeeze()
            true = torch.concat(true).squeeze()

            if trainer.datamodule.hparams.get("features") == "MS" or trainer.datamodule.hparams.get("features") == "S":
                subset_scaler = StandardScaler()
                subset_scaler.mean_ = trainer.datamodule.scaler.mean_[pl_module.base_idx]
                subset_scaler.scale_ = trainer.datamodule.scaler.scale_[pl_module.base_idx]
            else:
                msg = "features must be MS or S"
                raise ValueError(msg)

            for m_name, m_function in self.chosen_metrics.items():
                metric_dict[f"{subset_name}_{m_name}"] = m_function(true, pred)

                if subset_name == "test":
                    last_cut_off = 0
                    for cut_off, mw_id in zip(dataloader.dataset.cutoffs[1:], dataloader.dataset.mw_list, strict=False):
                        metric_dict[f"{subset_name}_{m_name}_{mw_id}"] = m_function(
                            true[last_cut_off:cut_off], pred[last_cut_off:cut_off]
                        )
                        last_cut_off = cut_off

        print("Saving metrics")
        for i in range(pl_module.hparams.pred_len):
            trainer.logger.log_metrics({k: v[i].item() for k, v in metric_dict.items()}, step=i + 1)

        self.metric_dict = metric_dict


class StepWiseMetricsCallback(Callback):
    """Callback with 2 tasks
    First initialize all metrics and losses so that they are displayed in the tensorboard hyperparameter tab.
    After fitting the model metrics are calculated for each forecast horizon on the predict, val and test dataloaders
    provided by the used trainers DataModule.
    Each metric is also for the subset of samples that are defined as flood samples.

    METRICS ARE CALCULATED FOR THE FIRST MODEL CHECKPOINT FOUND!

    SHOULD ALWAYS BE THE LAST CALLBACK IN THE CALLBACK LIST. Uses predict, which overwrites some values in the trainer.

    Args:
        chosen_metrics (list, optional): list of metric names and functions.
            Defaults to ['mse', 'mae', 'kge', 'rmse', 'r2', 'nse', 'p10', 'p20','wape','conf50','conf10','conf05'].
        filter_dict (dict, optional): dict with names of filter to apply to the x,true,pred and a function taking these values and returning a mask.
            By default this is 'all', so no filter and rain_20' which checks if the sum of the second last column of x is >= 20, so heavyish rain.

    """

    # pylint: disable-next=dangerous-default-value
    def __init__(self, col_wise=False) -> None:

        super().__init__()

        self.chosen_metrics = mt.get_metric_dict(["mse", "mae"])

    def on_train_start(self, trainer, pl_module) -> None:
        pl_module.logger.log_hyperparams(pl_module.hparams)

    def on_fit_end(self, trainer, pl_module) -> None:
        # This is necessary, because the trainer clears the callback_metrics after training
        # Technically i could look before, but logging them at the end is easier
        # metric_placeholder = trainer.callback_metrics.copy()

        train_loader = trainer.datamodule.predict_dataloader()
        val_loader = trainer.datamodule.val_dataloader()
        test_loader = trainer.datamodule.test_dataloader()

        trainer2 = pl.Trainer(
            # default_root_dir=self.log_dir,
            logger=False,
            accelerator=trainer.accelerator,
            devices=trainer.device_ids,
            callbacks=[TQDMProgressBar(refresh_rate=20)],
        )

        model = CustomLightningModule.from_disk(trainer.log_dir)
        model.eval()
        metric_dict = {}
        dataloaders = {"train": train_loader, "val": val_loader, "test": test_loader}

        if trainer.datamodule.hparams.get("features") == "M":
            n_features = len(model.target_idx)
            if pl_module.hparams.diff_comb:
                n_features //= 2
                subset_scaler = StandardScaler()
                subset_scaler.mean_ = trainer.datamodule.scaler.mean_[:n_features]
                subset_scaler.scale_ = trainer.datamodule.scaler.scale_[:n_features]

            elif trainer.datamodule.hparams.get("diff"):
                subset_scaler = StandardScaler()
                subset_scaler.mean_ = trainer.datamodule.scaler.mean_[:n_features]
                subset_scaler.scale_ = trainer.datamodule.scaler.scale_[:n_features]
            else:
                subset_scaler = trainer.datamodule.scaler
        else:
            msg = "features must be M"
            raise ValueError(msg)

        for subset_name, dataloader in dataloaders.items():
            # x,pred,true = zip(*trainer.predict(dataloaders=dataloader))
            x, pred, true = zip(*trainer2.predict(model=model, dataloaders=dataloader), strict=True)

            # for subset_name, subset in outputs.items():
            # x, pred, true = zip(*subset)
            x = torch.concat(x).squeeze()
            pred = torch.concat(pred).squeeze()
            true = torch.concat(true).squeeze()

            if pl_module.hparams.diff_comb:
                pred_direct = pred[:, :, :n_features]
                pred_differenced = pred[:, :, n_features:]
                x_base = x[:, -1, :n_features][:, None]
                pred_undifferenced = x_base + pred_differenced.cumsum(axis=1)
                # true_undifferenced = x_base + true[:,:,n_cols:].cumsum(axis=1)
                true_direct = true[:, :, :n_features]

                true_rescaled = torch.Tensor(
                    subset_scaler.transform(true_direct.reshape(-1, n_features)).reshape(true_direct.shape)
                )

                # TODO if val/test?
                for m_name, m_function in self.chosen_metrics.items():
                    metric_dict[f"{subset_name}_{m_name}"] = m_function(true_direct, pred_direct)
                    metric_dict[f"{subset_name}_{m_name}_diff"] = m_function(true_direct, pred_undifferenced)

                # TODO i reuse the same tensor to save memory, not sure if that actually works.
                pred_temp = torch.Tensor(
                    subset_scaler.transform(pred_direct.reshape(-1, n_features)).reshape(pred_direct.shape)
                )
                for m_name, m_function in self.chosen_metrics.items():
                    metric_dict[f"{subset_name}_{m_name}_rescaled"] = m_function(true_rescaled, pred_temp)
                pred_temp = torch.Tensor(
                    subset_scaler.transform(pred_undifferenced.reshape(-1, n_features)).reshape(
                        pred_undifferenced.shape
                    )
                )
                for m_name, m_function in self.chosen_metrics.items():
                    metric_dict[f"{subset_name}_{m_name}_diff_rescaled"] = m_function(true_rescaled, pred_temp)

                # print("mae",metric_dict["train_mae"].mean())
                # print("mae_diff",metric_dict["train_mae_diff"].mean())
                # metric_dict["train_mae"].mean() -> 67
                # metric_dict["train_mae_diff"].mean() -> 136
                # diff_better = metric_dict["train_mae_diff"] < metric_dict["train_mae"]
                # pred_diff_better = torch.where(diff_better, pred_differenced,pred_direct)
                # print("mae_diff_better",mt.mae_series(true_direct,pred_diff_better).mean())
                # tensor(59.2534)
                # pred_combined_individual =  combine_pred(pred_direct, pred_differenced, metric_dict["train_mae"], combined_metrics=False)

                # mt.mae_series(true_direct,pred_adjusted).mean() 59.5

                # mae_diff_better = mt.mae_series(true_direct,pred_diff_better)
                # mae_diff_better.mean() -> 62
                # pred_adjusted_diff_better = process_local_minima(pred_diff_better, pred_diff, mae_diff_better)
                # mae_diff_better_adjusted = mt.mae_series(true_direct,pred_adjusted_diff_better)
                # mae_diff_better_adjusted.mean() -> 55.45

                if subset_name != "train":
                    if subset_name == "val":
                        # test: _rescaled
                        pred_metric = metric_dict["train_mae_rescaled"]
                        pred_diff_metric = metric_dict["train_mae_diff_rescaled"]
                    elif subset_name == "test":
                        pred_metric = metric_dict["val_mae_rescaled"]
                        pred_diff_metric = metric_dict["val_mae_diff_rescaled"]
                    else:
                        msg = "Only 'val' and 'test' are allowed subset_names"
                        raise ValueError(msg)

                    diff_better = pred_diff_metric < pred_metric
                    pred_diff_better = torch.where(diff_better, pred_undifferenced, pred_direct)

                    # Take the preds from the differenced forecast where they are just better by metric
                    for m_name, m_function in self.chosen_metrics.items():
                        metric_dict[f"{subset_name}_{m_name}_naivecomb"] = m_function(true_direct, pred_diff_better)
                    pred_temp = torch.Tensor(
                        subset_scaler.transform(pred_diff_better.reshape(-1, n_features)).reshape(
                            pred_diff_better.shape
                        )
                    )
                    for m_name, m_function in self.chosen_metrics.items():
                        # metric_dict[f"{subset_name}_{m_name}_naivecomb_diff"] = m_function(true_rescaled,pred_temp) #TODO FALSCH? _rescaled korrekt?
                        metric_dict[f"{subset_name}_{m_name}_naivecomb_rescaled"] = m_function(
                            true_rescaled, pred_temp
                        )  # TODO FALSCH? _rescaled korrekt?

                    # Repeat the value of the local mae minima until the next minima
                    # TODO remove all non rescaled metrics?
                    pred_temp = combine_repeat(pred_direct, pred_metric, combined_metrics=True)
                    for m_name, m_function in self.chosen_metrics.items():
                        metric_dict[f"{subset_name}_{m_name}_combined_repeat"] = m_function(true_direct, pred_temp)
                    pred_temp = torch.Tensor(
                        subset_scaler.transform(pred_temp.reshape(-1, n_features)).reshape(pred_temp.shape)
                    )
                    for m_name, m_function in self.chosen_metrics.items():
                        metric_dict[f"{subset_name}_{m_name}_combined_repeat_rescaled"] = m_function(
                            true_rescaled, pred_temp
                        )
                    pred_temp = combine_repeat(pred_direct, pred_metric, combined_metrics=False)
                    for m_name, m_function in self.chosen_metrics.items():
                        metric_dict[f"{subset_name}_{m_name}_individual_repeat"] = m_function(true_direct, pred_temp)
                    pred_temp = torch.Tensor(
                        subset_scaler.transform(pred_temp.reshape(-1, n_features)).reshape(pred_temp.shape)
                    )
                    for m_name, m_function in self.chosen_metrics.items():
                        metric_dict[f"{subset_name}_{m_name}_individual_repeat_rescaled"] = m_function(
                            true_rescaled, pred_temp
                        )

                    # First take the values where we assume that the differenced prediction is already better or don't
                    for prefer_diff in [True, False]:
                        cur_pred = pred_diff_better if prefer_diff else pred_direct
                        # combine metrics over features or don't
                        for comb_str, combined_metrics in zip(["combined", "individual"], [True, False], strict=False):
                            for consider_metric_change in [True, False]:
                                if consider_metric_change:
                                    pred_temp = combine_pred_conditional(
                                        cur_pred,
                                        pred_differenced,
                                        pred_metric,
                                        pred_diff_metric,
                                        combined_metrics=combined_metrics,
                                    )
                                else:
                                    pred_temp = combine_pred(
                                        cur_pred, pred_differenced, pred_metric, combined_metrics=combined_metrics
                                    )

                                for m_name, m_function in self.chosen_metrics.items():
                                    metric_dict[
                                        f"{subset_name}_{m_name}_{prefer_diff}_{comb_str}_{consider_metric_change}"
                                    ] = m_function(true_direct, pred_temp)

                                pred_temp = torch.Tensor(
                                    subset_scaler.transform(pred_temp.reshape(-1, n_features)).reshape(pred_temp.shape)
                                )

                                for m_name, m_function in self.chosen_metrics.items():
                                    metric_dict[
                                        f"{subset_name}_{m_name}_{prefer_diff}_{comb_str}_{consider_metric_change}_rescaled"
                                    ] = m_function(true_rescaled, pred_temp)

            else:
                for m_name, m_function in self.chosen_metrics.items():
                    full_m_name = f"{subset_name}_{m_name}"
                    metric_dict[full_m_name] = m_function(true, pred)

                pred = torch.Tensor(subset_scaler.transform(pred.reshape(-1, n_features)).reshape(pred.shape))
                true = torch.Tensor(subset_scaler.transform(true.reshape(-1, n_features)).reshape(true.shape))

                for m_name, m_function in self.chosen_metrics.items():
                    full_m_name_rescaled = f"{subset_name}_{m_name}_rescaled"
                    metric_dict[full_m_name_rescaled] = m_function(true, pred)

        # import pandas as pd
        # interesting_cols = ['test_mae_rescaled','test_mae_diff_rescaled','test_mae_True_combined_True_rescaled','test_mae_True_combined_False_rescaled','test_mae_True_individual_True_rescaled','test_mae_True_individual_False_rescaled','test_mae_False_combined_True_rescaled','test_mae_False_combined_False_rescaled','test_mae_False_individual_True_rescaled','test_mae_False_individual_False_rescaled','test_mae_combined_repeat_rescaled','test_mae_individual_repeat_rescaled']
        # ['test_mae','test_mae_diff','test_mae_True_combined_True','test_mae_True_combined_False','test_mae_True_individual_True','test_mae_True_individual_False','test_mae_False_combined_True','test_mae_False_combined_False','test_mae_False_individual_True','test_mae_False_individual_False']
        # df_metrics =pd.DataFrame({col:metric_dict[col].mean(axis=1) for col in interesting_cols})

        # Just logging the calculated values
        for i in range(pl_module.hparams.pred_len):
            mean_metrics = {f"{k}": v[i].mean().item() for k, v in metric_dict.items()}
            trainer.logger.log_metrics(mean_metrics, step=i + 1)

        self.metric_dict = metric_dict
