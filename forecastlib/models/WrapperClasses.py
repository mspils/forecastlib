import io
from pathlib import Path

import lightning.pytorch as pl
import pandas as pd
import xgboost as xgb
import yaml
from xgboost.callback import EarlyStopping

try:
    import pastas as ps
except ImportError:
    print("Pastas not available")


from forecastlib.utils.tools import ConfigTracker, load_model_settings


class XGBoostWrapper(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.args = ConfigTracker(args)
        self.columns = self.args.columns
        self.pred_len = self.args.pred_len
        self.target_sensor_name = self.args.target_sensor_name
        self.diff = self.args.diff
        self.num_days = self.args.num_days
        self.num_ensembles = 51
        self.kind = "XGBRegressor"
        self.freq = "D"

        # self.features = self.args.features
        # self.model_id = self.args.model_id
        # self.scaler = args.scaler #This one we don't want to track, i think. Because it's in a pickled represantation in the data_moduls hparams
        # self.Model_class = self.model_dict[self.args.model]

        # self.model = XGBRegressor(**self.args)

        self.model_id = self.args["model_id"] if "model_id" in self.args else None
        self.model = xgb.XGBRegressor(
            device=self.args.device,
            n_estimators=self.args.n_estimators,
            learning_rate=self.args.learning_rate,
            max_depth=self.args.max_depth,
            subsample=self.args.subsample,
            colsample_bytree=self.args.colsample_bytree,
            reg_alpha=self.args.reg_alpha,
            reg_lambda=self.args.reg_lambda,
            callbacks=[EarlyStopping(rounds=self.args.early_stopping_rounds, save_best=True)],
        )
        self.log_dir = self.args.log_dir

        temp_hparams = {k: self.args[k] for k in self.args.accessed_attrs}

        temp_hparams = temp_hparams | {"freq": "D", "kind": "XGBRegressor"}
        if self.model_id is not None:
            temp_hparams["model_id"] = self.model_id
        #    "device": pickle.dumps(args.device)}

        self.save_hyperparameters(temp_hparams, ignore=["scaler", "device"])

    def save(self, log_dir=None):
        if log_dir is None:
            log_dir = self.log_dir

        Path(log_dir).mkdir(exist_ok=True, parents=True)
        self.model.save_model(Path(log_dir) / "xgboost_model.json")

        temp_hparams = {k: self.args[k] for k in self.args.accessed_attrs}
        temp_hparams = temp_hparams | {"freq": "D", "kind": "XGBRegressor"}
        if self.model_id is not None:
            temp_hparams["model_id"] = self.model_id

        with open(Path(self.log_dir) / "hparams.yaml", "w+") as f:
            yaml.dump(temp_hparams, f)

    @classmethod
    def from_disk(cls, model_dir, device=None):
        if isinstance(model_dir, str):
            model_dir = Path(model_dir)
        args = load_model_settings(model_dir, device)
        # checkpoint_path = next((model_dir / "checkpoints").iterdir())
        # model = CustomLightningModule.load_from_checkpoint(checkpoint_path,args=args)
        # model = cls.load_from_checkpoint(checkpoint_path,args=args)

        # yaml_data = yaml.load(yaml_clob, Loader=yaml.FullLoader)

        # model = xgb.XGBRegressor()
        # yaml_data = ConfigTracker(args)

        model = cls(args)
        model.model.load_model(model_dir / "xgboost_model.json")
        if model.model_id is None:
            model.model_id = model_dir.name

        return model

    @classmethod
    def from_db(cls, yaml_clob, model_blob, device=None):
        args = yaml.load(yaml_clob, Loader=yaml.FullLoader)

        # model = xgb.XGBRegressor()
        # yaml_data = ConfigTracker(yaml_data)

        model = cls(args)

        with io.BytesIO(model_blob) as checkpoint_stream:
            model.model.load_model(bytearray(checkpoint_stream.read()))

        return model

    def fit(
        self,
        X,
        y,
        sample_weight=None,
        base_margin=None,
        eval_set=None,
        verbose=50,
        xgb_model=None,
        sample_weight_eval_set=None,
        base_margin_eval_set=None,
        feature_weights=None,
    ):
        args = {
            k: v for k, v in locals().items() if k not in ("self", "__class__")
        }  # Kinda ugly, implemented the same way in the xgboost library
        self.model.fit(**args)

    def predict(
        self, X, *, output_margin: bool = False, validate_features: bool = True, base_margin=None, iteration_range=None
    ):

        return self.model.predict(X, output_margin, validate_features, base_margin, iteration_range)



class PastasWrapper(pl.LightningModule):
    """Wrapper for Pastas groundwater models.

    Pastas models use stress time series (precipitation, evaporation, etc.) for predictions.
    The predict method needs to accept these stress series, which may differ from those
    used during training. This is handled via the stress_data parameter.
    """

    def __init__(self, args):
        super().__init__()
        self.args = ConfigTracker(args)

        stress_classes = {
            "recharge": ps.RechargeModel,
            "tarso": ps.TarsoModel,
            "stress": ps.StressModel,
        }
        recharge_classes = {
            "flex": ps.rch.FlexModel,
            "berendrecht": ps.rch.Berendrecht,
            "linear": ps.rch.Linear,
            "peterson": ps.rch.Peterson,
        }
        rfunc_dict = {
            "exponential": ps.Exponential,
            "spline": ps.Spline,
            "gamma": ps.Gamma,
            "doubleexponential": ps.DoubleExponential,
        }
        if self.args.include_well:
            print("using wells is not tested/implemented yet")
        self.target_sensor_name = self.args.target_sensor_name
        self.evap_name = self.args.evap_name
        #self.temp_name = self.args.temp_name
        self.prec_name = self.args.prec_name

        #TODO well name?

        # Model metadata
        self.model_id = self.args["model_id"] if "model_id" in self.args else None
        self.kind = "PastasModel"
        self.freq = self.args["freq"] if "freq" in self.args else "D"
        self.log_dir = self.args.log_dir if "log_dir" in self.args else None

        # Store configuration for later reconstruction
        self.stress_model_str = self.args["stress_model"] if "stress_model" in self.args else "recharge"
        self.stress_model_class = stress_classes[self.stress_model_str]

        self.recharge_str = self.args["recharge_class"] if "recharge_class" in self.args else "linear"
        self.recharge_class = recharge_classes[self.recharge_str]

        self.rfunc_str = self.args["rfunc_class"] if "rfunc_class" in self.args else "exponential"
        self.rfunc_class = rfunc_dict[self.rfunc_str]
        self.rfunc = self.rfunc_class()

        self.warmup = self.args["warmup"] if "warmup" in self.args else 365

        self.constant = self.args.stress_model != "tarso"

        try:
            self.model = self.args.model
        except (KeyError, AttributeError):
            prec_temp = self.args.prec if self.recharge_class == ps.rch.Linear else self.args.prec * 1000

            self.model = ps.Model(self.args.obs_train, name=self.args.obs_train.name, constant=self.constant)

            if self.args.noise:
                self.model.add_noisemodel(ps.ArNoiseModel())

            if self.stress_model_str == "recharge":
                if self.recharge_class == ps.rch.FlexModel:
                    rch = self.recharge_class(gw_uptake=True)
                else:
                    rch = self.recharge_class()

                # recharge_model = ps.RechargeModel(prec, evap,recharge=rch,rfunc=rfunc, name="recharge")
                # Temperature literally never matters temp=temp_temp

                recharge_model = ps.RechargeModel(
                    prec=prec_temp, evap=self.args.evap, recharge=rch, rfunc=self.rfunc, name="recharge"
                )
                self.model.add_stressmodel(recharge_model)

            elif self.stress_model_str == "tarso":
                tarso_model = ps.TarsoModel(
                    self.args.prec,
                    self.args.evap,
                    dmin=self.model.oseries.series.min(),
                    dmax=self.model.oseries.series.max(),
                )
                self.model.add_stressmodel(tarso_model)
            elif self.stress_model_str == "stress":
                sm1 = ps.StressModel(self.args.evap, rfunc=self.rfunc, name="evap", settings="evap", up=False)
                sm2 = ps.StressModel(prec_temp, rfunc=self.rfunc, name="prec", settings="prec", up=True)
                self.model.add_stressmodel(sm1)
                self.model.add_stressmodel(sm2)

            if self.args.include_well and "well" in self.args:
                well_model = ps.StressModel(
                    self.args.well / 1000, rfunc=ps.Hantush(), name="well", settings="well", up=False
                )
                self.model.add_stressmodel(well_model)

        # Save hyperparameters
        temp_hparams = {k: self.args[k] for k in self.args.accessed_attrs if k != "log_dir"}
        temp_hparams = temp_hparams | {
            "kind": "PastasModel",
            "freq": self.freq,
            "warmup": self.warmup,
        }
        if self.model_id is not None:
            temp_hparams["model_id"] = self.model_id
        self.save_hyperparameters(
            temp_hparams, ignore=["log_dir", "obs_train", "stress_model_class", "rfunc_class", "recharge_class"]
        )

    def save(self, log_dir=None):
        """Save the pastas model and its hyperparameters."""
        if log_dir is None:
            log_dir = self.log_dir

        if log_dir is None:
            raise ValueError("log_dir must be provided either in args or as parameter")

        Path(log_dir).mkdir(exist_ok=True, parents=True)

        # Save pastas model
        if self.model is not None:
            self.model.to_file(Path(log_dir) / "pastas_model.pas")

        # Save hyperparameters
        temp_hparams = {
            k: self.args[k] for k in self.args.accessed_attrs if k not in ["log_dir", "obs_train", "prec", "evap"]
        }
        temp_hparams = temp_hparams | {
            "kind": "PastasModel",
            "freq": self.freq,
            "warmup": self.warmup,
        }
        if self.model_id is not None:
            temp_hparams["model_id"] = self.model_id

        with open(Path(log_dir) / "hparams.yaml", "w+") as f:
            yaml.dump(temp_hparams, f)

    @classmethod
    def from_disk(cls, model_dir, device=None):
        """Load a PastasWrapper model from disk.

        Args:
            model_dir: Path to directory containing 'pastas_model.pas' and 'hparams.yaml'
            device: Device parameter (for API compatibility, not used by pastas)

        Returns:
            PastasWrapper instance with loaded model

        """
        if isinstance(model_dir, str):
            model_dir = Path(model_dir)

        args = load_model_settings(model_dir, device)

        # Load the pastas model
        pastas_model_path = model_dir / "pastas_model.pas"
        if pastas_model_path.exists():
            model = ps.io.load(pastas_model_path)
        args.model = model

        wrapper = cls(args)
        if wrapper.model_id is None:
            wrapper.model_id = model_dir.name

        return wrapper

    @classmethod
    def from_db(cls, yaml_clob, model_blob, device=None):
        """Load a PastasWrapper model from forecastlib.database.

        Args:
            yaml_clob: YAML configuration as string/bytes
            model_blob: Serialized pastas model
            device: Device parameter (for API compatibility, not used by pastas)

        Returns:
            PastasWrapper instance with loaded model

        """
        if isinstance(yaml_clob, bytes):
            yaml_clob = yaml_clob.decode("utf-8")

        # Decode YAML if bytes
        if isinstance(yaml_clob, bytes):
            yaml_clob = yaml_clob.decode("utf-8")

        args = yaml.load(yaml_clob, Loader=yaml.FullLoader)

        # Deserialize and load pastas model from blob into a model object
        model = None
        with io.BytesIO(model_blob) as blob_stream:
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".pas", delete=False) as tmp:
                tmp.write(blob_stream.read())
                tmp_path = tmp.name

            try:
                model = ps.io.load(tmp_path)
            finally:
                Path(tmp_path).unlink()

        # Attach model into args so __init__ constructs wrapper with existing model
        try:
            args["model"] = model
        except Exception:
            # If args is not a dict-like, fall back to attribute assignment after wrapping
            pass

        wrapper = cls(args)
        return wrapper

    def fit(self, stress_data=None, tmin=None, tmax=None, warmup=None, report=False, **kwargs):
        """Fit a pastas model.

        Args:
            stress_data: Nested dict, one per stress model (optional).
                        {stress_model_name: {attribute_name: pandas.Series, ...}, ...}
                        Example: {'recharge': {'prec': prec_series, 'evap': evap_series}}
            tmin: Start time for solving (optional)
            tmax: End time for solving (optional)
            warmup: Number of days for model warmup (default: 365)
            report: Whether to print fitting report (default: False)
            **kwargs: Additional arguments passed to model.solve()

        Returns:
            self for method chaining

        """
        if self.model is None:
            raise ValueError("Model must be initialized before fitting. Use from_disk() or set model externally.")

        if warmup is None:
            warmup = self.warmup

        # Update stress data in model if provided
        if stress_data is not None:
            self._update_stress_data(stress_data)

        # Solve the model
        solve_kwargs = {"warmup": warmup, "report": report}
        if tmin is not None:
            solve_kwargs["tmin"] = tmin
        if tmax is not None:
            solve_kwargs["tmax"] = tmax
        solve_kwargs.update(kwargs)

        self.model.solve(**solve_kwargs)
        return self

    def predict(self, tmin=None, tmax=None, warmup=None, stress_data=None, return_components=False):
        """Make predictions with the pastas model.

        IMPORTANT: Pastas models need the full stress series (precipitation, evaporation, etc.)
        to make predictions. By default, the model uses the stress data it was initialized with.
        If you need different stress data (e.g., for scenario analysis), pass it via stress_data.

        Args:
            tmax: End time for simulation. If None, uses the last date in stress series.
            stress_data: Nested dict mapping stress model names to their timeseries data.
                        Structure: {stress_model_name: {attribute_name: pandas.Series, ...}, ...}

        Example:
                            {
                                'recharge': {
                                    'prec': new_precip_series,
                                    'evap': new_evap_series
                                },
                                'well': {
                                    'stress': new_well_series
                                }
                            }
                        This is useful for predictions beyond the training period or scenario analysis.
            return_components: If True, returns dict with simulation and contributions from each stress

        Returns:
            pandas Series with predictions (times values), or dict if return_components=True

        """
        if self.model is None:
            raise ValueError("Model must be fitted before prediction.")

        # Save original stress data if we need to override it
        original_stress_data = None
        if stress_data is not None:
            original_stress_data = self._store_original_stress_data()
            self._update_stress_data(stress_data)

        try:
            # Make simulation
            simulation = self.model.simulate(tmin=tmin, tmax=tmax, warmup=warmup)

            if return_components:
                # Get contributions from each stress model
                components = {}
                for sm_name in self.model.stressmodels:
                    try:
                        contribution = self.model.get_contribution(sm_name, tmax=tmax)
                        components[sm_name] = contribution
                    except Exception as e:
                        print(f"Could not get contribution for {sm_name}: {e}")

                return {"simulation": simulation, "components": components}
            return simulation
        finally:
            # Restore original stress data if we modified it
            if original_stress_data is not None:
                self._restore_stress_data(original_stress_data)

    def _update_stress_data(self, stress_data):
        """Update the stress data in the model.

        Args:
            stress_data: Nested dict, one per stress model.
                        {stress_model_name: {attribute_name: pandas.Series, ...}, ...}
                        Example: {'recharge': {'prec': prec_series, 'evap': evap_series}}

        """
        #well_model = ps.StressModel(well / 1000, rfunc=ps.Hantush(), name="well", settings="well", up=False)
        #ml.add_stressmodel(well_model)

        for stress_model_name, attributes_dict in stress_data.items():
            sm = self.model.stressmodels[stress_model_name]
            if stress_model_name == "recharge":
                sm.prec = ps.timeseries.TimeSeries(
                    attributes_dict["prec"].combine_first(sm.prec.series)
                )
                sm.evap = ps.timeseries.TimeSeries(
                    attributes_dict["evap"].combine_first(sm.evap.series)
                )

                if sm.temp is not None and "temp" in attributes_dict:
                    print("warning: temperature is not really implemented?")
                    sm.temp = ps.timeseries.TimeSeries(
                        attributes_dict["temp"].combine_first(sm.temp.series)
                    )
            elif stress_model_name == "well":
                sm.well = ps.timeseries.TimeSeries(
                    attributes_dict["well"].combine_first(sm.well.series)
                )
            else:
                raise NotImplementedError


    def update_oseries(self, new_observations: pd.Series) -> None:
        """Extend the model's oseries with more recent observations.

        New values overwrite existing ones where indices overlap; the original
        series fills any remaining gaps.

        Args:
            new_observations: Series of observed groundwater levels indexed by
                datetime, e.g. fetched from SensorManager since the last
                training date.

        """
        merged = new_observations.combine_first(self.model.oseries.series)
        self.model.oseries = ps.timeseries.TimeSeries(merged, name=self.model.oseries.name)

    def _store_original_stress_data(self):
        """Store a copy of current stress data for restoration."""
        original = {}
        for sm_name, sm in self.model.stressmodels.items():
            original[sm_name] = {}
            if hasattr(sm, "prec"):
                original[sm_name]["prec"] = sm.prec.series.copy()
            if hasattr(sm, "evap"):
                original[sm_name]["evap"] = sm.evap.series.copy()
            if hasattr(sm, "temp") and sm.temp is not None:
                original[sm_name]["temp"] = sm.temp.series.copy()
            if hasattr(sm, "well") and sm.well is not None:
                original[sm_name]["well"] = sm.well.series.copy()
        return original

    def _restore_stress_data(self, original_data):
        """Restore stress data from stored copy."""
        for stress_model_name, attributes_dict in original_data.items():
            sm = self.model.stressmodels[stress_model_name]

            if "prec" in attributes_dict:
                sm.prec = ps.timeseries.TimeSeries(attributes_dict["prec"])
            if "evap" in attributes_dict:
                sm.evap = ps.timeseries.TimeSeries(attributes_dict["evap"])
            if "temp" in attributes_dict:
                sm.temp = ps.timeseries.TimeSeries(attributes_dict["temp"])
            if "well" in attributes_dict:
                sm.well = ps.timeseries.TimeSeries(attributes_dict["well"])

            # Reprocess the stress data after restoration
            sm.update_stress(freq=self.model.settings["freq"])

    def get_model_config(self):
        """Get the model configuration as a dictionary."""
        return {
            "kind": self.kind,
            "freq": self.freq,
            #'recharge_class': self.recharge_class,
            #'rfunc_class': self.rfunc_class,
            #'noise_model': self.noise_model,
            "warmup": self.warmup,
        }

    def get_recommended_warmup(self, cutoff: float = 0.999) -> float:
        """Return the minimum recommended warmup period in days.

        Iterates over all stress models that have a response function and
        returns the maximum tmax across them, which is the time at which
        the impulse response has decayed to ``cutoff`` of its total effect.

        Args:
            cutoff: Fraction of the total response to capture (default 0.999).

        Returns:
            Warmup period in days as a float. Falls back to ``self.warmup``
            if no stress model exposes an rfunc.

        """
        if self.model is None:
            raise ValueError("Model must be fitted before querying warmup period.")

        tmax_values = []
        for sm_name, sm in self.model.stressmodels.items():
            if not hasattr(sm, "rfunc"):
                continue
            try:
                params = self.model.get_parameters(sm_name)
                tmax_values.append(sm.rfunc.get_tmax(params, cutoff=cutoff))
            except Exception:
                pass

        return max(tmax_values) if tmax_values else self.warmup
