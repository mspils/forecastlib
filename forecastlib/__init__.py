from forecastlib.models.LightningWrapper import (
    CustomLightningModule,
    EnsembleModule,
    UncertaintyLightningModule,
    WeightingModel,
)

__all__ = [
    "CustomLightningModule",
    "EnsembleModule",
    "UncertaintyLightningModule",
    "WeightingModel",
]

try:
    from forecastlib.models.WrapperClasses import PastasWrapper, XGBoostWrapper

    __all__ += ["PastasWrapper", "XGBoostWrapper"]
except ImportError:
    pass
