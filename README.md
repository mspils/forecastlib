# forecastlib

A [PyTorch Lightning](https://lightning.ai/)–based library of time series forecasting
models, data loaders and utilities. Adapted and reorganized from
[thuml/Time-Series-Library](https://github.com/thuml/Time-Series-Library): the models are
more or less the same, wrapped for Lightning with better logging, plus additions for
differencing, database-backed model/forecast storage and a Dash dashboard.

## Installation

Using [uv](https://docs.astral.sh/uv/):

```bash
uv add "forecastlib @ git+https://github.com/mspils/forecastlib"
```

or with pip:

```bash
pip install "forecastlib @ git+https://github.com/mspils/forecastlib"
```

### Optional extras

| Extra       | Enables                                                        |
|-------------|---------------------------------------------------------------|
| `database`  | `forecastlib.database`, `forecastlib.dashboard`, `forecastlib.app` (Oracle/SQLite logging + Dash dashboard) |
| `wrappers`  | `forecastlib.models.WrapperClasses` (Pastas + XGBoost wrappers) |
| `mamba`     | The `Mamba` model (needs a CUDA build of `mamba-ssm`; `MambaSimple` works without it) |

```bash
uv add "forecastlib[database,wrappers] @ git+https://github.com/mspils/forecastlib"
```

## Usage

```python
import forecastlib
from forecastlib import CustomLightningModule

# CustomLightningModule, EnsembleModule, UncertaintyLightningModule, WeightingModel
# are exported at the top level.
```

When using the `database` extra, configuration is read from a `.env` file — see
[.env.example](.env.example).

## Development

```bash
uv sync --extra dev
uv run poe lint
uv run poe test
```

## License

MIT — see [LICENSE](LICENSE).
