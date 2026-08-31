from datetime import timedelta

NUM_RECENT_FORECASTS = 5
NUM_RECENT_LOGS = 10
BUFFER_TIME = timedelta(hours=2, minutes=30)
PERIODS_EXT_FORECAST_TABLE = 10
FREQUENCY = "h"

# Wie viele Tage zurück in die Vergangenheit schauen, um zu prüfen, ob ein Eingangswert in das Modelll externe Vorhersagen hat.
# Ginge auch ohne, das wäre dann aber deutlich ineffizienter. Falls die externen Vorhersagen zuverlässig kommen lässt sich das auch auf wenige Tage reduzieren.
LOOKBACK_EXTERNAL_FORECAST_EXISTENCE = 100
DEBUG_USE_STATIC_TIMESTAMP = True
