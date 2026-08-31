"""Forecast monitoring dashboard application.

This module defines the main Dash application for monitoring and visualizing
forecast model performance and system logs.
"""

import logging

import dash_bootstrap_components as dbc
import pandas as pd
from dash import Dash, html

import forecastlib.database.db_tools as dbt
from forecastlib.dashboard.callback_handlers import CallbackHandler
from forecastlib.dashboard.database_service import DatabaseService
from forecastlib.dashboard.layout_components import (
    create_header,
    create_model_monitoring_tab,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Debug flags
DEBUG_USE_STATIC_TIMESTAMP = True
DEBUG = True
DEBUG_TIMESTAMP = pd.to_datetime("2023-05-13 14:00:00.000") if DEBUG_USE_STATIC_TIMESTAMP else None


class ForecastMonitor:
    """Main forecast monitoring dashboard application.

    This class orchestrates the Dash application, database service,
    and callback handlers.
    """

    def __init__(self, username: str | None = None, password: str | None = None, dsn: str | None = None):
        """Initialize ForecastMonitor with database and Dash setup.

        Args:
            username: Optional database username
            password: Optional database password
            dsn: Optional database DSN

        """
        # Initialize database engine
        self.engine = dbt.get_engine()
        self.db_service = DatabaseService(self.engine)

        # Create Dash app
        self.app = Dash(
            __name__,
            suppress_callback_exceptions=True,
            external_stylesheets=[dbc.themes.BOOTSTRAP],
            assets_url_path="dashboard",
        )

        # Setup layout and callbacks
        self._setup_layout()
        self._register_callbacks()

    def _setup_layout(self) -> None:
        """Setup the application layout."""
        header = create_header()

        # Initial tab content
        model_monitoring_tab = create_model_monitoring_tab()

        self.app.layout = html.Div(
            [header, html.Div(id="visible-tab-content", children=[model_monitoring_tab])],
            className="min-vh-100 bg-light",
        )

    def _register_callbacks(self) -> None:
        """Register all application callbacks."""
        handler = CallbackHandler(self.app, self.db_service, debug_timestamp=DEBUG_TIMESTAMP)
        handler.register_callbacks()

    def run(self, host: str = "0.0.0.0", port: int = 8050) -> None:
        """Run the Dash application.

        Args:
            host: Host to bind to
            port: Port to bind to

        """
        self.app.run(host=host, port=port, debug=DEBUG)


if __name__ == "__main__":
    monitor = ForecastMonitor()
    monitor.run()
