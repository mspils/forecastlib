"""Callback handlers for the forecast monitoring dashboard.

This module defines all Dash callbacks for the forecast monitoring application.
"""

from typing import Any

import dash_bootstrap_components as dbc
from dash import dcc, html
from dash.dependencies import MATCH, Input, Output, State

from forecastlib.dashboard.database_service import DatabaseService
from forecastlib.dashboard.layout_components import create_status_table
from forecastlib.dashboard.tables_and_plots import (
    create_historical_plot,
    create_historical_table,
    create_inp_forecast_status_table,
    create_log_table,
    create_metrics_plots,
    create_model_forecasts_plot,
    create_status_summary_chart,
)


class CallbackHandler:
    """Manages all dashboard callbacks."""

    def __init__(self, app: Any, db_service: DatabaseService, debug_timestamp=None) -> None:
        """Initialize the callback handler.

        Args:
            app: Dash application instance
            db_service: DatabaseService instance
            debug_timestamp: Optional debug timestamp for testing

        """
        self.app = app
        self.db_service = db_service
        self.debug_timestamp = debug_timestamp

    def register_callbacks(self) -> None:
        """Register all dashboard callbacks."""
        self._register_tab_callbacks()
        self._register_status_callbacks()
        self._register_detail_callbacks()
        self._register_system_log_callbacks()
        self._register_collapse_callbacks()

    def _register_tab_callbacks(self) -> None:
        """Register tab switching callbacks."""

        @self.app.callback(Output("visible-tab-content", "children"), Input("main-tabs", "active_tab"))
        def switch_tab(active_tab: str) -> Any:
            """Switch between main tabs."""
            from forecastlib.dashboard.layout_components import (
                create_model_metrics_tab,
                create_model_monitoring_tab,
                create_system_logs_tab,
            )

            if active_tab == "model-monitoring":
                return create_model_monitoring_tab()
            if active_tab == "system-logs":
                return create_system_logs_tab()
            if active_tab == "model-metrics":
                return create_model_metrics_tab()
            return html.P("No tab selected")

    def _register_status_callbacks(self) -> None:
        """Register model status update callbacks."""

        @self.app.callback(
            [Output("model-status-table", "children"), Output("status-summary-chart", "figure")],
            Input("status-update", "n_intervals"),
        )
        def update_dashboard(n: int) -> tuple[Any, Any]:
            """Update the status table and summary chart."""
            status = self.db_service.get_active_models_status(self.debug_timestamp)
            if not status:
                return dbc.Alert("Error fetching data", color="danger"), {}

            table_content = create_status_table(status["model_status"], status["last_check_time"])
            fig = create_status_summary_chart(status)

            return table_content, fig

    def _register_detail_callbacks(self) -> None:
        """Register callbacks for detailed model information."""

        @self.app.callback(
            [
                Output("fcst-view", "children"),
                Output("historical-view", "children"),
                Output("log-view", "children"),
                Output("metrics-view", "children"),
                Output("current-sensor-names", "data"),
            ],
            [Input("status-table", "selected_rows")],
            [State("status-table", "data")],
        )
        def update_right_column(
            selected_rows: list[int], table_data: list[dict]
        ) -> tuple[Any, Any, Any, Any, list[str]]:
            """Update the right column with details for the selected model."""
            if not selected_rows:
                return (
                    html.Div("Wähle ein Modell um Vorhersagen anzuzeigen."),
                    html.Div("Wähle ein Modell um Messwerte anzuzeigen."),
                    html.Div("Wähle ein Modell um Logs anzuzeigen."),
                    html.Div("Wähle ein Modell um Metriken anzuzeigen."),
                    [],
                )

            selected_row = table_data[selected_rows[0]]
            sensor_name = selected_row["sensor_name"]
            model_id = selected_row["model_id"]
            model_name = selected_row["model_name"]
            freq = selected_row["freq"]

            # Collect all necessary data
            logs = self.db_service.get_recent_logs(sensor_name)
            df_historical = self.db_service.get_historical_data(model_id, self.debug_timestamp)
            sensor_names = list(df_historical.columns) if not df_historical.empty else []
            df_inp_fcst, ext_forecast_names = self.db_service.get_input_forecasts(sensor_names, self.debug_timestamp)
            df_forecasts = self.db_service.get_recent_forecasts(model_id, freq, self.debug_timestamp)
            df_conf = self.db_service.get_confidence(model_id, flood=False, subset="test")

            # Create log view
            log_table = create_log_table(logs)
            log_view = html.Div([html.H4(f"Neueste Logs für {model_name}"), log_table])

            # Create historical view
            if not df_historical.empty:
                fig = create_historical_plot(df_historical, model_name)
                df_filtered = df_historical[df_historical.isna().any(axis=1)]
                historical_table = create_historical_table(df_filtered)
                historical_view = html.Div(
                    [
                        dcc.Graph(figure=fig),
                        html.H4(
                            "Zeitpunkte mit fehlenden Messdaten",
                            style={"marginTop": "20px", "marginBottom": "10px"},
                        ),
                        html.Div(historical_table, style={"width": "100%", "padding": "10px"}),
                    ]
                )
            else:
                historical_view = html.Div("Keine Messdaten verfügbar")

            # Create forecast view
            if not df_forecasts.empty:
                inp_fcst_table = create_inp_forecast_status_table(df_inp_fcst, ext_forecast_names)
                fig_fcst = create_model_forecasts_plot(df_forecasts, df_historical, df_inp_fcst, df_conf, sensor_name)
                fcst_view = html.Div(
                    [
                        html.H4(f"Pegelvorhersage Modell {model_name}"),
                        dcc.Graph(figure=fig_fcst),
                        html.H4(
                            "Status Eingangsvorhersagen",
                            style={"marginTop": "20px", "marginBottom": "10px"},
                        ),
                        html.Div(inp_fcst_table, style={"width": "100%", "padding": "10px"}),
                    ]
                )
            else:
                fcst_view = html.Div("No forecasts available")

            # Create metrics view
            metrics_view = html.Div(
                [
                    html.H4(f"Modell Metriken für {model_name}"),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.Label("Metric:", className="fw-bold mb-2"),
                                    dcc.Dropdown(
                                        id={
                                            "type": "metric-selector",
                                            "section": "metrics",
                                        },
                                        options=[
                                            {"label": metric.upper(), "value": metric}
                                            for metric in [
                                                "mae",
                                                "mse",
                                                "nse",
                                                "kge",
                                                "p10",
                                                "p20",
                                                "r2",
                                                "rmse",
                                                "wape",
                                                "conf05",
                                                "conf10",
                                                "conf50",
                                            ]
                                        ],
                                        value="mae",
                                        clearable=False,
                                        className="mb-4",
                                    ),
                                ],
                                width=4,
                            )
                        ]
                    ),
                    html.Div(id="metric-plots-container"),
                    html.Div(id="metric-info-container", className="mt-4"),
                ]
            )

            return fcst_view, historical_view, log_view, metrics_view, sensor_names

        @self.app.callback(
            [Output("metric-plots-container", "children"), Output("metric-info-container", "children")],
            [
                Input({"type": "metric-selector", "section": "metrics"}, "value"),
                Input("status-table", "selected_rows"),
            ],
            [State("status-table", "data")],
        )
        def update_metric_plots(metric_name: str, selected_rows: list[int], table_data: list[dict]) -> tuple[Any, Any]:
            """Update metric plots based on selected metric."""
            if not selected_rows or not metric_name:
                return html.Div(), html.Div()

            selected_row = table_data[selected_rows[0]]
            model_id = selected_row["model_id"]

            metrics_data, _edge_times = self.db_service.get_model_metrics(model_id, metric_name)
            if metrics_data is None or metrics_data.empty:
                return html.Div("Keine Metriken verfügbar"), html.Div()

            fig = create_metrics_plots(metrics_data, metric_name)
            plots = html.Div([dcc.Graph(figure=fig, className="mb-4")])

            info = html.Div(
                [
                    html.H5("Zeitraum der Metrikberechnung:", className="h6 mb-3"),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.Strong("Start Trainingszeitraum: "),
                                ],
                                width=6,
                            ),
                            dbc.Col(
                                [
                                    html.Strong("Start Validierungszeitraum: "),
                                ],
                                width=6,
                            ),
                            dbc.Col(
                                [
                                    html.Strong("Start Testzeitraum: "),
                                ],
                                width=6,
                            ),
                            dbc.Col(
                                [
                                    html.Strong("Ende Testzeitraum: "),
                                ],
                                width=6,
                            ),
                        ]
                    ),
                ],
                className="bg-light p-3 rounded",
            )

            return plots, info

    def _register_system_log_callbacks(self) -> None:
        """Register system log update callbacks."""

        @self.app.callback(Output("system-log-table", "data"), Input("logs-update", "n_intervals"))
        def update_system_logs(n: int) -> list[dict]:
            """Update the system log table."""
            return self.db_service.get_all_logs(limit=1000)

    def _register_collapse_callbacks(self) -> None:
        """Register collapse toggle callbacks."""

        @self.app.callback(
            [
                Output({"type": "collapse-content", "section": MATCH}, "is_open"),
                Output({"type": "collapse-button", "section": MATCH}, "children"),
            ],
            [Input({"type": "collapse-button", "section": MATCH}, "n_clicks")],
            [
                State({"type": "collapse-content", "section": MATCH}, "is_open"),
                State({"type": "collapse-button", "section": MATCH}, "children"),
            ],
        )
        def toggle_collapse(n_clicks: int, is_open: bool, current_children: list[Any]) -> tuple[bool, list[Any]]:
            """Toggle collapse state for sections."""
            if n_clicks:
                title = current_children[0]["props"]["children"]
                if is_open:
                    return False, [
                        html.Span(title, style={"flex": "1"}),
                        html.Span("►", className="ms-2"),
                    ]
                return True, [
                    html.Span(title, style={"flex": "1"}),
                    html.Span("▼", className="ms-2"),
                ]
            return is_open, current_children
