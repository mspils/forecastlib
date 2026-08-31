"""Layout components for the forecast monitoring dashboard.

This module provides functions to build various sections of the dashboard UI.
"""

from datetime import datetime

import dash_bootstrap_components as dbc
from dash import dash_table, dcc, html

from forecastlib.dashboard.styles import TAB_STYLE, TABLE_STYLE


def create_header() -> html.Div:
    """Create the main header bar of the dashboard.

    Returns:
        HTML Div containing the header

    """
    return html.Div(
        [
            dbc.Navbar(
                [
                    dbc.Container(
                        [
                            dbc.Row(
                                [
                                    dbc.Col(
                                        html.H1(
                                            "Pegel Dashboard",
                                            className="mb-0 text-white d-flex align-items-center",
                                        ),
                                        width="auto",
                                    ),
                                    dbc.Col(
                                        dbc.Tabs(
                                            [
                                                dbc.Tab(
                                                    label="Modell Monitoring",
                                                    tab_id="model-monitoring",
                                                    **TAB_STYLE,
                                                ),
                                                dbc.Tab(
                                                    label="System Logs",
                                                    tab_id="system-logs",
                                                    **TAB_STYLE,
                                                ),
                                                dbc.Tab(
                                                    label="Modell  Metrics",
                                                    tab_id="model-metrics",
                                                    **TAB_STYLE,
                                                ),
                                            ],
                                            id="main-tabs",
                                            active_tab="model-monitoring",
                                            className="nav-tabs border-0",
                                        ),
                                        width="auto",
                                        className="d-flex align-items-end",
                                    ),
                                ],
                            ),
                        ],
                        fluid=True,
                        className="px-4",
                    ),
                ],
                color="primary",
                dark=True,
            )
        ],
        className="mb-4",
    )


def create_collapsible_section(title: str, section_id: str, view_id: str, is_open: bool = True) -> dbc.Card:
    """Create a collapsible card section.

    Args:
        title: Title of the section
        section_id: ID for the section
        view_id: ID for the content div
        is_open: Whether the section starts open

    Returns:
        Bootstrap Card with collapsible content

    """
    return dbc.Card(
        [
            dbc.CardHeader(
                [
                    dbc.Button(
                        children=[
                            html.Span(title, style={"flex": "1"}),
                            html.Span("▼" if is_open else "►", className="ms-2"),
                        ],
                        color="link",
                        id={"type": "collapse-button", "section": section_id},
                        className="text-decoration-none text-dark h5 mb-0 w-100 d-flex align-items-center",
                    )
                ]
            ),
            dbc.Collapse(
                dbc.CardBody(html.Div(id=view_id)),
                id={"type": "collapse-content", "section": section_id},
                is_open=is_open,
            ),
        ],
        className="mb-4",
    )


def create_model_monitoring_tab() -> dbc.Container:
    """Create the main model monitoring tab layout.

    Returns:
        Bootstrap Container with the model monitoring layout

    """
    return dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(html.H3("Modell Status", className="h5 mb-0")),
                                    dbc.CardBody(
                                        [
                                            html.Div(id="model-status-table"),
                                            dcc.Interval(
                                                id="status-update",
                                                interval=600000,  # 10 minutes
                                            ),
                                        ]
                                    ),
                                ],
                                className="mb-4",
                            ),
                            dbc.Card(
                                [
                                    dbc.CardHeader(html.H3("Status Übersicht", className="h5 mb-0")),
                                    dbc.CardBody([dcc.Graph(id="status-summary-chart")]),
                                ]
                            ),
                        ],
                        xs=12,
                        md=5,
                        className="mb-4 mb-md-0",
                    ),
                    dbc.Col(
                        [
                            dcc.Store(id="current-sensor-names"),
                            create_collapsible_section("Vorhersagen", "forecasts", "fcst-view"),
                            create_collapsible_section("Messwerte", "historical", "historical-view"),
                            create_collapsible_section("Logs", "logs", "log-view"),
                            create_collapsible_section("Metriken", "metrics", "metrics-view"),
                        ],
                        xs=12,
                        md=7,
                    ),
                ]
            )
        ],
        fluid=True,
        className="py-4",
    )


def create_system_logs_tab() -> dbc.Container:
    """Create the system logs tab layout.

    Returns:
        Bootstrap Container with the system logs layout

    """
    return dbc.Container(
        [
            dbc.Card(
                [
                    dbc.CardBody(
                        [
                            html.H3("System Logs", className="mb-4"),
                            dcc.Interval(
                                id="logs-update",
                                interval=30000,  # 30 seconds
                            ),
                            dash_table.DataTable(
                                id="system-log-table",
                                columns=[
                                    {"name": "Zeitstempel", "id": "timestamp"},
                                    {"name": "Level", "id": "level"},
                                    {"name": "Pegel", "id": "sensor"},
                                    {"name": "Nachricht", "id": "message"},
                                    {"name": "Modul", "id": "module"},
                                    {"name": "Funktion", "id": "function"},
                                    {"name": "Zeile", "id": "line"},
                                    {"name": "Exception", "id": "exception"},
                                ],
                                **TABLE_STYLE,
                                page_size=20,
                                page_action="native",
                                sort_action="native",
                                sort_mode="multi",
                                filter_action="native",
                            ),
                        ]
                    )
                ]
            )
        ],
        fluid=True,
        className="py-4",
    )


def create_model_metrics_tab() -> dbc.Container:
    """Create the model metrics tab layout.

    Returns:
        Bootstrap Container with the model metrics layout

    """
    return dbc.Container(
        [
            dbc.Card(
                [
                    dbc.CardBody(
                        [
                            html.H3("Modell Metriken", className="mb-4"),
                            dash_table.DataTable(
                                id="metric-table",
                                columns=[
                                    {"name": "Zeitstempel", "id": "timestamp"},
                                    {"name": "Level", "id": "level"},
                                    {"name": "Pegel", "id": "sensor"},
                                    {"name": "Nachricht", "id": "message"},
                                    {"name": "Modul", "id": "module"},
                                    {"name": "Funktion", "id": "function"},
                                    {"name": "Zeile", "id": "line"},
                                    {"name": "Exception", "id": "exception"},
                                ],
                                **TABLE_STYLE,
                                page_size=20,
                                page_action="native",
                                sort_action="native",
                                sort_mode="multi",
                                filter_action="native",
                            ),
                        ]
                    )
                ]
            )
        ],
        fluid=True,
        className="py-4",
    )


def create_status_table(model_status: list, last_check_time: datetime) -> html.Div:
    """Create a status table for model forecasts.

    Args:
        model_status: List of model status dictionaries
        last_check_time: Timestamp of the status check

    Returns:
        HTML Div containing the table

    """
    header = html.Div(
        [
            html.H4(
                f"Status am {last_check_time.strftime('%Y-%m-%d %H:%M:%S')}",
                className="h6 mb-2",
            ),
            html.P(
                f"Vorhersagen erwartet für: {last_check_time.strftime('%Y-%m-%d %H:%M:00')}",
                className="text-muted small",
            ),
        ]
    )

    table = dash_table.DataTable(
        id="status-table",
        columns=[
            {"name": "Modell", "id": "model_name"},
            {"name": "Status", "id": "has_current_forecast"},
            {"name": "Neueste Vorhersage", "id": "last_forecast_time"},
            {"name": "Vorhersage erstellt", "id": "forecast_created"},
            {"name": "Pegel", "id": "sensor_name"},
            {"name": "model_id", "id": "model_id", "hideable": True},
            {"name": "freq", "id": "freq", "hideable": True},
            # {"name": "hparams", "id": "hparams", "hideable": True},
        ],
        data=[
            {
                "model_name": row["model_name"],
                "has_current_forecast": "✓" if row["has_current_forecast"] else "✗",
                "last_forecast_time": (
                    row["last_forecast_time"].strftime("%Y-%m-%d %H:%M:%S")
                    if row["last_forecast_time"]
                    else "No valid forecast"
                ),
                "forecast_created": (
                    row["forecast_created"].strftime("%Y-%m-%d %H:%M:%S") if row["forecast_created"] else "N/A"
                ),
                "sensor_name": row["sensor_name"],
                "model_id": row["model_id"],
                "freq": row["freq"],
                # "hparams": row["hparams"],
            }
            for row in model_status
        ],
        hidden_columns=["model_id", "freq"],  # ,"hparams"],
        **TABLE_STYLE,
        row_selectable="single",
        selected_rows=[],
    )

    return html.Div([header, table])
