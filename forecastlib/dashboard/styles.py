"""Dashboard styling definitions and color palettes.

This module defines consistent styling configurations for the Plotly-based
dashboard including colors, table styles, tab styles, and component layouts.
Uses Bootstrap color conventions for consistency.
"""

from typing import Any

import plotly.express as px

# Common colors (using Bootstrap colors for consistency)
COLORS: dict[str, str] = {
    "success": "#198754",
    "danger": "#dc3545",
    "light_gray": "#f4f4f4",
    "border": "rgba(0,0,0,.125)",
    "background": "#f8f9fa",
    "warning_bg": "#fff3e0",
    "error_bg": "#ffebee",
    "warning": "#ef6c00",
    "error": "#c62828",
}
"""Dictionary of color definitions for dashboard components."""


COLOR_SET: list[str] = px.colors.qualitative.Set3
"""Default color palette for charts and visualizations."""

# Table Styles (This is messy, the conditional style data is not the same for each table)
TABLE_STYLE: dict[str, Any] = {
    "style_table": {"overflowX": "auto", "borderRadius": "4px", "border": f"1px solid {COLORS['border']}"},
    "style_header": {
        "backgroundColor": COLORS["background"],
        "fontWeight": "600",
        "textAlign": "left",
        "padding": "12px 16px",
        "borderBottom": f"1px solid {COLORS['border']}",
    },
    "style_cell": {"textAlign": "left", "padding": "12px 16px", "fontSize": "14px"},
    "style_data_conditional": [
        {"if": {"row_index": "odd"}, "backgroundColor": "rgb(248, 249, 250)"},
        {
            "if": {"filter_query": '{has_current_forecast} = "✓"', "column_id": "has_current_forecast"},
            "color": COLORS["success"],
        },
        {
            "if": {"filter_query": '{has_current_forecast} = "✗"', "column_id": "has_current_forecast"},
            "color": COLORS["danger"],
        },
        {"if": {"filter_query": '{level} = "ERROR"'}, "backgroundColor": COLORS["error_bg"], "color": COLORS["error"]},
        {
            "if": {"filter_query": '{level} = "WARNING"'},
            "backgroundColor": COLORS["warning_bg"],
            "color": COLORS["warning"],
        },
    ],
}
"""Base table styling configuration with headers, cells, and conditional formatting."""

TAB_STYLE: dict[str, Any] = {
    "active_label_class_name": "fw-bold",
    "label_style": {"color": "rgba(255, 255, 255, 0.9)", "padding": "1rem 1.5rem"},
    "active_label_style": {"color": "white", "background": "rgba(255, 255, 255, 0.1)"},
}
"""Tab component styling for active/inactive states."""


# Historical Table Style
HISTORICAL_TABLE_STYLE: dict[str, Any] = {
    "style_table": {"maxHeight": "1200px", "maxWidth": "1600px", "overflowY": "auto", "width": "100%"},
    "style_cell": {
        "textAlign": "right",
        "padding": "5px",
        "minWidth": "100px",
        "whiteSpace": "normal",
        "fontSize": "12px",
    },
    "style_header": {"backgroundColor": COLORS["light_gray"], "fontWeight": "bold", "textAlign": "center"},
}
"""Styling for historical data tables with scrolling and compact layout."""

# Forecast Status Table Style
FORECAST_STATUS_TABLE_STYLE: dict[str, Any] = {
    "style_table": {"overflowX": "auto", "width": "100%"},
    "style_cell": {"textAlign": "center", "padding": "5px", "minWidth": "100px", "fontSize": "12px"},
    "style_header": {"backgroundColor": COLORS["light_gray"], "fontWeight": "bold", "textAlign": "center"},
}
"""Styling for forecast status summary tables."""

# Plot Layouts
STATUS_SUMMARY_LAYOUT: dict[str, Any] = {
    "height": 300,
    "margin": {"l": 30, "r": 30, "t": 50, "b": 30},
    "showlegend": True,
}
"""Default layout configuration for status summary plots."""

MODEL_FORECASTS_LAYOUT = {
    "height": 600,
    "xaxis_title": "Time",
    "yaxis_title": "Gauge [cm]",
    "legend": {"yanchor": "top", "y": 0.99, "xanchor": "left", "x": 1.05},
}

HISTORICAL_PLOT_LAYOUT = {
    "xaxis_title": "Time",
    "yaxis_title": "Value",
    "height": 600,
    "showlegend": True,
    "legend": {"yanchor": "top", "y": 0.99, "xanchor": "left", "x": 1.05},
    "margin": {"r": 150},
}

INPUT_FORECASTS_LAYOUT = {
    "showlegend": True,
    "legend": {
        "yanchor": "top",
        "y": 0.99,
        "xanchor": "left",
        "x": 1.05,
        "groupclick": "togglegroup",
        "itemsizing": "constant",
        "tracegroupgap": 5,
    },
    "margin": {"r": 150},
}

# Plot Colors and Traces
FORECAST_COLORS = {"historical": "black", "historical_width": 2}


def get_conditional_styles_for_columns(columns: list[str], timestamp_col: str = "tstamp") -> list[dict[str, Any]]:
    """Generate conditional styles for columns with blank values."""
    styles = [
        {
            "if": {"filter_query": f"{{{col}}} is blank", "column_id": col},
            "backgroundColor": "#ffebee",
            "color": "#c62828",
        }
        for col in columns
        if col != timestamp_col
    ]

    styles.append({"if": {"column_id": timestamp_col}, "textAlign": "left", "minWidth": "150px"})

    return styles


def get_forecast_status_conditional_styles(sensor_names: list[str]) -> list[dict[str, Any]]:
    """Generate conditional styles for forecast status table."""
    styles = []
    for col in sensor_names:
        styles.extend(
            [
                {
                    "if": {"filter_query": f'{{{col}}} = "Missing"', "column_id": col},
                    "backgroundColor": "#ffebee",
                    "color": "#c62828",
                },
                {
                    "if": {"filter_query": f'{{{col}}} = "OK"', "column_id": col},
                    "backgroundColor": "#e8f5e9",
                    "color": "#2e7d32",
                },
            ]
        )
    return styles
