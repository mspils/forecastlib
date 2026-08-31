from datetime import datetime, timedelta

import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
from dash import dash_table, html
from plotly.subplots import make_subplots

from forecastlib.dashboard.constants import PERIODS_EXT_FORECAST_TABLE
from forecastlib.dashboard.styles import (
    COLOR_SET,
    COLORS,
    FORECAST_STATUS_TABLE_STYLE,
    HISTORICAL_TABLE_STYLE,
    STATUS_SUMMARY_LAYOUT,
    TABLE_STYLE,
    get_conditional_styles_for_columns,
    get_forecast_status_conditional_styles,
)


def add_opacity(c, opacity) -> str:
    return f"rgba{c[3:-1]},{opacity})"


def create_status_summary_chart(status):
    """Creates a simple pie chart showing the percentage of models with current forecasts."""
    total_models = len(status["model_status"])
    models_with_forecast = sum(1 for model in status["model_status"] if model["has_current_forecast"])
    models_without_forecast = total_models - models_with_forecast

    fig = go.Figure(
        data=[
            go.Pie(
                labels=["Current Forecast", "Missing Forecast"],
                values=[models_with_forecast, models_without_forecast],
                hole=0.4,
                marker_colors=[COLORS["success"], COLORS["danger"]],
                textinfo="percent",
                hovertemplate="Status: %{label}<br>Count: %{value}<br>Percentage: %{percent}<extra></extra>",
            )
        ]
    )

    fig.update_layout(
        title={
            "text": f"Model Status ({models_with_forecast} of {total_models} models have current forecasts)",
            "y": 0.95,
        },
        **STATUS_SUMMARY_LAYOUT,
    )

    return fig


def create_model_forecasts_plot(df_forecasts, df_historical, df_inp_fcst, df_conf, sensor_name):
    """Create a plot showing recent model forecasts alongside historical data.

    Args:
        df_forecasts: DataFrame from get_recent_forecasts with columns: tstamp, member, horizon_step, value, created
        df_historical: DataFrame with historical sensor data indexed by tstamp
        df_inp_fcst: DataFrame with input forecasts
        df_conf: DataFrame with confidence intervals
        sensor_name: Name of the target sensor for this model

    """
    if df_forecasts.empty:
        return go.Figure().add_annotation(
            text="No forecasts available"
        )  # Convert long format to wide format (h1, h2, ..., h48)
    df_forecasts_wide = df_forecasts.pivot_table(
        index=["tstamp", "member"], columns="horizon_step", values="value", aggfunc="first"
    ).reset_index()

    df_forecasts_wide = df_forecasts_wide.round(2)

    start_times = sorted(set(df_forecasts_wide["tstamp"]) | set(df_inp_fcst["tstamp"]))
    start_time_colors = {t: COLOR_SET[i % len(COLOR_SET)] for i, t in enumerate(start_times)}

    fig = go.Figure()

    if not df_historical.empty:
        df_measured = df_historical[start_times[0] - timedelta(days=3) :]
        # Add historical data
        fig.add_trace(
            go.Scatter(
                x=df_measured.index,
                y=df_measured[sensor_name],
                name=f"Messwerte - {sensor_name}",
                line={"color": "black", "width": 2},
                mode="lines",
            )
        )

    show_legend_check = set()
    for (start_time, member), df in df_forecasts.groupby(["tstamp", "member"]):
        most_recent = df_forecasts["tstamp"].max() == start_time

        show_legend = start_time not in show_legend_check
        show_legend_check.add(start_time)
        legend_group = f"{start_time.strftime('%Y-%m-%d %H:%M')}"

        x = df["target_time"]
        y = df["value"]

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                name=legend_group,
                legendgroup=legend_group,
                showlegend=show_legend,
                line={
                    "color": start_time_colors[start_time],
                    "width": 3 if member == 0 else 1,
                },
                hovertemplate=(f"Member: {member:2} - Value: %{{y:.2f}}"),
                visible=True if most_recent else "legendonly",
            ),
        )

        if most_recent and member == 0 and not df_conf.empty:
            for alpha, conf_name, opacity in zip(
                ["05", "10", "50"], ["95%", "90%", "50%"], [0.1, 0.2, 0.3], strict=True
            ):
                subset = "test"
                metric_name = f"{subset}_conf{alpha}"

                fillcolor = add_opacity(start_time_colors[start_time], opacity)

                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=y + df_conf[metric_name].values,
                        name=f"{conf_name} CI Hauptlauf",
                        showlegend=False,
                        mode="lines",
                        line={"width": 0},
                        fillcolor=fillcolor,
                        visible=True if most_recent else "legendonly",
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=y - df_conf[metric_name].values,
                        name=f"{conf_name} CI Hauptlauf",
                        mode="lines",
                        line={"width": 0},
                        fill="tonexty",
                        fillcolor=fillcolor,
                        visible=True if most_recent else "legendonly",
                    )
                )

    # members = df_forecasts_wide['member'].unique()
    # show_legend_check = set()
    # for member in members:
    #     member_data = df_forecasts_wide[df_forecasts_wide['member'] == member]
    #     for _, row in member_data.iterrows():
    #         start_time = row['tstamp']
    #         if start_time in show_legend_check:
    #             show_legend = False
    #         else:
    #             show_legend = True
    #             show_legend_check.add(start_time)

    #         legend_group = f'{start_time.strftime("%Y-%m-%d %H:%M")}'

    #         if FREQUENCY == "d":
    #             x = [start_time + timedelta(days=step+1) for step in row.index[2:]]
    #         else:
    #             x = [start_time + timedelta(hours=step+1) for step in row.index[2:]]
    #         y = row[2:].values

    #         if all(v is None for v in y):
    #             continue
    #         most_recent = start_time == df_forecasts_wide["tstamp"].max()
    #         fig.add_trace(
    #             go.Scatter(
    #                 x=x,
    #                 y=y,
    #                 name=legend_group,
    #                 legendgroup=legend_group,
    #                 showlegend=show_legend,
    #                 line=dict(
    #                     color=start_time_colors[start_time],
    #                     width=3 if member == 0 else 1,
    #                 ),
    #                 hovertemplate=(
    #                     f'Member: {member:2} - '
    #                     'Value: %{y:.2f}'
    #                 ),
    #                 visible=True if most_recent else 'legendonly'
    #             ),
    #         )
    #         if most_recent and member == 0 and not df_conf.empty:
    #             for alpha, conf_name, opacity in zip(['05', '10', '50'], ['95%', '90%', '50%'], [0.1, 0.2, 0.3]):
    #                 subset = 'test'
    #                 metric_name = f"{subset}_conf{alpha}"

    #                 fillcolor = add_opacity(start_time_colors[start_time], opacity)

    #                 fig.add_trace(
    #                     go.Scatter(
    #                         x=x,
    #                         y=y + df_conf[metric_name].values,
    #                         name=f"{conf_name} CI Hauptlauf",
    #                         showlegend=False,
    #                         mode='lines',
    #                         line=dict(width=0),
    #                         fillcolor=fillcolor,
    #                         visible=True if most_recent else 'legendonly')
    #                 )
    #                 fig.add_trace(
    #                     go.Scatter(
    #                         x=x,
    #                         y=y - df_conf[metric_name].values,
    #                         name=f"{conf_name} CI Hauptlauf",
    #                         mode='lines',
    #                         line=dict(width=0),
    #                         fill='tonexty',
    #                         fillcolor=fillcolor,
    #                         visible=True if most_recent else 'legendonly')
    #                )

    temp_layout_dict = {"yaxis": {"title": {"text": "Pegel [cm]"}}}
    fig.update_layout(**temp_layout_dict)

    # Add input forecasts: pivot long -> wide and plot per sensor
    if not df_inp_fcst.empty:
        # Pivot input forecasts (long format) to wide per (tstamp, member, sensor_name)
        df_inp_wide = df_inp_fcst.pivot_table(
            index=["tstamp", "member", "sensor_name"], columns="horizon_step", values="value", aggfunc="first"
        ).reset_index()

        # Determine max horizon for input forecasts
        try:
            max_inp_h = int(df_inp_fcst["horizon_step"].max())
        except Exception:
            max_inp_h = 48

        # Rename numeric horizon columns to h1..hN
        for col in list(df_inp_wide.columns):
            if isinstance(col, int):
                df_inp_wide.rename(columns={col: f"h{col}"}, inplace=True)

        sensors = df_inp_wide["sensor_name"].unique()
        show_legend_check = set()

        for sensor_idx, sensor in enumerate(sensors, 1):
            temp_layout_dict = {
                f"yaxis{sensor_idx + 1}": {
                    "title": {"text": "Niederschlag" if "Precip,h.Cmd" in sensor else "Pegel [cm]"},
                    "anchor": "free",
                    "overlaying": "y",
                    "autoshift": True,
                }
            }
            fig.update_layout(**temp_layout_dict)

            sensor_data = df_inp_wide[df_inp_wide["sensor_name"] == sensor]
            members = sensor_data["member"].unique()

            for member in members:
                member_data = sensor_data[sensor_data["member"] == member]
                for _, row in member_data.iterrows():
                    start_time = row["tstamp"]
                    legend_group = f"{start_time.strftime('%Y-%m-%d %H:%M')} {sensor}"
                    if legend_group in show_legend_check:
                        show_legend = False
                    else:
                        show_legend = True
                        show_legend_check.add(legend_group)

                    timestamps = [start_time + timedelta(hours=i) for i in range(1, max_inp_h + 1)]
                    values = [row.get(f"h{i}") for i in range(1, max_inp_h + 1)]
                    most_recent = start_time == df_forecasts_wide["tstamp"].max()

                    fig.add_trace(
                        go.Scatter(
                            x=timestamps,
                            y=values,
                            name=legend_group,
                            legendgroup=legend_group,
                            showlegend=show_legend,
                            line={
                                "color": start_time_colors.get(start_time, COLOR_SET[0]),
                                "width": 3 if member == 0 else 1,
                                "dash": "solid" if member == 0 else "dot",
                            },
                            hovertemplate=(f"Member: {member:2} - Value: %{{y:.2f}}<br>"),
                            visible=True if most_recent else "legendonly",
                            yaxis=f"y{sensor_idx + 1}",
                        )
                    )

    fig.update_layout(
        hovermode="x",
        legend={
            "yanchor": "top",
            "y": 0.99,
            "xanchor": "left",
            "x": 1.05,
        },
    )
    return fig


def create_log_table(logs_data):
    """Creates a configured DataTable for logging display.

    Args:
        logs_data: List of dictionaries containing log data

    Returns:
        dash_table.DataTable: Configured table for log display

    """
    columns = [
        {"name": "Time", "id": "timestamp"},
        {"name": "Level", "id": "level"},
        {"name": "Message", "id": "message"},
        {"name": "Exception", "id": "exception"},
    ]

    return dash_table.DataTable(
        columns=columns,
        data=logs_data,
        **TABLE_STYLE,
        # style_data_conditional=style_data_conditional,
        # style_table=style_table,
        # style_cell=style_cell
    )


def create_historical_plot(df_historical, model_name):
    """Creates a plotly figure for historical sensor data.

    Args:
        df_historical: DataFrame with sensor measurements
        model_name: String name of the model

    Returns:
        plotly.graph_objects.Figure: Configured plot

    """
    fig = go.Figure()

    # Add a trace for each sensor
    for column in df_historical.columns:
        fig.add_trace(
            go.Scatter(
                x=df_historical.index,
                y=df_historical[column],
                name=column,
                mode="lines",
                hovertemplate="%{y:.2f}<extra>%{x}</extra>",
            )
        )

    # Update layout
    fig.update_layout(
        # title=f'Sensor Measurements - Last 144 Hours for {model_name}',
        xaxis_title="Time",
        yaxis_title="Value",
        height=600,
        showlegend=True,
        legend={"yanchor": "top", "y": 0.99, "xanchor": "left", "x": 1.05},
        margin={"r": 150},
    )

    return fig


def create_historical_table(df_historical):
    """Creates a formatted DataTable for historical sensor data."""
    df_table = df_historical.reset_index()
    df_table["tstamp"] = df_table["tstamp"].dt.strftime("%Y-%m-%d %H:%M")

    columns = [{"name": col, "id": col} for col in df_table.columns]
    style_data_conditional = get_conditional_styles_for_columns(df_table.columns)

    # TODO move buttons etc. to app.py
    hist_table = dash_table.DataTable(
        id="historical-table",
        columns=columns,
        data=df_table.to_dict("records"),
        style_data_conditional=style_data_conditional,
        fixed_columns={"headers": True, "data": 1},
        sort_action="native",
        sort_mode="single",
        **HISTORICAL_TABLE_STYLE,
    )

    # return dbc.Card([
    #    dbc.CardHeader([
    #        dbc.Row([
    #            dbc.Col(
    #                dbc.Button(
    #                    "Show All Data",
    #                    id='toggle-missing-values',
    #                    color="primary",
    #                    className="mb-3",
    #                    n_clicks=0
    #                ),
    #                width="auto"
    #            )
    #        ])
    #    ]),
    #    dbc.CardBody(hist_table)
    # ])
    return html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Button(
                            "Show All Data", id="toggle-missing-values", color="primary", className="mb-3", n_clicks=0
                        ),
                        width="auto",
                    )
                ]
            ),
            hist_table,
        ]
    )

    # return hist_table


def create_inp_forecast_status_table(df_forecast, ext_forecast_names):
    """Creates a status table showing availability of forecasts for each sensor at 3-hour intervals.

    Args:
        df_forecast: DataFrame with columns tstamp, sensor_name containing forecast data
        ext_forecast_names: List of external forecast names

    Returns:
        dash.html.Div: Div containing configured DataTable

    """
    # Get unique sensor names
    # sensor_names = sorted(df_forecast['sensor_name'].unique())

    # Create index of last 48 hours at 3-hour intervals
    # last_required_hour = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0) adding a timezone messes with the comparison with the timestamps in the database
    last_required_hour = datetime.now().replace(minute=0, second=0, microsecond=0)
    while last_required_hour.hour % 3 != 0:
        last_required_hour -= timedelta(hours=1)

    time_range = pd.date_range(
        end=last_required_hour,
        periods=PERIODS_EXT_FORECAST_TABLE,  # 48 hours / 3 + 1
        freq="3h",
    )

    # Initialize result DataFrame with NaN
    # status_df = pd.DataFrame(index=time_range, columns=sensor_names)
    status_df = pd.DataFrame(index=time_range, columns=ext_forecast_names)

    # For each sensor and timestamp, check if data exists
    # for sensor in sensor_names:
    for sensor in ext_forecast_names:
        sensor_data = df_forecast[df_forecast["sensor_name"] == sensor]
        for timestamp in time_range:
            # TODO ENSEMBLES
            has_data = any(
                sensor_data["tstamp"] == timestamp
                # (sensor_data['tstamp'] <= timestamp) &
                # (sensor_data['tstamp'] + timedelta(hours=48) >= timestamp)
            )
            status_df.loc[timestamp, sensor] = "OK" if has_data else "Missing"

    # Reset index to make timestamp a column
    status_df = status_df.reset_index()
    status_df["index"] = status_df["index"].dt.strftime("%Y-%m-%d %H:%M")

    # Configure table styles
    style_data_conditional = get_forecast_status_conditional_styles(ext_forecast_names)

    return html.Div(
        dash_table.DataTable(
            id="forecast-status-table",
            columns=[{"name": "Timestamp", "id": "index"}, *[{"name": col, "id": col} for col in ext_forecast_names]],
            data=status_df.to_dict("records"),
            style_data_conditional=style_data_conditional,
            fixed_columns={"headers": True, "data": 1},
            sort_action="native",
            sort_mode="single",
            **FORECAST_STATUS_TABLE_STYLE,
        )
    )


def create_metrics_plots(df_metrics, metric_name):
    """Create plots for model metrics."""
    fig = make_subplots(rows=2, cols=1, subplot_titles=["Gesamt", "Flutbereich (95. Percentil)"], vertical_spacing=0.1)

    for col in df_metrics:
        row = 2 if "flood" in col else 1
        set_name = col.split("_")[0]
        if set_name == "train":
            color = COLOR_SET[0]
        elif set_name == "val":
            color = COLOR_SET[1]
        elif set_name == "test":
            color = COLOR_SET[2]
        else:
            color = COLOR_SET[3]

        fig.add_trace(
            go.Scatter(
                x=df_metrics.index,
                y=df_metrics[col],
                name=set_name,
                legendgroup=set_name,
                showlegend=row == 1,
                mode="lines+markers",
                line={"color": color},
            ),
            row=row,
            col=1,
        )

    fig.update_layout(
        xaxis_title="Vorhersagehorizont",
        yaxis_title=metric_name.upper(),
        height=600,
        showlegend=True,
        legend={"yanchor": "top", "y": 0.99, "xanchor": "left", "x": 1.05},
        margin={"r": 50},
    )
    return fig
