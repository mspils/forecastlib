"""Dashboard layout helper components for creating UI elements.

Provides utility functions for building collapsible sections and other
reusable dashboard components.
"""

from typing import Any

from dash import html


def create_collapsible_section(title: str, content: Any, is_open: bool = True) -> html.Div:
    """Create a collapsible section UI component.

    Creates a dashboard section with a toggle button that can expand/collapse
    the content area.

    Args:
        title: Section title text displayed in the header.
        content: Dash component(s) to display in the collapsible content area.
        is_open: Whether section starts expanded. Defaults to True.

    Returns:
        Dash Div component containing the collapsible section.

    """
    return html.Div(
        [
            html.Div(
                [
                    html.H3(title, style={"display": "inline-block", "marginRight": "10px"}),
                    html.Button(
                        "▼" if is_open else "▶",
                        id={"type": "collapse-button", "section": title},
                        style={
                            "border": "none",
                            "background": "none",
                            "fontSize": "20px",
                            "cursor": "pointer",
                        },
                    ),
                ],
                style={"marginBottom": "10px"},
            ),
            html.Div(
                content,
                id={"type": "collapse-content", "section": title},
                style={"display": "block" if is_open else "none"},
            ),
        ]
    )
