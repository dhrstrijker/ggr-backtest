"""Dash application factory."""

import dash
from dash import html, dcc
import dash_bootstrap_components as dbc

from .layouts.base import create_base_layout
from .callbacks import register_all_callbacks


def create_app(data_store) -> dash.Dash:
    """
    Create and configure the Dash application.

    Args:
        data_store: DataStore instance with pre-computed results

    Returns:
        Configured Dash application
    """
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        suppress_callback_exceptions=True,
        title="GGR Pairs Trading Dashboard",
        meta_tags=[
            {"name": "viewport", "content": "width=device-width, initial-scale=1"}
        ],
    )

    # Store data_store reference for callbacks
    app.data_store = data_store

    # Set base layout
    app.layout = create_base_layout(data_store)

    # Register all callbacks
    register_all_callbacks(app, data_store)

    return app
