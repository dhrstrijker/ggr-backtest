"""Navigation and routing callbacks."""

from dash import html, callback, Input, Output, State, no_update
import dash_bootstrap_components as dbc

from ..layouts.page1_fund_overview import create_fund_overview_layout
from ..layouts.page2_live_monitor import create_live_monitor_layout
from ..layouts.page3_pair_inspector import create_pair_inspector_layout


def register_navigation_callbacks(app, data_store):
    """Register navigation-related callbacks."""

    @app.callback(
        Output("page-content", "children"),
        Input("url", "pathname"),
    )
    def render_page(pathname):
        """Render the appropriate page based on URL."""
        if pathname == "/" or pathname == "/overview":
            return create_fund_overview_layout(data_store)
        elif pathname == "/live":
            return create_live_monitor_layout(data_store)
        elif pathname == "/pairs":
            return create_pair_inspector_layout(data_store)
        else:
            # Default to overview for unknown paths
            return create_fund_overview_layout(data_store)

    @app.callback(
        [Output("nav-overview", "active"),
         Output("nav-live", "active"),
         Output("nav-pairs", "active")],
        Input("url", "pathname"),
    )
    def update_nav_active(pathname):
        """Update which nav item is active."""
        return (
            pathname == "/" or pathname == "/overview",
            pathname == "/live",
            pathname == "/pairs",
        )

    @app.callback(
        Output("wait-mode-store", "data"),
        Input("wait-mode-toggle", "value"),
    )
    def update_wait_mode(toggle_value):
        """Update wait mode store when toggle changes."""
        return 1 if toggle_value else 0

    @app.callback(
        Output("navbar-collapse", "is_open"),
        Input("navbar-toggler", "n_clicks"),
        State("navbar-collapse", "is_open"),
    )
    def toggle_navbar(n_clicks, is_open):
        """Toggle navbar collapse on mobile."""
        if n_clicks:
            return not is_open
        return is_open
