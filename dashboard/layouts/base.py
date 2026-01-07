"""Base layout with navigation and global components."""

from dash import html, dcc
import dash_bootstrap_components as dbc


def create_base_layout(data_store) -> html.Div:
    """
    Create the base layout with navigation, global toggles, and page content.

    Args:
        data_store: DataStore instance with configuration info

    Returns:
        Base layout component
    """
    # Get period info for banner
    formation_info = data_store.get_formation_period_info()
    trading_info = data_store.get_trading_period_info()

    return html.Div([
        # URL routing
        dcc.Location(id="url", refresh=False),

        # Stores for global state
        dcc.Store(id="wait-mode-store", data=1),  # Default Wait-1-Day
        dcc.Store(id="return-calc-store", data="committed"),

        # Navigation bar
        dbc.Navbar(
            dbc.Container([
                # Brand
                dbc.NavbarBrand(
                    "GGR Pairs Trading",
                    href="/",
                    className="ms-2 fw-bold",
                ),

                # Toggle for mobile
                dbc.NavbarToggler(id="navbar-toggler"),

                # Navigation items
                dbc.Collapse(
                    dbc.Nav([
                        dbc.NavItem(dbc.NavLink("Fund Overview", href="/", id="nav-overview")),
                        dbc.NavItem(dbc.NavLink("Live Monitor", href="/live", id="nav-live")),
                        dbc.NavItem(dbc.NavLink("Pair Inspector", href="/pairs", id="nav-pairs")),

                        # Separator
                        dbc.NavItem(html.Span("|", className="nav-link text-muted")),

                        # Wait-One-Day Toggle
                        dbc.NavItem(
                            html.Div([
                                dbc.Label("Wait-1-Day:", className="me-2 text-light mb-0"),
                                dbc.Switch(
                                    id="wait-mode-toggle",
                                    value=True,
                                    className="mt-1",
                                ),
                            ], className="d-flex align-items-center"),
                            className="ms-3",
                        ),
                    ], navbar=True),
                    id="navbar-collapse",
                    navbar=True,
                ),
            ], fluid=True),
            color="dark",
            dark=True,
            className="mb-0",
        ),

        # Rolling window info banner
        dbc.Alert(
            [
                dbc.Row([
                    dbc.Col([
                        html.Strong("Trading Window: "),
                        html.Span(
                            f"{trading_info['start'].strftime('%Y-%m-%d')} to "
                            f"{trading_info['end'].strftime('%Y-%m-%d')}",
                            className="me-4",
                        ),
                        html.Strong("Pairs Formed: "),
                        html.Span(
                            f"{formation_info['start'].strftime('%Y-%m-%d')} to "
                            f"{formation_info['end'].strftime('%Y-%m-%d')}",
                            className="me-4",
                        ),
                        html.Strong("Top Pairs: "),
                        html.Span(f"{len(data_store.pairs)}"),
                    ], width=12),
                ]),
            ],
            color="info",
            className="mb-0 py-2 rounded-0",
        ),

        # Page content container
        html.Div(id="page-content", className="container-fluid py-3"),

        # Footer
        html.Footer(
            dbc.Container([
                html.Hr(),
                html.P(
                    "GGR Distance Method Backtester - Based on Gatev, Goetzmann, and Rouwenhorst (2006)",
                    className="text-muted text-center small mb-0",
                ),
            ], fluid=True),
            className="mt-auto py-3",
        ),
    ], className="d-flex flex-column min-vh-100")


def create_page_header(title: str, subtitle: str = None) -> html.Div:
    """Create a page header component."""
    children = [html.H4(title, className="mb-0")]
    if subtitle:
        children.append(html.P(subtitle, className="text-muted mb-0"))

    return html.Div(
        dbc.Row([
            dbc.Col(children, width=12),
        ]),
        className="mb-4",
    )
