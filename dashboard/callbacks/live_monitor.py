"""Callbacks for Page 2: Live Trading Monitor."""

from dash import html, callback, Input, Output, State, no_update
import dash_bootstrap_components as dbc

from ..components.metrics_card import format_currency, format_with_color


def get_spread_color(distance: float) -> str:
    """Get Bootstrap color class based on distance magnitude."""
    if abs(distance) > 2.0:
        return "danger"  # Red - beyond entry threshold
    elif abs(distance) < 0.5:
        return "success"  # Green - near convergence
    else:
        return "warning"  # Yellow - in between


def get_spread_text_color(distance: float) -> str:
    """Get text color class based on distance magnitude."""
    color = get_spread_color(distance)
    return f"text-{color}"


def register_live_monitor_callbacks(app, data_store):
    """Register Live Monitor page callbacks."""

    @app.callback(
        Output("current-date-display", "children"),
        Input("date-slider", "value"),
    )
    def update_date_display(date_index):
        """Update the current date display."""
        dates = data_store.get_trading_dates()
        if date_index is None or date_index >= len(dates):
            return ""

        selected_date = dates[date_index]
        return selected_date.strftime("%B %d, %Y")

    @app.callback(
        [Output("open-positions-count", "children"),
         Output("unrealized-pnl", "children"),
         Output("realized-pnl", "children"),
         Output("signals-today", "children")],
        [Input("date-slider", "value"),
         Input("wait-mode-store", "data")],
    )
    def update_summary_metrics(date_index, wait_mode):
        """Update summary metric cards for selected date."""
        dates = data_store.get_trading_dates()
        if date_index is None or date_index >= len(dates):
            return "0", "$0.00", "$0.00", "0"

        selected_date = dates[date_index]

        # Get open positions
        positions = data_store.get_position_at_date(selected_date, wait_mode)
        open_count = len(positions)

        # Calculate unrealized P&L
        unrealized = sum(
            data_store.calculate_unrealized_pnl(pos, selected_date)
            for pos in positions
        )

        # Get realized P&L
        realized = data_store.get_realized_pnl_at_date(selected_date, wait_mode)

        # Count signals today (entries or exits)
        trades = data_store.get_all_trades(wait_mode)
        signals_count = sum(
            1 for t in trades
            if t.entry_date == selected_date or t.exit_date == selected_date
        )

        # Format outputs
        open_display = str(open_count)
        unrealized_display = format_with_color(unrealized, format_currency)
        realized_display = format_with_color(realized, format_currency)
        signals_display = str(signals_count)

        return open_display, unrealized_display, realized_display, signals_display

    @app.callback(
        Output("pairs-status-table", "children"),
        [Input("date-slider", "value"),
         Input("wait-mode-store", "data")],
    )
    def update_pairs_table(date_index, wait_mode):
        """Update the pairs status table for selected date."""
        dates = data_store.get_trading_dates()
        if date_index is None or date_index >= len(dates):
            return html.P("No data available")

        selected_date = dates[date_index]

        # Get open positions
        positions = data_store.get_position_at_date(selected_date, wait_mode)
        position_pairs = {pos["pair"]: pos for pos in positions}

        # Build table rows
        rows = []
        for pair in data_store.pairs:
            # Get current distance
            distance = data_store.get_current_distance(pair, selected_date)
            if distance is None:
                distance = 0.0

            # Check if position is open
            position = position_pairs.get(pair)

            if position:
                status = "OPEN"
                status_color = "primary"
                direction = "Long" if position["direction"] == 1 else "Short"
                direction_badge = dbc.Badge(
                    direction,
                    color="success" if position["direction"] == 1 else "danger",
                    className="ms-1",
                )
                days_open = (selected_date - position["entry_date"]).days
                pnl = data_store.calculate_unrealized_pnl(position, selected_date)
                pnl_display = format_with_color(pnl, format_currency)
            else:
                status = "WAITING"
                status_color = "secondary"
                direction_badge = html.Span()
                days_open = "-"
                pnl_display = "-"

            # Color code spread based on distance
            spread_color = get_spread_text_color(distance)
            spread_display = html.Span(
                f"{distance:+.2f}σ",
                className=f"{spread_color} fw-bold",
            )

            rows.append(
                html.Tr([
                    html.Td(f"{pair[0]} / {pair[1]}", className="fw-bold"),
                    html.Td([dbc.Badge(status, color=status_color), direction_badge]),
                    html.Td(spread_display, className="text-center"),
                    html.Td(str(days_open), className="text-center"),
                    html.Td(pnl_display, className="text-end"),
                ])
            )

        return dbc.Table(
            [
                html.Thead(
                    html.Tr([
                        html.Th("Pair"),
                        html.Th("Status"),
                        html.Th("Current Distance", className="text-center"),
                        html.Th("Days Open", className="text-center"),
                        html.Th("P&L", className="text-end"),
                    ])
                ),
                html.Tbody(rows),
            ],
            striped=True,
            bordered=True,
            hover=True,
            responsive=True,
        )

    @app.callback(
        Output("date-slider", "value"),
        [Input("reset-button", "n_clicks"),
         Input("end-button", "n_clicks"),
         Input("animation-interval", "n_intervals")],
        [State("date-slider", "value"),
         State("date-slider", "max"),
         State("animation-interval", "disabled")],
    )
    def control_slider(reset_clicks, end_clicks, n_intervals, current_value, max_value, interval_disabled):
        """Control slider position with buttons and animation."""
        from dash import ctx

        if not ctx.triggered:
            return no_update

        trigger = ctx.triggered[0]["prop_id"].split(".")[0]

        if trigger == "reset-button":
            return 0
        elif trigger == "end-button":
            return max_value
        elif trigger == "animation-interval" and not interval_disabled:
            # Advance slider
            if current_value < max_value:
                return current_value + 1
            else:
                return max_value

        return no_update

    @app.callback(
        [Output("animation-interval", "disabled"),
         Output("play-button", "children")],
        Input("play-button", "n_clicks"),
        State("animation-interval", "disabled"),
    )
    def toggle_animation(n_clicks, is_disabled):
        """Toggle animation play/pause."""
        if n_clicks is None:
            return True, html.I(className="bi bi-play-fill")

        if is_disabled:
            # Start playing
            return False, html.I(className="bi bi-pause-fill")
        else:
            # Pause
            return True, html.I(className="bi bi-play-fill")
