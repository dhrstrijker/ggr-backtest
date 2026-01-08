"""Callbacks for Page 3: Pair Inspector (Staggered Methodology)."""

from urllib.parse import parse_qs

from dash import html, callback, Input, Output, State, no_update
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd

from src.signals import calculate_spread, calculate_formation_stats, calculate_distance
from ..components.metrics_card import format_currency, format_percentage


def register_pair_inspector_callbacks(app, data_store):
    """Register Pair Inspector page callbacks."""

    @app.callback(
        Output("current-pair", "data"),
        Input("url", "search"),
    )
    def parse_pair_from_url(search):
        """Parse pair from URL query parameter."""
        if not search:
            return None

        params = parse_qs(search.lstrip("?"))
        pair_key = params.get("pair", [None])[0]
        return pair_key

    @app.callback(
        Output("pair-inspector-title", "children"),
        Input("current-pair", "data"),
    )
    def update_page_title(pair_value):
        """Update the page title with the current pair name."""
        if not pair_value:
            return "Pair Inspector"

        sym_a, sym_b = pair_value.split("_")
        return f"Pair Inspector: {sym_a} / {sym_b}"

    @app.callback(
        Output("normalized-prices-chart", "figure"),
        [Input("current-pair", "data"),
         Input("wait-mode-store", "data")],
    )
    def update_prices_chart(pair_value, wait_mode):
        """Update the normalized prices chart (GGR Figure 1 style)."""
        if not pair_value:
            return go.Figure()

        # Parse pair
        sym_a, sym_b = pair_value.split("_")
        pair = (sym_a, sym_b)

        # Get full price data
        if sym_a not in data_store.close_prices.columns or sym_b not in data_store.close_prices.columns:
            return go.Figure()

        prices_a = data_store.close_prices[sym_a].dropna()
        prices_b = data_store.close_prices[sym_b].dropna()

        # Normalize prices (base 1.0 = $1 invested)
        norm_a = prices_a / prices_a.iloc[0]
        norm_b = prices_b / prices_b.iloc[0]

        # Get trades for this pair from all cycles
        trades = data_store.get_trades_for_pair(pair, wait_mode)

        # Create figure
        fig = go.Figure()

        # Add shaded regions for trades FIRST (so they're behind the lines)
        for trade in trades:
            fill_color = "rgba(46, 134, 171, 0.1)" if trade.direction == 1 else "rgba(233, 79, 55, 0.1)"

            fig.add_vrect(
                x0=trade.entry_date,
                x1=trade.exit_date,
                fillcolor=fill_color,
                line_width=0,
                layer="below",
            )

        # Price lines
        fig.add_trace(go.Scatter(
            x=norm_a.index,
            y=norm_a.values,
            mode="lines",
            name=sym_a,
            line=dict(color="#2E86AB", width=2),
            hovertemplate=f"{sym_a}<br>Date: %{{x}}<br>Value: $%{{y:.3f}}<extra></extra>",
        ))

        fig.add_trace(go.Scatter(
            x=norm_b.index,
            y=norm_b.values,
            mode="lines",
            name=sym_b,
            line=dict(color="#E94F37", width=2),
            hovertemplate=f"{sym_b}<br>Date: %{{x}}<br>Value: $%{{y:.3f}}<extra></extra>",
        ))

        # Add trade markers
        for trade in trades:
            marker_color = "#2E86AB" if trade.direction == 1 else "#E94F37"
            direction_text = "Long" if trade.direction == 1 else "Short"

            # Entry marker (on stock A line)
            if trade.entry_date in norm_a.index:
                entry_val = norm_a.loc[trade.entry_date]
                fig.add_trace(go.Scatter(
                    x=[trade.entry_date],
                    y=[entry_val],
                    mode="markers",
                    marker=dict(
                        symbol="triangle-up" if trade.direction == 1 else "triangle-down",
                        size=14,
                        color=marker_color,
                        line=dict(width=1, color="white"),
                    ),
                    name="Entry",
                    showlegend=False,
                    hovertemplate=(
                        f"<b>ENTRY ({direction_text})</b><br>"
                        f"Date: {trade.entry_date.strftime('%Y-%m-%d')}<br>"
                        f"Distance: {trade.entry_distance:.2f}σ<extra></extra>"
                    ),
                ))

            # Exit marker
            if trade.exit_date in norm_a.index:
                exit_val = norm_a.loc[trade.exit_date]
                fig.add_trace(go.Scatter(
                    x=[trade.exit_date],
                    y=[exit_val],
                    mode="markers",
                    marker=dict(
                        symbol="x",
                        size=12,
                        color=marker_color,
                        line=dict(width=2),
                    ),
                    name="Exit",
                    showlegend=False,
                    hovertemplate=(
                        f"<b>EXIT ({trade.exit_reason})</b><br>"
                        f"Date: {trade.exit_date.strftime('%Y-%m-%d')}<br>"
                        f"P&L: {format_currency(trade.pnl)}<extra></extra>"
                    ),
                ))

        # Layout
        fig.update_layout(
            template="plotly_white",
            hovermode="x unified",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
            ),
            margin=dict(l=50, r=20, t=30, b=50),
            xaxis_title="Date",
            yaxis_title="Normalized Value ($1 invested)",
            yaxis_tickformat="$.2f",
        )

        return fig

    @app.callback(
        [Output("cycle-selector", "options"),
         Output("cycle-selector", "value")],
        [Input("current-pair", "data"),
         Input("wait-mode-store", "data")],
    )
    def populate_cycle_selector(pair_value, wait_mode):
        """Populate the cycle selector with cycles that have trades for this pair."""
        if not pair_value:
            return [], None

        sym_a, sym_b = pair_value.split("_")
        pair = (sym_a, sym_b)

        # Get staggered result
        result = data_store.get_staggered_result(wait_mode)
        if not result:
            return [], None

        # Get all trades for this pair to check which cycles have trades
        all_trades = data_store.get_trades_for_pair(pair, wait_mode)
        cycles_with_trades = {t.cycle_id for t in all_trades if t.cycle_id is not None}

        # Find cycles that have trades for this pair
        options = []
        for cycle in result.cycles:
            if cycle.cycle_id in cycles_with_trades:
                trade_count = sum(1 for t in all_trades if t.cycle_id == cycle.cycle_id)
                label = (
                    f"Cycle {cycle.cycle_id}: "
                    f"{cycle.trading_start.strftime('%Y-%m-%d')} to "
                    f"{cycle.trading_end.strftime('%Y-%m-%d')} "
                    f"({trade_count} trades)"
                )
                options.append({"label": label, "value": cycle.cycle_id})

        # Default to most recent cycle
        default_value = options[-1]["value"] if options else None

        return options, default_value

    @app.callback(
        Output("spread-distance-chart", "figure"),
        [Input("current-pair", "data"),
         Input("wait-mode-store", "data"),
         Input("cycle-selector", "value")],
    )
    def update_distance_chart(pair_value, wait_mode, selected_cycle_id):
        """Update the distance chart with CORRECT GGR methodology.

        Uses static formation σ and normalizes from trading period start,
        matching the actual backtest calculation.
        """
        if not pair_value or selected_cycle_id is None:
            return go.Figure()

        # Parse pair
        sym_a, sym_b = pair_value.split("_")
        pair = (sym_a, sym_b)

        # Get the selected cycle
        result = data_store.get_staggered_result(wait_mode)
        if not result:
            return go.Figure()

        cycle = None
        for c in result.cycles:
            if c.cycle_id == selected_cycle_id:
                cycle = c
                break

        if not cycle or not cycle.pairs or pair not in cycle.pairs:
            return go.Figure()

        # Get entry threshold from config
        entry_threshold = data_store.config.get("entry_threshold", 2.0)

        # Check symbols exist
        if sym_a not in data_store.close_prices.columns or sym_b not in data_store.close_prices.columns:
            return go.Figure()

        # === CORRECT GGR CALCULATION ===
        # 1. Get formation period data and calculate STATIC formation σ
        formation_close_a = data_store.close_prices[sym_a].loc[cycle.formation_start:cycle.formation_end]
        formation_close_b = data_store.close_prices[sym_b].loc[cycle.formation_start:cycle.formation_end]
        formation_spread = calculate_spread(formation_close_a, formation_close_b, normalize=True)
        formation_stats = calculate_formation_stats(formation_spread)
        formation_std = formation_stats['std']

        # 2. Get trading period data and calculate distance using STATIC σ
        trading_close_a = data_store.close_prices[sym_a].loc[cycle.trading_start:cycle.trading_end]
        trading_close_b = data_store.close_prices[sym_b].loc[cycle.trading_start:cycle.trading_end]
        trading_spread = calculate_spread(trading_close_a, trading_close_b, normalize=True)
        distance = calculate_distance(trading_spread, formation_std)

        # Get trades for this pair in this cycle
        all_trades = data_store.get_trades_for_pair(pair, wait_mode)
        # Filter to trades that belong to this cycle (by cycle_id, not date overlap)
        cycle_trades = [
            t for t in all_trades
            if t.cycle_id == selected_cycle_id
        ]

        # Create figure
        fig = go.Figure()

        # Threshold lines
        fig.add_hline(
            y=entry_threshold,
            line_dash="dash",
            line_color="#E94F37",
            annotation_text=f"+{entry_threshold}σ",
            annotation_position="right",
            opacity=0.7,
        )
        fig.add_hline(
            y=-entry_threshold,
            line_dash="dash",
            line_color="#2E86AB",
            annotation_text=f"-{entry_threshold}σ",
            annotation_position="right",
            opacity=0.7,
        )
        fig.add_hline(
            y=0,
            line_dash="solid",
            line_color="gray",
            line_width=2,
            annotation_text="Exit (0)",
            annotation_position="right",
        )

        # Distance line (using correct GGR calculation)
        fig.add_trace(go.Scatter(
            x=distance.index,
            y=distance.values,
            mode="lines",
            name="Spread Distance",
            line=dict(color="#4A4A4A", width=1.5),
            hovertemplate="Date: %{x}<br>Distance: %{y:.2f}σ<extra></extra>",
        ))

        # Add trade markers for this cycle
        for trade in cycle_trades:
            marker_color = "#2E86AB" if trade.direction == 1 else "#E94F37"

            # Entry marker
            fig.add_trace(go.Scatter(
                x=[trade.entry_date],
                y=[trade.entry_distance],
                mode="markers",
                marker=dict(
                    symbol="circle",
                    size=10,
                    color=marker_color,
                    line=dict(width=1, color="white"),
                ),
                showlegend=False,
                hovertemplate=f"<b>ENTRY</b><br>{trade.entry_date.strftime('%Y-%m-%d')}<br>Distance: {trade.entry_distance:.2f}σ<extra></extra>",
            ))

            # Exit marker
            fig.add_trace(go.Scatter(
                x=[trade.exit_date],
                y=[trade.exit_distance],
                mode="markers",
                marker=dict(
                    symbol="x",
                    size=10,
                    color=marker_color,
                    line=dict(width=2),
                ),
                showlegend=False,
                hovertemplate=f"<b>EXIT</b><br>{trade.exit_date.strftime('%Y-%m-%d')}<br>Distance: {trade.exit_distance:.2f}σ<extra></extra>",
            ))

        # Layout with info about formation σ
        fig.update_layout(
            template="plotly_white",
            hovermode="x unified",
            showlegend=False,
            margin=dict(l=50, r=80, t=30, b=50),
            xaxis_title="Date",
            yaxis_title="Distance (σ)",
            yaxis_zeroline=True,
            title=dict(
                text=f"Formation σ = {formation_std:.4f}",
                font=dict(size=11, color="gray"),
                x=0.02,
                y=0.98,
            ),
        )

        return fig

    @app.callback(
        Output("pair-stats-sidebar", "children"),
        [Input("current-pair", "data"),
         Input("wait-mode-store", "data")],
    )
    def update_pair_stats(pair_value, wait_mode):
        """Update the pair statistics sidebar."""
        if not pair_value:
            return html.P("No pair selected", className="text-muted")

        # Parse pair
        sym_a, sym_b = pair_value.split("_")
        pair = (sym_a, sym_b)

        # Get aggregated pair stats
        pair_stats = data_store.pair_stats.get(pair, {})

        # Get trades for this pair
        trades = data_store.get_trades_for_pair(pair, wait_mode)

        # Get cycles where this pair traded
        cycles_traded = data_store.get_cycles_for_pair(pair, wait_mode)

        # Calculate pair-specific metrics
        total_pnl = sum(t.pnl for t in trades)
        win_count = sum(1 for t in trades if t.pnl > 0)
        win_rate = (win_count / len(trades) * 100) if trades else 0
        avg_holding = sum(t.holding_days for t in trades) / len(trades) if trades else 0

        return html.Div([
            html.H6("Staggered Stats", className="border-bottom pb-2 mb-3"),
            html.P([
                html.Strong("Cycles Traded: "),
                f"{len(cycles_traded)}",
            ], className="mb-1"),
            html.P([
                html.Strong("Total Trades: "),
                str(len(trades)),
            ], className="mb-1"),
            html.P([
                html.Strong("Total P&L: "),
                html.Span(
                    format_currency(total_pnl),
                    className="text-success" if total_pnl >= 0 else "text-danger",
                ),
            ], className="mb-1"),
            html.P([
                html.Strong("Win Rate: "),
                f"{win_rate:.1f}%",
            ], className="mb-1"),
            html.P([
                html.Strong("Avg Holding: "),
                f"{avg_holding:.1f} days",
            ], className="mb-3"),

            html.H6("Cycle IDs", className="border-bottom pb-2 mb-3"),
            html.P(
                ", ".join(str(c) for c in cycles_traded[:10]) + ("..." if len(cycles_traded) > 10 else ""),
                className="small text-muted",
            ) if cycles_traded else html.P("None", className="text-muted small"),
        ])

    @app.callback(
        Output("pair-trades-table", "children"),
        [Input("current-pair", "data"),
         Input("wait-mode-store", "data")],
    )
    def update_trades_table(pair_value, wait_mode):
        """Update the trade history table (showing all trades across cycles)."""
        if not pair_value:
            return html.P("No pair selected", className="text-muted")

        # Parse pair
        sym_a, sym_b = pair_value.split("_")
        pair = (sym_a, sym_b)

        # Get trades from all cycles
        trades = data_store.get_trades_for_pair(pair, wait_mode)

        if not trades:
            return html.P("No trades for this pair across all cycles", className="text-muted")

        # Build table rows
        rows = []
        for i, trade in enumerate(trades, 1):
            direction_badge = dbc.Badge(
                "Long" if trade.direction == 1 else "Short",
                color="success" if trade.direction == 1 else "danger",
            )

            pnl_color = "text-success" if trade.pnl >= 0 else "text-danger"

            rows.append(
                html.Tr([
                    html.Td(str(i)),
                    html.Td(direction_badge),
                    html.Td(trade.entry_date.strftime("%Y-%m-%d")),
                    html.Td(trade.exit_date.strftime("%Y-%m-%d")),
                    html.Td(f"{trade.entry_distance:+.2f}σ"),
                    html.Td(f"{trade.exit_distance:+.2f}σ"),
                    html.Td(str(trade.holding_days), className="text-center"),
                    html.Td(html.Span(format_currency(trade.pnl), className=pnl_color)),
                    html.Td(dbc.Badge(trade.exit_reason, color="secondary")),
                ])
            )

        return dbc.Table(
            [
                html.Thead(
                    html.Tr([
                        html.Th("#"),
                        html.Th("Direction"),
                        html.Th("Entry Date"),
                        html.Th("Exit Date"),
                        html.Th("Entry Dist"),
                        html.Th("Exit Dist"),
                        html.Th("Days", className="text-center"),
                        html.Th("P&L"),
                        html.Th("Exit Reason"),
                    ])
                ),
                html.Tbody(rows),
            ],
            striped=True,
            bordered=True,
            hover=True,
            responsive=True,
            size="sm",
        )
