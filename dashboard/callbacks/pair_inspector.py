"""Callbacks for Page 3: Pair Inspector."""

from dash import html, callback, Input, Output, no_update
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd

from ..components.metrics_card import format_currency, format_percentage


def register_pair_inspector_callbacks(app, data_store):
    """Register Pair Inspector page callbacks."""

    @app.callback(
        Output("normalized-prices-chart", "figure"),
        [Input("pair-selector", "value"),
         Input("wait-mode-store", "data")],
    )
    def update_prices_chart(pair_value, wait_mode):
        """Update the normalized prices chart (GGR Figure 1 style)."""
        if not pair_value:
            return go.Figure()

        # Parse pair
        sym_a, sym_b = pair_value.split("_")
        pair = (sym_a, sym_b)

        # Get price data
        prices_a = data_store.trading_prices[sym_a]
        prices_b = data_store.trading_prices[sym_b]

        # Normalize prices (base 1.0 = $1 invested)
        norm_a = prices_a / prices_a.iloc[0]
        norm_b = prices_b / prices_b.iloc[0]

        # Get trades for this pair
        results = data_store.get_results(wait_mode)
        trades = results[pair].trades if pair in results else []

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
            entry_idx = norm_a.index.get_loc(trade.entry_date) if trade.entry_date in norm_a.index else None
            if entry_idx is not None:
                fig.add_trace(go.Scatter(
                    x=[trade.entry_date],
                    y=[norm_a.iloc[entry_idx]],
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
            exit_idx = norm_a.index.get_loc(trade.exit_date) if trade.exit_date in norm_a.index else None
            if exit_idx is not None:
                fig.add_trace(go.Scatter(
                    x=[trade.exit_date],
                    y=[norm_a.iloc[exit_idx]],
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
        Output("spread-distance-chart", "figure"),
        [Input("pair-selector", "value"),
         Input("wait-mode-store", "data")],
    )
    def update_distance_chart(pair_value, wait_mode):
        """Update the distance chart with entry/exit threshold bands."""
        if not pair_value:
            return go.Figure()

        # Parse pair
        sym_a, sym_b = pair_value.split("_")
        pair = (sym_a, sym_b)

        # Get distance series
        distance = data_store.distance_series.get(pair)
        if distance is None:
            return go.Figure()

        # Get trades
        results = data_store.get_results(wait_mode)
        trades = results[pair].trades if pair in results else []

        # Get entry threshold from config
        entry_threshold = data_store.config.get("entry_threshold", 2.0)

        # Create figure
        fig = go.Figure()

        # Add threshold bands
        # Upper entry zone (> 2σ)
        fig.add_hrect(
            y0=entry_threshold, y1=distance.max() + 0.5,
            fillcolor="rgba(233, 79, 55, 0.1)",
            line_width=0,
            annotation_text="Entry Zone (Short)",
            annotation_position="top right",
        )

        # Lower entry zone (< -2σ)
        fig.add_hrect(
            y0=distance.min() - 0.5, y1=-entry_threshold,
            fillcolor="rgba(46, 134, 171, 0.1)",
            line_width=0,
            annotation_text="Entry Zone (Long)",
            annotation_position="bottom right",
        )

        # Threshold lines
        fig.add_hline(
            y=entry_threshold,
            line_dash="dash",
            line_color="#E94F37",
            annotation_text=f"+{entry_threshold}σ",
            annotation_position="right",
        )
        fig.add_hline(
            y=-entry_threshold,
            line_dash="dash",
            line_color="#2E86AB",
            annotation_text=f"-{entry_threshold}σ",
            annotation_position="right",
        )
        fig.add_hline(
            y=0,
            line_dash="solid",
            line_color="gray",
            line_width=2,
            annotation_text="Exit (0)",
            annotation_position="right",
        )

        # Distance line
        fig.add_trace(go.Scatter(
            x=distance.index,
            y=distance.values,
            mode="lines",
            name="Distance",
            line=dict(color="#4A4A4A", width=1.5),
            hovertemplate="Date: %{x}<br>Distance: %{y:.2f}σ<extra></extra>",
        ))

        # Add trade markers
        for trade in trades:
            marker_color = "#2E86AB" if trade.direction == 1 else "#E94F37"

            # Entry marker
            if trade.entry_date in distance.index:
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
                    hovertemplate=f"<b>ENTRY</b><br>{trade.entry_date.strftime('%Y-%m-%d')}<extra></extra>",
                ))

            # Exit marker
            if trade.exit_date in distance.index:
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
                    hovertemplate=f"<b>EXIT</b><br>{trade.exit_date.strftime('%Y-%m-%d')}<extra></extra>",
                ))

        # Layout
        fig.update_layout(
            template="plotly_white",
            hovermode="x unified",
            showlegend=False,
            margin=dict(l=50, r=80, t=20, b=50),
            xaxis_title="Date",
            yaxis_title="Distance (σ)",
            yaxis_zeroline=True,
        )

        return fig

    @app.callback(
        Output("pair-stats-sidebar", "children"),
        [Input("pair-selector", "value"),
         Input("wait-mode-store", "data")],
    )
    def update_pair_stats(pair_value, wait_mode):
        """Update the pair statistics sidebar."""
        if not pair_value:
            return html.P("Select a pair to view statistics")

        # Parse pair
        sym_a, sym_b = pair_value.split("_")
        pair = (sym_a, sym_b)

        # Get SSD
        ssd = data_store.ssd_matrix.loc[sym_a, sym_b]

        # Get correlation
        corr = data_store.pair_correlations.get(pair, 0)

        # Get formation stats
        form_stats = data_store.formation_stats.get(pair, {})
        formation_std = form_stats.get("std", 0)

        # Get trades for this pair
        results = data_store.get_results(wait_mode)
        trades = results[pair].trades if pair in results else []

        # Calculate pair-specific metrics
        total_pnl = sum(t.pnl for t in trades)
        win_count = sum(1 for t in trades if t.pnl > 0)
        win_rate = (win_count / len(trades) * 100) if trades else 0
        avg_holding = sum(t.holding_days for t in trades) / len(trades) if trades else 0

        return html.Div([
            html.H6("Formation Period", className="border-bottom pb-2 mb-3"),
            html.P([
                html.Strong("SSD: "),
                f"{ssd:.6f}",
            ], className="mb-1"),
            html.P([
                html.Strong("Correlation: "),
                f"{corr:.4f}",
            ], className="mb-1"),
            html.P([
                html.Strong("Formation σ: "),
                f"{formation_std:.6f}",
            ], className="mb-3"),

            html.H6("Trading Period", className="border-bottom pb-2 mb-3"),
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

            html.H6("Sector", className="border-bottom pb-2 mb-3"),
            html.P([
                html.Strong(f"{sym_a}: "),
                "Tanker Shipping",
            ], className="mb-1 small"),
            html.P([
                html.Strong(f"{sym_b}: "),
                "Tanker Shipping",
            ], className="mb-1 small"),
        ])

    @app.callback(
        Output("pair-trades-table", "children"),
        [Input("pair-selector", "value"),
         Input("wait-mode-store", "data")],
    )
    def update_trades_table(pair_value, wait_mode):
        """Update the trade history table."""
        if not pair_value:
            return html.P("Select a pair to view trade history")

        # Parse pair
        sym_a, sym_b = pair_value.split("_")
        pair = (sym_a, sym_b)

        # Get trades
        results = data_store.get_results(wait_mode)
        trades = results[pair].trades if pair in results else []

        if not trades:
            return html.P("No trades for this pair", className="text-muted")

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
