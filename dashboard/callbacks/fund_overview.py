"""Callbacks for Page 1: Fund Overview (Staggered Methodology)."""

from dash import html, callback, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

from ..components.metrics_card import format_currency, format_percentage, format_with_color


def register_fund_overview_callbacks(app, data_store):
    """Register Fund Overview page callbacks."""

    @app.callback(
        [Output("total-pnl-metric", "children"),
         Output("annualized-return-metric", "children"),
         Output("sharpe-ratio-metric", "children"),
         Output("max-drawdown-metric", "children"),
         Output("win-rate-metric", "children"),
         Output("total-trades-metric", "children"),
         Output("current-mode-indicator", "children")],
        Input("wait-mode-store", "data"),
    )
    def update_header_metrics(wait_mode):
        """Update header metric cards with dollar-based GGR metrics."""
        ggr_metrics = data_store.get_ggr_metrics(wait_mode)
        trade_metrics = data_store.get_metrics(wait_mode)
        result = data_store.get_staggered_result(wait_mode)

        # Total P&L (dollar amount) - from GGR metrics
        total_pnl = ggr_metrics.get("total_pnl", 0) if ggr_metrics else 0
        pnl_color = "success" if total_pnl >= 0 else "danger"
        pnl_display = html.Span(f"${total_pnl:,.0f}", className=f"text-{pnl_color} fw-bold fs-4")

        # Annualized return (from committed capital) - from GGR metrics
        ann_return = ggr_metrics.get("ann_return_committed", 0) if ggr_metrics else 0
        ann_return_display = format_with_color(ann_return * 100, format_percentage)

        # Sharpe ratio - from GGR metrics
        sharpe = ggr_metrics.get("sharpe_ratio", 0) if ggr_metrics else 0
        sharpe_color = "success" if sharpe > 1 else "warning" if sharpe > 0 else "danger"
        sharpe_display = html.Span(f"{sharpe:.2f}", className=f"text-{sharpe_color} fw-bold fs-4")

        # Max drawdown - from GGR metrics (based on realized P&L)
        max_dd_pct = ggr_metrics.get("max_drawdown_pct", 0) if ggr_metrics else 0
        max_dd_display = format_with_color(max_dd_pct * 100, format_percentage)

        # Win rate - from trade metrics
        win_rate = trade_metrics.get("win_rate", 0)
        win_rate_display = html.Span(f"{win_rate:.1%}", className="fw-bold fs-4")

        # Total trades - from trade metrics
        total_trades = trade_metrics.get("total_trades", 0)
        total_cycles = result.total_portfolios if result else 0
        trades_display = html.Span(f"{total_trades}", className="fw-bold fs-4")

        # Mode indicator
        mode_text = "Wait-1-Day" if wait_mode == 1 else "Wait-0-Day (Same Day)"
        mode_indicator = html.Div([
            dbc.Badge(mode_text, color="primary", className="me-2"),
            dbc.Badge(f"{total_cycles} Portfolio Cycles", color="info", className="me-2"),
            dbc.Badge(f"{len(data_store.get_all_pairs())} Unique Pairs", color="secondary"),
        ])

        return pnl_display, ann_return_display, sharpe_display, max_dd_display, win_rate_display, trades_display, mode_indicator

    @app.callback(
        Output("equity-curve-chart", "figure"),
        Input("wait-mode-store", "data"),
    )
    def update_equity_chart(wait_mode):
        """Update the cumulative P&L chart."""
        cumulative_pnl = data_store.get_cumulative_pnl(wait_mode)

        fig = go.Figure()

        if len(cumulative_pnl) > 0:
            pnl_color = "#2E86AB" if cumulative_pnl.iloc[-1] >= 0 else "#E94F37"
            fig.add_trace(
                go.Scatter(
                    x=cumulative_pnl.index,
                    y=cumulative_pnl.values,
                    mode="lines",
                    name="Cumulative P&L",
                    line=dict(color=pnl_color, width=2.5),
                    fill="tozeroy",
                    fillcolor=f"rgba(233, 79, 55, 0.1)" if cumulative_pnl.iloc[-1] < 0 else "rgba(46, 134, 171, 0.1)",
                    hovertemplate="Date: %{x}<br>P&L: $%{y:,.0f}<extra></extra>",
                ),
            )

        # Add zero line
        fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)

        # Layout
        fig.update_layout(
            template="plotly_white",
            hovermode="x unified",
            margin=dict(l=60, r=20, t=30, b=50),
            xaxis_title="Date",
            yaxis_title="Cumulative P&L ($)",
            yaxis_tickprefix="$",
            yaxis_tickformat=",",
            showlegend=False,
        )

        return fig

    @app.callback(
        Output("monthly-returns-chart", "figure"),
        Input("wait-mode-store", "data"),
    )
    def update_monthly_returns_chart(wait_mode):
        """Update the monthly P&L bar chart."""
        monthly_pnl = data_store.get_monthly_pnl(wait_mode)

        # Create figure
        fig = go.Figure()

        if len(monthly_pnl) > 0:
            colors = ["#28a745" if p >= 0 else "#dc3545" for p in monthly_pnl.values]
            fig.add_trace(go.Bar(
                x=monthly_pnl.index.astype(str),
                y=monthly_pnl.values,
                marker_color=colors,
                hovertemplate="Month: %{x}<br>P&L: $%{y:,.0f}<extra></extra>",
            ))

        # Add zero line
        fig.add_hline(y=0, line_color="black", line_width=1)

        # Layout
        fig.update_layout(
            template="plotly_white",
            margin=dict(l=60, r=20, t=10, b=50),
            xaxis_title="Month",
            yaxis_title="Monthly P&L ($)",
            yaxis_tickprefix="$",
            yaxis_tickformat=",",
            showlegend=False,
        )

        return fig

    @app.callback(
        Output("risk-metrics-table", "children"),
        Input("wait-mode-store", "data"),
    )
    def update_risk_metrics(wait_mode):
        """Update the risk metrics table with GGR metrics."""
        ggr_metrics = data_store.get_ggr_metrics(wait_mode)

        # All from GGR metrics (based on realized P&L)
        sharpe = ggr_metrics.get("sharpe_ratio", 0)
        max_dd = ggr_metrics.get("max_drawdown", 0)
        max_dd_pct = ggr_metrics.get("max_drawdown_pct", 0)
        capital_committed = ggr_metrics.get("capital_committed", 0)
        years = ggr_metrics.get("years", 0)
        avg_active = ggr_metrics.get("avg_active_portfolios", 0)

        rows = [
            ("Sharpe Ratio (Annualized)", f"{sharpe:.2f}"),
            ("Max Drawdown", f"{format_currency(max_dd)} ({max_dd_pct:.1%})"),
            ("Capital Committed", format_currency(capital_committed)),
            ("Trading Period", f"{years:.1f} years"),
            ("Avg Active Portfolios", f"{avg_active:.1f}"),
        ]

        return dbc.Table([
            html.Tbody([
                html.Tr([
                    html.Td(label, className="fw-bold"),
                    html.Td(value, className="text-end"),
                ]) for label, value in rows
            ])
        ], bordered=True, hover=True, size="sm")

    @app.callback(
        Output("trade-stats-table", "children"),
        Input("wait-mode-store", "data"),
    )
    def update_trade_stats(wait_mode):
        """Update the trade statistics table with P&L focus."""
        trade_metrics = data_store.get_metrics(wait_mode)
        ggr_metrics = data_store.get_ggr_metrics(wait_mode)
        trades = data_store.get_all_trades(wait_mode)

        # From GGR metrics
        total_pnl = ggr_metrics.get("total_pnl", 0)
        pairs_traded = ggr_metrics.get("pairs_traded", 0)

        # From trade metrics
        total_trades = trade_metrics.get("total_trades", 0)
        win_rate = trade_metrics.get("win_rate", 0)
        profit_factor = trade_metrics.get("profit_factor", 0)

        # Calculate avg win/loss from trades (break-even excluded)
        wins = [t.pnl for t in trades if t.pnl > 0]
        losses = [t.pnl for t in trades if t.pnl < 0]  # Strict less than
        avg_win = sum(wins) / len(wins) if wins else 0
        avg_loss = abs(sum(losses) / len(losses)) if losses else 0  # Use abs for consistency

        # Calculate avg holding days
        holding_days = [t.holding_days for t in trades]
        avg_holding = sum(holding_days) / len(holding_days) if holding_days else 0

        # Format Total P&L with color
        pnl_class = "text-success" if total_pnl >= 0 else "text-danger"
        pnl_formatted = html.Span(format_currency(total_pnl), className=pnl_class)

        rows = [
            ("Total P&L", pnl_formatted),
            ("Total Trades", str(total_trades)),
            ("Win Rate", format_percentage(win_rate * 100)),
            ("Profit Factor", f"{profit_factor:.2f}x" if profit_factor else "N/A"),
            ("Avg Win", format_currency(avg_win)),
            ("Avg Loss", format_currency(avg_loss)),
            ("Avg Holding Days", f"{avg_holding:.1f}"),
            ("Unique Pairs", str(pairs_traded)),
        ]

        return dbc.Table([
            html.Tbody([
                html.Tr([
                    html.Td(label, className="fw-bold"),
                    html.Td(value, className="text-end") if isinstance(value, str) else html.Td(value, className="text-end"),
                ]) for label, value in rows
            ])
        ], bordered=True, hover=True, size="sm")
