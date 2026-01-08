"""Callbacks for Page 1: Fund Overview (Staggered Methodology)."""

from dash import html, callback, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..components.metrics_card import format_currency, format_percentage, format_with_color


def register_fund_overview_callbacks(app, data_store):
    """Register Fund Overview page callbacks."""

    @app.callback(
        [Output("annualized-return-metric", "children"),
         Output("sharpe-ratio-metric", "children"),
         Output("active-portfolios-metric", "children"),
         Output("total-trades-metric", "children"),
         Output("current-mode-indicator", "children")],
        Input("wait-mode-store", "data"),
    )
    def update_header_metrics(wait_mode):
        """Update header metric cards."""
        metrics = data_store.get_metrics(wait_mode)
        result = data_store.get_staggered_result(wait_mode)

        # Annualized return
        ann_return = metrics.get("annualized_return", 0)
        ann_return_display = format_with_color(ann_return * 100, format_percentage)

        # Sharpe ratio
        sharpe = metrics.get("sharpe_ratio", 0)
        sharpe_color = "success" if sharpe > 1 else "warning" if sharpe > 0 else "danger"
        sharpe_display = html.Span(f"{sharpe:.2f}", className=f"text-{sharpe_color} fw-bold fs-4")

        # Average active portfolios
        avg_active = result.active_portfolios_over_time.mean() if result else 0
        active_display = html.Div([
            html.Span(f"{avg_active:.1f}", className="fw-bold fs-4"),
            html.Br(),
            html.Small(f"of ~6 target", className="text-muted"),
        ])

        # Total trades
        total_trades = metrics.get("total_trades", 0)
        total_cycles = result.total_portfolios if result else 0
        trades_display = html.Div([
            html.Span(f"{total_trades}", className="fw-bold fs-4"),
            html.Br(),
            html.Small(f"across {total_cycles} cycles", className="text-muted"),
        ])

        # Mode indicator
        mode_text = "Wait-1-Day" if wait_mode == 1 else "Wait-0-Day (Same Day)"
        mode_indicator = html.Div([
            dbc.Badge(mode_text, color="primary", className="me-2"),
            dbc.Badge(f"{total_cycles} Portfolio Cycles", color="info", className="me-2"),
            dbc.Badge(f"{len(data_store.get_all_pairs())} Unique Pairs", color="secondary"),
        ])

        return ann_return_display, sharpe_display, active_display, trades_display, mode_indicator

    @app.callback(
        Output("equity-curve-chart", "figure"),
        Input("wait-mode-store", "data"),
    )
    def update_equity_chart(wait_mode):
        """Update the cumulative returns chart."""
        # Get cumulative returns (from staggered result)
        cumulative_returns = data_store.get_cumulative_returns(wait_mode)

        # Get SPY returns
        spy_returns = data_store.get_spy_returns()

        # Create figure
        fig = go.Figure()

        # Strategy line (cumulative returns)
        if len(cumulative_returns) > 0:
            fig.add_trace(go.Scatter(
                x=cumulative_returns.index,
                y=cumulative_returns.values * 100,  # Convert to percentage
                mode="lines",
                name="GGR Strategy",
                line=dict(color="#2E86AB", width=2),
                hovertemplate="Date: %{x}<br>Return: %{y:.2f}%<extra></extra>",
            ))

        # SPY line
        if len(spy_returns) > 0:
            fig.add_trace(go.Scatter(
                x=spy_returns.index,
                y=spy_returns.values * 100,  # Convert to percentage
                mode="lines",
                name="S&P 500 (SPY)",
                line=dict(color="#E94F37", width=2, dash="dash"),
                hovertemplate="Date: %{x}<br>Return: %{y:.2f}%<extra></extra>",
            ))

        # Add zero line
        fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)

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
            yaxis_title="Cumulative Return (%)",
            yaxis_tickformat=".1f",
        )

        return fig

    @app.callback(
        Output("monthly-returns-chart", "figure"),
        Input("wait-mode-store", "data"),
    )
    def update_monthly_returns_chart(wait_mode):
        """Update the monthly returns bar chart."""
        monthly_returns = data_store.get_monthly_returns(wait_mode)

        # Create figure
        fig = go.Figure()

        if len(monthly_returns) > 0:
            colors = ["green" if r >= 0 else "red" for r in monthly_returns.values]
            fig.add_trace(go.Bar(
                x=monthly_returns.index,
                y=monthly_returns.values * 100,  # Convert to percentage
                marker_color=colors,
                hovertemplate="Month: %{x}<br>Return: %{y:.2f}%<extra></extra>",
            ))

        # Add zero line
        fig.add_hline(y=0, line_color="black", line_width=1)

        # Layout
        fig.update_layout(
            template="plotly_white",
            margin=dict(l=50, r=20, t=10, b=50),
            xaxis_title="Month",
            yaxis_title="Monthly Return (%)",
            yaxis_tickformat=".1f",
            showlegend=False,
        )

        return fig

    @app.callback(
        Output("risk-metrics-table", "children"),
        Input("wait-mode-store", "data"),
    )
    def update_risk_metrics(wait_mode):
        """Update the risk metrics table."""
        metrics = data_store.get_metrics(wait_mode)

        sharpe = metrics.get("sharpe_ratio", 0)
        max_dd = metrics.get("max_drawdown", 0)
        ann_vol = metrics.get("annualized_volatility", 0)
        monthly_vol = metrics.get("monthly_volatility", 0)

        rows = [
            ("Sharpe Ratio (Annualized)", f"{sharpe:.2f}"),
            ("Max Drawdown", format_percentage(max_dd * 100)),
            ("Annualized Volatility", format_percentage(ann_vol * 100)),
            ("Monthly Volatility", format_percentage(monthly_vol * 100)),
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
        """Update the trade statistics table."""
        metrics = data_store.get_metrics(wait_mode)
        trades = data_store.get_all_trades(wait_mode)

        total_trades = metrics.get("total_trades", 0)
        win_rate = metrics.get("win_rate", 0)

        # Calculate avg win/loss from trades
        wins = [t.pnl for t in trades if t.pnl > 0]
        losses = [t.pnl for t in trades if t.pnl <= 0]
        avg_win = sum(wins) / len(wins) if wins else 0
        avg_loss = sum(losses) / len(losses) if losses else 0

        # Calculate avg holding days
        holding_days = [t.holding_days for t in trades]
        avg_holding = sum(holding_days) / len(holding_days) if holding_days else 0

        # Count long/short
        long_trades = sum(1 for t in trades if t.direction == 1)
        short_trades = sum(1 for t in trades if t.direction == -1)

        rows = [
            ("Total Trades", str(total_trades)),
            ("Win Rate", format_percentage(win_rate * 100)),
            ("Avg Win", format_currency(avg_win)),
            ("Avg Loss", format_currency(avg_loss)),
            ("Avg Holding Days", f"{avg_holding:.1f}"),
            ("Long Trades", str(long_trades)),
            ("Short Trades", str(short_trades)),
        ]

        return dbc.Table([
            html.Tbody([
                html.Tr([
                    html.Td(label, className="fw-bold"),
                    html.Td(value, className="text-end"),
                ]) for label, value in rows
            ])
        ], bordered=True, hover=True, size="sm")
