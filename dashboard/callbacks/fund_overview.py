"""Callbacks for Page 1: Fund Overview."""

from dash import html, callback, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..components.metrics_card import format_currency, format_percentage, format_with_color


def register_fund_overview_callbacks(app, data_store):
    """Register Fund Overview page callbacks."""

    @app.callback(
        [Output("total-return-metric", "children"),
         Output("ytd-return-metric", "children"),
         Output("active-pairs-metric", "children"),
         Output("capital-usage-metric", "children"),
         Output("current-mode-indicator", "children")],
        [Input("wait-mode-store", "data"),
         Input("return-calc-toggle", "value")],
    )
    def update_header_metrics(wait_mode, return_calc):
        """Update header metric cards."""
        metrics = data_store.get_metrics(wait_mode)
        trades = data_store.get_all_trades(wait_mode)

        # Total return
        total_return = metrics.get("total_return", 0)
        total_return_pct = metrics.get("total_return_pct", 0)

        total_return_display = html.Div([
            format_with_color(total_return_pct, format_percentage),
            html.Br(),
            html.Small(format_currency(total_return), className="text-muted"),
        ])

        # YTD return (same as total for now since we're in one period)
        # In a real system, this would filter by current year
        ytd_display = format_with_color(total_return_pct, format_percentage)

        # Active pairs (pairs with at least one trade)
        active_pairs = len(set(t.pair for t in trades))
        total_pairs = len(data_store.pairs)
        pairs_display = f"{active_pairs} / {total_pairs}"

        # Capital usage
        total_capital = data_store.config["capital_per_trade"] * data_store.config["top_n_pairs"]
        capital_deployed = active_pairs * data_store.config["capital_per_trade"]
        usage_pct = (capital_deployed / total_capital) * 100 if total_capital > 0 else 0
        usage_display = f"{usage_pct:.0f}%"

        # Mode indicator
        mode_text = "Wait-1-Day" if wait_mode == 1 else "Wait-0-Day (Same Day)"
        calc_text = "Committed Capital" if return_calc == "committed" else "Fully Invested"
        mode_indicator = html.Div([
            dbc.Badge(mode_text, color="primary", className="me-2"),
            dbc.Badge(calc_text, color="secondary"),
        ])

        return total_return_display, ytd_display, pairs_display, usage_display, mode_indicator

    @app.callback(
        Output("equity-curve-chart", "figure"),
        [Input("wait-mode-store", "data"),
         Input("return-calc-toggle", "value")],
    )
    def update_equity_chart(wait_mode, return_calc):
        """Update the equity curve chart with SPY comparison."""
        # Get strategy returns
        strategy_returns = data_store.get_strategy_returns(wait_mode, return_calc)

        # Get SPY returns
        spy_returns = data_store.get_spy_returns()

        # Align dates
        common_dates = strategy_returns.index.intersection(spy_returns.index)

        if len(common_dates) == 0:
            # Fallback if no common dates
            common_dates = strategy_returns.index

        # Create figure
        fig = go.Figure()

        # Strategy line
        fig.add_trace(go.Scatter(
            x=strategy_returns.index,
            y=strategy_returns.values,
            mode="lines",
            name="GGR Strategy",
            line=dict(color="#2E86AB", width=2),
            hovertemplate="Date: %{x}<br>Return: %{y:.2f}%<extra></extra>",
        ))

        # SPY line
        if len(spy_returns) > 0:
            fig.add_trace(go.Scatter(
                x=spy_returns.index,
                y=spy_returns.values,
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
        Output("risk-metrics-table", "children"),
        Input("wait-mode-store", "data"),
    )
    def update_risk_metrics(wait_mode):
        """Update the risk metrics table."""
        metrics = data_store.get_metrics(wait_mode)

        sharpe = metrics.get("sharpe_ratio", 0)
        max_dd = metrics.get("max_drawdown_pct", 0)
        volatility = metrics.get("volatility", 0) if "volatility" in metrics else None

        # Calculate volatility if not in metrics
        if volatility is None:
            equity = data_store.get_equity_curve(wait_mode)
            daily_returns = equity.pct_change().dropna()
            volatility = daily_returns.std() * (252 ** 0.5) * 100 if len(daily_returns) > 0 else 0

        rows = [
            ("Sharpe Ratio", f"{sharpe:.2f}"),
            ("Max Drawdown", format_percentage(max_dd)),
            ("Annualized Volatility", format_percentage(volatility)),
            ("Profit Factor", f"{metrics.get('profit_factor', 0):.2f}"),
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

        rows = [
            ("Total Trades", str(metrics.get("total_trades", 0))),
            ("Win Rate", format_percentage(metrics.get("win_rate", 0))),
            ("Avg Win", format_currency(metrics.get("avg_win", 0))),
            ("Avg Loss", format_currency(metrics.get("avg_loss", 0))),
            ("Avg Holding Days", f"{metrics.get('avg_holding_days', 0):.1f}"),
            ("Long Trades", str(metrics.get("long_trades", 0))),
            ("Short Trades", str(metrics.get("short_trades", 0))),
        ]

        return dbc.Table([
            html.Tbody([
                html.Tr([
                    html.Td(label, className="fw-bold"),
                    html.Td(value, className="text-end"),
                ]) for label, value in rows
            ])
        ], bordered=True, hover=True, size="sm")
