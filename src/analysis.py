"""Performance analysis and visualization module."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .backtest import Trade


def calculate_metrics(
    trades: list[Trade],
    equity_curve: pd.Series,
    risk_free_rate: float = 0.02,
) -> dict[str, Any]:
    """
    Calculate performance metrics from backtest results.

    Args:
        trades: List of completed trades
        equity_curve: Equity curve series
        risk_free_rate: Annual risk-free rate for Sharpe calculation

    Returns:
        Dictionary with performance metrics
    """
    if not trades:
        return {
            "total_trades": 0,
            "total_return": 0,
            "total_return_pct": 0,
            "sharpe_ratio": 0,
            "max_drawdown": 0,
            "max_drawdown_pct": 0,
            "win_rate": 0,
            "avg_win": 0,
            "avg_loss": 0,
            "profit_factor": 0,
            "avg_holding_days": 0,
        }

    # Basic stats
    pnls = [t.pnl for t in trades]
    pnl_pcts = [t.pnl_pct for t in trades]
    total_pnl = sum(pnls)
    initial_capital = equity_curve.iloc[0]
    total_return_pct = (equity_curve.iloc[-1] - initial_capital) / initial_capital

    # Win rate
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    win_rate = len(wins) / len(pnls) if pnls else 0

    # Average win/loss
    avg_win = np.mean(wins) if wins else 0
    avg_loss = abs(np.mean(losses)) if losses else 0

    # Profit factor
    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 1
    profit_factor = gross_profit / gross_loss if gross_loss != 0 else float("inf")

    # Sharpe ratio (annualized)
    returns = equity_curve.pct_change().dropna()
    if len(returns) > 0 and returns.std() > 0:
        # Assume ~252 trading days per year
        excess_returns = returns - risk_free_rate / 252
        sharpe = np.sqrt(252) * excess_returns.mean() / returns.std()
    else:
        sharpe = 0

    # Max drawdown
    rolling_max = equity_curve.expanding().max()
    drawdown = equity_curve - rolling_max
    max_drawdown = drawdown.min()
    max_drawdown_pct = (max_drawdown / rolling_max[drawdown.idxmin()]) if max_drawdown < 0 else 0

    # Average holding period
    avg_holding_days = np.mean([t.holding_days for t in trades])

    # Trade breakdown by direction
    long_trades = [t for t in trades if t.direction == 1]
    short_trades = [t for t in trades if t.direction == -1]

    return {
        "total_trades": len(trades),
        "total_return": total_pnl,
        "total_return_pct": total_return_pct,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
        "max_drawdown_pct": max_drawdown_pct,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "avg_holding_days": avg_holding_days,
        "long_trades": len(long_trades),
        "short_trades": len(short_trades),
        "long_win_rate": len([t for t in long_trades if t.pnl > 0]) / len(long_trades) if long_trades else 0,
        "short_win_rate": len([t for t in short_trades if t.pnl > 0]) / len(short_trades) if short_trades else 0,
    }


def print_metrics(metrics: dict[str, Any]) -> None:
    """Print formatted performance metrics."""
    print("=" * 50)
    print("BACKTEST RESULTS")
    print("=" * 50)
    print(f"Total Trades:     {metrics['total_trades']}")
    print(f"Total Return:     ${metrics['total_return']:,.2f} ({metrics['total_return_pct']:.2%})")
    print(f"Sharpe Ratio:     {metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown:     ${metrics['max_drawdown']:,.2f} ({metrics['max_drawdown_pct']:.2%})")
    print("-" * 50)
    print(f"Win Rate:         {metrics['win_rate']:.2%}")
    print(f"Avg Win:          ${metrics['avg_win']:,.2f}")
    print(f"Avg Loss:         ${metrics['avg_loss']:,.2f}")
    print(f"Profit Factor:    {metrics['profit_factor']:.2f}")
    print(f"Avg Holding Days: {metrics['avg_holding_days']:.1f}")
    print("-" * 50)
    print(f"Long Trades:      {metrics['long_trades']} ({metrics['long_win_rate']:.2%} win rate)")
    print(f"Short Trades:     {metrics['short_trades']} ({metrics['short_win_rate']:.2%} win rate)")
    print("=" * 50)


def trades_to_dataframe(trades: list[Trade]) -> pd.DataFrame:
    """Convert list of trades to DataFrame."""
    if not trades:
        return pd.DataFrame()
    return pd.DataFrame([t.to_dict() for t in trades])


def plot_equity_curve(
    equity_curve: pd.Series,
    title: str = "Equity Curve",
    show_drawdown: bool = True,
) -> go.Figure:
    """
    Plot equity curve with optional drawdown overlay.

    Args:
        equity_curve: Equity curve series
        title: Chart title
        show_drawdown: Whether to show drawdown in secondary axis

    Returns:
        Plotly Figure object
    """
    if show_drawdown:
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            row_heights=[0.7, 0.3],
            vertical_spacing=0.05,
        )

        # Equity curve
        fig.add_trace(
            go.Scatter(
                x=equity_curve.index,
                y=equity_curve.values,
                mode="lines",
                name="Equity",
                line=dict(color="blue", width=2),
            ),
            row=1, col=1,
        )

        # Drawdown
        rolling_max = equity_curve.expanding().max()
        drawdown = (equity_curve - rolling_max) / rolling_max * 100

        fig.add_trace(
            go.Scatter(
                x=drawdown.index,
                y=drawdown.values,
                mode="lines",
                name="Drawdown %",
                fill="tozeroy",
                line=dict(color="red", width=1),
                fillcolor="rgba(255, 0, 0, 0.2)",
            ),
            row=2, col=1,
        )

        fig.update_yaxes(title_text="Equity ($)", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)

    else:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=equity_curve.index,
                y=equity_curve.values,
                mode="lines",
                name="Equity",
                line=dict(color="blue", width=2),
            )
        )
        fig.update_yaxes(title_text="Equity ($)")

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        hovermode="x unified",
        template="plotly_white",
        height=500,
    )

    return fig


def plot_trade(
    close_prices: pd.DataFrame,
    trade: Trade,
    distance: pd.Series,
    lookback_days: int = 60,
) -> go.Figure:
    """
    Plot a single trade with price series and distance (GGR methodology).

    Args:
        close_prices: DataFrame with close prices
        trade: Trade object to visualize
        distance: Distance series (spread / formation_std)
        lookback_days: Days before entry to show

    Returns:
        Plotly Figure object
    """
    sym_a, sym_b = trade.pair

    # Find date range
    entry_idx = close_prices.index.get_loc(trade.entry_date)
    start_idx = max(0, entry_idx - lookback_days)
    exit_idx = close_prices.index.get_loc(trade.exit_date)
    end_idx = min(len(close_prices), exit_idx + 10)

    date_range = close_prices.index[start_idx:end_idx]
    prices_a = close_prices[sym_a].loc[date_range]
    prices_b = close_prices[sym_b].loc[date_range]
    d = distance.loc[date_range]

    # Normalize for comparison
    norm_a = prices_a / prices_a.iloc[0] * 100
    norm_b = prices_b / prices_b.iloc[0] * 100

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.6, 0.4],
        vertical_spacing=0.1,
    )

    # Price series (normalized)
    fig.add_trace(
        go.Scatter(
            x=norm_a.index, y=norm_a.values,
            mode="lines", name=sym_a,
            line=dict(color="blue", width=2),
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=norm_b.index, y=norm_b.values,
            mode="lines", name=sym_b,
            line=dict(color="orange", width=2),
        ),
        row=1, col=1,
    )

    # Entry/exit markers
    fig.add_vline(x=trade.entry_date, line_dash="dash", line_color="green", row=1)
    fig.add_vline(x=trade.exit_date, line_dash="dash", line_color="red", row=1)
    fig.add_vline(x=trade.entry_date, line_dash="dash", line_color="green", row=2)
    fig.add_vline(x=trade.exit_date, line_dash="dash", line_color="red", row=2)

    # Distance (GGR: static σ from formation)
    fig.add_trace(
        go.Scatter(
            x=d.index, y=d.values,
            mode="lines", name="Distance (σ)",
            line=dict(color="purple", width=2),
        ),
        row=2, col=1,
    )

    # Threshold lines (GGR: ±2σ entry, 0 exit)
    fig.add_hline(y=2, line_dash="dot", line_color="red", row=2,
                  annotation_text="+2σ Entry")
    fig.add_hline(y=-2, line_dash="dot", line_color="green", row=2,
                  annotation_text="-2σ Entry")
    fig.add_hline(y=0, line_dash="solid", line_color="black", row=2,
                  annotation_text="Exit (crossing)")

    direction = "Long" if trade.direction == 1 else "Short"
    fig.update_layout(
        title=f"{sym_a}/{sym_b} - {direction} Trade (P&L: ${trade.pnl:.2f})",
        template="plotly_white",
        height=600,
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Normalized Price", row=1, col=1)
    fig.update_yaxes(title_text="Distance (σ)", row=2, col=1)

    return fig


def plot_ssd_heatmap(
    ssd_matrix: pd.DataFrame,
    title: str = "SSD Heatmap (Lower = More Similar)",
) -> go.Figure:
    """
    Plot SSD matrix as a heatmap.

    Args:
        ssd_matrix: SSD matrix from calculate_ssd_matrix
        title: Chart title

    Returns:
        Plotly Figure object
    """
    fig = go.Figure(
        data=go.Heatmap(
            z=ssd_matrix.values,
            x=ssd_matrix.columns,
            y=ssd_matrix.index,
            colorscale="RdYlGn_r",  # Red = high (different), Green = low (similar)
            hoverongaps=False,
        )
    )

    fig.update_layout(
        title=title,
        template="plotly_white",
        height=500,
        width=600,
    )

    return fig


def plot_pair_prices(
    close_prices: pd.DataFrame,
    pair: tuple[str, str],
    title: str | None = None,
) -> go.Figure:
    """
    Plot normalized prices for a pair.

    Args:
        close_prices: DataFrame with close prices
        pair: Tuple of (symbol_a, symbol_b)
        title: Optional chart title

    Returns:
        Plotly Figure object
    """
    sym_a, sym_b = pair
    prices_a = close_prices[sym_a]
    prices_b = close_prices[sym_b]

    # Normalize
    norm_a = prices_a / prices_a.iloc[0] * 100
    norm_b = prices_b / prices_b.iloc[0] * 100

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=norm_a.index, y=norm_a.values,
            mode="lines", name=sym_a,
            line=dict(width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=norm_b.index, y=norm_b.values,
            mode="lines", name=sym_b,
            line=dict(width=2),
        )
    )

    if title is None:
        title = f"{sym_a} vs {sym_b} (Normalized)"

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Normalized Price (base 100)",
        template="plotly_white",
        hovermode="x unified",
    )

    return fig


def plot_pair_analysis(
    close_prices: pd.DataFrame,
    pair: tuple[str, str],
    trades: list[Trade],
    distance: pd.Series,
    entry_threshold: float = 2.0,
) -> go.Figure:
    """
    Plot comprehensive pair analysis with all trades marked.

    Shows:
    - Row 1: Normalized prices for both stocks with entry/exit markers
    - Row 2: Distance (in σ) with threshold lines and trade regions

    Per GGR methodology: Exit occurs when spread crosses zero, not at a threshold.

    Args:
        close_prices: DataFrame with close prices
        pair: Tuple of (symbol_a, symbol_b)
        trades: List of Trade objects for this pair
        distance: Distance series (spread / formation_std) for this pair
        entry_threshold: Entry threshold in σ for reference lines

    Returns:
        Plotly Figure object
    """
    sym_a, sym_b = pair

    # Normalize prices
    prices_a = close_prices[sym_a]
    prices_b = close_prices[sym_b]
    norm_a = prices_a / prices_a.iloc[0] * 100
    norm_b = prices_b / prices_b.iloc[0] * 100

    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.6, 0.4],
        vertical_spacing=0.08,
        subplot_titles=(f"{sym_a} vs {sym_b} (Normalized Prices)", "Distance (σ from formation)"),
    )

    # Row 1: Normalized prices
    fig.add_trace(
        go.Scatter(
            x=norm_a.index, y=norm_a.values,
            mode="lines", name=sym_a,
            line=dict(color="blue", width=2),
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=norm_b.index, y=norm_b.values,
            mode="lines", name=sym_b,
            line=dict(color="orange", width=2),
        ),
        row=1, col=1,
    )

    # Row 2: Distance (in σ)
    fig.add_trace(
        go.Scatter(
            x=distance.index, y=distance.values,
            mode="lines", name="Distance (σ)",
            line=dict(color="purple", width=2),
        ),
        row=2, col=1,
    )

    # Add threshold lines (GGR: entry at ±2σ, exit on crossing zero)
    fig.add_hline(y=entry_threshold, line_dash="dash", line_color="red",
                  annotation_text=f"Short Entry (+{entry_threshold}σ)", row=2, col=1)
    fig.add_hline(y=-entry_threshold, line_dash="dash", line_color="green",
                  annotation_text=f"Long Entry (-{entry_threshold}σ)", row=2, col=1)
    fig.add_hline(y=0, line_color="black", line_width=2,
                  annotation_text="Exit (crossing)", row=2, col=1)

    # Add trade markers and shaded regions
    colors = {"long": "green", "short": "red"}

    for i, trade in enumerate(trades):
        direction = "long" if trade.direction == 1 else "short"
        color = colors[direction]

        # Shaded region for trade duration (on both subplots)
        fig.add_vrect(
            x0=trade.entry_date, x1=trade.exit_date,
            fillcolor=color, opacity=0.15,
            line_width=0,
            row=1, col=1,
        )
        fig.add_vrect(
            x0=trade.entry_date, x1=trade.exit_date,
            fillcolor=color, opacity=0.15,
            line_width=0,
            row=2, col=1,
        )

        # Entry marker
        fig.add_trace(
            go.Scatter(
                x=[trade.entry_date],
                y=[distance.loc[trade.entry_date] if trade.entry_date in distance.index else trade.entry_distance],
                mode="markers",
                marker=dict(symbol="triangle-up" if direction == "long" else "triangle-down",
                           size=12, color=color),
                name=f"Entry #{i+1}" if i == 0 else None,
                showlegend=(i == 0),
                hovertemplate=f"Entry #{i+1}<br>Date: %{{x}}<br>Distance: {trade.entry_distance:.2f}σ<br>Direction: {direction}<extra></extra>",
            ),
            row=2, col=1,
        )

        # Exit marker
        fig.add_trace(
            go.Scatter(
                x=[trade.exit_date],
                y=[distance.loc[trade.exit_date] if trade.exit_date in distance.index else trade.exit_distance],
                mode="markers",
                marker=dict(symbol="x", size=10, color=color),
                name=f"Exit #{i+1}" if i == 0 else None,
                showlegend=False,
                hovertemplate=f"Exit #{i+1}<br>Date: %{{x}}<br>Distance: {trade.exit_distance:.2f}σ<br>P&L: ${trade.pnl:.2f}<br>Reason: {trade.exit_reason}<extra></extra>",
            ),
            row=2, col=1,
        )

    # Calculate summary stats
    total_trades = len(trades)
    total_pnl = sum(t.pnl for t in trades)
    win_rate = len([t for t in trades if t.pnl > 0]) / total_trades if total_trades > 0 else 0

    fig.update_layout(
        title=f"{sym_a}/{sym_b} Pair Analysis | Trades: {total_trades} | P&L: ${total_pnl:,.2f} | Win Rate: {win_rate:.1%}",
        template="plotly_white",
        height=700,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_yaxes(title_text="Normalized Price (base 100)", row=1, col=1)
    fig.update_yaxes(title_text="Distance (σ)", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)

    return fig


def generate_pair_report(
    close_prices: pd.DataFrame,
    pair: tuple[str, str],
    trades: list[Trade],
    distance: pd.Series,
    config: dict,
) -> dict:
    """
    Generate a detailed report for a single pair.

    Args:
        close_prices: DataFrame with close prices
        pair: Tuple of (symbol_a, symbol_b)
        trades: List of trades for this pair
        distance: Distance series (spread / formation_std)
        config: Backtest configuration dict

    Returns:
        Dictionary with pair statistics and figure
    """
    sym_a, sym_b = pair

    # Calculate pair-level stats
    total_trades = len(trades)
    if total_trades > 0:
        winning_trades = [t for t in trades if t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl <= 0]

        total_pnl = sum(t.pnl for t in trades)
        win_rate = len(winning_trades) / total_trades
        avg_win = sum(t.pnl for t in winning_trades) / len(winning_trades) if winning_trades else 0
        avg_loss = sum(t.pnl for t in losing_trades) / len(losing_trades) if losing_trades else 0
        avg_holding = sum(t.holding_days for t in trades) / total_trades

        long_trades = [t for t in trades if t.direction == 1]
        short_trades = [t for t in trades if t.direction == -1]
    else:
        total_pnl = win_rate = avg_win = avg_loss = avg_holding = 0
        long_trades = short_trades = []

    # Create the visualization
    fig = plot_pair_analysis(
        close_prices, pair, trades, distance,
        entry_threshold=config.get("entry_threshold", 2.0),
    )

    # Create trades table
    trades_data = []
    for i, t in enumerate(trades, 1):
        trades_data.append({
            "#": i,
            "Entry": t.entry_date.strftime("%Y-%m-%d"),
            "Exit": t.exit_date.strftime("%Y-%m-%d"),
            "Direction": "Long" if t.direction == 1 else "Short",
            "Days": t.holding_days,
            "Entry σ": f"{t.entry_distance:.2f}",
            "Exit σ": f"{t.exit_distance:.2f}",
            "P&L": f"${t.pnl:.2f}",
            "Return": f"{t.pnl_pct:.2%}",
            "Exit Reason": t.exit_reason,
        })

    return {
        "pair": f"{sym_a}/{sym_b}",
        "total_trades": total_trades,
        "total_pnl": total_pnl,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "avg_holding_days": avg_holding,
        "long_trades": len(long_trades),
        "short_trades": len(short_trades),
        "figure": fig,
        "trades_table": pd.DataFrame(trades_data) if trades_data else pd.DataFrame(),
    }


def print_pair_report(report: dict) -> None:
    """Print a formatted pair report."""
    print("=" * 70)
    print(f"PAIR ANALYSIS: {report['pair']}")
    print("=" * 70)
    print(f"\nTotal Trades:      {report['total_trades']}")
    print(f"Total P&L:         ${report['total_pnl']:,.2f}")
    print(f"Win Rate:          {report['win_rate']:.1%}")
    print(f"Avg Win:           ${report['avg_win']:,.2f}")
    print(f"Avg Loss:          ${report['avg_loss']:,.2f}")
    print(f"Avg Holding Days:  {report['avg_holding_days']:.1f}")
    print(f"Long Trades:       {report['long_trades']}")
    print(f"Short Trades:      {report['short_trades']}")
    print("\nTrades:")
    print("-" * 70)
    if not report['trades_table'].empty:
        print(report['trades_table'].to_string(index=False))
    else:
        print("No trades for this pair.")
    print("=" * 70)


def plot_zscore_series(
    zscore: pd.Series,
    entry_threshold: float = 2.0,
    exit_threshold: float = 0.5,
    title: str = "Z-Score Series",
) -> go.Figure:
    """
    Plot z-score series with threshold lines.

    Args:
        zscore: Z-score series
        entry_threshold: Entry threshold value
        exit_threshold: Exit threshold value
        title: Chart title

    Returns:
        Plotly Figure object
    """
    fig = go.Figure()

    # Z-score line
    fig.add_trace(
        go.Scatter(
            x=zscore.index, y=zscore.values,
            mode="lines", name="Z-Score",
            line=dict(color="purple", width=2),
        )
    )

    # Fill areas
    fig.add_hrect(
        y0=entry_threshold, y1=10,
        fillcolor="red", opacity=0.1,
        line_width=0,
        annotation_text="Short Entry Zone",
    )
    fig.add_hrect(
        y0=-10, y1=-entry_threshold,
        fillcolor="green", opacity=0.1,
        line_width=0,
        annotation_text="Long Entry Zone",
    )

    # Threshold lines
    fig.add_hline(y=entry_threshold, line_dash="dash", line_color="red",
                  annotation_text=f"Entry ({entry_threshold}σ)")
    fig.add_hline(y=-entry_threshold, line_dash="dash", line_color="green",
                  annotation_text=f"Entry (-{entry_threshold}σ)")
    fig.add_hline(y=exit_threshold, line_dash="dot", line_color="gray",
                  annotation_text=f"Exit ({exit_threshold}σ)")
    fig.add_hline(y=-exit_threshold, line_dash="dot", line_color="gray")
    fig.add_hline(y=0, line_color="black")

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Z-Score",
        template="plotly_white",
        hovermode="x unified",
    )

    return fig


# =============================================================================
# Trade Analysis Functions (for ggr_trade_analysis.ipynb)
# =============================================================================


def run_parameter_grid(
    close_prices: pd.DataFrame,
    open_prices: pd.DataFrame,
    pairs: list[tuple[str, str]],
    entry_thresholds: list[float] | None = None,
    formation_days_list: list[int] | None = None,
    max_holding_days: int = 126,
    capital_per_trade: float = 10000.0,
    commission: float = 0.001,
) -> pd.DataFrame:
    """
    Run backtest across parameter grid and return performance metrics.

    Args:
        close_prices: Full close prices DataFrame (will be sliced for formation/trading)
        open_prices: Full open prices DataFrame
        pairs: List of pairs to trade
        entry_thresholds: Entry thresholds to test (default: [1.5, 2.0, 2.5, 3.0])
        formation_days_list: Formation period lengths to test (default: [126, 189, 252])
        max_holding_days: Max holding days for config
        capital_per_trade: Capital per trade for config
        commission: Commission rate for config

    Returns:
        DataFrame with columns: entry_threshold, formation_days, sharpe_ratio,
                               total_return, win_rate, num_trades
    """
    from .backtest import run_backtest, BacktestConfig, combine_results

    if entry_thresholds is None:
        entry_thresholds = [1.5, 2.0, 2.5, 3.0]
    if formation_days_list is None:
        formation_days_list = [126, 189, 252]

    results = []
    total_combinations = len(entry_thresholds) * len(formation_days_list)

    for i, formation_days in enumerate(formation_days_list):
        # Split data: formation period is the first `formation_days` days
        # Trading period is the remainder
        formation_close = close_prices.iloc[:formation_days]
        trading_close = close_prices.iloc[formation_days:]
        trading_open = open_prices.iloc[formation_days:]

        if len(trading_close) < 20:
            print(f"Skipping formation_days={formation_days}: insufficient trading data")
            continue

        for j, entry_threshold in enumerate(entry_thresholds):
            combo_num = i * len(entry_thresholds) + j + 1
            print(f"Running {combo_num}/{total_combinations}: entry_threshold={entry_threshold}, formation_days={formation_days}")

            config = BacktestConfig(
                entry_threshold=entry_threshold,
                max_holding_days=max_holding_days,
                capital_per_trade=capital_per_trade,
                commission=commission,
            )

            try:
                backtest_results = run_backtest(
                    formation_close=formation_close,
                    trading_close=trading_close,
                    trading_open=trading_open,
                    pairs=pairs,
                    config=config,
                )

                all_trades, combined_equity = combine_results(
                    backtest_results,
                    initial_capital=capital_per_trade * len(pairs),
                )

                metrics = calculate_metrics(all_trades, combined_equity)

                results.append({
                    "entry_threshold": entry_threshold,
                    "formation_days": formation_days,
                    "sharpe_ratio": metrics["sharpe_ratio"],
                    "total_return": metrics["total_return"],
                    "total_return_pct": metrics["total_return_pct"],
                    "win_rate": metrics["win_rate"],
                    "num_trades": metrics["total_trades"],
                    "max_drawdown_pct": metrics["max_drawdown_pct"],
                    "profit_factor": metrics["profit_factor"],
                })
            except Exception as e:
                print(f"Error with entry_threshold={entry_threshold}, formation_days={formation_days}: {e}")
                results.append({
                    "entry_threshold": entry_threshold,
                    "formation_days": formation_days,
                    "sharpe_ratio": 0,
                    "total_return": 0,
                    "total_return_pct": 0,
                    "win_rate": 0,
                    "num_trades": 0,
                    "max_drawdown_pct": 0,
                    "profit_factor": 0,
                })

    return pd.DataFrame(results)


def plot_parameter_heatmap(
    grid_results: pd.DataFrame,
    metric: str = "sharpe_ratio",
    title: str | None = None,
) -> go.Figure:
    """
    Plot parameter sensitivity heatmap.

    Args:
        grid_results: DataFrame from run_parameter_grid()
        metric: Column to use for coloring (sharpe_ratio, total_return_pct, win_rate)
        title: Chart title (auto-generated if None)

    Returns:
        Plotly Figure with heatmap
    """
    # Pivot to matrix form
    pivot = grid_results.pivot(
        index="formation_days",
        columns="entry_threshold",
        values=metric,
    )

    # Create annotations with values
    annotations = []
    for i, row in enumerate(pivot.index):
        for j, col in enumerate(pivot.columns):
            value = pivot.loc[row, col]
            if metric == "sharpe_ratio":
                text = f"{value:.2f}"
            elif metric in ["total_return_pct", "win_rate", "max_drawdown_pct"]:
                text = f"{value:.1%}"
            else:
                text = f"{value:.0f}"
            annotations.append(
                dict(
                    x=col, y=row,
                    text=text,
                    showarrow=False,
                    font=dict(color="white" if abs(value) > 0.5 else "black"),
                )
            )

    metric_labels = {
        "sharpe_ratio": "Sharpe Ratio",
        "total_return_pct": "Total Return %",
        "win_rate": "Win Rate",
        "num_trades": "Number of Trades",
        "max_drawdown_pct": "Max Drawdown %",
        "profit_factor": "Profit Factor",
    }

    if title is None:
        title = f"Parameter Sensitivity: {metric_labels.get(metric, metric)}"

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index,
            colorscale="RdYlGn",
            hoverongaps=False,
            colorbar=dict(title=metric_labels.get(metric, metric)),
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Entry Threshold (σ)",
        yaxis_title="Formation Period (days)",
        template="plotly_white",
        annotations=annotations,
        height=400,
        width=600,
    )

    return fig


def calculate_trading_correlation(
    trading_close: pd.DataFrame,
    pair: tuple[str, str],
) -> float:
    """
    Calculate correlation between pair's daily returns during trading period.

    Args:
        trading_close: Trading period close prices
        pair: Tuple of (symbol_a, symbol_b)

    Returns:
        Pearson correlation coefficient of daily returns
    """
    sym_a, sym_b = pair
    returns_a = trading_close[sym_a].pct_change().dropna()
    returns_b = trading_close[sym_b].pct_change().dropna()
    return returns_a.corr(returns_b)


def plot_correlation_scatter(
    trades_df: pd.DataFrame,
    formation_correlations: dict[tuple[str, str], float],
    trading_correlations: dict[tuple[str, str], float],
    title: str = "Formation vs Trading Correlation by Win/Loss",
) -> go.Figure:
    """
    Scatter plot of formation vs trading correlation colored by trade outcome.

    Args:
        trades_df: DataFrame of trades with 'pair' and 'pnl' columns
        formation_correlations: Dict mapping pair -> formation correlation
        trading_correlations: Dict mapping pair -> trading correlation
        title: Chart title

    Returns:
        Plotly scatter figure
    """
    # Add correlations to trades
    data = []
    for _, trade in trades_df.iterrows():
        # Parse pair string back to tuple
        pair_str = trade["pair"]
        pair_parts = pair_str.split("/")
        pair = (pair_parts[0], pair_parts[1])

        if pair in formation_correlations and pair in trading_correlations:
            data.append({
                "pair": pair_str,
                "formation_corr": formation_correlations[pair],
                "trading_corr": trading_correlations[pair],
                "pnl": trade["pnl"],
                "outcome": "Winner" if trade["pnl"] > 0 else "Loser",
            })

    plot_df = pd.DataFrame(data)

    if plot_df.empty:
        fig = go.Figure()
        fig.update_layout(title="No data available")
        return fig

    fig = go.Figure()

    # Winners
    winners = plot_df[plot_df["outcome"] == "Winner"]
    if not winners.empty:
        fig.add_trace(
            go.Scatter(
                x=winners["formation_corr"],
                y=winners["trading_corr"],
                mode="markers",
                name="Winners",
                marker=dict(color="green", size=10, opacity=0.7),
                text=winners["pair"],
                hovertemplate="Pair: %{text}<br>Formation Corr: %{x:.3f}<br>Trading Corr: %{y:.3f}<br>P&L: $%{customdata:.2f}<extra></extra>",
                customdata=winners["pnl"],
            )
        )

    # Losers
    losers = plot_df[plot_df["outcome"] == "Loser"]
    if not losers.empty:
        fig.add_trace(
            go.Scatter(
                x=losers["formation_corr"],
                y=losers["trading_corr"],
                mode="markers",
                name="Losers",
                marker=dict(color="red", size=10, opacity=0.7),
                text=losers["pair"],
                hovertemplate="Pair: %{text}<br>Formation Corr: %{x:.3f}<br>Trading Corr: %{y:.3f}<br>P&L: $%{customdata:.2f}<extra></extra>",
                customdata=losers["pnl"],
            )
        )

    # Diagonal line (correlation unchanged)
    fig.add_trace(
        go.Scatter(
            x=[-1, 1], y=[-1, 1],
            mode="lines",
            name="Correlation Unchanged",
            line=dict(color="gray", dash="dash"),
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Formation Period Correlation",
        yaxis_title="Trading Period Correlation",
        template="plotly_white",
        height=500,
        width=600,
        xaxis=dict(range=[-0.2, 1.1]),
        yaxis=dict(range=[-0.2, 1.1]),
    )

    return fig


def plot_mae_analysis(
    trades: list[Trade],
    title: str = "Maximum Adverse Excursion vs Final P&L",
) -> go.Figure:
    """
    Scatter plot of max adverse spread vs final P&L.

    Args:
        trades: List of Trade objects (must have max_adverse_spread)
        title: Chart title

    Returns:
        Plotly scatter figure
    """
    if not trades:
        fig = go.Figure()
        fig.update_layout(title="No trades to display")
        return fig

    # Extract data
    data = []
    for t in trades:
        # Use absolute MAE for visualization (distance from 0)
        mae = abs(t.max_adverse_spread)
        data.append({
            "mae": mae,
            "pnl": t.pnl,
            "outcome": "Winner" if t.pnl > 0 else "Loser",
            "pair": f"{t.pair[0]}/{t.pair[1]}",
            "entry_distance": abs(t.entry_distance),
        })

    plot_df = pd.DataFrame(data)

    fig = go.Figure()

    # Winners
    winners = plot_df[plot_df["outcome"] == "Winner"]
    if not winners.empty:
        fig.add_trace(
            go.Scatter(
                x=winners["mae"],
                y=winners["pnl"],
                mode="markers",
                name="Winners",
                marker=dict(color="green", size=10, opacity=0.7),
                text=winners["pair"],
                hovertemplate="Pair: %{text}<br>Max Adverse: %{x:.2f}σ<br>P&L: $%{y:.2f}<extra></extra>",
            )
        )

    # Losers
    losers = plot_df[plot_df["outcome"] == "Loser"]
    if not losers.empty:
        fig.add_trace(
            go.Scatter(
                x=losers["mae"],
                y=losers["pnl"],
                mode="markers",
                name="Losers",
                marker=dict(color="red", size=10, opacity=0.7),
                text=losers["pair"],
                hovertemplate="Pair: %{text}<br>Max Adverse: %{x:.2f}σ<br>P&L: $%{y:.2f}<extra></extra>",
            )
        )

    # Zero line
    fig.add_hline(y=0, line_color="black", line_width=1)

    # Add trendline annotation area markers
    fig.add_vrect(
        x0=4, x1=plot_df["mae"].max() + 0.5,
        fillcolor="red", opacity=0.1,
        line_width=0,
        annotation_text="High MAE Zone (falling knives)",
        annotation_position="top left",
    )

    fig.update_layout(
        title=title,
        xaxis_title="Maximum Adverse Spread (|σ|)",
        yaxis_title="Final P&L ($)",
        template="plotly_white",
        height=500,
        width=700,
    )

    return fig


def plot_duration_histogram(
    trades: list[Trade],
    title: str = "Holding Duration: Winners vs Losers",
) -> go.Figure:
    """
    Overlayed histograms of holding_days for winners vs losers.

    Args:
        trades: List of Trade objects
        title: Chart title

    Returns:
        Plotly figure with overlayed histograms
    """
    if not trades:
        fig = go.Figure()
        fig.update_layout(title="No trades to display")
        return fig

    winners = [t.holding_days for t in trades if t.pnl > 0]
    losers = [t.holding_days for t in trades if t.pnl <= 0]

    fig = go.Figure()

    if winners:
        fig.add_trace(
            go.Histogram(
                x=winners,
                name=f"Winners (n={len(winners)})",
                opacity=0.6,
                marker_color="green",
                nbinsx=20,
            )
        )

    if losers:
        fig.add_trace(
            go.Histogram(
                x=losers,
                name=f"Losers (n={len(losers)})",
                opacity=0.6,
                marker_color="red",
                nbinsx=20,
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Holding Days",
        yaxis_title="Number of Trades",
        template="plotly_white",
        barmode="overlay",
        height=400,
        width=600,
    )

    # Add median lines
    if winners:
        fig.add_vline(x=np.median(winners), line_dash="dash", line_color="darkgreen",
                      annotation_text=f"Winner Median: {np.median(winners):.0f}d")
    if losers:
        fig.add_vline(x=np.median(losers), line_dash="dash", line_color="darkred",
                      annotation_text=f"Loser Median: {np.median(losers):.0f}d")

    return fig


def plot_ssd_by_outcome(
    trades_df: pd.DataFrame,
    pairs_ranking: pd.DataFrame,
    title: str = "Formation SSD by Trade Outcome",
) -> go.Figure:
    """
    Box plot of SSD values grouped by win/loss.

    Args:
        trades_df: DataFrame of trades
        pairs_ranking: DataFrame from rank_all_pairs() with ssd column
        title: Chart title

    Returns:
        Plotly box plot figure
    """
    # Merge trades with SSD data
    trades_with_ssd = []

    for _, trade in trades_df.iterrows():
        pair_str = trade["pair"]
        pair_parts = pair_str.split("/")

        # Find SSD for this pair
        ssd_row = pairs_ranking[
            ((pairs_ranking["symbol_a"] == pair_parts[0]) & (pairs_ranking["symbol_b"] == pair_parts[1])) |
            ((pairs_ranking["symbol_a"] == pair_parts[1]) & (pairs_ranking["symbol_b"] == pair_parts[0]))
        ]

        if not ssd_row.empty:
            trades_with_ssd.append({
                "ssd": ssd_row["ssd"].values[0],
                "outcome": "Winner" if trade["pnl"] > 0 else "Loser",
                "pnl": trade["pnl"],
            })

    plot_df = pd.DataFrame(trades_with_ssd)

    if plot_df.empty:
        fig = go.Figure()
        fig.update_layout(title="No data available")
        return fig

    fig = go.Figure()

    # Winners box
    winners = plot_df[plot_df["outcome"] == "Winner"]
    if not winners.empty:
        fig.add_trace(
            go.Box(
                y=winners["ssd"],
                name=f"Winners (n={len(winners)})",
                marker_color="green",
                boxpoints="all",
                jitter=0.3,
                pointpos=-1.5,
            )
        )

    # Losers box
    losers = plot_df[plot_df["outcome"] == "Loser"]
    if not losers.empty:
        fig.add_trace(
            go.Box(
                y=losers["ssd"],
                name=f"Losers (n={len(losers)})",
                marker_color="red",
                boxpoints="all",
                jitter=0.3,
                pointpos=-1.5,
            )
        )

    fig.update_layout(
        title=title,
        yaxis_title="Formation Period SSD",
        template="plotly_white",
        height=450,
        width=500,
    )

    return fig


def plot_exit_reason_pie(
    trades: list[Trade],
    title: str = "Exit Reason Distribution",
) -> go.Figure:
    """
    Pie chart of exit reasons.

    Args:
        trades: List of Trade objects
        title: Chart title

    Returns:
        Plotly pie chart
    """
    if not trades:
        fig = go.Figure()
        fig.update_layout(title="No trades to display")
        return fig

    # Count exit reasons
    exit_counts = {}
    for t in trades:
        reason = t.exit_reason
        exit_counts[reason] = exit_counts.get(reason, 0) + 1

    labels = list(exit_counts.keys())
    values = list(exit_counts.values())

    # Color mapping
    color_map = {
        "crossing": "green",
        "max_holding": "orange",
        "end_of_data": "red",
    }
    colors = [color_map.get(l, "gray") for l in labels]

    # Calculate health percentage
    total = sum(values)
    crossing_pct = exit_counts.get("crossing", 0) / total * 100 if total > 0 else 0
    health_text = f"Strategy Health: {crossing_pct:.1f}% Converged"

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                marker_colors=colors,
                textinfo="label+percent",
                textposition="inside",
                hovertemplate="Exit Reason: %{label}<br>Count: %{value}<br>Percentage: %{percent}<extra></extra>",
            )
        ]
    )

    fig.update_layout(
        title=f"{title}<br><sup>{health_text}</sup>",
        template="plotly_white",
        height=400,
        width=500,
    )

    return fig


def plot_pnl_by_exit_reason(
    trades: list[Trade],
    title: str = "P&L Distribution by Exit Reason",
) -> go.Figure:
    """
    Violin/box plot of P&L grouped by exit_reason.

    Args:
        trades: List of Trade objects
        title: Chart title

    Returns:
        Plotly violin plot
    """
    if not trades:
        fig = go.Figure()
        fig.update_layout(title="No trades to display")
        return fig

    # Group trades by exit reason
    data = {}
    for t in trades:
        reason = t.exit_reason
        if reason not in data:
            data[reason] = []
        data[reason].append(t.pnl)

    # Color mapping
    color_map = {
        "crossing": "green",
        "max_holding": "orange",
        "end_of_data": "red",
    }

    fig = go.Figure()

    for reason, pnls in data.items():
        fig.add_trace(
            go.Violin(
                y=pnls,
                name=f"{reason} (n={len(pnls)})",
                box_visible=True,
                meanline_visible=True,
                line_color=color_map.get(reason, "gray"),
                fillcolor=color_map.get(reason, "gray"),
                opacity=0.6,
            )
        )

    # Zero line
    fig.add_hline(y=0, line_color="black", line_width=1)

    # Add summary statistics
    annotations = []
    for i, (reason, pnls) in enumerate(data.items()):
        mean_pnl = np.mean(pnls)
        annotations.append(
            dict(
                x=i, y=max(pnls) + 50,
                text=f"Avg: ${mean_pnl:.0f}",
                showarrow=False,
                font=dict(size=10),
            )
        )

    fig.update_layout(
        title=title,
        yaxis_title="P&L ($)",
        template="plotly_white",
        height=450,
        width=600,
        showlegend=True,
        annotations=annotations,
    )

    return fig


# =============================================================================
# Staggered Portfolio Analysis Functions
# =============================================================================


def calculate_staggered_metrics(
    result: "StaggeredResult",
    risk_free_rate: float = 0.02,
) -> dict[str, Any]:
    """
    Calculate performance metrics for staggered portfolio methodology.

    Args:
        result: StaggeredResult from run_staggered_backtest()
        risk_free_rate: Annual risk-free rate for Sharpe calculation

    Returns:
        Dictionary with performance metrics
    """
    from .staggered import StaggeredResult

    monthly_returns = result.monthly_returns.dropna()

    if len(monthly_returns) == 0:
        return {
            "total_months": 0,
            "annualized_return": 0,
            "annualized_volatility": 0,
            "sharpe_ratio": 0,
            "max_drawdown": 0,
            "total_portfolios": result.total_portfolios,
            "avg_active_portfolios": 0,
            "total_trades": 0,
            "avg_monthly_return": 0,
        }

    # Monthly statistics
    avg_monthly_return = monthly_returns.mean()
    monthly_std = monthly_returns.std()

    # Annualized metrics (12 months per year)
    annualized_return = (1 + avg_monthly_return) ** 12 - 1
    annualized_volatility = monthly_std * np.sqrt(12)

    # Sharpe ratio (annualized)
    monthly_rf = risk_free_rate / 12
    excess_returns = monthly_returns - monthly_rf
    if monthly_std > 0:
        sharpe = np.sqrt(12) * excess_returns.mean() / monthly_std
    else:
        sharpe = 0

    # Max drawdown from cumulative returns
    cumulative = (1 + monthly_returns).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    # Active portfolios statistics
    active_counts = result.active_portfolios_over_time.dropna()
    avg_active = active_counts.mean() if len(active_counts) > 0 else 0

    # Total trades
    total_trades = len(result.all_trades) if result.all_trades else 0

    # Win rate across all trades
    if result.all_trades:
        wins = [t for t in result.all_trades if t.pnl > 0]
        win_rate = len(wins) / total_trades if total_trades > 0 else 0
    else:
        win_rate = 0

    return {
        "total_months": len(monthly_returns),
        "avg_monthly_return": avg_monthly_return,
        "monthly_volatility": monthly_std,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
        "total_portfolios": result.total_portfolios,
        "avg_active_portfolios": avg_active,
        "total_trades": total_trades,
        "win_rate": win_rate,
    }


def print_staggered_metrics(metrics: dict[str, Any]) -> None:
    """Print formatted staggered backtest metrics."""
    print("=" * 60)
    print("STAGGERED PORTFOLIO BACKTEST RESULTS (GGR Methodology)")
    print("=" * 60)
    print(f"Total Months:           {metrics['total_months']}")
    print(f"Total Portfolios:       {metrics['total_portfolios']}")
    print(f"Avg Active Portfolios:  {metrics['avg_active_portfolios']:.1f}")
    print("-" * 60)
    print(f"Avg Monthly Return:     {metrics['avg_monthly_return']:.2%}")
    print(f"Monthly Volatility:     {metrics['monthly_volatility']:.2%}")
    print(f"Annualized Return:      {metrics['annualized_return']:.2%}")
    print(f"Annualized Volatility:  {metrics['annualized_volatility']:.2%}")
    print(f"Sharpe Ratio:           {metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown:           {metrics['max_drawdown']:.2%}")
    print("-" * 60)
    print(f"Total Trades:           {metrics['total_trades']}")
    print(f"Win Rate:               {metrics['win_rate']:.2%}")
    print("=" * 60)


def plot_staggered_returns(
    result: "StaggeredResult",
    title: str = "GGR Staggered Portfolio Performance",
) -> go.Figure:
    """
    Plot cumulative returns with active portfolio count overlay.

    Args:
        result: StaggeredResult from run_staggered_backtest()
        title: Chart title

    Returns:
        Plotly Figure object
    """
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.5, 0.25, 0.25],
        vertical_spacing=0.05,
        subplot_titles=(
            "Cumulative Returns",
            "Monthly Returns",
            "Active Portfolios",
        ),
    )

    # Row 1: Cumulative returns
    cumulative = result.cumulative_returns.dropna()
    fig.add_trace(
        go.Scatter(
            x=cumulative.index,
            y=cumulative.values * 100,  # Convert to percentage
            mode="lines",
            name="Cumulative Return",
            line=dict(color="blue", width=2),
        ),
        row=1, col=1,
    )

    # Add zero line
    fig.add_hline(y=0, line_color="black", line_width=1, row=1, col=1)

    # Row 2: Monthly returns bar chart
    monthly = result.monthly_returns.dropna()
    colors = ["green" if r >= 0 else "red" for r in monthly.values]
    fig.add_trace(
        go.Bar(
            x=monthly.index,
            y=monthly.values * 100,  # Convert to percentage
            name="Monthly Return",
            marker_color=colors,
        ),
        row=2, col=1,
    )
    fig.add_hline(y=0, line_color="black", line_width=1, row=2, col=1)

    # Row 3: Active portfolio count
    active = result.active_portfolios_over_time.dropna()
    fig.add_trace(
        go.Scatter(
            x=active.index,
            y=active.values,
            mode="lines+markers",
            name="Active Portfolios",
            line=dict(color="purple", width=2),
            marker=dict(size=4),
        ),
        row=3, col=1,
    )

    # Add target line at 6 portfolios
    fig.add_hline(
        y=6, line_dash="dash", line_color="gray",
        annotation_text="Target: 6 portfolios",
        row=3, col=1,
    )

    fig.update_yaxes(title_text="Return (%)", row=1, col=1)
    fig.update_yaxes(title_text="Return (%)", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=3, col=1)
    fig.update_xaxes(title_text="Date", row=3, col=1)

    fig.update_layout(
        title=title,
        template="plotly_white",
        height=700,
        hovermode="x unified",
        showlegend=False,
    )

    return fig


def plot_portfolio_timeline(
    result: "StaggeredResult",
    max_cycles: int = 20,
    title: str = "Portfolio Cycle Timeline",
) -> go.Figure:
    """
    Gantt-style chart showing portfolio formation/trading periods.

    Args:
        result: StaggeredResult from run_staggered_backtest()
        max_cycles: Maximum number of cycles to display
        title: Chart title

    Returns:
        Plotly Figure object
    """
    cycles = result.cycles[:max_cycles]

    fig = go.Figure()

    for i, cycle in enumerate(cycles):
        # Formation period bar
        fig.add_trace(
            go.Scatter(
                x=[cycle.formation_start, cycle.formation_end],
                y=[i, i],
                mode="lines",
                line=dict(color="lightblue", width=20),
                name="Formation" if i == 0 else None,
                showlegend=(i == 0),
                hovertemplate=(
                    f"Cycle {cycle.cycle_id}<br>"
                    f"Formation: {cycle.formation_start.strftime('%Y-%m-%d')} to {cycle.formation_end.strftime('%Y-%m-%d')}<br>"
                    f"Pairs: {len(cycle.pairs) if cycle.pairs else 0}"
                    "<extra></extra>"
                ),
            )
        )

        # Trading period bar
        fig.add_trace(
            go.Scatter(
                x=[cycle.trading_start, cycle.trading_end],
                y=[i, i],
                mode="lines",
                line=dict(color="green", width=20),
                name="Trading" if i == 0 else None,
                showlegend=(i == 0),
                hovertemplate=(
                    f"Cycle {cycle.cycle_id}<br>"
                    f"Trading: {cycle.trading_start.strftime('%Y-%m-%d')} to {cycle.trading_end.strftime('%Y-%m-%d')}<br>"
                    f"Pairs: {len(cycle.pairs) if cycle.pairs else 0}"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Portfolio Cycle",
        template="plotly_white",
        height=max(400, len(cycles) * 30),
        yaxis=dict(
            tickmode="array",
            tickvals=list(range(len(cycles))),
            ticktext=[f"Cycle {c.cycle_id}" for c in cycles],
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig
