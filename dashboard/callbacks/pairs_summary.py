"""Callbacks for Page 3: Pairs Summary (Staggered Methodology)."""

from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc

from ..components.metrics_card import format_currency, format_percentage


def register_pairs_summary_callbacks(app, data_store):
    """Register Pairs Summary page callbacks."""

    @app.callback(
        Output("pairs-summary-table", "children"),
        Input("wait-mode-store", "data"),
    )
    def update_pairs_table(wait_mode):
        """Build the pairs summary table showing all pairs across all cycles."""
        # Get all unique pairs from all cycles for the current wait mode
        all_pairs = data_store.get_all_pairs(wait_mode)

        if not all_pairs:
            return html.P("No pairs found in backtest results", className="text-muted")

        # Build stats for each pair for the current wait mode
        pairs_data = []
        for pair in all_pairs:
            trades = data_store.get_trades_for_pair(pair, wait_mode)
            cycles = data_store.get_cycles_for_pair(pair, wait_mode)

            total_pnl = sum(t.pnl for t in trades)
            num_trades = len(trades)
            win_count = sum(1 for t in trades if t.pnl > 0)
            loss_count = sum(1 for t in trades if t.pnl < 0)
            # Win rate excludes break-even trades (matching src/analysis.py)
            decided_trades = win_count + loss_count
            win_rate = (win_count / decided_trades * 100) if decided_trades > 0 else 0
            avg_pnl = total_pnl / num_trades if num_trades > 0 else 0
            avg_holding = sum(t.holding_days for t in trades) / num_trades if num_trades > 0 else 0

            pairs_data.append({
                "pair": pair,
                "cycles_traded": len(cycles),
                "num_trades": num_trades,
                "win_rate": win_rate,
                "total_pnl": total_pnl,
                "avg_pnl": avg_pnl,
                "avg_holding": avg_holding,
            })

        # Sort by total P&L descending
        pairs_data.sort(key=lambda x: x["total_pnl"], reverse=True)

        # Build table rows
        rows = []
        for rank, data in enumerate(pairs_data, 1):
            sym_a, sym_b = data["pair"]
            pair_key = f"{sym_a}_{sym_b}"

            pnl_color = "text-success" if data["total_pnl"] >= 0 else "text-danger"
            avg_pnl_color = "text-success" if data["avg_pnl"] >= 0 else "text-danger"

            rows.append(
                html.Tr(
                    [
                        html.Td(str(rank)),
                        html.Td(html.Strong(f"{sym_a} / {sym_b}")),
                        html.Td(str(data["cycles_traded"]), className="text-center"),
                        html.Td(str(data["num_trades"]), className="text-center"),
                        html.Td(f"{data['win_rate']:.1f}%", className="text-center"),
                        html.Td(f"{data['avg_holding']:.0f}", className="text-center"),
                        html.Td(
                            html.Span(format_currency(data["total_pnl"]), className=pnl_color),
                            className="text-end",
                        ),
                        html.Td(
                            html.Span(format_currency(data["avg_pnl"]), className=avg_pnl_color),
                            className="text-end",
                        ),
                        html.Td(
                            dcc.Link(
                                dbc.Button(
                                    "Inspect",
                                    color="primary",
                                    size="sm",
                                    className="py-0",
                                ),
                                href=f"/pairs?pair={pair_key}",
                            ),
                            className="text-center",
                        ),
                    ],
                    className="align-middle",
                )
            )

        return dbc.Table(
            [
                html.Thead(
                    html.Tr([
                        html.Th("#", style={"width": "40px"}),
                        html.Th("Pair"),
                        html.Th("Cycles", className="text-center", title="Number of portfolio cycles this pair was selected"),
                        html.Th("Trades", className="text-center"),
                        html.Th("Win Rate", className="text-center"),
                        html.Th("Avg Days", className="text-center", title="Average holding days"),
                        html.Th("Total P&L", className="text-end"),
                        html.Th("Avg P&L", className="text-end"),
                        html.Th("Action", className="text-center", style={"width": "80px"}),
                    ])
                ),
                html.Tbody(rows),
            ],
            striped=True,
            bordered=True,
            hover=True,
            responsive=True,
        )
