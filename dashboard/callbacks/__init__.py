"""Dashboard callback modules."""

from .navigation import register_navigation_callbacks
from .fund_overview import register_fund_overview_callbacks
from .live_monitor import register_live_monitor_callbacks
from .pair_inspector import register_pair_inspector_callbacks
from .pairs_summary import register_pairs_summary_callbacks


def register_all_callbacks(app, data_store):
    """Register all callbacks for the dashboard."""
    register_navigation_callbacks(app, data_store)
    register_fund_overview_callbacks(app, data_store)
    register_live_monitor_callbacks(app, data_store)
    register_pair_inspector_callbacks(app, data_store)
    register_pairs_summary_callbacks(app, data_store)


__all__ = [
    "register_all_callbacks",
    "register_navigation_callbacks",
    "register_fund_overview_callbacks",
    "register_live_monitor_callbacks",
    "register_pair_inspector_callbacks",
    "register_pairs_summary_callbacks",
]
