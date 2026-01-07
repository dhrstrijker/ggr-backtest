"""Dashboard layout modules."""

from .base import create_base_layout
from .page1_fund_overview import create_fund_overview_layout
from .page2_live_monitor import create_live_monitor_layout
from .page3_pair_inspector import create_pair_inspector_layout

__all__ = [
    "create_base_layout",
    "create_fund_overview_layout",
    "create_live_monitor_layout",
    "create_pair_inspector_layout",
]
