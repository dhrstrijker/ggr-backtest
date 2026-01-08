"""Dashboard layout modules."""

from .base import create_base_layout
from .page1_fund_overview import create_fund_overview_layout
from .page3_pair_inspector import create_pair_inspector_layout
from .page4_pairs_summary import create_pairs_summary_layout

__all__ = [
    "create_base_layout",
    "create_fund_overview_layout",
    "create_pair_inspector_layout",
    "create_pairs_summary_layout",
]
