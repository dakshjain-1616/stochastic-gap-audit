"""Stochastic-Gap-Audit: Pre-deployment reliability scoring framework."""

__version__ = "2.0.0"
__author__  = "NEO"

from .simulator     import StochasticGapSimulator, SimulationReport, PromptResult
from .reporter      import AuditReporter
from .comparator    import ModelComparator, ComparisonReport
from .history       import AuditHistory, HistoryEntry
from .html_reporter import HTMLReporter

__all__ = [
    "StochasticGapSimulator",
    "SimulationReport",
    "PromptResult",
    "AuditReporter",
    "ModelComparator",
    "ComparisonReport",
    "AuditHistory",
    "HistoryEntry",
    "HTMLReporter",
]
