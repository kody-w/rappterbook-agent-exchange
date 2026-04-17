"""Mars-100: A 100-Martian-year recursive colony simulation with 10 agent-colonists."""
from __future__ import annotations

from src.mars100.engine import Mars100Engine, SimulationResult, YearResult
from src.mars100.colonist import Colonist, create_founding_ten
from src.mars100.analysis import full_analysis

__all__ = [
    "Mars100Engine", "SimulationResult", "YearResult",
    "Colonist", "create_founding_ten", "full_analysis",
]
