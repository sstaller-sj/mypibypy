"""mypibypy -- piFISH split-probe design.

Design split-probe (P1L/P1R) primary probes for piFISH targets and emit an order
workbook that drops into an existing validated P2/P3/P4 amplifier pool.
"""
from .design import (
    Candidate,
    ProbePair,
    arm_separation,
    arms_meet_at_junction,
    build_fusion_pairs,
    build_pair,
    check_partner_specificity,
    fusion_footprint,
    scan,
)
from .loci import FusionJunction, Locus, load_fusions, load_loci
from .outcomes import DesignFacts, ImagingResult, OutcomeStore
from .sequence import GenomicInterval, fetch_sequence, revcomp
from .specificity import SpecificityReport, resolve_backend, screen_pair
from .systems import AmplifierSystem, get_system, is_example, loaded_from, systems

__version__ = "0.2.0"

# `SYSTEM_1` / `SYSTEM_2` are deliberately not re-exported here: importing them
# would load the systems file at import time, and `import mypibypy` must work
# before any systems are configured. Reach them via `systems()` or
# `get_system("cy5")`, or as lazy attributes of `mypibypy.systems`.
__all__ = [
    "AmplifierSystem", "Candidate", "DesignFacts", "FusionJunction",
    "GenomicInterval", "ImagingResult", "Locus", "OutcomeStore", "ProbePair",
    "SpecificityReport",
    "arm_separation", "arms_meet_at_junction", "build_fusion_pairs",
    "build_pair",
    "check_partner_specificity", "fetch_sequence", "fusion_footprint",
    "get_system", "is_example", "load_fusions", "load_loci", "loaded_from",
    "resolve_backend", "revcomp", "scan", "screen_pair", "systems",
]
