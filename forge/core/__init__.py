"""Forge core — framework-agnostic protocols, types, and configuration.

This package has ZERO external framework dependencies (no areal, no torch, etc.).
Only stdlib + typing are allowed here.
"""

from forge.core.protocols import InferenceBridge, RewardBackend, TrainBackend

__all__ = [
    "InferenceBridge",
    "RewardBackend",
    "TrainBackend",
]
