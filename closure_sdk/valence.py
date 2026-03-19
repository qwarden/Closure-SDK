"""The chain — translates ball geometry into labeled color channels.

The ball holds full-color quaternions. This module is the prism that
splits them into human-readable channels using the Hopf fibration.

Two operations:

    expose(element)              — any point on the ball → Valence.
                                   Works at every step of composition,
                                   not just at incident time.

    expose_incident(inc, drift)  — a localized incident → IncidentValence.
                                   Same channels, plus structural labels:
                                   which positions, what payload, what broke.

The 3+1 channels:

    W  (scalar, S¹ fiber)   — the coherence axis. Has or hasn't.
                               Parametrized by e (self-evident, axiom 2).
    R, G, B  (vector, S²)   — the interaction axes. Where and how far.
                               Parametrized by π (observed, axiom 1).

Two incident types, same algebraic object, different broken axis:

    Missing record      — W broke (existence). One position is None.
    Content mismatch    — RGB broke (position). Both positions present, different.

This module translates. It never decides which side is correct.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .hopf import hopf_decompose
from .canon import IncidentReport


def _to_quaternion(element: NDArray[np.float64]) -> NDArray[np.float64]:
    """Pad a closure element to a full quaternion for Hopf decomposition."""
    q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    n = min(len(element), 4)
    q[:n] = element[:n]
    return q


@dataclass(frozen=True)
class Valence:
    """The color channels of any point on the ball.

    sigma  — how far from identity (the magnitude dial).
    base   — direction on S² as (R, G, B). What kind of divergence.
    phase  — angle on S¹ as W. Which fiber.
    """

    sigma: float
    base: tuple[float, float, float]  # S² direction → (R, G, B)
    phase: float  # S¹ fiber → W


@dataclass(frozen=True)
class IncidentValence:
    """A fully labeled incident in color channels.

    Carries everything Valence has (sigma, base, phase) plus the
    structural context of the incident:

    position_a    — where in stream A (None if the record is absent from A).
    position_b    — where in stream B (None if the record is absent from B).
    payload       — the actual record bytes.
    axis          — "existence" (W broke, record missing) or
                    "position" (RGB broke, record moved).
    displacement  — how many positions apart (None if missing).
    """

    # From the algebra
    position_a: int | None
    position_b: int | None
    payload: bytes

    # Derived labels
    axis: str  # "existence" | "position"
    displacement: int | None

    # Hopf channels
    sigma: float
    base: tuple[float, float, float]  # S² direction → (R, G, B)
    phase: float  # S¹ fiber → W


def expose(element: NDArray[np.float64]) -> Valence:
    """The prism. Takes any point on the ball and splits it into color
    channels via Hopf. Call this after every ingest to watch the
    channels evolve in real time, or on any diff to see what kind of
    divergence it is.
    """
    q = _to_quaternion(element)
    hopf = hopf_decompose(q)

    return Valence(
        sigma=float(hopf["sigma"]),
        base=tuple(float(x) for x in hopf["base"]),
        phase=float(hopf["phase"]),
    )


def expose_incident(incident: IncidentReport, drift_element: NDArray[np.float64]) -> IncidentValence:
    """The labeler. Takes a localized incident and the drift quaternion,
    splits the quaternion into channels, and attaches structural labels:
    which axis broke, which positions, how far apart. This is the
    endpoint of the chain — what the application layer reads.
    """
    q = _to_quaternion(drift_element)
    hopf = hopf_decompose(q)

    if incident.source_index is not None and incident.target_index is not None:
        axis = "position"
        displacement = abs(incident.source_index - incident.target_index)
    else:
        axis = "existence"
        displacement = None

    return IncidentValence(
        position_a=incident.source_index,
        position_b=incident.target_index,
        payload=incident.record,
        axis=axis,
        displacement=displacement,
        sigma=float(hopf["sigma"]),
        base=tuple(float(x) for x in hopf["base"]),
        phase=float(hopf["phase"]),
    )
