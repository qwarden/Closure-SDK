"""The prism — Hopf fibration, internal to the SDK.

This is the math that splits a quaternion into its color channels.
Any point on S³ (the ball) decomposes into three things:

    sigma  — how far from identity. The total magnitude.
    base   — a direction on S² (a regular sphere). Three numbers: R, G, B.
             This is the interaction part (axiom 1, vector, π-parametrized).
    phase  — an angle on S¹ (a circle). One number: W.
             This is the coherence part (axiom 2, scalar, e-parametrized).

The Hopf fibration is what makes S³ special among spheres — it has a
canonical way to split into a base direction and a fiber angle. That
split IS the 3+1 channel decomposition.

This module is internal. Users call expose() in valence.py, which
wraps these functions and returns clean Valence objects.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def hopf_project(q: NDArray[np.float64]) -> NDArray[np.float64]:
    """Extract the base direction (R, G, B) from a quaternion.
    Projects S³ down to S² — the interaction component. Returns a
    unit vector in R³.
    """
    w, x, y, z = q[0], q[1], q[2], q[3]
    u = 2 * (x * z + w * y)
    v = 2 * (y * z - w * x)
    t = w * w + z * z - x * x - y * y
    return np.array([u, v, t], dtype=np.float64)


def hopf_fiber_phase(q: NDArray[np.float64]) -> float:
    """Extract the fiber phase (W) from a quaternion. This is the S¹
    angle within the fiber over the base point — the coherence
    component. Returns radians in [-π, π].
    """
    w, x, y, z = q[0], q[1], q[2], q[3]
    phase = np.arctan2(2 * (w * x + y * z), w * w + z * z - x * x - y * y)
    return float(phase)


def hopf_decompose(q: NDArray[np.float64]) -> dict:
    """Full split. Takes a quaternion and returns all three channels:
    sigma (magnitude), base (R, G, B direction on S²), and phase
    (W angle on S¹). This is the prism applied to one point.
    """
    base = hopf_project(q)
    phase = hopf_fiber_phase(q)
    cos_half = np.clip(q[0], -1.0, 1.0)
    sigma = 2.0 * float(np.arccos(abs(cos_half)))
    return {
        "base": base,
        "phase": phase,
        "sigma": sigma,
    }
