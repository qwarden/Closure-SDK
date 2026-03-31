"""The ball's tools — operations on S³.

Records enter as bytes, land on the ball as points, and every question
about coherence reduces to measuring distances between those points.

Primitives — the four irreducible operations:
    embed      — the entry door. Bytes become a point on the ball.
    compose    — the closure operation. Two points become one.
    invert     — the undo. Every point has an opposite.
    sigma      — the thermometer. Distance from the north pole (identity).

Derived — built from primitives for convenience:
    diff       — the gap between two points (invert, then compose).
    compare    — the gap as a yes/no verdict (diff, then sigma, then threshold).

Every operation is O(1) and lives entirely on S³ (unit quaternions).
"""

from __future__ import annotations

import numpy as np
import closure_rs

from .state import ClosureState, CompareResult

# The S³ group object. Created once, used by every operation.
_SPHERE = closure_rs.sphere()


def embed(record: bytes) -> ClosureState:
    """The SDK entry door. Takes raw bytes, hashes them with SHA-256,
    and places the result on S³ as a unit quaternion.

    The SDK stays in cryptographic mode on purpose. Geometric byte
    composition is used by Closure DNA, not by the verification SDK.
    """
    elem = closure_rs.closure_element_from_raw_bytes("Sphere", [record], hashed=True)
    return ClosureState(group="Sphere", element=np.array(elem))


def compose(left: ClosureState, right: ClosureState) -> ClosureState:
    """The closure operation. Quaternion multiplication: left · right.
    This is how running products are built — embed record after record
    and compose each one in. Order matters (non-commutative), which is
    what makes position detectable.
    """
    combined = _SPHERE.compose(left.element, right.element)
    return ClosureState(group="Sphere", element=np.array(combined))


def invert(state: ClosureState) -> ClosureState:
    """The undo. Every point on the ball has an opposite — its inverse.
    Compose a point with its inverse and you get identity (north pole).
    This is what lets the algebra subtract one stream from another and
    what generates the duality between incident types.
    """
    inv = _SPHERE.inverse(state.element)
    return ClosureState(group="Sphere", element=np.array(inv))


def sigma(state: ClosureState) -> float:
    """The thermometer. Measures how far a point sits from identity
    (the north pole). σ = 0 means perfect coherence. σ > 0 means
    something diverged, and the number tells you exactly how much.
    """
    return float(_SPHERE.distance_from_identity(state.element))


def diff(a: ClosureState, b: ClosureState) -> ClosureState:
    """The gap. Computes invert(a) then compose with b — the element
    that, composed with a, would give you b. This is the divergence
    object itself, still on the ball, still full-color. You can measure
    it (sigma) or split it into channels (expose).
    """
    delta = _SPHERE.compose(_SPHERE.inverse(a.element), b.element)
    return ClosureState(group="Sphere", element=np.array(delta))


def compare(
    a: ClosureState,
    b: ClosureState,
    *,
    threshold: float = 1e-10,
) -> CompareResult:
    """The quick verdict. Computes diff, measures sigma, compares
    against a threshold. Returns the drift number and a coherent flag.
    If coherent is True, the streams match. If False, something broke.
    """
    delta = _SPHERE.compose(_SPHERE.inverse(a.element), b.element)
    drift = float(_SPHERE.distance_from_identity(delta))
    return CompareResult(drift=drift, coherent=drift < threshold)
