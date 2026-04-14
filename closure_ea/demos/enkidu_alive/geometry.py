"""Quaternion geometry for Enkidu Alive.

Pure S³ algebra. No neural network. No training.
Hamilton product, inverse, sigma, Hopf decomposition.

Grid movement uses commuting rotations on the Clifford torus
inside S³. UP/DOWN rotate in the (w,x) plane. LEFT/RIGHT rotate
in the (y,z) plane. These commute because they act on orthogonal
planes — so grid position is always exactly (x_steps, y_steps)
with no cross-terms. The full S³ non-commutativity is preserved
in the algebra; we simply choose movement generators that live
on the torus T² ⊂ S³.

This is geometrically correct: the Clifford torus is a real
submanifold of S³, and the Hopf fibration still applies.
"""

import numpy as np


# --- Quaternion operations ---

def qmul(q1, q2):
    """Hamilton product of two quaternions [w, x, y, z]."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return normalize(np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ]))


def normalize(q):
    """Project back to unit sphere."""
    n = np.linalg.norm(q)
    if n < 1e-12:
        return IDENTITY.copy()
    return q / n


def invert(q):
    """Quaternion conjugate = inverse on S³. Three sign flips."""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def sigma(q):
    """Geodesic distance from identity. σ = arccos(|w|)."""
    return np.arccos(np.clip(abs(q[0]), 0.0, 1.0))


def gap(a, b):
    """The quaternion that takes a to b: invert(a) · b."""
    return qmul(invert(a), b)


def distance(a, b):
    """Geodesic distance between two quaternions."""
    return sigma(gap(a, b))


# --- Hopf decomposition ---

def hopf_decompose(q):
    """Decompose quaternion into σ, base (R,G,B), phase (W).

    Returns dict with sigma, R, G, B, W.
    """
    w, x, y, z = q
    sig = sigma(q)

    # Base: project to S² via vector part / ||vector||
    vec_norm = np.sqrt(x*x + y*y + z*z)
    if vec_norm < 1e-12:
        R, G, B = 0.0, 0.0, 0.0
    else:
        R = x / vec_norm
        G = y / vec_norm
        B = z / vec_norm

    # Fiber: phase angle
    W = np.arctan2(vec_norm, w)

    return {"sigma": sig, "R": R, "G": G, "B": B, "W": W}


# --- Constants ---

IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])

# Movement quaternions on the Clifford torus T² ⊂ S³.
#
# The Clifford torus parametrizes S³ as:
#   q(α, β) = (cos α, sin α, cos β · 0, sin β · 0) -- no, let's be precise.
#
# We use two orthogonal rotation planes:
#   UP/DOWN:    rotation in the (w, x) plane by angle θ
#               q = (cos(θ/2), sin(θ/2), 0, 0)
#   LEFT/RIGHT: rotation in the (w, z) plane by angle θ — BUT we need
#               these to commute with UP/DOWN.
#
# Two quaternion rotations commute when they act on orthogonal 2-planes.
# Rotation in (w,x): q = cos(α) + sin(α)·i  →  affects w,x only
#                     when applied as q·p·q⁻¹ this rotates the (j,k) plane.
#
# For a RUNNING PRODUCT (not conjugation), q₁·q₂ commutes when both
# are in the same complex subplane. Two independent subplanes of ℍ:
#   Plane 1: span{1, i}  →  q = a + bi  (complex numbers in w,x)
#   Plane 2: span{1, j}  →  BUT 1 is shared, so these DON'T commute.
#
# The correct commuting decomposition for running products on S³:
# Represent position as two INDEPENDENT angles (α, β) on T².
# The quaternion at position (α, β) is:
#   q(α, β) = (cos α · cos β,  sin α · cos β,  cos α · sin β,  sin α · sin β)
# This is the Clifford torus parametrization. Movements in α and β commute.
#
# One grid step = θ change in α or β.

_THETA = np.pi / 16  # one grid cell — 32 steps per full circle, usable range ±15

def _clifford(alpha, beta):
    """Clifford torus parametrization: (α, β) → unit quaternion."""
    return np.array([
        np.cos(alpha) * np.cos(beta),
        np.sin(alpha) * np.cos(beta),
        np.cos(alpha) * np.sin(beta),
        np.sin(alpha) * np.sin(beta),
    ])

def _clifford_angles(q):
    """Extract (α, β) from a Clifford torus quaternion."""
    w, x, y, z = q
    alpha = np.arctan2(x, w)
    beta = np.arctan2(y, w / np.cos(np.arctan2(x, w)) if abs(np.cos(np.arctan2(x, w))) > 1e-12 else 1.0)
    return alpha, beta


# Instead of composing quaternions for grid movement, we track position
# as (α, β) angles on the Clifford torus and convert to quaternion.
# This guarantees commutativity and exact grid positions.

def position_to_quaternion(x, y):
    """Grid position (x, y) → quaternion on the Clifford torus."""
    alpha = y * _THETA  # up/down = α
    beta = x * _THETA   # left/right = β
    return _clifford(alpha, beta)


def quaternion_to_position(q):
    """Quaternion → approximate grid position (x, y).

    Reads off the Clifford torus angles and converts to grid steps.
    """
    w, x, y, z = q
    # α from the (w, x) components
    alpha = np.arctan2(x, w)
    # β from the (w, y) components — but need to account for α
    cos_a = np.cos(alpha)
    if abs(cos_a) > 1e-8:
        beta = np.arctan2(y / cos_a, 1.0) if abs(w / cos_a) < 1e-8 else np.arctan2(y, w / cos_a)
    else:
        sin_a = np.sin(alpha)
        beta = np.arctan2(z / sin_a, 1.0) if abs(sin_a) > 1e-8 else 0.0

    grid_x = beta / _THETA
    grid_y = alpha / _THETA
    return (round(float(grid_x), 1), round(float(grid_y), 1))


# Movement: add/subtract _THETA to α or β, recompute quaternion.
# This is equivalent to composing on the torus, but exact.

MOVE_DELTAS = {
    "up":    (0, 1),     # +1 in y (α)
    "down":  (0, -1),    # -1 in y
    "right": (1, 0),     # +1 in x (β)
    "left":  (-1, 0),    # -1 in x
}

MOVE_NAMES = list(MOVE_DELTAS.keys())


class EnkiduState:
    """Enkidu's position on the Clifford torus.

    Tracks grid position (x, y) as integers and computes the
    corresponding quaternion. Movement is exact — no floating
    point drift.
    """

    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    @property
    def quaternion(self):
        return position_to_quaternion(self.x, self.y)

    @property
    def position(self):
        return (self.x, self.y)

    def move(self, direction):
        """Take one step. Returns the new quaternion."""
        dx, dy = MOVE_DELTAS[direction]
        self.x += dx
        self.y += dy
        return self.quaternion

    def sigma_from(self, target_q):
        """Distance from current position to a target quaternion."""
        return distance(self.quaternion, target_q)

    def sigma_home(self):
        """Distance from home (identity)."""
        return sigma(self.quaternion)

    def hopf(self):
        """Hopf decomposition of current state."""
        return hopf_decompose(self.quaternion)

    def best_step_toward(self, target_x, target_y):
        """Which direction reduces Manhattan distance to target?

        Uses grid coordinates directly — no ambiguity from S³ wrapping.
        Returns (move_name, manhattan_distance_after_move).
        """
        best_name = None
        best_dist = float("inf")

        for name, (dx, dy) in MOVE_DELTAS.items():
            nx, ny = self.x + dx, self.y + dy
            d = abs(nx - target_x) + abs(ny - target_y)
            if d < best_dist:
                best_dist = d
                best_name = name

        return best_name, best_dist

    def copy(self):
        return EnkiduState(self.x, self.y)
