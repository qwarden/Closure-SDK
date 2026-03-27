"""
Cell — one cell in the lattice.

A Cell wraps a Kernel with a substrate-specific adapter.
The Kernel handles the algebra (compose, sigma, emit).
The Cell handles the interface (embed, teach, classify).

A Cell has:
    - a kernel (the closure recurrence on S³)
    - an adapter (the directional truth from outside)
    - a genome (what it has learned — positions on S³)

Three outcomes per cycle:
    - CLOSURE (σ < ε): purpose fulfilled, emit upward, reset
    - DEATH (σ → π): coherence lost — detected by the LATTICE, not the cell
    - ALIVE (ε < σ < π): still accumulating
"""

import numpy as np
import math
from closure_ea.kernel import Kernel, compose, inverse, sigma, identity


def geodesic_step(a, b, t=0.1):
    """Step along the geodesic from a toward b on S³. SLERP."""
    d = np.dot(a, b)
    if d < 0: b = -b; d = -d
    if d > 0.9999: return b.copy()
    theta = math.acos(min(d, 1.0))
    s = math.sin(theta)
    if s < 1e-8: return a.copy()
    r = math.sin((1-t)*theta)/s * a + math.sin(t*theta)/s * b
    n = np.linalg.norm(r)
    return r/n if n > 1e-8 else a.copy()


class Adapter:
    """Base adapter. Substrate-specific subclasses override embed().

    The adapter provides:
        - embed(event) → quaternion on S³ (deterministic)
        - teach(event, vote) → update the genome from kernel feedback
        - classify(gap) → 'missing' or 'reorder' via Hopf channels

    The adapter does NOT provide:
        - closure algebra (that's the kernel)
        - hierarchy management (that's the lattice)
        - lifecycle decisions (that's the lattice)
        - semantics (those emerge from composition)
    """

    def __init__(self, damping=0.05):
        self.genome = {}      # event_key → (position on S³, count)
        self.damping = damping

    def embed(self, event_key):
        """Map an event to S³. Override per substrate."""
        if event_key in self.genome:
            return self.genome[event_key][0].copy()
        # Default: hash to S³
        import hashlib
        h = hashlib.sha256(
            event_key.encode() if isinstance(event_key, str) else event_key
        ).digest()
        q = np.array([
            int.from_bytes(h[i:i+4], 'little') for i in range(0, 16, 4)
        ], dtype=np.float64)
        q -= q.mean()
        n = np.linalg.norm(q)
        pos = q / n if n > 1e-8 else identity()
        self.genome[event_key] = (pos, 0)
        return pos.copy()

    def embed_exact(self, event_key, position):
        """Set an exact position for an event. The adapter's truth.
        count = -1 marks this as truth-locked: teach() will not overwrite.
        """
        self.genome[event_key] = (position.copy(), -1)

    def teach(self, event_key, vote):
        """Update the genome from the kernel's vote.
        Truth-locked positions (count == -1) are not updated.
        """
        if event_key in self.genome:
            pos, count = self.genome[event_key]
            if count == -1:
                return  # adapter truth — do not overwrite
            new_pos = geodesic_step(pos, vote, self.damping)
            self.genome[event_key] = (new_pos, count + 1)
        else:
            self.genome[event_key] = (vote.copy(), 1)

    def classify(self, gap_q):
        """Classify a gap as missing or reorder via Hopf decomposition.

        W-dominant (scalar) → missing: something absent
        RGB-dominant (vector) → reorder: something misplaced

        This is not a decision. It is a measurement of which kind
        of departure from identity the current state represents.
        """
        w = abs(gap_q[0])
        rgb = math.sqrt(gap_q[1]**2 + gap_q[2]**2 + gap_q[3]**2)
        if w > rgb:
            return 'missing'
        return 'reorder'

    @property
    def size(self):
        return len(self.genome)


class Cell:
    """One cell. A kernel wrapped with an adapter.

    Feed it events. It composes, checks closure, learns.
    The adapter provides the truth. The kernel does the algebra.
    The genome accumulates what's learned.

    The cell does NOT decide its own lifecycle.
    The lattice observes the cell's gap and decides.
    """

    def __init__(self, adapter, epsilon=0.15, parent_kernel=None):
        self.adapter = adapter
        self.kernel = Kernel(epsilon=epsilon, parent=parent_kernel)

    def ingest(self, event_key):
        """One event from the substrate.

        1. Adapter embeds the event → quaternion
        2. Record σ BEFORE composition
        3. Kernel composes it → result + vote
        4. Record σ AFTER composition
        5. Classify the gap: missing (W-dominant) or reorder (RGB-dominant)
        6. Teach DIRECTIONALLY: only when σ dropped (event helped close)

        Returns: (result, gap, incident_type)
            result: 'open' or 'closure'
            gap: current σ
            incident_type: 'missing' or 'reorder'
        """
        q = self.adapter.embed(event_key)
        sigma_before = self.kernel.gap

        result, vote = self.kernel.ingest(q)

        sigma_after = self.kernel.gap

        # Classify the remaining gap
        incident_type = self.adapter.classify(self.kernel.C)

        return result, self.kernel.gap, incident_type

    def ingest_with_truth(self, event_key, position):
        """Ingest with adapter-provided exact position.
        The adapter sets the truth, the kernel composes it.
        No learning from vote — the position IS the truth.
        """
        self.adapter.embed_exact(event_key, position)
        result, vote = self.kernel.ingest(position)
        incident_type = self.adapter.classify(self.kernel.C)
        return result, self.kernel.gap, incident_type

    @property
    def gap(self):
        """Current σ — distance from closure."""
        return self.kernel.gap

    @property
    def prediction(self):
        """C⁻¹ — what the cell expects next. The downward signal."""
        return self.kernel.prediction

    @property
    def incident_type(self):
        """Current gap classified: 'missing' or 'reorder'."""
        return self.adapter.classify(self.kernel.C)

    @property
    def genome_size(self):
        return self.adapter.size

    def reset(self):
        self.kernel.reset()
