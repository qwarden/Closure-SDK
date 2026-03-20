"""Decompose any divergence into color channels.

The Hopf fibration splits S³ → S² × S¹:
  W     — existence (has or hasn't)
  R,G,B — position and displacement
  σ     — total magnitude
"""

import closure_sdk as closure

seer = closure.Seer()
for r in [b"event-1", b"event-2", b"event-3"]:
    seer.ingest(r)

valence = closure.expose(seer.state().element)

print(f"sigma: {valence.sigma:.6f}")
print(f"R: {valence.base[0]:+.6f}  G: {valence.base[1]:+.6f}  B: {valence.base[2]:+.6f}")
print(f"W: {valence.phase:+.6f}")
