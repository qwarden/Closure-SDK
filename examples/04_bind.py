"""Identity binding — verify two systems hold the same data.

Only elements cross the wire. Neither side sees the other's content.
Neither side can fake agreement. The spheres cannot lie to each other.
"""

import closure_sdk as closure

# System A composes its records
seer_a = closure.Seer()
for r in [b"record-1", b"record-2", b"record-3"]:
    seer_a.ingest(r)

# System B composes the same records independently
seer_b = closure.Seer()
for r in [b"record-1", b"record-2", b"record-3"]:
    seer_b.ingest(r)

# Exchange elements, never data
result = closure.bind(seer_a.state().element, seer_b.state().element)
print(f"Relation: {result.relation}  sigma: {result.sigma:.6f}")

# System C has different data
seer_c = closure.Seer()
for r in [b"record-1", b"record-2", b"record-X"]:
    seer_c.ingest(r)

result2 = closure.bind(seer_a.state().element, seer_c.state().element)
print(f"Relation: {result2.relation}  sigma: {result2.sigma:.6f}")
