"""Classify incidents in real time as records arrive.

Enkidu holds unmatched records for one grace period. If the match
arrives, silent resolve. If not, promote to missing. If the match
arrives after promotion, reclassify to reorder.
"""

import closure_sdk as closure

stream = closure.Enkidu()

# Source gets a record, target doesn't (yet)
stream.ingest(b"late-record", 0, "source")

# Grace period expires — promoted to missing
missing = stream.advance_cycle()
print(f"After grace: {len(missing)} missing")
print(f"  {missing[0].incident_type}  record={missing[0].record}")

# Target gets the record late — reclassify
result = stream.ingest(b"late-record", 5, "target")
print(f"\nLate arrival: reclassified to {result.incident_type}")
print(f"  src={result.source_index}  tgt={result.target_index}")
