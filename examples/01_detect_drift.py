"""Detect drift between two streams of records.

Compose both on S³, compare. One call, any scale.
"""

import closure_sdk as closure

# Two systems processing the same events
source = closure.Seer()
target = closure.Seer()

records = [f"tx-{i}".encode() for i in range(1000)]

for r in records:
    source.ingest(r)
    target.ingest(r)

# Identical → coherent
result = source.compare(target)
print(f"Coherent: {result.coherent}  drift: {result.drift:.6f}")

# Corrupt one record on the target side
target2 = closure.Seer()
corrupted = list(records)
corrupted[500] = b"TAMPERED"
for r in corrupted:
    target2.ingest(r)

result2 = source.compare(target2)
print(f"Coherent: {result2.coherent}  drift: {result2.drift:.6f}")
