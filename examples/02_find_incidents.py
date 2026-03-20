"""Find every incident between two complete sequences.

Gilgamesh composes both on S³, walks both chains, classifies each
mismatch from the Hopf fiber. Two types: missing (W axis) or
reorder (RGB axis).
"""

import closure_sdk as closure

source = [b"a", b"b", b"c", b"d", b"e"]
target = [b"a", b"d", b"c", b"e"]  # b missing, c and d swapped

faults = closure.gilgamesh(source, target)

for f in faults:
    print(f"{f.incident_type:8s}  src={f.source_index}  tgt={f.target_index}  record={f.record}")
