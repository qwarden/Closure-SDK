# Closure SDK

A primitive data structure — the geometry of ordered information.

## What is this

A primitive data structure, in the same sense that a stack, a queue, or
a blockchain is a primitive data structure. A stack is LIFO; a queue is
FIFO; a hash map is key-value; a blockchain is a hash chain. This one
composes ordered data on S³ — the 3-sphere of unit quaternions, which
is the richest space where sequential composition is still associative.
The structure follows from two axioms and Hurwitz's theorem, and every
other ordered structure projects from it.

Raw bytes go in, SHA-256 hashes them, and the hash composes on the
sphere. The result is a point on S³ that cannot be reversed to the
original data but can be composed, diffed, inverted, and decomposed
into color channels — a hash you can do algebra on.

Two systems that compose the same data land on the same point, and they
can verify this by exchanging elements without ever exchanging the data
itself. Because the elements are built on a cryptographic hash, neither
side can fake agreement: the spheres cannot lie to each other, even
though neither one knows the other's content or the hash that created it.

Any data that can be serialized to bytes can be composed — databases,
blockchains, gene sequences, financial ledgers, network packets,
satellite telemetry, event streams, anything ordered.

## Why give your data this shape

Because you get properties that do not exist anywhere else in this
combination:

    Instant coherence     — did my data change? One comparison, any scale.
    Exact magnitude       — how much changed, as a precise geometric distance.
    What kind             — missing record and reordering look different.
                            Two types only, algebraic inverses of each other.
    Where it broke        — find the exact record in O(log n). At one million
                            records, 20 comparisons.
    Color channels        — the Hopf fibration splits every divergence into
                            3+1 channels: W (exists or doesn't) + RGB (where
                            and how far). Available at every step.
    Algebra               — combine summaries across shards, subtract one
                            from another, diff two snapshots, patch a third.
                            Without touching raw data.
    Cryptographic summary — embed() hashes with SHA-256 before composing.
                            The element cannot be reversed to the original
                            data, but can be composed, diffed, inverted, and
                            decomposed into channels. A hash you can do
                            algebra on.
    Identity binding      — two systems verify they hold the same data by
                            exchanging elements, never the data itself.
                            Neither side can fake agreement. Neither side
                            learns the other's content.
    Streaming             — classify incidents as records arrive, before
                            either stream is complete.
    Proven guarantees     — exact sensitivity (Theorem 1) and uniform
                            detectability (Theorem 2), both proven.

## Quick start

```bash
pip install -e '.[dev]'
```

```python
import closure_sdk as closure

# Compose records on S³
source = closure.Seer()
target = closure.Seer()

for record in source_stream:
    source.ingest(record)
for record in target_stream:
    target.ingest(record)

# Did anything change?
result = source.compare(target)
if not result.coherent:
    print(f"Drift detected: {result.drift:.6f}")
```

```python
# Find every incident between two complete sequences
faults = closure.gilgamesh(source_records, target_records)
for f in faults:
    print(f"{f.incident_type}  record={f.record!r}  src={f.source_index}  tgt={f.target_index}")
```

```python
# Classify incidents in real time as records arrive
stream = closure.Enkidu()
for record, position in arrivals:
    result = stream.ingest(record, position, side)
    if result:
        print(f"Reclassified: {result.incident_type}")

# Roll the clock — unresolved records become missing
missing = stream.advance_cycle()
```

```python
# Two systems verify they hold the same data — only elements cross the wire
result = closure.bind(my_element, their_element)
# result.relation: "equal", "inverse", or "disordered"
```

```python
# See the color channels of any composition
valence = closure.expose(element)
# valence.sigma, valence.base (R, G, B), valence.phase (W)
```

## What's in this repo

| Path | What it is |
|---|---|
| `closure_sdk/` | The SDK — 22 symbols. Compose, compare, localize, classify, bind, expose. |
| `closure_rs/` | The engine — Rust core with Python bindings. S³ composition, embedding, search. |
| `rust/` | Rust source for the engine. |
| `tests/` | 192 tests covering algebra, convergence, paths, hierarchy, streaming, binding. |
| `benchmarks/` | Performance comparisons against SHA-256, hash chains, and Merkle trees. |
| `CLOSURE_SDK.md` | Full technical documentation — theory, architecture, API reference. |

## Architecture

```
raw bytes → embed (SHA-256 → S³) → compose on the sphere → measure

    Seer        sensor, O(1), detects drift
    Oracle      recorder, O(n), locates in O(log n)
    Witness     reference template, verifies against known-good

    Gilgamesh   static: two complete sequences → every incident
    Enkidu      stream: records arrive in real time → classify on arrival

    Bind        two elements → equal, inverse, or disordered
    Expose      any element → color channels (σ, RGB, W)
```

## Tests

```bash
pytest tests/ -q
```

## License

AGPL-3.0-only
