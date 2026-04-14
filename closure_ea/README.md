# closure_ea

A digital brain on S³. Self-contained Rust crate.

**Paper:** [A Geometric Computer](docs/GeometricComputer.pdf) · [Zenodo](https://zenodo.org/records/19578024)

---

## What this is

A complete cognitive architecture running on one manifold: S³, the unit 3-sphere.
Every brain state is a unit quaternion. Every update is a Hamilton product —
the only arithmetic in the system, and the richest space where sequential
composition remains associative and non-commutative.

The brain runs a single loop: ingest a carrier, compare the current state against
what the genome predicts, measure the geodesic distance. When that distance
crosses π/4 — the BKT phase boundary on S³ — the brain closes: it writes to
the genome, updates the hierarchy, and corrects its prediction. Learning happens
through repeated closure. Convergence is a fixed-point theorem.

Five layers build upward from S³: substrate → memory → execution → brain → learning.
Each layer uses only the layer beneath it. The same Hamilton product that encodes a
unit quaternion rotation also runs a Minsky machine, reads the genome, and integrates
the perceptual field.

---

## What you get

    Identity maintenance      A?=A at any scale. One comparison.
                              σ = 0 at perfect coherence.
                              σ = π/4 at the Hopf equator — the closure threshold.

    Two incident types        Missing record (W axis breaks) or reorder
                              (RGB axes break). Algebraic inverses.
                              There is no third type.

    Carrier channels          R = salience (X) — the [Total, Unknown] commutator.
                              G = total    (Y) — the full field; the prior.
                              B = unknown  (Z) — what has not been integrated.
                              Known = G − B. Yellow = learned. Blue = novel.

    Factorized addressing     Axis queries find semantic type (S²).
                              Angle queries find cyclic position (S¹).
                              Full queries find the exact carrier (S³).
                              Independent channels — neither knows about the other.

    Field resonance           ZREAD reads the genome by proximity, not exact match.
                              Tunable radius. Path-ordered coalition accumulation.
                              Query strength falls as cos(σ), cuts off at σ = π/3.

    Genome persistence        DNA layer bootstraps from orbit seeds.
                              Epigenetic layer learns from ingest.
                              Consolidation reorganizes epigenetic; DNA is permanent.

    Neuromodulation           Arousal tone  — EMA of step pressure / SIGMA_BALANCE.
                              Coherence tone — EMA of signed SFE change / (π/2).
                              Session-ephemeral. Observational: does not control
                              write rate or consolidation threshold directly.

    Curriculum learning       (input, target) pairs drive convergence.
                              σ-gap measures error. Convergence is geometric.

    Turing completeness       2-counter Minsky machine and FRACTRAN run natively
                              on the carrier substrate.

    BrainState serialization  Full state serializes to JSON. Exact round-trip.

---

## Architecture

```
SUBSTRATE
│
├── sphere        compose(a, b)  inverse(a)  sigma(a)  slerp(a, b, t)  IDENTITY
│                 Hamilton product on S³. Star involution. Geodesic distance.
│
├── embed         domain_embed(bytes)  f64_to_sphere4  i64_to_sphere4
│                 Vocabulary  MusicEncoder
│                 SHA-256 → carrier on S³.
│
├── verify        verify(a, b)  SIGMA_BALANCE  SIGMA_IDENTITY
│                 VERIFY: A?=A. Returns VerificationEvent with σ, closure kind,
│                 and Hopf decomposition.
│
└── hopf          hopf_decompose(q)  carrier_from_hopf(base, phase)
                  address_distance(q, addr, mode)
                  AddressMode::{Full, Base, Phase, Scalar}
                  SALIENCE_AXIS  TOTAL_AXIS  UNKNOWN_AXIS
                  VectorChannel  HopfChannel
                  Hopf fibration S³ → S² × S¹ and factorized addressing.

MEMORY
│
├── buffer        Buffer  BufferEntry
│                 Transient input window. EMBED writes here; ZREAD reads it.
│
├── genome        Genome  GenomeConfig  GenomeEntry  Layer::{DNA, Epigenetic, Category}
│                 Nearest-neighbor reads. Law-3 ingest. BKT topology control.
│                 ZREAD_T_MIN  BKT_THRESHOLD  DOBRUSHIN_DELTA_2  DOBRUSHIN_DELTA_3
│
└── field         zread(genome, buffer, query, mode, channel)
                  resonate(genome, query, channel)
                  zread_at_query_channel_with_mode
                  PopulationSource  ResonanceHit

EXECUTION
│
├── carrier       VerificationCell  EulerPlane  TwistSheet  CouplingState
│                 NeighborCoupling  CarrierObservation
│
├── execution     MinskyMachine  FractranMachine  OrbitRuntime  orbit_generator
│                 OrbitSeed  MinskyInstr  Fraction
│
└── zeta          Zeta functions over the carrier lattice.

BRAIN
│
├── hierarchy     Hierarchy  ClosureEvent  ClosureLevel  ClosureKind  ClosureRole
│                 Recursive closure detection and genome emission.
│
├── localization  localize(genome, query)  LocalizedInterval
│                 O(log n) minimal closure interval search.
│
├── consolidation consolidate(genome)  ConsolidationReport  PromotionCandidate
│                 Merge, prune, reorganize epigenetic layer. Never touches DNA.
│
├── neuromodulation  NeuromodState
│                    arousal_tone  coherence_tone
│                    update(step_pressure, valence)
│
└── three_cell    ThreeCell  BrainState  Step  UpdateReport  EvaluationReport
                  PendingPrediction  PredictionFeedback
                  ThreeCell::ingest(carrier) runs the full loop:
                    buffer → ZREAD → RESONATE → VERIFY
                    → Cell A composition → closure detection
                    → Cell C integration → hierarchy emission → genome ingest

LEARNING
└── teach         teach(brain, inputs, targets)
                  run_curriculum_passes(brain, curriculum, passes)
                  CurriculumReport  CurriculumTrace  CurriculumWindow
                  σ-gap measures error. Convergence is geometric.
```

---

## Experiments

Eleven numbered experiments plus two live terminal visualizers.
All pass `cargo run --example <name> --release`.

| # | name | subject |
|---|------|---------|
| 1 | `exp_arithmetic` | Exact modular arithmetic on S³ for Z/nZ orbits |
| 2 | `exp_bkt_phase_transition` | BKT phase boundary at σ = π/4 (the Hopf equator) |
| 3 | `exp_riemann_zeros` | Riemann zeros as local minima in the carrier field |
| 4 | `exp_associative_memory` | Associative recall by σ-radius, not exact key match |
| 5 | `exp_turing` | Turing completeness via 2-counter Minsky machine |
| 6 | `exp_fractran` | FRACTRAN: Turing-complete prime-native computation |
| 7 | `exp_prime_resonance` | Riemann zeros as prime eigenstates on S³ |
| 8 | `exp_collatz` | Collatz sequence: orbit convergence on the carrier lattice |
| 9 | `exp_fiber_memory` | Fiber memory: axis and angle as independent address channels |
| — | `exp_su2_gates` | SU(2) gate dictionary and single-qubit completeness |
| — | `exp_neuromodulated_learning` | Arousal and coherence modulation of the learning regime |

Live visualizers — require a terminal with true-color support:

```
cargo run --example gol_live --release           # Conway's GoL on S³, pattern selector
cargo run --example gray_game_live --release     # Gray Game, 3 interference modes
                                                 # (key 1: spectrum/integration
                                                 #  key 2: resonance/creation
                                                 #  key 3: edge — critical point)
```

---

## Running

```
cargo run --example exp_arithmetic --release
cargo run --example exp_fiber_memory --release

cargo run --example gol_live --release           # Conway's GoL on S³
cargo run --example gray_game_live --release     # Gray Game, 3 interference modes

cargo test --release     # 55 tests, all passing
```
