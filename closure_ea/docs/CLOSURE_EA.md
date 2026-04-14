# Closure-Ea — Technical Reference

A complete cognitive architecture on one manifold: S³, the unit 3-sphere.
Substrate, memory, execution, brain, and learning — all in one Rust crate,
all using one arithmetic operation: the Hamilton product.

---

## Part I — First-Principles Comparison

Before diving into modules, here is the system as a stack.

### The System at a Glance

| Layer | What it does | Main code |
|---|---|---|
| **Substrate** | Unit quaternions on S³, Hamilton product, geodesic distance, SLERP | `sphere.rs` |
| **Hopf anatomy** | Splits every carrier into S² base, S¹ fiber phase, and W-depth; defines address modes and channel views | `hopf.rs` |
| **Input boundary** | Maps domain data into carriers while preserving the Hopf write contract | `embed.rs` |
| **Persistent memory** | DNA anchors, epigenetic traces, response memory | `genome.rs` |
| **Field machine** | ZREAD coalition read, RESONATE winner selection, response eligibility | `field.rs` |
| **Living body** | Cell A, Cell C, buffer, hierarchy, prediction staging, correction | `three_cell.rs` |
| **Body state** | Slow arousal/coherence integration from step pressure and valence | `neuromodulation.rs`, `three_cell.rs` |
| **Learning drivers** | Directed curricula over the living body | `teach.rs` |
| **Sleep / abstraction** | Consolidation, promotion, category birth, multi-level hierarchy | `consolidation.rs`, `hierarchy.rs` |
| **Programs and domains** | Arithmetic, Minsky, FRACTRAN, Gray Game, zeros, tracks | `examples/`, `zeta.rs`, track adapters |

The purpose of this table is architectural: map every essential cognitive and computational function to its exact code location, and verify no function is missing.

### Table 1 — Brain vs Unix core vs Von Neumann vs Transformer vs closure_ea

| Function | Biological Brain | Unix Kernel | Von Neumann CPU | Transformer | closure_ea |
|---|---|---|---|---|---|
| **Input boundary** | Sensory transduction (photoreceptors → spikes) | `read(2)` syscall | Instruction fetch | Tokenizer + embedding layer | `domain_embed()`, `bytes_to_sphere4()` → carrier on S³ |
| **Transient working space** | Working memory (prefrontal cortex, ~7 items) | Page cache, process stack | Registers + cache | KV cache (context window) | `Buffer` (rolling transient entries) |
| **Long-term perceptual memory** | Neocortical long-term potentiation | Filesystem (inode tree) | Main memory (RAM) | Learned weight matrix | `Genome` — Epigenetic layer |
| **Structural anchors, invariants** | Brainstem reflex arcs, genetic constraints | Kernel text segment (read-only) | Boot ROM, microcode | Pre-trained weight initialization | `Genome` — DNA layer (never mutated) |
| **Prediction / anticipation** | Predictive coding (Rao & Ballard 1999) | Prefetch / speculative exec | Branch predictor | Autoregressive next-token probability | `commit_prediction()` → `PendingPrediction` |
| **Reality feedback / error signal** | Prediction-error neuron (PE signal) | `errno`, interrupt return | Branch misprediction flush | Cross-entropy loss, backprop | `evaluate_prediction()` → `credit_response()` |
| **Correction credit assignment** | Hebbian STDP (spike-timing) | File fsync, journal commit | Write-back cache flush | Gradient descent on weights | `record_co_resonance()` + `credit_response()` |
| **Hard attention / winner selection** | Lateral inhibition → winner-takes-all | Scheduler `pick_next_task()` | ALU result bus | Argmax over attention scores | `RESONATE` — nearest-neighbor on S³ |
| **Soft attention / coalition read** | Oscillatory synchrony, population code | `mmap` (all pages accessible) | Memory bus broadcast | Scaled dot-product attention + softmax | `ZREAD` — path-ordered Hamilton product |
| **Attention neighborhood gate** | Receptive field boundary | `mprotect` access control | Cache line boundary | Attention mask | `ZREAD_T_MIN = 0.5` (t < 0.5 → entry contributes IDENTITY) |
| **Lateral inhibition** | GABAergic interneurons suppress local competitors | CPU bus arbitration | — | — | `apply_lateral_inhibition()` in `collect_response_eligibility()` |
| **Coalition evidence accumulation** | Hebbian co-activation (LTP) | Filesystem access times | — | Attention weight accumulation | `record_co_resonance()` on raw eligibility |
| **Memory consolidation / sleep** | Hippocampal replay, synaptic homeostasis | `fsck`, journal replay, page reclaim | — | — | `consolidate()` — merge + prune + promote |
| **Category formation / abstraction** | Cortical column selectivity, concept cells | Filesystem directory entries | — | Latent space clustering | `collect_promotion_candidates()` → `genomes[1]` |
| **Hierarchical abstraction** | Cortical hierarchy (V1→V2→IT→PFC) | VFS layer, block device layer | — | Transformer layers (stacked) | `Hierarchy` with `genomes[0..n]` |
| **External free energy / surprise** | Prediction error (PE), neural surprise | `EAGAIN`, retry count | Pipeline stall | Cross-entropy loss value | `Step.prediction_error` = σ(cell_c, incoming) |
| **Internal free energy / self-tension** | Interoceptive self-model error | — | — | — | `Step.self_free_energy` = σ(cell_c, ZREAD(cell_c)) |
| **Signed coherence change** | Hedonic valence, dopamine signal | — | — | — | `Step.valence` = prev_sfe − sfe |
| **Slow body state** | Neuromodulatory tone, arousal/coherence integration | Brain-state regulation | — | — | `NeuromodState` from step pressure + valence |
| **Homeostatic identity drive** | FEP minimization, autonomic regulation | Watchdog daemon | Interrupt service routine | — | `self_observe()` → fixed-point seeking via convergence to Fix(ER) |
| **Arithmetic substrate** | Spiking convolutions, synaptic weights | Integer / float ALU | ALU (ADD, MUL, etc.) | Matrix multiply + activation | `compose()` — Hamilton product only |
| **Distance metric** | Tuning curve overlap, population vector distance | Inode gap, block distance | — | Cosine similarity (dot product) | `sigma()` = acos(\|w\|) — geodesic on S³ |
| **Interpolation / partial update** | STDP rate, LTP magnitude | — | — | Softmax weight blend | `slerp()` — geodesic interpolation on S³ |
| **Learned state encoding** | Synaptic weight vector | File data block | Memory word | Weight matrix row | `GenomeEntry.value` — unit quaternion on S³ |
| **Address / key** | Place cell firing field | Inode number, hash key | Memory address | Key vector (K in QKV) | `GenomeEntry.address` — VerificationCell on S³ |
| **Similarity scoring** | Firing rate ∝ overlap | Hash collision | — | Q·Kᵀ dot product | `coupling_from_gap()` = cos(σ) ∈ [0,1] |
| **Novelty detection / create** | Dentate gyrus pattern separation | `creat()` new inode | Cache miss → DRAM fetch | OOV token → UNK | `GenomeConfig.novelty_threshold` → `create` path in `Genome::ingest()` |
| **Reinforcement / familiarity** | LTP, increased synaptic strength | Page hit → cache warmth | Cache hit | High attention score on familiar token | `reinforce` path in `Genome::ingest()` |
| **Position / sequence encoding** | Grid cells, place cells, theta phase | File offset, inode mtime | Program counter | Positional encoding | `position_phase` in `domain_embed()` — S¹ fiber coordinate |
| **Two-headed read (multi-head attn)** | Left/right hemisphere dual processing | — | — | Multi-head attention | Hopf channel split: `HopfChannel::W` + `HopfChannel::RGB` |
| **Persistence / checkpointing** | Long-term potentiation, memory consolidation | `fsync()` to disk | — | Model checkpoint (.pt file) | `Genome::save_to_file()` / `load_from_file()` |
| **Observer cloning / fork** | Split-brain, identical twins? | `fork(2)` | — | Model copy / ensemble member | `ThreeCell::from_brain_state()` |
| **Causal credit (immediate)** | STDP causal window (pre before post) | — | — | Gradient at that layer | `effective_eligibility` → `stage_prediction()`, `credit_response()` |
| **Causal credit (structural)** | LTP coalition (joint activation history) | — | — | — | `raw_eligibility` → `record_co_resonance()` → promotion |
| **Curriculum / training** | Supervised practice, motor learning | — | — | SGD over training set | `teach()`, `run_curriculum_passes()` in `teach.rs` |
| **Sequence learning** | Temporal binding, episodic memory | Pipe, sequential read | — | Autoregressive generation | `ingest_sequence()`, sequential edges in `GenomeEntry` |
| **Phase boundary / stability gate** | Synaptic consolidation threshold (LTP vs LTD) | OOM killer threshold | — | Dropout rate | `BKT_THRESHOLD = 0.48` (Berezinskii–Kosterlitz–Thouless) |
| **Fibration / coordinate decomposition** | Tonotopic + phase maps | — | — | Positional decomposition | `hopf_decompose()` — S³ → S² × S¹ |

---

### What this table shows

The architecture is functionally complete: input, working memory, long-term
memory, prediction, feedback, attention, consolidation, hierarchy, and
arithmetic all have explicit code locations.

Three features have no standard analogue:

- `self_free_energy` — the brain measures tension between its accumulated model
  and its own field. An internal coherence observable, available at every step.
- `valence` — signed coherence change per step. Positive valence means the last
  step drove the brain toward self-consistency.
- `raw_eligibility / effective_eligibility` — structural coalition evidence
  (category formation) and immediate causal credit (prediction correction) are
  kept separate. One drives category birth; the other drives present correction.

---

## Part II — Architecture Manual

### 2.1 The manifold: S³

Everything lives on the unit 3-sphere. A carrier is a unit quaternion
`[w, x, y, z]` with `w² + x² + y² + z² = 1`.

The one arithmetic operation is the Hamilton product.
It serves simultaneously as:

- sequential composition,
- rotation composition,
- memory interaction,
- field integration,
- program execution.

Non-commutativity is load-bearing:

```text
compose(a, b) ≠ compose(b, a)
```

That difference is the system's native encoding of order, causality,
and directional change.

**Key primitives** (`sphere.rs`):

| Function | Math | Meaning |
|---|---|---|
| `IDENTITY` | `[1,0,0,0]` | Rest state. The brain's home point. Every orbit converges toward it. |
| `compose(a, b)` | Hamilton product + renormalize | Sequential composition of two experiences |
| `inverse(a)` | `[w, -x, -y, -z]` | Reversal, conjugate. compose(a, inverse(a)) = IDENTITY |
| `sigma(a)` | `acos(\|w\|)` | Geodesic distance from IDENTITY on S³. σ=0 means identity; σ=π/2 means maximum distance |
| `slerp(a, b, t)` | Spherical linear interp | Geodesic interpolation. Used for all soft updates |

`sigma` is the system's distance measurement. Every threshold, coupling
weight, novelty test, and phase boundary is expressed in units of `σ`.

---

### 2.2 The Hopf fibration: structure within S³

S³ carries a natural Hopf fibration:

```text
S³ → S² × S¹
```

Every carrier decomposes into three linked coordinates:

- **S² base** — semantic type, axis, role, or field direction
- **S¹ fiber phase** — cyclic position, orbit slot, beat, or sequence phase
- **W-depth** — existence depth / proximity to identity

This is the codebase's write contract. When a domain is encoded cleanly:

- type goes to **S²**
- position goes to **S¹**
- depth/presence is read from **W**

#### Hopf anatomy by domain

| Domain | S² base encodes | S¹ fiber encodes | W-depth expresses |
|---|---|---|---|
| **Tokens / vocabulary** | token identity / semantic type | position in sequence | strength of presence |
| **Music** | harmonic role | beat / bar phase | event depth from identity / silence |
| **Arithmetic / orbit programs** | orbit family / generator axis | slot within the orbit | distance from the identity pole |
| **Prime / zeta scans** | prime-direction contribution in the field | scan phase / accumulation phase | scalar depth of the running product |
| **Gray Game / fields** | local field direction of the cell state | emergent phase of the local carrier | hemisphere / equator placement of the state |

#### Address modes

The Hopf split gives the system several legitimate ways to compare carriers:

- `Full` — full S³ geodesic distance
- `Base` — compare only S² semantic direction
- `Phase` — compare only S¹ fiber position
- `Scalar` — compare only W-depth

That is what lets the same substrate support semantic lookup, sequence
position, orbit arithmetic, role identity, and field geometry.

#### RGB semantic channels

The S² base carries the RGB semantic axes used by the runtime:

- **R / X / index 1 = Salience** — the [Total, Unknown] commutator; the residual mismatch
- **G / Y / index 2 = Total** — the full field; known + unknown = everything
- **B / Z / index 3 = Unknown** — what has not been integrated; the novel input; the error

Defined in `hopf.rs` as `SALIENCE_AXIS`, `TOTAL_AXIS`, `UNKNOWN_AXIS`.

The `X` component holds the cross term `y1·z2 − z1·y2` — the G/B commutator.
At runtime:

- **Total** = the full field aggregate; everything the genome currently holds
- **Unknown** = what has not yet been integrated into the model; the error
- **Known** = Total − Unknown = Yellow channel; what has been learned
- **Salience** = the residual direction of their mismatch

`SemanticFrame` in `hopf.rs` names the runtime triple `total`, `known`, `residual` —
where `known` is Cell C (the accumulated model) — and measures `salience_sigma`,
`w_gap`, and `rgb_gap` on S³.

The Hopf split keeps three kinds of structure orthogonal: what something is (S²),
where it is in a cycle (S¹), and how strongly it exists (W). Different domains
share one manifold without flattening role, position, and depth into one coordinate.

---

### 2.3 The input boundary: EMBED

```
bytes → S³ carrier
```

`embed.rs` is the input boundary. It is where outside structure is
lifted into the manifold.

Three main entry styles are used:

| Mode | Function | Use case |
|---|---|---|
| Geometric | `bytes_to_sphere4()` | Locality-preserving. Similar bytes land near each other on S³. Sequential composition. |
| Cryptographic | `domain_embed()` | SHA-256 → Box-Muller → S³. Same bytes always → same point. Locality destroyed. Used for domain labels. |
| Vocabulary | `Vocabulary::register()` | Explicit placement for structured domains (e.g., RNA nucleotides must be within σ=π/4 of each other for ZREAD to reach across them) |

The guiding rule is:

- use `domain_embed()` when the domain naturally has **type + position**
- use `bytes_to_sphere4()` when the input is one opaque continuous signal
- use `Vocabulary::register()` when the domain has a known geometric write law

Examples already present in the repo:

- language / tokens: semantic type → S², sequence position → S¹
- music: harmonic role → S², beat phase → S¹
- parity-aware domains: alternate fiber branch through `parity_phase_gate()`

EMBED is intentionally one-way. The system compares carriers geometrically;
it does not recover raw external bytes by inversion.

---

### 2.4 The genome: three-layer persistent memory

The genome (`genome.rs`) is the brain's permanent memory. It is a flat array of `GenomeEntry` records, each holding:

| Field | Type | Meaning |
|---|---|---|
| `address` | VerificationCell | What this entry responds to (how it was approached) |
| `value` | `[f64; 4]` | What this entry holds (what it returns when queried) |
| `layer` | `Layer` | DNA / Epigenetic / Response |
| `support` | `usize` | How many times this entry was reinforced |
| `activation_count` | `usize` | How many correction events touched it |
| `co_resonance` | `Vec<(usize, f64)>` | Hebbian coalition history with other entries |
| `zread_read_count` | `u64` | Denominator for mean co-resonance (all reads, not just joint) |
| `salience_sum/count` | `f64` / `usize` | Mean prediction-time salience accumulated across corrections |
| `coherence_sum/count` | `f64` / `usize` | Mean body-state coherence accumulated across corrections |

**Three layers with different write laws:**

```
DNA layer           — seeded at bootstrap, never modified. Structural anchors.
                      No arithmetic can overwrite these.

Epigenetic layer    — written by perception (ingest). Every closure event that
                      survives BKT pruning lives here. Subject to merge, prune,
                      consolidate during sleep.

Response layer      — written by reality correction (evaluate_prediction).
                      What the evaluative loop recorded when reality returned.
                      Drives category formation.
```

**Ingest routing** (the write law for epigenetic entries):

```
σ(incoming, nearest_address) ≤ reinforce_threshold  →  reinforce: no geometric change
σ(incoming, nearest_address) ≤ novelty_threshold    →  correct:   SLERP value toward incoming
σ(incoming, nearest_address) > novelty_threshold    →  create:    new entry
```

DNA entries are always skipped in the correct path — the brain cannot modify its brainstem.

**Critical constants:**

| Constant | Value | Derivation | Meaning |
|---|---|---|---|
| `BKT_THRESHOLD` | 0.48 | 0.96/√4 (S³ BKT critical coupling) | Phase boundary between order and disorder. Entries below this are pruned during sleep. |
| `ZREAD_T_MIN` | 0.5 | cos(π/3) | Participation gate. Below this, an entry contributes IDENTITY to any ZREAD. Not arbitrary: π/3 is the natural Partzuf boundary on S³. |
| `CO_RESONANCE_FLOOR` | ≈0.2304 | BKT_THRESHOLD² | Minimum joint coupling for a Response pair to qualify as a stable coalition. Two BKT-alive entries in their worst joint read. |

---

### 2.5 The field machine: RESONATE and ZREAD

The field machine (`field.rs`) reads the genome. Two distinct operations:

**RESONATE** — hard attention, winner selection

```
Given query q, return the single genome entry with the smallest σ(q, entry.address).
```

Used for: current prediction, EMBED-to-genome lookup, orbit position recall.

The result is a `ResonanceHit { index, carrier, gap }` — the closest entry and how far away it was.

**ZREAD** — soft attention, population read

```
Given query q, compose every genome entry's value weighted by its coupling t = cos(σ(q, entry.address)).
Entries with t < ZREAD_T_MIN contribute IDENTITY (neutral element of Hamilton product).
Result: one carrier that integrates the whole genome's response to q.
```

Used for: Cell C integration, self-observation, perceptual field read.

Crucially, ZREAD is **path-ordered** (insertion order = memory sequence order). It is not a commutative sum. Reordering entries changes the result. This is not a limitation — it is the causal structure of the system.

**ZREAD is equivariant under S³ rotations:**
For any rotation g: `ZREAD(g·q·g⁻¹, g·population·g⁻¹) = g · ZREAD(q, population) · g⁻¹`.
This holds exactly because σ is rotation-invariant, compose is equivariant, and both iterate in the same slot order.

**Lateral inhibition** — online winner suppression

```
raw_coalition = all Response entries with t ≥ ZREAD_T_MIN
t_winner = max t in coalition
for each loser j:
    o_j = max coupling between j's address and any winner's address
    t'_j = max(0, t_j − o_j × t_winner)
effective_coalition = {(i, t'_i) | t'_i ≥ ZREAD_T_MIN}
```

Near-duplicate entries (high address overlap with winner) are suppressed. Entries far from the winner are unchanged. This sharpens category discrimination during prediction without erasing coalition evidence.

**Critical split:**
- `raw_eligibility` → `record_co_resonance()` (structural evidence, category formation)
- `effective_eligibility` → `stage_prediction()`, `credit_response()` (causal output, immediate credit)

These answer different questions. Using the inhibited set for co-resonance would prevent category formation between nearby entries. Using the raw set for prediction would let near-duplicates pollute the output.

---

### 2.6 The diabolo: ThreeCell

`ThreeCell` is the complete brain instance. It holds:

```
cell_a: [f64; 4]      — fast oscillator, running product of raw input
cell_c: [f64; 4]      — slow accumulator, closure-event integral
hierarchy: Hierarchy  — multi-level genome stack (genomes[0..n])
buffer: Buffer        — transient entries from current sequence
pending_prediction    — staged output waiting for reality's judgment
prev_sfe: f64         — previous self_free_energy (for valence)
neuromod: NeuromodState — slow body state (arousal/coherence)
```

The body has one living cycle with two phases:

- **System 1** — perceptual integration through `ingest()`
- **System 2** — reality feedback through `evaluate_prediction()`

The crate also exposes helper/query verbs around this body. Keeping them
distinct makes the architecture easier to read.

#### Living body verbs

```rust
ThreeCell::ingest(carrier) → Step
```
One complete perception step. The 7-step loop:
1. Push carrier to buffer
2. ZREAD over genome ∪ buffer → integrated field
3. RESONATE(field) → nearest genome entry
4. LOAD that entry's value
5. VERIFY(loaded, field) → VerificationEvent with σ
6. If σ < threshold → closure: localize, compose into cell_c, emit to hierarchy
7. Hierarchy emits to genome if level closes

Returns `Step` with all observable signals (prediction_error, self_free_energy, valence, closure, etc.).
This also updates the slow body state from the same step:

- `arousal_tone` integrates level-0 step pressure
- `coherence_tone` integrates signed valence

```rust
ThreeCell::evaluate_prediction(actual) → Option<EvaluationReport>
```
Reality returns. If a prediction was staged:
1. Compute correction: σ(predicted, actual)
2. `credit_response(eligibility, actual)` → move eligible Response entries toward actual
3. `learn_response(context, actual)` → create/update the explicit Response entry at that context
4. Run consolidation if σ-pressure crossed `SIGMA_BALANCE`

These two verbs are the living loop.

#### Prediction / query helpers

| Verb | Visibility | Purpose |
|---|---|---|
| `commit_prediction(predicted, source)` | `pub` | The canonical learning verb. Collects both raw and effective eligibility, records co-resonance from raw, stages prediction from effective. Always use this. |
| `stage_prediction(predicted, ctx, src, elig)` | `pub(crate)` | Raw lifecycle verb. Stores pending prediction. Use only when you need explicit eligibility control (internal lifecycle tests). |
| `evaluate_prediction(actual)` | `pub` | Reality feedback. Applies correction to staged prediction. |
| `force_consolidate()` | `pub` | Explicit consolidation outside the automatic pressure trigger. |

Two additional helpers sit beside the body:

- `teach()` in `teach.rs` — curriculum harness around the living loop:
  runs input through `ingest_sequence`, stages prediction, calls `evaluate_prediction(actual)`.

- `evaluate()` in `three_cell.rs` — read-only query:
  composes a sequence externally and runs `RESONATE` without mutating the body.

---

### 2.7 The hierarchy: multi-level abstraction

`Hierarchy` holds a stack of genomes: `genomes[0]` is the perceptual level; `genomes[1]` is the first abstract level; etc.

Promotion happens when a `Response` coalition in `genomes[0]` accumulates:
- `activation_count ≥ PROMOTION_MIN_ACTIVATIONS` (= 2, the prime-2 orbit period)
- `mean_co_resonance(i, j) ≥ CO_RESONANCE_FLOOR` with at least one partner

A promoted cluster is injected into `genomes[1]` via `genome.ingest()`. The same ingest routing applies at every level. The hierarchy is self-similar by construction.

---

### 2.8 Consolidation: sleep

`consolidate()` runs over the epigenetic layer (never DNA) in three passes:

**1. Merge** — two BKT-alive epigenetic entries within `merge_threshold` of each other SLERP into one. Their activation counts, support, and edges union. The result is a broader attractor basin.

**2. Prune** — entries with mean ZREAD coupling below `BKT_THRESHOLD` (0.48) are removed. These are in the disorder phase and do not carry load-bearing signal.

**3. Reorganize** — sequential edges are remapped after removals.

**When to run:** automatically when internal σ-pressure crosses a threshold; explicitly via `cell.sleep()` (after a complete sequence) or `cell.force_consolidate()`.
The level-0 automatic trigger is `SIGMA_BALANCE`.

**What it preserves:** DNA entries are never touched. The ratchet effect guarantees that permanently-seeded anchors survive all consolidation passes.

---

### 2.9 Measurement signals: what the brain reports per step

Every `ingest()` returns a `Step`. The diagnostic fields:

| Field | Formula | Meaning |
|---|---|---|
| `prediction_error` | σ(cell_c, incoming) | External free energy. How surprised was the accumulated model at this input? |
| `self_free_energy` | σ(cell_c, ZREAD(cell_c)) | Internal free energy. How surprised is cell_c at its own genome? |
| `valence` | prev_sfe − sfe | Signed coherence change. > 0 means this step drove the brain toward self-consistency. |
| `arousal_tone` | low-pass(step_pressure / SIGMA_BALANCE) | Slow activation integral from the architecture's own per-step pressure mass. |
| `coherence_tone` | low-pass(valence / (π/2)) | Slow integral of whether recent updates improved or destabilized self-consistency. |
| `cell_a_sigma` | σ(cell_a) | Distance of the raw accumulator from identity. Measures total drift from rest. |
| `cell_c_sigma` | σ(cell_c) | Distance of the prediction state from identity. |

**Free energy convergence:** as training progresses, `self_free_energy → 0`. When it reaches zero, cell_c is a fixed point of the genome's own field — the brain's accumulated model is self-consistent. This is convergence to `Fix(ER)` (the fixed point of the entropy reduction functor).

**Valence interpretation:**
- First ingest: valence < 0 (empty brain has perfect self-consistency; first observation disrupts it)
- Repeated familiar input: eventually valence > 0 (brain integrates the pattern, self-consistency improves)
- Genuine novelty: persistent valence < 0 (pattern not yet in genome, disruption not yet resolved)

**Neuromodulatory interpretation:**
- `arousal_tone` rises when a step contributes substantial level-0 σ-pressure
- `coherence_tone` rises when recent steps reduce self-free-energy
- `coherence_tone` is written into Response-entry history during `credit_response()`
- promotion reads that history through `mean_coherence()`
- criterion 6 only speaks once the entry has at least `PROMOTION_MIN_ACTIVATIONS`
  worth of coherence history

The current runtime uses neuromodulation to accumulate body state and gate
promotion coherence history. The write-rate multiplier is baseline `1.0`,
and consolidation fires at `SIGMA_BALANCE`.

---

### 2.10 The curriculum: teach.rs

`teach.rs` provides directed learning drivers over the living body.
It is the codebase's explicit answer to:

```text
How do I present ordered experience to the brain and let reality correct it?
```

```rust
// Teach one (input, target) pair:
let loss = teach(&mut brain, &[input_carrier], &target_carrier);
// loss = σ(predicted, target)
```

The prediction comes from the brain's `field_read` during input ingestion.
That means `teach()` is body-native:

1. `ingest_sequence(input)`
2. take the last `field_read` as the prediction
3. `commit_prediction(predicted, source)`
4. `evaluate_prediction(target)`

This is different from `evaluate()`, which is read-only query.

**Convergence:** repeated presentation of (input, target) pairs causes the genome to reinforce the target slot's weight until RESONATE returns it as argmax. No gradient descent. Convergence is driven by the geometry of repeated closure.

---

### 2.11 Phased constants: where numbers come from

All thresholds in the system are derived, not tuned. The derivation chain:

```
S³ BKT critical coupling: 0.96/√4 = 0.48
                                    ↓
               BKT_THRESHOLD = 0.48   (disorder-order phase boundary)
                                    ↓
        CO_RESONANCE_FLOOR = 0.48² ≈ 0.2304   (min joint coupling of two alive entries)

Neighborhood geometry: π/3 is the natural Partzuf boundary
    cos(π/3) = 0.5
                                    ↓
                     ZREAD_T_MIN = 0.5   (participation gate)

Dobrushin prime frame (independent corroboration):
    t(p=2) = 1/√2 ≈ 0.707  [PRIME_2_COUPLING]
    t(p=3) = 1/√3 ≈ 0.577  [PRIME_3_COUPLING]
    t(p=5) = 1/√5 ≈ 0.447  [PRIME_5_COUPLING]
    BKT_THRESHOLD = 0.48 ∈ (PRIME_5_COUPLING, PRIME_3_COUPLING)
    Primes 2 and 3 are above the Parochet; prime 5 and above are below.
```

Nothing is free-parameter tuned. The derivation is self-consistent across three independent frames (BKT physics, S³ neighborhood geometry, Dobrushin prime structure).

---

## Part III — Module Reference

### `sphere.rs` — S³ primitives
The substrate. Five functions. No state. No allocations.

### `hopf.rs` — Hopf fibration
Decomposition, coupling weights, address modes, channel classification. The geometry underlying all ZREAD and RESONATE operations.

### `embed.rs` — Input boundary
Bytes → S³. Vocabulary for structured placements. MusicEncoder for pitch/rhythm. The only place external data enters.

### `verify.rs` — VERIFY primitive
Given two carriers, returns: σ (geodesic distance), ClosureKind (identity/balance/open), HopfDominance (which Hopf component dominates the gap), full VerificationEvent. Used on every ingest step.

### `carrier.rs` — VerificationCell
Wraps a carrier with its geometric metadata: Euler plane, phase, turns, sheet, coherence width. The `VerificationCell::from_geometry_or_default()` constructor is the only way to create a genome address.

### `buffer.rs` — Transient buffer
Rolling window of recent inputs. Present for ZREAD reads alongside genome entries. Cleared on every ingest (entries expire).

### `genome.rs` — Persistent memory
Three-layer array of GenomeEntry. Ingest routing, BKT constants, co-resonance recording, Response layer write path.

### `field.rs` — Field machine
RESONATE (nearest neighbor), ZREAD (path-ordered population read), Response eligibility collection (raw + inhibited), lateral inhibition.

### `localization.rs` — Closure localization
Given a sequence of carriers, find the minimal backward window that achieves closure. Used by Cell A to produce the localized packet composed into Cell C.

### `hierarchy.rs` — Multi-level emission
ClosureEvent propagation upward through genome levels. Cascade detection. Each level is a separate Genome instance with its own consolidation.

### `consolidation.rs` — Sleep
Merge + prune + reorganize over epigenetic layer. Promotion candidate detection. Category birth.

### `three_cell.rs` — The diabolo
The complete brain: Cell A, Cell C, Hierarchy, Buffer, pending prediction, two verbs (ingest, evaluate_prediction), curriculum helpers, BrainState serialization.

### `neuromodulation.rs` — Slow body state
Per-step integration of level-0 pressure and signed valence into
`arousal_tone` and `coherence_tone`. Session-ephemeral state carried by
`ThreeCell`, with coherence history forwarded into `GenomeEntry` during
`credit_response()`.

### `teach.rs` — Curriculum drivers
Directed learning over the live runtime. Curriculum traces for reproducible experiments. Convergence measurement.

### `zeta.rs` — Geometric zero detection
Hurwitz-zeta Euler product computation on S³. Riemann zero detection via W-component minimum tracking. The analytical thread connecting the architecture to the critical line.

---

## Part IV — Domain Adapter Pattern

The core crate is domain-independent. Domain-specific encoding lives in
`examples/` as adapters following this pattern:

```
1. Register domain vocabulary (Vocabulary::register, or use domain_embed directly)
2. Optional: seed DNA with domain axioms (genome.seed_dna)
3. Training loop: ingest input, commit_prediction, evaluate_prediction(label)
4. Consolidation: cell.sleep() after each sequence
5. Evaluation: cell.ingest(input).prediction_error
```

**Reference examples:**

- `exp_arithmetic` — Z/nZ orbits; orbit seeding and modular arithmetic on S³
- `exp_riemann_zeros` — Euler product scan; W-component minima detection
- `exp_associative_memory` — Sigma-radius recall; genome population and query
- `gray_game_live` — Cellular automaton domain; three interference regimes (spectrum, resonance, edge)
- `exp_neuromodulated_learning` — Full supervised loop with neuromod observables

**Domain-specific vs core:**

- Vocabulary placement → adapter
- Domain axiom seeding → adapter
- Effector output (decoding the prediction) → adapter
- All geometry, all memory, all learning → core

---

## Part V — What Is Not Here (By Design)

| Feature | Why absent |
|---|---|
| Gradient descent | Convergence driven by geometric closure, not optimization |
| Softmax | Coupling function cos(σ) gates by geometry; no normalization required |
| Floating-point loss surface | Loss = σ (geodesic distance). Already dimensionless and bounded [0, π/2]. |
| Backpropagation | No computational graph. Write path is forward-only (SLERP toward correction). |
| Batch normalization | SLERP toward IDENTITY serves the same smoothing purpose |
| Dropout | BKT pruning replaces stochastic regularization with a principled phase boundary |
| Adam/SGD optimizer | No optimizer. Convergence is a fixed-point theorem, not a tuning problem. |
| Tokenizer vocabulary size | No discrete vocabulary. Every byte sequence has a continuous carrier. |
| Architecture hyperparameters | Four GenomeConfig parameters, all derived from geometry (see §2.11) |
