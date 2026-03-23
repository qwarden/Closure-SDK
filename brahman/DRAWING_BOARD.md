# Drawing Board — Generative Closure from First Principles

## Foundation

Binary closure is the atom of learning. The bracket experiment proved
it: gradient descent on S³ discovers compositional inverses from data
alone. `(` maps to quaternion q. `)` maps to q⁻¹. A valid sequence
composes to identity. σ = arccos(|w|) = 0.

This is not a toy problem. This IS learning at its irreducible case.

Every compositional relationship between any two elements reduces to
a closure question: does A·B → identity? Binary.

When composition fails, the Hopf fibration decomposes S³ = S² × S¹
into exactly two orthogonal failure axes:

- **Missing** (S¹ fiber / W): an element that should exist doesn't
- **Reorder** (S² base / RGB): all elements exist, wrong arrangement

These are exhaustive by geometry. There is no third axis.

No event in the universe escapes these two categories.

---

## Experimental record

### Experiment 1: Pure geometry (original Drawing Board hypothesis)

**Hypothesis:** Strip ALL neural overhead. Embedding table only +
Hamilton product + σ loss. Generate via geodesic nearest neighbor
to C⁻¹.

**Model:** S3Pure — `nn.Parameter(V × m × 4)`, no attention, no
prediction head, no cross-entropy.

```
S3Pure on brackets: 3 tokens × 4 quaternion components = 12 parameters.
Loss: σ(valid) + max(0, margin - σ(corrupted)). No cross-entropy.
200 epochs, 5,000 sequences.

Inverse discovery:    FAIL — σ( ( · ) ) = 1.26  (should be ~0)
σ separation:         MARGINAL — t=2.81, pair accuracy 55.4%
                      (σ valid = 0.74, σ corrupted = 0.83)
Generation:           FAIL — deterministic: 0%, temp=0.3: 16%
                      Always generates ((· — stuck in local minimum.
```

**Follow-up: per-step σ (dense gradient signal).**

Hypothesis: σ_final gives one gradient signal at the end of a long
chain. Use σ at EVERY position (sum across all steps) to give each
token a direct, one-step gradient.

```
S3Pure with per-step σ loss (sum across all positions):
300 epochs, same 12 parameters.

Inverse discovery:    FAIL — σ( ( · ) ) = 1.04
σ separation:         FAIL — t=1.05, pair accuracy 50.4%
Generation:           FAIL — generates ((((((((((((((((((((
                      with σ = 0.045 (near zero!)
```

**Root cause:** "Minimize σ everywhere" has a degenerate solution:
collapse ALL embeddings to near-identity. Then σ ≈ 0 for everything.
The model generates infinite open brackets because embed(`(`) ≈
identity, so composing any number stays near identity and σ stays low.

**Conclusion:** σ alone cannot distinguish "closes because inverses
cancel" from "everything is identity." Both give low σ. The
optimization landscape around the σ-only loss has a degenerate
global minimum that gradient descent finds before the inverse solution.

### Experiment 2: S3Transformer with full overhead

**Model:** S3Transformer — embedding network (Linear→GELU→Linear),
4 attention layers with learned quaternion rotations (S3Attention),
quaternion composition residual (S3Block), prediction head
(Linear→GELU→Linear). Cross-entropy + closure loss.

```
Brackets (closure_weight=0.3):
  S3Transformer, 1,351 params. 4 layers, m=1. 30 epochs, 3.5 min CPU.
  σ separation:  t=83.32, pair accuracy 98.8%
  Generation:    94.3% valid, avg length 19.7 tokens
  Inverses:      σ( ( · ) ) ≈ 0.04 — DISCOVERED

Grid walk (closure_weight=0.0, pure next-token prediction):
  S3Transformer, 2,205 params. 4 layers, m=2. 100 epochs, CPU.
  Generation:    98.3% closed walks
  σ separation:  FLAT (t=1.09) — geometry not used for detection
  Inverses:      NOT discovered (σ(UP·DOWN) ≈ 0.81)
```

**Key finding:** Pure next-token prediction on closed walks achieves
98.3% generation quality WITHOUT any σ training. The model learns
closure from token statistics — attention counts what's in the prefix
and predicts what's needed. The geometry provides the manifold, but
the model doesn't need to be told about closure explicitly.

**Key finding:** σ separation and generation quality compete.
closure_weight=0.3 gives σ separation but collapses generation.
closure_weight=0.0 gives generation but σ goes flat. For detection,
train with σ. For generation, train with cross-entropy. They serve
different purposes.

### Experiment 3: Minimum overhead sweep (grid walk)

**Question:** The full S3Transformer works. Pure geometry fails. What
is the MINIMUM neural overhead that still generates closed walks?

**Method:** Systematic ablation on grid walk (5-token vocab, m=2).
Pure next-token prediction (closure_weight=0.0). 20,000 closed walks
training set, 50 epochs, 500 generated walks per test. Varied:
- Embedding: "network" (Linear→GELU→Linear) vs "lookup" (nn.Embedding)
- Head: "network" (Linear→GELU→Linear) vs "linear" (single nn.Linear)
- Attention layers: 4, 2, 1, 0

**Training data: closed walks ONLY.** No corrupted examples, no
contrastive loss, no negative samples. The model sees only valid
sequences. First principles: if the data has a geometric shape
(closure on S³), the model learns that shape from positive examples
alone. This matches the bracket experiments where generation succeeded
from valid sequences only.

```
Config              Params  Closed/500    %     Time
──────────────────  ──────  ──────────  ─────  ─────
4L net+net           1,309   479/500    95.8%   94s   ← full model (baseline)
4L net+lin             901   391/500    78.2%   96s   ← stripped head
4L lookup+net          893   341/500    68.2%   93s   ← stripped embed
4L lookup+lin          485   246/500    49.2%   89s   ← stripped both
2L lookup+lin          421   202/500    40.4%   50s   ← fewer layers
1L lookup+lin          389   164/500    32.8%   31s   ← minimal layers
0L lookup+lin          357    49/500     9.8%   11s   ← no attention
0L net+net           1,181    45/500     9.0%   15s   ← no attention, full networks
0L net+lin             773    56/500    11.2%   13s   ← no attention, full embed
```

**Analysis — what each component contributes:**

**Attention layers are essential.** 0 layers = ~10% regardless of
embed/head configuration (0L net+net = 9.0%, 0L net+lin = 11.2%,
0L lookup+lin = 9.8%). Without attention, each position sees only
its own embedding + positional embedding. It cannot attend to previous
tokens. It cannot count how many UPs vs DOWNs have occurred. It
cannot predict the next token based on context. The ~10% baseline
is the random closure rate for walks of this length distribution.

Attention is not overhead — it IS the context mechanism. The S³
geometry provides the manifold and the composition algebra, but the
model needs a way to see the whole sequence. Attention on S³ (geodesic
dot product + learned rotations) is that mechanism. The question is
how many layers, not whether to have them.

**The embedding network matters more than the prediction head.**
Stripping the embed (4L lookup+net = 68.2%) costs more than stripping
the head (4L net+lin = 78.2%). The embed is the translator INTO the
geometry — it maps raw tokens to points on S³. A richer embedding
network (Linear→GELU→Linear with hidden layer) gives gradient descent
more surface to navigate the mapping from discrete tokens to the
continuous manifold. nn.Embedding is a direct lookup — each token gets
one fixed point, with no hidden layer to provide nonlinear optimization
paths between tokens.

The prediction head reads OUT of the geometry. A single Linear layer
(8D → 5 logits for grid walk) loses 17 percentage points vs the full
network head. The full head (Linear→GELU→Linear) has a hidden layer
that can compute nonlinear functions of the geometric state before
producing logits. This matters because the relationship between a
point on S³ and the correct next token is not a linear function — it
depends on the geodesic structure. But it matters LESS than the embed
because the geometry does most of the work in between.

**Layer count degrades gradually.** 4L → 2L → 1L with minimal
embed/head (lookup+lin): 49.2% → 40.4% → 32.8%. Each layer adds
one round of geodesic attention + quaternion composition. More layers
= more rounds of "look at context, compose, look again." The grid
walk task (5 tokens, short sequences) is simple enough that even 1
layer captures some structure, but 4 layers captures significantly
more.

**Stripping embed AND head together is catastrophic.** 4L lookup+lin
= 49.2%, barely above chance for this task. The geometry alone (with
attention) can route information, but without adequate translation
layers at both ends, it can't map between token space and S³ space
well enough for accurate prediction.

### What the sweep proves about the architecture

**The minimum viable model for generation has three non-negotiable
components:**

1. **Embedding network** — not a bare lookup table, but a small
   network (at least one hidden layer) that maps tokens to S³. This
   provides the optimization surface gradient descent needs to learn
   the token-to-quaternion mapping. The Drawing Board's bare
   `nn.Parameter` table failed because it lacks this optimization
   surface. nn.Embedding (direct lookup) degrades to 68% even with
   full attention.

2. **Attention layers** — at least 1 layer of S3Attention, which
   computes geodesic dot products between unit quaternions at each
   position, weighted by learned rotations (4×4 per S³ factor). This
   is how the model sees context. 0 layers = random. 1 layer = 33%.
   4 layers = 96%. The quaternion composition residual in each S3Block
   (qmul of input with attended output, then normalize) is the
   nonlinear transform that replaces the FFN in standard transformers.

3. **Prediction head** — at least a single Linear layer mapping from
   S³ to token logits. A full network head (with hidden layer) is
   better (+17 points) but not strictly required. The head's job is
   simpler than the embed's — it reads a point on a known manifold,
   rather than mapping from an arbitrary discrete space into one.

**The minimum that works well:** Full embedding network + 4 attention
layers + full prediction head = 1,309 parameters at 95.8%.

**The minimum that works at all:** Full embedding network + 1 attention
layer + single Linear head ≈ ~500 parameters. Not tested in this exact
config but interpolated from the sweep: 1L with full embed should be
~50-60% (between 1L lookup+lin at 32.8% and 4L net+lin at 78.2%).

**The theoretical minimum:** The embed network + attention + head
architecture IS the minimum. None of the three components can be
removed. They can be made smaller (fewer layers, smaller hidden dims)
but not eliminated.

---

## Revised mechanism

### Architecture: S3Transformer with cross-entropy

The experiments converge on one architecture for generation:

```
Input: integer tokens
  ↓
Embedding network: token → unit quaternion on (S³)^m
  Linear(vocab_size, hidden) → GELU → Linear(hidden, 4m) → normalize
  ↓
Positional embedding: add learned positional quaternions, normalize
  ↓
N × S3Block:
  S3Attention: geodesic dot product (4×4 learned rotation per factor)
  Residual: qmul(input, attended) → normalize
  ↓
Prediction head: unit quaternion → logits over vocabulary
  Linear(4m, hidden) → GELU → Linear(hidden, vocab_size)
  ↓
Loss: cross_entropy(predicted_next, actual_next)
```

**σ is computed at every position for free** (arccos of the w-component
of each S³ factor, averaged). It is NOT part of the loss. It is a
diagnostic channel that separates coherent from incoherent sequences
as an emergent property of the learned geometry.

**Training data: coherent sequences only.** No corrupted examples
needed. The model learns the shape of the data. If the data is closed
walks, it learns closure. If the data is grammatical English, it learns
grammar. First principles: things default to their true shape.

### Parameter budget

For grid walk (5 tokens, m=2, 4 layers, hidden=32):
```
Embedding:  Linear(5, 32) + Linear(32, 8)    =  416 params
Positions:  34 × 8                             =  272 params
Attention:  4 layers × (2 factors × 4×4)      =  128 params
Head:       Linear(8, 32) + Linear(32, 5)      =  421 params
Biases:                                         =   72 params
Total:                                          = 1,309 params  → 95.8% closed
```

For brackets (3 tokens, m=1, 4 layers, hidden=32):
```
Total:                                          = 1,351 params  → 94.3% valid
```

For character-level language (96 tokens, m=20, 6 layers, hidden=64):
```
Embedding:  nn.Embedding(96, 64) + Linear(64, 80)  =  11,264 params
Positions:  513 × 80                                 =  41,040 params
Attention:  6 layers × (20 factors × 4×4)            =   1,920 params
Head:       Linear(80, 64) + Linear(64, 96)           =  11,264 params
Biases:                                               =     400 params
Total:                                                ≈  65,888 params
```

The attention layers account for 1,920 of 65,888 parameters (2.9%).
The embedding + head account for 22,528 (34.2%). Positional embeddings
account for 41,040 (62.3%). The geometry (quaternion composition,
normalization, geodesic distance) is parameter-free.

For language, positional embeddings dominate because max_seq_len is
large. This is a strong argument for the recursive architecture: if
each level processes short sequences (10-15 tokens), positional
embeddings shrink from 41K to ~1K per level.

---

## The minimum viable model problem

### What we know

The sweep establishes a clear hierarchy of component importance:

```
Attention layers >> Embedding network > Prediction head
```

Attention is binary — without it, the model is random. The embed
network is the most important learned component — it's the bridge
from discrete token space to continuous S³. The head is the least
critical learned component but still contributes significantly.

### What we don't yet know (for language)

The grid walk has 5 tokens. Language has 96. The grid walk has trivial
structure (count displacements). Language has grammar, semantics,
narrative. The sweep results give the shape of the answer but not the
exact numbers for language.

**Specific unknowns for the language model:**

1. **How many attention layers?** Grid walk needs 4 for 96%. Language
   is more complex — may need 6-8. But with the recursive architecture,
   each level processes short sequences with simple structure, so
   fewer layers per level may suffice.

2. **How many S³ factors (m)?** Grid walk uses m=2 (2 DOF for x/y
   displacement — natural match). The Colab used m=20 (80D). The
   bracket test passed with m=1. The right m for language is unknown
   — this is the dimensionality experiment from BRAHMAN.md. The sweep
   should be m=1,2,4,8,16,20,32 on the same corpus.

3. **Embedding hidden dimension?** Grid walk uses hidden=32. Language
   may need more. The embed network maps 96 tokens to (S³)^m — with
   m=20 that's 96 inputs to 80 outputs through a hidden layer. The
   hidden dimension controls the capacity of this mapping.

4. **Does nn.Embedding + Linear work as well as Linear + GELU + Linear?**
   The Colab notebook used nn.Embedding(vocab_size, hidden) → GELU →
   Linear(hidden, dim). This is a hybrid — direct lookup into a
   hidden space, then nonlinear projection to S³. May be better than
   one-hot → Linear → GELU → Linear because nn.Embedding avoids the
   sparse one-hot multiplication. Needs testing.

### The critical experiment: minimum overhead for language

Before running expensive multi-hour TinyStories training, we need the
same sweep we did for grid walks but on brackets (which train in
minutes) with the language-relevant question: **does the component
hierarchy (attention >> embed > head) hold for richer vocabularies
and longer sequences?**

If it does, the minimum viable language model is:
- Full embedding network (Linear→GELU→Linear or nn.Embedding→GELU→Linear)
- N attention layers (N to be determined by sweep)
- Full or single-Linear prediction head
- Pure cross-entropy, σ as diagnostic
- Trained on coherent text only

If the hierarchy changes for language (e.g., head becomes more
important with larger vocabularies), the minimum shifts accordingly.

---

## Recursive Enkidu

### The recursion

A closure element C is 4m numbers. Those numbers are data. Data
embeds on S³. So closure elements from level N become tokens at
level N+1.

Each level is the same operation:

```
EnkiduLevel:
    input:   stream of quaternions (raw tokens or closure elements from below)
    state:   running product C (identity initially)
    output:  closure elements (tokens for the next level)

    for each input quaternion q:
        C = normalize(C · q)
        σ = arccos(|C.w|)

        if σ < threshold:
            emit C as a token to level N+1
            reset C = identity

        if σ > 0:
            hopf = decompose(C)   → (σ, R, G, B, W)
            if |W| > |RGB|: incident = missing
            if |RGB| > |W|: incident = reorder
```

### What the levels discover

Level 0 operates on character embeddings. When a subsequence of
characters composes to near-identity, that subsequence is a
coherent unit — a morpheme, a word. Not defined by a dictionary.
Defined by algebraic closure.

Level 1 operates on level-0 closure elements. When a subsequence
of words composes to near-identity, that's a coherent phrase or
clause.

Level 2 operates on level-1 closure elements. Coherent paragraphs,
arguments, ideas.

Same Enkidu at every level. Same two failure modes. The hierarchy
emerges from the algebra, not from architectural decisions.

### Why the recursion solves the training bottleneck

The TinyStories flat run threw 512 characters at a single transformer.
Result: 16K tok/s, ~3 hours per epoch, gibberish at BPC 3.5. The
attention computed O(T²) scores across 512 positions with m=20
factors. The model was trying to learn character patterns, word
structure, grammar, and narrative simultaneously in one flat pass.

The sweep shows why this fails: attention is the essential component,
but attention cost scales as O(T²). With T=512, each attention layer
computes 262,144 score entries per sample. With m=20 factors, that's
5.2M geodesic dot products per layer. 6 layers = 31M per sample.

The recursive architecture sidesteps this entirely:

```
Flat model:        1 level  × T=512    → O(512²) = 262,144 attention scores/layer
Recursive model:   3 levels × T≈10 each → O(10²) × 3 = 300 attention scores/layer
```

That's a ~870× reduction in attention computation. Each level
processes SHORT sequences:

- Level 0 sees ~5–10 characters (a word-length unit)
- Level 1 sees ~3–8 word-level closure elements (a phrase)
- Level 2 sees a few phrase-level elements (a sentence)

Long-range structure accumulates across levels instead of within
them. A paragraph is not modeled by attention across 2000 characters
— it's modeled by level 2 attending across 5–10 sentence-level
closure elements.

### The anti-collapse property of recursion

The Drawing Board's S3Pure failed because σ alone lets embeddings
collapse to identity. Cross-entropy prevents this in the flat model.
The recursion may provide its own anti-collapse mechanism:

If characters collapse to identity at level 0, then all character
compositions produce the same closure element (identity). All
"words" become identical. Level 1 receives a stream of identical
tokens. Level 1's prediction loss (cross-entropy on the next
word-level element) is maximally high because it can't distinguish
any input. The gradient from level 1's loss flows back through the
composition operation (differentiable: just the running product C
at the threshold crossing) to the character embedding table, pushing
character embeddings apart.

The hierarchy creates its own anti-collapse pressure. Degenerate
embeddings at level 0 → degenerate tokens at level 1 → high loss
at level 1 → gradient pushes level 0 apart.

**Testable prediction:** Train recursive Enkidu with prediction loss
at level 1 ONLY (no character-level cross-entropy). If level-1 loss
alone forces character embeddings into distinct, compositionally
meaningful positions, then the recursion IS the anti-collapse
mechanism, and character-level cross-entropy is redundant overhead
at level 0.

If this works, the minimum overhead at level 0 drops to: embedding
network + attention layers + NO prediction head. Level 0 only needs
to compose characters into closure elements for level 1. The
prediction head is only needed at the TOP level (where generation
targets originate) and at the BOTTOM level (where characters are
emitted).

### Generative recursion

When Enkidu at level N has σ > 0:

**Missing (W-axis):** C⁻¹ is the closure element that would
complete the composition. Pass C⁻¹ down to level N-1 as a
generation target. Level N-1 generates a sequence of its tokens
whose composition approximates C⁻¹. If N-1 = 0, those tokens
are characters. If N-1 > 0, recurse down.

**Reorder (RGB-axis):** The existing level N-1 subsequences are
correct but misordered. The RGB displacement vector indicates
the direction of misalignment. Permute the existing subsequences
to minimize RGB displacement.

Generation recurses downward. Classification recurses upward.
Same algebra both directions.

### Minimum overhead per level (revised with sweep data)

The sweep proved three components are non-negotiable in a flat model.
In the recursive architecture, the requirements differ by level:

**Level 0 (characters → word-level closure elements):**
- Embedding network: REQUIRED (maps discrete tokens to S³)
- Attention layers: REQUIRED (context mechanism — 0 layers = random)
- Prediction head: REQUIRED for flat training, MAYBE redundant if
  level-1 loss provides anti-collapse (see testable prediction above)
- Training signal: cross-entropy on next character (or level-1 loss
  flowing down — to be tested)

**Level 1+ (closure elements → higher closure elements):**
- Embedding: MAYBE TRIVIAL. Input tokens are already unit quaternions
  on (S³)^m — they're closure elements from the level below. The
  "embedding" may be identity (no network needed). Or a small
  network may help gradient flow even though the input is already
  on the manifold. To be tested.
- Attention layers: REQUIRED (same argument as level 0 — need context)
- Prediction head: REQUIRED at the top generative level. Intermediate
  levels may only need to emit closure elements (no prediction).
- Training signal: cross-entropy on next closure element at that level

**The most optimistic minimum:** neural overhead exists only at the
boundaries — level 0 (raw data → S³) and the top level (S³ → generated
output). Everything in between is pure algebra: attention (geodesic
dot products, no learned embed/head) + quaternion composition.

---

## Implementation path

### Step A: Bracket overhead sweep (validates sweep for richer tasks)

Repeat the grid walk overhead sweep on brackets. Same configs (embed
type, head type, layer count). Brackets have 3 tokens (simpler than
grid walk's 5) but longer sequences (up to 32 tokens) and deeper
nesting structure. If the component hierarchy holds (attention >>
embed > head), the finding generalizes.

Also test: nn.Embedding→GELU→Linear vs Linear→GELU→Linear (one-hot)
as embedding. The Colab notebook used the former. Determine if
nn.Embedding is equivalent or better.

### Step B: Recursive composition on nested brackets

Multi-level brackets: `{[()]}`. 6-token vocab: `(`, `)`, `[`, `]`,
`{`, `}`, EOS.

Build a 2-level recursive model:
- Level 0: small S3Transformer, processes characters, emits closure
  elements when σ < threshold
- Level 1: same architecture, processes level-0 closure elements,
  predicts next element

Train on valid nested brackets only. Test:
- Does level 0 discover `()` as a closure unit?
- Does level 1 discover `[()]` and `{[()]}` as higher units?
- Does generation produce valid nested brackets?
- Does level-1 loss alone (no level-0 cross-entropy) learn correct
  level-0 embeddings? (anti-collapse test)

### Step C: Character-level recursive training on text

Train level 0 on short character windows (10-15 chars) from real text.
When level 0 emits closure elements (σ < threshold), collect them as
training data for level 1. Train level 1 on sequences of word-level
closure elements.

Compare against the flat TinyStories run:
- Throughput: should be >>16K tok/s (short sequences, small models)
- BPC: at character level, measure prediction accuracy
- σ separation: at each level, measure coherence detection
- Generation: prompt → level 2 target → level 1 decomposition →
  level 0 character emission

### Step D: Minimum overhead sweep for language

Same sweep as grid walks but on character-level text:
- Embed type: network vs lookup vs nn.Embedding+GELU+Linear
- Head type: network vs linear
- Layers per level: 1, 2, 4, 6
- m factors: 1, 2, 4, 8, 16, 20
- Recursive vs flat

This determines the minimum viable language model configuration.

---

## What this replaces (with experimental evidence)

| Standard transformer          | S³ recursive closure             | Evidence |
|-------------------------------|----------------------------------|----------|
| Embedding layer (V × d)       | Small network (V → hidden → 4m)  | Sweep: network >> lookup |
| Positional encoding            | Composition order IS position     | Theorem 1 |
| Multi-head attention (Q/K/V)   | Geodesic distance + 4×4 rotations| Sweep: 0L = random, 4L = 96% |
| Feed-forward network           | Quaternion multiplication (qmul) | S3Block: no FFN, works |
| Layer normalization             | Unit sphere constraint (free)    | normalize() replaces LayerNorm |
| Residual connections            | Quaternion composition (exact)   | F.normalize(qmul(x, attended)) |
| Softmax over vocabulary         | Cross-entropy prediction head    | Sweep: head contributes +17pts |
| σ loss / contrastive           | NOT NEEDED for generation        | Grid walk: 0.0 closure = 98.3% |
| Long-range O(T²) attention    | Recursive hierarchy (O(T²) on short T) | Theoretical: 870× reduction |
| ~175B parameters               | Small model × N recursive levels | To be validated on language |

---

## Open questions (ordered by priority)

1. **Does the component hierarchy hold for language?** The grid walk
   sweep found attention >> embed > head. If this changes with 96
   tokens and real text structure, the minimum viable model changes.
   Test: bracket sweep (Step A), then character text sweep (Step D).

2. **Does recursive anti-collapse work?** If level-1 loss alone
   forces level-0 embeddings apart, the minimum overhead at level 0
   drops (no prediction head needed). This is the most impactful
   unknown for the recursive architecture.

3. **How many S³ factors for language?** m=2 for grid walk (2 DOF).
   m=1 for brackets. Language needs more — but how many? The
   dimensionality sweep (m=1..32) on the same corpus answers this.

4. **Emission threshold.** σ < ε triggers level emission. ε
   determines granularity. Could be learned (one scalar per level)
   or fixed. Tight ε = character fragments, loose ε = long phrases.

5. **Level count for English.** Characters → words → phrases →
   sentences → paragraphs = 4-5 levels. But the algebra might
   discover different boundaries (morphemes? clauses?). The emission
   threshold and the data together determine this.

6. **Do upper levels need embedding networks?** Their input tokens
   are already on S³ (closure elements). The embed step may be
   identity. If so, neural overhead exists only at level 0.

7. **Factor specialization.** With m factors, do different S³
   factors learn different linguistic aspects? Testable: freeze
   all but one factor, measure σ separation on syntax vs semantics
   vs phonetics tasks.
