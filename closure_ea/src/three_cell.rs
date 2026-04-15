//! The diabolo — Cell B riding the critical line between the two
//! machines.
//!
//! Phase 4 of BRAIN.md. The three-cell loop reframed around the
//! seven-step receive-and-verify loop of §9:
//!
//! ── Kabbalistic correspondence ──────────────────────────────────────
//!
//! **The Observer (Senchal 2026, Def 2.2)** — ThreeCell IS the bounded
//! observer O defined by sampling functor S_O: R → R_O, constrained by
//! boundedness B_O, persistence P_O, and relevance Rel_O. The genome
//! is R_O: the observer's compressed slice of the Ruliad. Every ingest
//! call is one application of the entropy reduction functor ER.
//!
//! **Cell A (fast oscillator) = solenoidal component** — the running
//! Hamilton product of raw input. Entropy-preserving: it records what
//! actually arrived without compressing it. This is the Q-mediated
//! circulation in Friston's HHD — the deterministic, entropy-neutral
//! component that drives active inference. Currently the least
//! developed cell; the solenoidal gap (Senchal §9.3, open question Q8)
//! lives here.
//!
//! **Cell C (slow accumulator) = gradient component** — accumulates
//! closure events. Entropy-reducing: each closure is a compression that
//! drives the observer toward Fix(ER). This is the Γ-driven dissipative
//! component of NESS dynamics. Well-implemented; the convergence
//! theorems (Thm 5.2, 6.7) apply to this cell's trajectory.
//!
//! **Seven-step loop = one Sefirotic emanation** — each ingest is one
//! application of S: C_n → C_{n+1}, reducing |Mor(C)| by enforcing the
//! VERIFY closure constraint. The loop is the quantitative instantiation
//! of the Sefirotic tree: embed → buffer → ZREAD (Partzuf activation) →
//! RESONATE (hard selection) → LOAD → VERIFY → emit.
//!
//! **ThreeCell = a Partzuf of Partzufim** — the cell is itself a stable
//! network (Partzuf) whose internal components (genome clusters) are the
//! lower Partzufim. The cell's output is the "face" of this meta-Partzuf:
//! the coherent carrier produced by the whole genome responding to input.
//!
//! **Self-observation** — a complete Observer must be able to sample its
//! own Ruliad. `self_observe()` runs ZREAD with the cell's own current
//! state as query, returning the genome's integrated response to itself.
//! The σ-gap between cell state and self-carrier is free energy (FEP).
//! Cell C exposes its Hopf base, current W-depth, and lifetime minimum W
//! so the observer's slow self-model can be inspected and persisted.
//!
//! ```text
//!   1.  bytes ← outside world
//!   2.  c ← EMBED(bytes)
//!   3.  buffer ← buffer ∪ {c}
//!   4.  f ← ZREAD over (genome ∪ buffer)
//!   5.  s' ← RESONATE(f)
//!   6.  s ← LOAD(slot of s' in genome)
//!   7.  ev ← VERIFY(s, s')
//! ```
//!
//! The brain has two living verbs:
//!
//! - [`ThreeCell::ingest`] for perception
//! - [`ThreeCell::evaluate_prediction`] for delayed reality feedback
//!
//! Steps 1–2 (bytes through EMBED) happen outside the diabolo; the
//! caller passes a carrier. Steps 3–7 happen inside perception.
//! The evaluation half-loop writes correction back into memory after
//! reality judges the previously staged output.
//!
//! Cell A (the fast-oscillator running product) composes the **raw
//! input**. The seven-step loop runs in parallel to observe what the
//! field machine makes of that input, producing a
//! [`crate::verify::VerificationEvent`] on every step; Cell A
//! continues accumulating the received sequence independently. The
//! two mechanisms are complementary: the loop measures the gradient;
//! Cell A records what the brain has actually received.
//!
//! Cell C is the slow accumulator of closure events. On each closure,
//! the localized packet gets composed into Cell C and emitted through
//! the hierarchy into the genome.

use crate::buffer::Buffer;
use crate::consolidation::{consolidate, ConsolidationReport};
use crate::field::{
    collect_response_eligibility, collect_response_eligibility_raw, resonate,
    resonate_channel_with_mode, zread,
    zread_at_query_channel_with_mode, PopulationSource, ResonanceHit,
};
use crate::genome::{Genome, GenomeConfig, Layer};
use crate::hierarchy::{
    rebuild_from_prefix, ClosureEvent, ClosureKind, ClosureLevel, ClosureRole, Hierarchy,
};
use crate::hopf;
use crate::hopf::{semantic_frame, AddressMode, HopfChannel, SemanticFrame};
use crate::localization::{localize, localize_balance, localized_excursion_peak};
use crate::neuromodulation::NeuromodState;
use crate::sphere::{compose, inverse, sample_vmf_s3, sigma, Rng, IDENTITY};
use crate::verify::{verify, VerificationEvent, SIGMA_BALANCE};
use serde::{Deserialize, Serialize};

/// What one `ingest` call produced.
///
/// The diabolo doesn't act on this — it already acted by composing.
/// The caller reads the step to observe what happened.
#[derive(Debug)]
pub struct Step {
    /// The carrier the caller handed in (after EMBED).
    pub input: [f64; 4],
    /// Result of step 5 — the field machine's collapsed read.
    /// `None` iff the population (genome ∪ buffer) was empty when
    /// the carrier was pushed, which only happens on the very first
    /// `ingest` call against a fresh diabolo.
    pub field_read: Option<ResonanceHit>,
    /// Result of step 7 — the round-trip verification event between
    /// the field-collapsed read and the raw input. `None` on the
    /// same empty-population path as `field_read`.
    pub verification: Option<VerificationEvent>,
    /// Cell A's sigma *after* composing the raw input into the
    /// running product. On closure steps this reflects the state
    /// after the prefix rebuild.
    pub cell_a_sigma: f64,
    /// Sigma of Cell C — the brain's accumulated prediction.
    pub cell_c_sigma: f64,
    /// Prediction error this step: σ(cell_c, incoming_carrier).
    ///
    /// External free energy — geodesic distance between what Cell C
    /// predicted and what actually arrived from the world.
    pub prediction_error: f64,
    /// Self free energy this step: σ(cell_c, zread_at_query(cell_c)).
    ///
    /// The brain reading its own genome through the Hopf field.
    /// Cell C queries the genome with itself as the query, producing
    /// the field's integrated response to the brain's current state.
    /// The gap between Cell C and that response is self-free-energy:
    /// how surprised is the accumulated model at its own genome?
    ///
    /// When self_free_energy → 0: Cell C is a fixed point of the
    /// genome's field — the brain's model is self-consistent.
    /// When self_free_energy is large: the genome has structure that
    /// Cell C has not yet integrated — the brain is still learning.
    ///
    /// This is the critical-line measurement. The zeros of self-free
    /// energy over curriculum time trace the Riemann spiral: stable
    /// convergence events where the brain's past model aligns with
    /// its own learned structure.
    pub self_free_energy: f64,
    /// Signed coherence signal for this step.
    ///
    /// valence = self_free_energy(t-1) - self_free_energy(t)
    ///
    ///  \> 0 : coherence-driving — this carrier reduced the brain's
    ///         self-inconsistency; the update brought Cell C closer to
    ///         being a fixed point of its own genome.
    ///  \< 0 : decoherence-driving — this carrier increased self-
    ///         inconsistency; the update pushed Cell C away from the
    ///         genome's attractor (genuine novelty or disruptive signal).
    ///  ≈ 0  : neutral — no material change in self-consistency.
    ///         Also 0.0 on the very first ingest, when no previous SFE
    ///         measurement exists (no delta without two measurements).
    pub valence: f64,
    /// S¹ fiber classification of this position.
    /// `false` = even position = trivial fiber (identity phase).
    /// `true`  = odd position  = non-trivial fiber (π-phase).
    /// The caller applies the gate at the embedding stage; this records
    /// which fiber was active when the carrier entered the machine.
    pub fiber_nontrivial: bool,
    /// Set iff Cell A's composition triggered a closure.
    pub closure: Option<ClosureEvent>,
    /// Any higher-level closure events from upward emission.
    pub hierarchy_events: Vec<ClosureEvent>,
    /// Consolidation reports from this step, one per level that fired.
    pub consolidation_reports: Vec<RuntimeConsolidationReport>,
    /// Semantic triple for this step.
    ///
    /// `total` = the ZREAD aggregate over (genome ∪ buffer) queried by the
    /// incoming carrier — the field's collective belief about the input.
    /// `known` = Cell C before this step's update — the accumulated model.
    /// `residual` = `compose(total, inverse(known))`.
    /// `salience_sigma` = `sigma(residual)` — geodesic distance between field
    /// and model. Zero when the model perfectly predicted the field; approaches
    /// π/2 at maximum mismatch. Used to gate learning at evaluation time.
    pub semantic: SemanticFrame,
    /// Slow body state AFTER this step's update.
    ///
    /// `arousal_tone` — low-pass integral of `step_pressure / SIGMA_BALANCE`,
    /// where step_pressure is the architecture's per-step activation mass
    /// (sum of prediction error, SFE excess, genome growth, closure excursion).
    /// High when the brain has been processing strongly activating or surprising
    /// content recently. Low during familiar, well-predicted input.
    ///
    /// `coherence_tone` — low-pass integral of `valence / (π/2)`.
    /// Positive when recent updates have been improving self-consistency.
    /// Negative when recent updates have been destabilizing the model.
    pub arousal_tone: f64,
    pub coherence_tone: f64,
    /// Effective von Mises-Fisher κ used for Cell A noise this step.
    /// `None` if noise is disabled. Lower κ = more exploration.
    pub noise_kappa: Option<f64>,
}

/// Runtime consolidation report: structural consolidation plus the pressure
/// that caused it.
#[derive(Clone, Debug)]
pub struct RuntimeConsolidationReport {
    /// Genome level consolidated.
    pub level: usize,
    /// Accumulated σ-pressure immediately before consolidation.
    pub pressure_before: f64,
    /// Remaining σ-pressure after consolidation.
    pub pressure_after: f64,
    /// Level-0 self-free-energy before consolidation.
    pub self_free_energy_before: Option<f64>,
    /// Level-0 self-free-energy after consolidation.
    pub self_free_energy_after: Option<f64>,
    /// Structural merge/prune report for the level genome.
    pub structural: ConsolidationReport,
}

/// Summary of what `ThreeCell::drive_sequence` produced for the
/// caller: how many Cell A closures fired during the sequence and
/// how many new genome entries were created (or 0 if everything
/// reinforced existing slots).
#[derive(Clone, Copy, Debug)]
pub struct SequenceOutcome {
    pub closures_fired: usize,
    pub genome_growth: usize,
}

/// Uniform update report: what changed during any body-mutating operation.
///
/// Every method that writes to the brain's body — `update`, `update_sequence`,
/// `sleep` — returns or can produce an `UpdateReport`. This makes the
/// living/reading boundary explicit: if you hold an `UpdateReport`, the
/// brain's body was changed.
///
/// Read-only verbs (`evaluate`, `generate`, `cell_a`, `cell_c`, etc.) do NOT
/// return `UpdateReport`. Callers that need the detailed per-step trace use
/// `ingest` directly.
#[derive(Clone, Debug)]
pub struct UpdateReport {
    /// Number of new genome entries created (epigenetic layer growth).
    pub genome_delta: usize,
    /// Number of Cell A closures that fired.
    pub closures_fired: usize,
    /// Hierarchy depth reached after this update (new levels created by cascade).
    pub hierarchy_depth: usize,
    /// True iff at least one genome level was consolidated during this update.
    pub consolidation_fired: bool,
    /// Consolidation reports, one per level that fired (if any).
    pub consolidation_reports: Vec<RuntimeConsolidationReport>,
}

/// Report from one delayed reality-evaluation write.
///
/// This is not perception. It is the second half of the figure-8:
/// the previous output met reality, the difference was computed, and the
/// correction was written back into memory.
#[derive(Clone, Debug)]
pub struct EvaluationReport {
    pub feedback: PredictionFeedback,
    pub genome_delta: usize,
    pub closures_fired: usize,
    pub hierarchy_depth: usize,
    pub consolidation_fired: bool,
    pub consolidation_reports: Vec<RuntimeConsolidationReport>,
}

/// The exact emitting locus of a prediction.
///
/// A prediction is a field event. Its source is part of the event —
/// not recomputed afterward by a secondary lookup.
///
/// - `GenomeSlot(i)`: prediction came from learned memory slot `i`.
///   Credit assignment writes directly to that slot.
/// - `GeometricFallback(c)`: prediction came from bare geometry with
///   no genome involvement. The fallback carrier `c` is materialized
///   into memory before correction is written.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum PredictionSource {
    GenomeSlot(usize),
    GeometricFallback([f64; 4]),
    /// The prediction was produced by a ZREAD soft-field read — a
    /// population-weighted aggregate, not a single slot. No slot is
    /// materialized; edge reinforcement is skipped. `distribute_credit`
    /// handles the write path.
    ZreadAggregate,
}

/// A prediction staged for evaluation by the next reality tick.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PendingPrediction {
    /// What the observer sent out to reality.
    pub predicted: [f64; 4],
    /// The context carrier that produced the prediction.
    /// Used as the key for the Response layer write (learn_response).
    pub context: [f64; 4],
    /// Exact emitting locus — intrinsic to the prediction event.
    pub source: PredictionSource,
    /// EligibilityTrace: Response entries that were active at prediction time.
    /// Each element is (genome_index, coupling_t).  Used for causal credit
    /// assignment when reality returns — only these entries receive the update.
    pub eligibility: Vec<(usize, f64)>,
    /// Cycle at which the prediction was staged.
    pub cycle: usize,
    /// Signed salience at the time this prediction was staged.
    ///
    /// `compose(predicted, inverse(context))[1]` — the X-component (R / salience
    /// axis) of the rotation from the model (context = Cell C at staging time)
    /// to the prediction. Unlike `sigma` (geodesic distance, symmetric under
    /// swap), this component is **antisymmetric**: swapping predicted and context
    /// exactly negates it.
    ///
    ///   salience_x > 0 : prediction departs in the salient-forward direction.
    ///   salience_x ≈ 0 : prediction lies in the model's existing plane.
    ///   salience_x < 0 : prediction departs in the anti-salient direction.
    ///
    /// At evaluation time, `credit_response` uses `(1.0 + salience_x).clamp(0.0, 2.0)`
    /// as a multiplicative amplifier: salient-forward (> 0) amplifies above baseline,
    /// neutral (= 0) gives exactly baseline rate, anti-salient (< 0) attenuates,
    /// and full anti-salient (≤ −1.0) zeroes the update entirely.
    ///
    /// Set from `last_salience_x`, which is loaded from
    /// `semantic.residual[1]` during the preceding `ingest()` call.
    /// This bridges the perception-time SemanticFrame to the learning path.
    #[serde(default)]
    pub salience_x: f64,
}

/// What reality said about the previous staged prediction.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct PredictionFeedback {
    /// Output previously sent to reality.
    pub predicted: [f64; 4],
    /// Reality that came back on the next tick.
    pub actual: [f64; 4],
    /// Context that produced the prediction.
    pub context: [f64; 4],
    /// `compose(actual, inverse(predicted))`
    pub correction: [f64; 4],
    /// `sigma(correction)` — zero means exact hit.
    pub sigma: f64,
    /// Binary truth signal at this level.
    pub correct: bool,
}

/// Complete serializable snapshot of a `ThreeCell` runtime.
///
/// Captures every field needed to resume an interrupted session exactly
/// where it left off. The hierarchy event log is included (`events: Vec<ClosureEvent>`)
/// and is restored by [`ThreeCell::from_brain_state`] via `Hierarchy::from_parts`.
/// `total_closures()` reads `hierarchy.events.len()`; this count must survive the
/// round trip to remain correct after restore.
///
/// Construct with [`ThreeCell::to_brain_state`].
/// Restore with [`ThreeCell::from_brain_state`].
/// File I/O: [`ThreeCell::save_state_to_file`] / [`ThreeCell::load_state_from_file`].
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BrainState {
    pub cell_a: [f64; 4],
    pub cell_a_count: usize,
    pub cell_a_threshold: f64,
    pub cell_a_excursion_peak: f64,
    pub cell_a_balance_excursion_peak: f64,
    pub cell_a_history: Vec<[f64; 4]>,
    pub cell_a_parity: usize,
    pub cell_c: [f64; 4],
    /// Running minimum of |W| observed across all closures.
    /// W = cell_c[0]. This tracks the deepest excursion of the self-model
    /// away from the existence axis — the lowest "depth" Cell C ever reached.
    /// Initialized to 1.0 (identity; W = cos(0) = 1).
    pub cell_c_min_w: f64,
    /// Running product of self-difference carriers:
    /// compose(cell_c, inverse(ZREAD(cell_c))).
    pub self_difference_product: [f64; 4],
    pub self_difference_count: usize,
    pub self_difference_history: Vec<[f64; 4]>,
    pub self_balance_excursion_peak: f64,
    pub buffer: Buffer,
    /// Per-level genomes. `genomes[0]` is the primary genome.
    pub genomes: Vec<Genome>,
    /// Per-level closure detector state (running product, history, threshold).
    pub levels: Vec<ClosureLevel>,
    pub default_threshold: f64,
    pub default_genome_config: GenomeConfig,
    pub cycle_count: usize,
    pub consolidation_pressure: Vec<f64>,
    /// Previous output waiting for reality's next evaluation tick.
    pub pending_prediction: Option<PendingPrediction>,
    /// Full event log. `total_closures()` reads this length; must be persisted.
    pub events: Vec<ClosureEvent>,
}

/// Recompute `cell_a_balance_excursion_peak` from a prefix slice after a
/// closure drains part of the history. Returns the maximum deviation of the
/// running product from the balance locus (σ = π/4) seen in the prefix.
/// Initialized to SIGMA_BALANCE when the prefix is empty (boot state).
fn balance_excursion_peak_from_prefix(prefix: &[[f64; 4]]) -> f64 {
    if prefix.is_empty() {
        return SIGMA_BALANCE;
    }
    let mut p = IDENTITY;
    let mut peak = 0.0_f64;
    for &c in prefix {
        p = compose(&p, &c);
        let dev = (sigma(&p) - SIGMA_BALANCE).abs();
        if dev > peak {
            peak = dev;
        }
    }
    peak
}

/// The diabolo on the critical line.
///
/// Phase 3 left the buffer, the genome, and the field machine
/// separate. Phase 4 wires them through Cell A (the fast running
/// product) and Cell C (the slow accumulator). The diabolo is the
/// composition of the three — one verb, seven steps, `ingest`.
#[derive(Clone)]
pub struct ThreeCell {
    /// Cell A running product on S³. Fast oscillator. Composes the
    /// raw inputs — what the brain has actually received.
    cell_a: [f64; 4],
    cell_a_count: usize,
    cell_a_threshold: f64,
    cell_a_excursion_peak: f64,
    /// Maximum deviation from the balance locus (σ = π/4) seen in the
    /// current oscillation. Initialized to SIGMA_BALANCE so the first
    /// approach to the balance basin is recognized as a genuine event.
    cell_a_balance_excursion_peak: f64,
    /// Ordered history of the carriers that fed Cell A's current
    /// oscillation. Used by Law 8 localization; cleared on closure.
    cell_a_history: Vec<[f64; 4]>,

    /// S¹ parity counter. Even positions (0, 2, 4, …) are in the trivial
    /// fiber; odd positions (1, 3, 5, …) are in the non-trivial fiber.
    /// At odd positions the input is pre-composed with the equatorial
    /// carrier before entering Cell A, implementing the S¹ double-cover.
    cell_a_parity: usize,

    /// Cell C running product on S³. Slow accumulator of localized
    /// closure packets. Cell C IS the brain's prediction.
    cell_c: [f64; 4],
    /// Running minimum of |W| across all Cell C accumulations.
    /// Tracks the deepest excursion of the self-model away from identity.
    cell_c_min_w: f64,
    /// Running product of self-difference carriers. This is the autobiographical
    /// loop: the observer compares its slow self-model against its own field
    /// response, then composes those differences over time. Critical-line
    /// crossings here become level-1 FixedPoint PrimeStates.
    self_difference_product: [f64; 4],
    self_difference_count: usize,
    self_difference_history: Vec<[f64; 4]>,
    self_balance_excursion_peak: f64,

    /// Transient buffer — holds incoming carriers for a configurable
    /// lifetime. The field machine reads `genome ∪ buffer` as one
    /// population.
    buffer: Buffer,

    /// Hierarchy: higher-level closure levels + the genome.
    hierarchy: Hierarchy,

    /// Monotonic count of `ingest` calls.
    cycle_count: usize,

    /// Accumulated σ-pressure per memory level. Consolidation fires when
    /// pressure crosses the balance locus, not from a clock.
    consolidation_pressure: Vec<f64>,

    /// Previous output waiting to be judged by the next reality input.
    pending_prediction: Option<PendingPrediction>,

    /// Self-free-energy from the previous ingest step, used to compute
    /// per-step valence.  `None` until the first ingest step completes:
    /// no previous measurement means no delta, so valence = 0.0.
    prev_sfe: Option<f64>,

    /// Slow body state — integrates fast signals across ingest steps.
    /// Session-ephemeral: resets to (0, 0) on brain restore, like prev_sfe.
    neuromod: NeuromodState,

    /// X-component (R/salience axis, index 1) of the semantic residual from
    /// the last `ingest()` call: `compose(f, inverse(cell_c))[1]`.
    ///
    /// Antisymmetric under swap of total and known (unlike sigma, which is
    /// symmetric). Forwarded into `PendingPrediction.salience_x` by
    /// `stage_prediction()` so the learning path uses the perception-time
    /// semantic signal, not a separately recomputed proxy.
    ///
    /// Not persisted in BrainState — reset to 0.0 at start of session and
    /// updated on every ingest. This means the first `commit_prediction` after
    /// a session restore gets salience_x = 0 (no learning amplification) until
    /// the next ingest updates it, which is the safe default.
    last_salience_x: f64,

    // ── Solenoidal noise (HHD completion) ────────────────────────────────
    //
    // The von Mises-Fisher perturbation that completes the Helmholtz-Hodge
    // decomposition. Without noise, Cell A is a second gradient — it
    // accumulates deterministically and converges to local minima. With
    // noise in the history, the trajectory can hop between basins.
    //
    // The BKT phase boundary in the genome acts as the noise filter:
    // noisy partial products that don't couple with anything real fall
    // below BKT_THRESHOLD and get pruned. Signal survives; noise dies.

    /// PRNG for solenoidal noise. `None` = deterministic (noise disabled).
    noise_rng: Option<Rng>,
    /// Base κ for von Mises-Fisher sampling. Default = 1/BKT_THRESHOLD ≈ 2.08.
    /// Modulated by neuromodulation tones at each step.
    noise_base_kappa: f64,
}

impl ThreeCell {
    /// Build a new diabolo.
    ///
    /// The genome is seeded with the **identity carrier** at boot
    /// (BRAIN.md §18: "The genome at boot contains exactly one
    /// carrier: the identity at [1, 0, 0, 0]."). This is the brain's
    /// permanent stack reference: every §9 step 6 LOAD has at least
    /// one slot to read from, so the round trip is always defined.
    ///
    /// * `cell_a_threshold` — σ below which Cell A closes.
    /// * `hierarchy_threshold` — σ threshold for hierarchy levels 1+.
    /// * `buffer_lifetime` — how many cycles an input stays in the
    ///   buffer before the chunked-learning sweep evaluates it
    ///   (BRAIN.md §29.1).
    /// * `genome_config` — thresholds governing
    ///   reinforce/correct/create decisions on the genome.
    pub fn new(
        cell_a_threshold: f64,
        hierarchy_threshold: f64,
        buffer_lifetime: usize,
        genome_config: GenomeConfig,
    ) -> Self {
        assert!(buffer_lifetime >= 2, "buffer_lifetime must be >= 2 (alpha = 1 − 1/n requires n ≥ 2 for a non-degenerate EMA)");
        let mut hierarchy = Hierarchy::new(hierarchy_threshold, genome_config);

        // §18: identity anchor — the brain's permanent stack reference.
        // Every §9 step-6 LOAD has at least one slot to read from.
        hierarchy.genomes[0].seed_dna(IDENTITY, 0, 0.0, 0.0);

        // Equatorial anchor — the crossing point of the diabolo (σ = π/4).
        // This is the Hopf equator: |w| = |x| = 1/√2, y = z = 0.
        // It is the only carrier that sits exactly at the balance locus —
        // the "You" in the picture. Without it the genome has no reference
        // for the FixedPoint zero. Carrier(2) in the prime observer lives here.
        let s2 = std::f64::consts::FRAC_1_SQRT_2;
        hierarchy.genomes[0].seed_dna([s2, s2, 0.0, 0.0], 0, 0.0, 0.0);

        // Prime-3 anchor — tetrahedral axis (σ = arccos(1/√3) ≈ 0.9553).
        // This carrier sits below the equator on the prime-3 orbit. Its
        // Dobrushin contraction coefficient δ(3) = 1 − 1/√3 ≈ 0.423 is
        // what drives the ×3+1 Collatz convergence. Without this anchor
        // the genome cannot address the three-body orbit directly.
        let s3 = 1.0_f64 / 3.0_f64.sqrt();
        hierarchy.genomes[0].seed_dna([s3, s3, s3, 0.0], 0, 0.0, 0.0);

        Self {
            cell_a: IDENTITY,
            cell_a_count: 0,
            cell_a_threshold,
            cell_a_excursion_peak: 0.0,
            cell_a_balance_excursion_peak: SIGMA_BALANCE,
            cell_a_history: Vec::new(),
            cell_a_parity: 0,
            cell_c: IDENTITY,
            cell_c_min_w: 1.0,
            self_difference_product: IDENTITY,
            self_difference_count: 0,
            self_difference_history: Vec::new(),
            self_balance_excursion_peak: SIGMA_BALANCE,
            buffer: Buffer::new(buffer_lifetime),
            hierarchy,
            cycle_count: 0,
            consolidation_pressure: vec![0.0],
            pending_prediction: None,
            prev_sfe: None,
            neuromod: NeuromodState::new(buffer_lifetime),
            last_salience_x: 0.0,
            noise_rng: None,
            noise_base_kappa: 0.0,
        }
    }

    /// Enable solenoidal noise on Cell A with the given RNG seed.
    ///
    /// `base_kappa` is the resting concentration for the von Mises-Fisher
    /// distribution. Recommended: `1.0 / BKT_THRESHOLD` ≈ 2.08 — the
    /// critical noise level where perturbations reach the closure threshold.
    /// Modulated each step by the neuromodulation tones.
    pub fn enable_noise(&mut self, seed: u64, base_kappa: f64) {
        self.noise_rng = Some(Rng::new(seed));
        self.noise_base_kappa = base_kappa;
    }

    /// Disable solenoidal noise, returning to deterministic Cell A.
    pub fn disable_noise(&mut self) {
        self.noise_rng = None;
    }

    /// Whether solenoidal noise is active.
    pub fn noise_enabled(&self) -> bool {
        self.noise_rng.is_some()
    }

    /// Reconstitute a brain from a previously saved genome.
    ///
    /// All fields are identical to `new`, except the genome is installed
    /// directly — no `seed_dna` call, because the identity DNA entry is
    /// already present in the serialized genome. Cell A, Cell C, and the
    /// buffer start fresh (Cell C regenerates from the first few closures;
    /// Cell A has no persistent state across restarts). The hierarchy levels
    /// also start empty — they accumulate from the first carrier onward.
    ///
    /// Use this after `Genome::load_from_file` to resume a saved brain.
    pub fn with_genome(
        genome: Genome,
        cell_a_threshold: f64,
        hierarchy_threshold: f64,
        buffer_lifetime: usize,
    ) -> Self {
        assert!(buffer_lifetime >= 2, "buffer_lifetime must be >= 2 (alpha = 1 − 1/n requires n ≥ 2 for a non-degenerate EMA)");
        let hierarchy = Hierarchy::with_genome(genome, hierarchy_threshold);
        Self {
            cell_a: IDENTITY,
            cell_a_count: 0,
            cell_a_threshold,
            cell_a_excursion_peak: 0.0,
            cell_a_balance_excursion_peak: SIGMA_BALANCE,
            cell_a_history: Vec::new(),
            cell_a_parity: 0,
            cell_c: IDENTITY,
            cell_c_min_w: 1.0,
            self_difference_product: IDENTITY,
            self_difference_count: 0,
            self_difference_history: Vec::new(),
            self_balance_excursion_peak: SIGMA_BALANCE,
            buffer: Buffer::new(buffer_lifetime),
            hierarchy,
            cycle_count: 0,
            consolidation_pressure: vec![0.0],
            pending_prediction: None,
            prev_sfe: None,
            neuromod: NeuromodState::new(buffer_lifetime),
            last_salience_x: 0.0,
            noise_rng: None,
            noise_base_kappa: 0.0,
        }
    }

    fn evaluate_pending_prediction(&mut self, actual: &[f64; 4]) -> Option<PredictionFeedback> {
        let pending = self.pending_prediction.take()?;
        let correction = compose(actual, &inverse(&pending.predicted));
        let sigma_gap = sigma(&correction);
        let correct = sigma_gap <= self.cell_a_threshold;

        // Pattern-match on the intrinsic prediction source. No secondary lookup.
        //
        // teach_response_at: the address of the source slot stays fixed (ZREAD
        // continues to find it from the original context query). Only the value
        // is SLERPed toward the true response at rate σ(gap)/π. This is
        // heteroassociative learning: the slot learns to return `actual` when
        // queried from its original address.
        let source_idx: Option<usize> = match pending.source {
            PredictionSource::GenomeSlot(i) => Some(i),
            PredictionSource::GeometricFallback(fallback) => {
                // Materialize the fallback carrier into memory first.
                let outcome = self.hierarchy.genomes[0].ingest(&fallback, 1, 0.0, 0.0);
                Some(outcome.index())
            }
            // ZREAD aggregate: no single slot emitted this prediction.
            // Edge reinforcement is skipped; distribute_credit handles the write.
            PredictionSource::ZreadAggregate => None,
        };

        let genome = &mut self.hierarchy.genomes[0];

        // 1. Causal credit assignment — EligibilityTrace → Response layer.
        //    Only the Response entries active at prediction time are updated.
        //    Learning rate is scaled by both correction magnitude (sigma_gap)
        //    and the salience that was recorded when the prediction was staged
        //    (how novel the prediction was vs the model at that moment).
        genome.credit_response(
            &pending.eligibility,
            actual,
            sigma_gap,
            pending.salience_x,
            self.neuromod.coherence_tone,
        );

        // 2. Response write — geometry separates contexts; consolidation merges
        //    compatible nearby responses into stable attractors (myelination).
        genome.learn_response(&pending.context, actual);

        // Also maintain sequential edge statistics for generation.
        let actual_idx = genome.nearest_index(actual);
        if let (Some(src), Some(act)) = (source_idx, actual_idx) {
            genome.reinforce_edge(src, act, 1);
        }
        if !correct {
            if let Some(src) = source_idx {
                let predicted_idx = genome.nearest_index(&pending.predicted);
                if let Some(pred) = predicted_idx {
                    genome.weaken_edge(src, pred, 1);
                }
            }
        }

        Some(PredictionFeedback {
            predicted: pending.predicted,
            actual: *actual,
            context: pending.context,
            correction,
            sigma: sigma_gap,
            correct,
        })
    }

    /// Delayed reality-evaluation verb.
    ///
    /// Perception and evaluation are not the same operation. `ingest`
    /// handles new world input. `evaluate_prediction` handles the returned
    /// truth value for the previously staged output and writes the correction
    /// into memory.
    pub fn evaluate_prediction(&mut self, actual: &[f64; 4]) -> Option<EvaluationReport> {
        let size_before = self.hierarchy.genomes[0].len();
        let closures_before = self.total_closures();
        self.cycle_count += 1;

        let feedback = self.evaluate_pending_prediction(actual)?;
        let self_free_energy = self.self_observe();
        let level0_pressure =
            feedback.sigma + (self_free_energy - self.cell_a_threshold).max(0.0);
        self.add_consolidation_pressure(0, level0_pressure);
        let consolidation_reports = self.consolidate_ready_levels(self_free_energy);
        let consolidation_fired = !consolidation_reports.is_empty();

        Some(EvaluationReport {
            feedback,
            genome_delta: self.hierarchy.genomes[0]
                .len()
                .saturating_sub(size_before),
            closures_fired: self.total_closures() - closures_before,
            hierarchy_depth: self.hierarchy.depth(),
            consolidation_fired,
            consolidation_reports,
        })
    }

    fn ensure_pressure_level(&mut self, level: usize) {
        while self.consolidation_pressure.len() <= level {
            self.consolidation_pressure.push(0.0);
        }
    }

    fn add_consolidation_pressure(&mut self, level: usize, pressure: f64) {
        if pressure <= 0.0 || !pressure.is_finite() {
            return;
        }
        self.ensure_pressure_level(level);
        self.consolidation_pressure[level] += pressure;
    }

    fn consolidate_level(
        &mut self,
        level: usize,
        self_free_energy_before: Option<f64>,
        force: bool,
    ) -> Option<RuntimeConsolidationReport> {
        if level >= self.hierarchy.genomes.len() {
            return None;
        }
        self.ensure_pressure_level(level);
        let pressure_before = self.consolidation_pressure[level];
        let consolidation_threshold = SIGMA_BALANCE;
        if !force && pressure_before < consolidation_threshold {
            return None;
        }

        let mut structural = consolidate(&mut self.hierarchy.genomes[level]);

        // Category birth: after consolidation, scan the genome for Response
        // clusters that have earned hierarchical promotion.  Promotion writes
        // the cluster's stable fixed point into genomes[1] as a new attractor,
        // provided no level-1 entry already covers it.
        //
        // Only runs on level-0 consolidation — that is where Response entries
        // live and where category birth evidence accumulates.
        if level == 0 {
            // Use candidates collected by consolidate() before its merge/prune pass.
            // Coalition evidence (co_resonance) is intact in those candidates;
            // reading it again after merge would find an erased co_resonance list.
            let candidates = std::mem::take(&mut structural.promotion_candidates);
            let novelty_threshold = self.hierarchy.genomes[0].config.novelty_threshold;
            let mut promoted = 0usize;
            for candidate in candidates {
                // Nearness check against existing level-1 entries.
                let already_covered = self
                    .hierarchy
                    .genome_at(1)
                    .map(|g| {
                        g.entries.iter().any(|e| {
                            sigma(&compose(&candidate.carrier, &crate::sphere::inverse(&e.address.geometry())))
                                < novelty_threshold
                        })
                    })
                    .unwrap_or(false);
                if !already_covered {
                    self.hierarchy.genome_at_mut(1).ingest(
                        &candidate.carrier,
                        candidate.activation_count,
                        candidate.closure_sigma,
                        candidate.excursion_peak,
                    );
                    promoted += 1;
                }
            }
            structural.promoted_categories = promoted;
        }

        let self_free_energy_after = if level == 0 {
            Some(self.self_observe())
        } else {
            None
        };

        let pressure_after = if level == 0 {
            self_free_energy_after
                .map(|s| (s - self.cell_a_threshold).max(0.0))
                .unwrap_or(0.0)
        } else {
            0.0
        };
        self.consolidation_pressure[level] = pressure_after;

        Some(RuntimeConsolidationReport {
            level,
            pressure_before,
            pressure_after,
            self_free_energy_before,
            self_free_energy_after,
            structural,
        })
    }

    /// Force a consolidation pass at level 0 regardless of pressure.
    ///
    /// Used in tests and in tracks that want to consolidate at a known moment
    /// (e.g. end-of-episode) rather than waiting for pressure to accumulate.
    /// Returns the same report slice as the internal consolidation path.
    pub fn force_consolidate(&mut self) -> Vec<RuntimeConsolidationReport> {
        let sfe = self.self_observe();
        let mut reports = Vec::new();
        if let Some(r) = self.consolidate_level(0, Some(sfe), true) {
            reports.push(r);
        }
        reports
    }

    fn consolidate_ready_levels(
        &mut self,
        level0_self_free_energy: f64,
    ) -> Vec<RuntimeConsolidationReport> {
        let levels = self
            .hierarchy
            .genomes
            .len()
            .max(self.consolidation_pressure.len());
        let mut reports = Vec::new();
        for level in 0..levels {
            let before = if level == 0 {
                Some(level0_self_free_energy)
            } else {
                None
            };
            if let Some(report) = self.consolidate_level(level, before, false) {
                reports.push(report);
            }
        }
        reports
    }

    /// Autobiographical PrimeState detector.
    ///
    /// The self-difference carrier is:
    ///
    /// ```text
    /// compose(cell_c, inverse(ZREAD(cell_c)))
    /// ```
    ///
    /// We compose that difference over time. When the running self-difference
    /// product crosses the critical line (σ = π/4), the observer has learned
    /// something about the gap between its accumulated self-model and its own
    /// memory field. That non-trivial zero is written as a level-1 FixedPoint:
    ///
    /// ```text
    /// self critical line -> genomes[1]
    /// ```
    ///
    /// No planner, no external training loop, no new primitive: just
    /// compose/inverse/sigma over the observer's own mirror.
    fn observe_self_prime_state(&mut self) -> Option<ClosureEvent> {
        let diff = self.self_difference_carrier();
        let prev_sigma = sigma(&self.self_difference_product);

        self.self_difference_history.push(diff);
        self.self_difference_product = compose(&self.self_difference_product, &diff);
        self.self_difference_count += 1;

        let current_sigma = sigma(&self.self_difference_product);
        let balance_dev = (current_sigma - SIGMA_BALANCE).abs();
        let prev_balance_dev = (prev_sigma - SIGMA_BALANCE).abs();
        if balance_dev > self.self_balance_excursion_peak {
            self.self_balance_excursion_peak = balance_dev;
        }

        if balance_dev <= self.cell_a_threshold
            && self.self_balance_excursion_peak > self.cell_a_threshold
            && prev_balance_dev >= balance_dev
        {
            let interval =
                localize_balance(&self.self_difference_history, self.cell_a_threshold);
            let oscillation_peak = self.self_balance_excursion_peak;
            self.self_balance_excursion_peak = 0.0;

            // A single self-difference hit is an observation, not an
            // autobiographical learning moment. PrimeStates require support.
            if interval.support <= 1 {
                return None;
            }

            let (hopf_base, hopf_phase) = hopf::decompose(&interval.product);
            let packet_peak = localized_excursion_peak(
                &self.self_difference_history,
                interval.start,
                interval.end,
            );

            let event = ClosureEvent {
                carrier: interval.product,
                sigma: interval.sigma,
                support: interval.support,
                oscillation_depth: self.self_difference_count,
                excursion_peak: packet_peak,
                oscillation_excursion_peak: oscillation_peak,
                level: 1,
                role: ClosureRole::FixedPoint,
                kind: hopf_classify(&interval.product),
                hopf_base,
                hopf_phase,
                interval,
            };

            self.hierarchy.genome_at_mut(1).ingest(
                &event.carrier,
                event.support,
                event.sigma,
                event.excursion_peak,
            );
            self.hierarchy.events.push(event.clone());
            return Some(event);
        }

        None
    }

    /// The brain's only input verb.
    ///
    /// Runs the seven-step receive-and-verify loop of BRAIN.md §9 with
    /// the **two-machine separation enforced** at steps 5–7:
    ///
    /// 1–3. Push the carrier into the buffer and tick the buffer.
    /// 4.   `f ← ZREAD(genome ∪ buffer)` — the field machine's
    ///      parameterized integration.
    /// 5.   `s' ← RESONATE(f)` — the field machine's collapsed read.
    ///      May land on either side of the population.
    /// 6.   `s ← LOAD(slot of s' in genome)` — the **stack-only**
    ///      reference read. Performed via `genome.nearest(&s')`,
    ///      then loading the value at that slot. The genome always
    ///      has at least the identity entry seeded at boot, so this
    ///      LOAD is always defined.
    /// 7.   `ev ← VERIFY(s, s')` — Cell B's round-trip closure
    ///      between what the genome alone holds and what the field
    ///      collapsed to. **The gradient lives in this gap.**
    ///
    /// Then, alongside the round trip:
    ///
    /// * Cell A composes the **raw input** into its running product.
    ///   Cell A is the brain's record of what it has actually received.
    /// * If VERIFY closes (Identity or Balance) and the field hit
    ///   came from the genome (not a self-match against the
    ///   just-pushed buffer entry), the **just-pushed buffer entry
    ///   earns a closure tag** via `record_closure`.
    /// * The buffer is drained at the chunk boundary
    ///   (`Buffer::drain_expired`); expired entries with `closures > 0`
    ///   are promoted into the genome via `Genome::ingest`.
    ///   This is the **only normal path** that grows the epigenetic
    ///   layer (BRAIN.md §29.1: "There is no per-event STORE.").
    /// * If Cell A's running product satisfies Law 1 (return +
    ///   excursion + local minimum + support), localize the closing
    ///   interval (Law 8). The carriers in `[interval.start,
    ///   interval.end]` get `record_closure`'d on the corresponding
    ///   buffer entries (those still alive). The closure cascades
    ///   upward through the hierarchy via `emit_closure`, but **the
    ///   hierarchy never writes to the genome** — promotion happens
    ///   only via the chunk-boundary drain.
    pub fn ingest(&mut self, carrier: &[f64; 4]) -> Step {
        self.cycle_count += 1;

        // ── Steps 1–3: receive path. Push and tick. ──────────────
        self.buffer.push(*carrier);
        self.buffer.tick();
        let just_pushed_buffer_idx = self.buffer.len() - 1;
        let genome_size_before = self.hierarchy.genomes[0].len();

        // ── Cell C prediction error — FEP surprise measurement. ──
        //
        // Cell C is the brain's running prediction: the composition
        // of every closure packet the brain has emitted so far. Before
        // processing the new token, measure how far that prediction is
        // from what actually arrived.
        //
        //   prediction_error = σ(compose(cell_c, inverse(carrier)))
        //
        // This is the geodesic distance on S³ between the prediction
        // and the observation — free energy under the von Mises-Fisher
        // prior. In the flat-space limit: −log P(token | cell_c).
        //
        // When prediction_error > cell_a_threshold, the brain was
        // genuinely surprised. Tag the just-pushed buffer entry with
        // a closure immediately so it earns genome promotion at the
        // chunk boundary — the brain should update its model toward
        // this token. The tagging is the same call-site as the VERIFY
        // path below; both can fire on the same step independently.
        //
        // When prediction_error ≤ cell_a_threshold, the brain predicted
        // this token correctly; no extra promotion signal is needed.
        let prediction_error = sigma(&compose(&self.cell_c, &inverse(carrier)));
        if prediction_error > self.cell_a_threshold {
            self.buffer.entries_mut()[just_pushed_buffer_idx].record_closure(
                1,
                prediction_error,
                prediction_error,
            );
        }

        // ── Step 4: ZREAD — soft attention over (genome ∪ buffer). ─────
        //    channel: the query's own Hopf half selects the head.
        //      W-domain carriers (|w| ≥ |rgb|) read the existence layer.
        //      RGB-domain carriers (|rgb| > |w|) read the position layer.
        //      Multi-head is forced by the fibration, not synthesized.
        //    mode: AddressMode::Full — standard full-geodesic coupling.
        //      To switch the brain to factorized address semantics (e.g.
        //      Base-only for type queries, Phase-only for position queries),
        //      change the mode argument here and in record_zread_contributions
        //      on the next line. Channel and mode are orthogonal.
        //    coupling: coupling_from_gap(address_distance(q, a, mode), mode)
        //      maps gap → SLERP fraction t. Entries with t < 0.5 (σ > π/3)
        //      are excluded. No softmax, no ℝ summation, no normalization. ─
        let query_channel = {
            let w = carrier[0].abs();
            let rgb = (carrier[1].powi(2) + carrier[2].powi(2) + carrier[3].powi(2)).sqrt();
            if w >= rgb {
                HopfChannel::W
            } else {
                HopfChannel::Rgb
            }
        };
        let f = zread_at_query_channel_with_mode(
            carrier,
            query_channel,
            AddressMode::Full,
            &self.hierarchy.genomes[0],
            &self.buffer,
        );

        // ── Semantic triple: total / known / residual. ───────────────────────
        //    total = f (the genome's collective reading of this input, ZREAD)
        //    known = cell_c (the accumulated model, before this step's update)
        //    residual = compose(total, inverse(known))
        //    salience_sigma = sigma(residual) = geodesic distance field→model
        //
        //    This is the runtime instantiation of the RGB semantic split:
        //    R = salience (residual), G = total (prior field), B = known (model).
        //    Captured here so evaluation time can weight credit assignment by
        //    how salient the field was relative to the model at perception time.
        let semantic = semantic_frame(&f, &self.cell_c);
        // Bridge the perception-time salience direction to the learning path.
        // residual[1] is the X-component (R/salience axis) of compose(f, inverse(cell_c)).
        // Antisymmetric: compose(f, inv(cell_c))[1] = -compose(cell_c, inv(f))[1].
        // Forwarded into PendingPrediction.salience_x via stage_prediction().
        self.last_salience_x = semantic.residual[1];

        // Record per-entry ZREAD coupling strengths for BKT pruning.
        // Pass the same channel and mode used above so accumulated
        // t-statistics match the active read path exactly.
        self.hierarchy.genomes[0].record_zread_contributions(
            carrier,
            query_channel,
            AddressMode::Full,
        );

        // ── Step 5: RESONATE(f) → s' — the field's collapsed read.
        //    Restricted to the same Hopf channel as ZREAD: the query's
        //    channel selects the head; both reads operate on the same
        //    half of the population.
        let s_prime = resonate_channel_with_mode(
            &f,
            query_channel,
            AddressMode::Full,
            &self.hierarchy.genomes[0],
            &self.buffer,
        );

        // ── Step 6: LOAD slot-of-s' from the GENOME ALONE — the
        //    stack-only reference read. If the hit came from the
        //    genome, load by exact slot index (entry.address can
        //    diverge from entry.value after Law 5 corrections). If
        //    the hit came from the buffer, fall back to nearest-on-
        //    genome to find a stack reference. The genome always has
        //    at least the seeded identity, so step 6 never fails. ─
        let verification = s_prime.as_ref().map(|hit| {
            let stack = self.load_genome_value_for_hit(hit);
            // ── Step 7: VERIFY(s, s') — round-trip closure. ───────
            verify(&stack, &hit.carrier)
        });

        // Tag the just-pushed buffer entry if VERIFY closed AND the
        // field hit was not the just-pushed entry itself. A self-match
        // is the degenerate case where the brain has nothing else to
        // read against; it doesn't count as "this entry coupled with
        // the existing population."
        if let (Some(ev), Some(hit)) = (verification.as_ref(), s_prime.as_ref()) {
            let is_self_match = matches!(hit.source, PopulationSource::Buffer)
                && hit.index == just_pushed_buffer_idx;
            // §27: both Identity and Balance closures count.
            let fired = ev.closes() || ev.balances();
            if fired && !is_self_match {
                let entry = &mut self.buffer.entries_mut()[just_pushed_buffer_idx];
                // Round-trip-only closure: there is no localized
                // trajectory packet here, so support = 1, sigma =
                // ev.sigma, and the excursion peak is just |ev.sigma|
                // (no broader trajectory was measured at step 7).
                entry.record_closure(1, ev.sigma, ev.sigma.abs());
            }
        }

        // ── Cell A: S¹ fiber classification + raw input composition. ─
        //    The S¹ parity counter classifies which fiber the current
        //    position is in: even = trivial fiber, odd = non-trivial fiber.
        //    This is preclassification, not carrier mutation. The carrier
        //    is already on S³ when ingest() receives it; the parity records
        //    which S¹ sheet it belongs to so downstream logic (ClosureRole
        //    detection, genome query routing) can use it.
        //
        //    The actual S¹ gate (pre-composing with EQUATORIAL before S³
        //    lift) is the caller's responsibility at the embedding stage —
        //    bytes_to_sphere4(parity_transform(data, parity)). ThreeCell
        //    only records the fiber classification.
        let _fiber_is_nontrivial = self.cell_a_parity % 2 == 1;
        self.cell_a_parity += 1;

        let prev_a_sigma = sigma(&self.cell_a);

        // ── Solenoidal noise: the HHD completion. ────────────────────────
        //
        // If noise is enabled, perturb the carrier with a von Mises-Fisher
        // sample before it enters Cell A's history. The noisy carrier is
        // what Cell A "perceives" — both the history and the running product
        // see the same perturbed input.
        //
        // κ is modulated by the neuromodulation tones: high coherence →
        // high κ (less noise, exploit); low coherence → low κ (more noise,
        // explore). The BKT phase boundary in the genome filters the result:
        // noisy partials that don't couple with anything real get pruned.
        let (cell_a_carrier, step_kappa) = if let Some(rng) = &mut self.noise_rng {
            let base_kappa = self.neuromod.kappa(self.noise_base_kappa);
            // Log-normal multiplicative noise on κ itself. The exploration
            // intensity fluctuates stochastically — sometimes the brain is
            // very noisy, sometimes quiet, even within a single pass. This
            // is the thermal fluctuation of the noise floor: the temperature
            // of the temperature.
            let kappa = base_kappa * (rng.normal() * 0.5).exp();
            let noisy = sample_vmf_s3(carrier, kappa, rng);
            (noisy, Some(kappa))
        } else {
            (*carrier, None)
        };

        self.cell_a_history.push(cell_a_carrier);
        self.cell_a = compose(&self.cell_a, &cell_a_carrier);
        self.cell_a_count += 1;

        let a_sigma = sigma(&self.cell_a);
        if a_sigma > self.cell_a_excursion_peak {
            self.cell_a_excursion_peak = a_sigma;
        }

        // Track balance-basin deviation. Large = far from π/4.
        let balance_dev = (a_sigma - SIGMA_BALANCE).abs();
        let prev_balance_dev = (prev_a_sigma - SIGMA_BALANCE).abs();
        if balance_dev > self.cell_a_balance_excursion_peak {
            self.cell_a_balance_excursion_peak = balance_dev;
        }

        // ── Cell A closure check — Law 1 + Law 8. ────────────────
        let mut closure = None;
        let mut hierarchy_events = Vec::new();

        if a_sigma <= self.cell_a_threshold
            && self.cell_a_excursion_peak > self.cell_a_threshold
            && prev_a_sigma >= a_sigma
        {
            let interval = localize(&self.cell_a_history, self.cell_a_threshold);

            if interval.support > 1 {
                let prefix_end = interval.start;
                let (a_base, a_phase) = hopf::decompose(&interval.product);
                let a_kind = hopf_classify(&interval.product);

                let packet_peak =
                    localized_excursion_peak(&self.cell_a_history, interval.start, interval.end);

                let event = ClosureEvent {
                    carrier: interval.product,
                    sigma: interval.sigma,
                    support: interval.support,
                    oscillation_depth: self.cell_a_count,
                    excursion_peak: packet_peak,
                    oscillation_excursion_peak: self.cell_a_excursion_peak,
                    level: 0,
                    role: ClosureRole::Carry,
                    kind: a_kind,
                    hopf_base: a_base,
                    hopf_phase: a_phase,
                    interval,
                };

                // ── Bootstrap persistence: orbit positions → genome.
                //
                // Each partial product within the closing window is
                // an orbit position the brain just discovered. For
                // bootstrap with ε self-composed n times, this writes
                // ε¹, ε², …, ε^(n−1) as distinct genome entries
                // (BRAIN.md §19 Checkpoints 2–3: integers as orbit
                // lengths, distinct ε^k as counting). Carriers near
                // identity are skipped — the boot identity already
                // occupies that slot, and the closing carrier itself
                // is at σ ≈ 0 by construction.
                //
                // This is the **per-oscillation** persistence path:
                // triggered by a structural cycle completion, not by
                // a single ingest event. It complements the per-event
                // round-trip path (round trip closes → tag buffer
                // entry → chunk drain promotes), which only catches
                // single matches against an existing basis. Bootstrap
                // needs the cold-start path because the round-trip
                // path can't grow the basis from {identity} alone.
                {
                    let mut partial = IDENTITY;
                    for k in 0..event.support {
                        let idx = event.interval.start + k;
                        partial = compose(&partial, &self.cell_a_history[idx]);
                        if sigma(&partial) > 1e-6 {
                            self.hierarchy.genomes[0].ingest(
                                &partial,
                                event.support,
                                event.sigma,
                                event.excursion_peak,
                            );
                        }
                    }
                }

                // Cell C accumulates the localized closure packet.
                self.cell_c = compose(&self.cell_c, &event.carrier);
                let w = self.cell_c[0].abs();
                if w < self.cell_c_min_w {
                    self.cell_c_min_w = w;
                }

                // Cascade upward through the hierarchy. The hierarchy
                // itself is silent on `Genome::ingest` (no per-event
                // STORE there); persistence happens through the orbit-
                // position writes above. Per-level buffers / EMIT-as-
                // buffer-entry (BRAIN.md §28) is still an open seam.
                hierarchy_events = self.hierarchy.emit_closure(&event);

                // Retain the unresolved prefix [0, prefix_end) and
                // rebuild Cell A state from it.
                let prefix = self.cell_a_history[..prefix_end].to_vec();
                let (new_a, new_peak) = rebuild_from_prefix(&prefix);
                self.cell_a = new_a;
                self.cell_a_count = prefix.len();
                self.cell_a_excursion_peak = new_peak;
                self.cell_a_balance_excursion_peak = balance_excursion_peak_from_prefix(&prefix);
                self.cell_a_history = prefix;

                closure = Some(event);
            }
        }

        // ── Cell A balance closure check — the second privileged locus. ──
        //
        // The balance locus (σ = π/4) is the Hopf equator where the W and
        // RGB channels carry equal weight. Structural/arrangement events
        // close here. This is a **pass-through observation**: the event is
        // emitted and the hierarchy is notified, but the history is NOT
        // drained and the running product continues uninterrupted. Only
        // identity closure (cycle completion, σ → 0) drains the history.
        //
        // After firing, `balance_excursion_peak` resets to 0 so the next
        // detection requires a fresh excursion away from π/4.
        if closure.is_none()
            && balance_dev <= self.cell_a_threshold
            && self.cell_a_balance_excursion_peak > self.cell_a_threshold
            && prev_balance_dev >= balance_dev
        {
            let interval = localize_balance(&self.cell_a_history, self.cell_a_threshold);
            let (b_base, b_phase) = hopf::decompose(&interval.product);
            let packet_peak =
                localized_excursion_peak(&self.cell_a_history, interval.start, interval.end);

            let bal_event = ClosureEvent {
                carrier: interval.product,
                sigma: interval.sigma,
                support: interval.support,
                oscillation_depth: self.cell_a_count,
                excursion_peak: packet_peak,
                oscillation_excursion_peak: self.cell_a_balance_excursion_peak,
                level: 0,
                role: ClosureRole::FixedPoint,
                kind: ClosureKind::Arrangement,
                hopf_base: b_base,
                hopf_phase: b_phase,
                interval,
            };

            // closure(level 0) -> memory(level 0): the FixedPoint packet
            // is a distinct, non-trivial carrier at σ ≈ π/4. Write it
            // to genomes[0] so the invariant holds for both closure roles
            // at level 0. (Carry partial positions are already written
            // above; this covers the FixedPoint role.)
            if bal_event.support > 1 {
                self.hierarchy.genomes[0].ingest(
                    &bal_event.carrier,
                    bal_event.support,
                    bal_event.sigma,
                    bal_event.excursion_peak,
                );
            }

            // Emit upward; history is NOT drained — orbit continues.
            self.cell_c = compose(&self.cell_c, &bal_event.carrier);
            let w_bal = self.cell_c[0].abs();
            if w_bal < self.cell_c_min_w {
                self.cell_c_min_w = w_bal;
            }
            hierarchy_events = self.hierarchy.emit_closure(&bal_event);

            // Reset so the next pass through π/4 requires a fresh excursion.
            self.cell_a_balance_excursion_peak = 0.0;

            closure = Some(bal_event);
        }

        // ── Chunk-boundary promotion. The ONLY normal path that
        //    grows the genome (BRAIN.md §29.1). Buffer entries whose
        //    lifetime expired this cycle and that earned at least one
        //    closure during their lifetime get promoted; the rest
        //    decay silently. ──────────────────────────────────────
        let expired = self.buffer.drain_expired();
        for entry in expired {
            if entry.closures > 0 {
                self.hierarchy.genomes[0].ingest(
                    &entry.carrier,
                    entry.support,
                    entry.closure_sigma,
                    entry.excursion_peak,
                );
            }
        }

        let self_free_energy = self.self_observe();
        if let Some(self_event) = self.observe_self_prime_state() {
            hierarchy_events.push(self_event);
        }

        // ── Runtime metabolism. Consolidation pressure is σ-mass, not a clock. ─
        let genome_growth = self.hierarchy.genomes[0]
            .len()
            .saturating_sub(genome_size_before);
        let mut level0_pressure = (prediction_error - self.cell_a_threshold).max(0.0)
            + (self_free_energy - self.cell_a_threshold).max(0.0)
            + genome_growth as f64 * self.cell_a_threshold;
        if let Some(ev) = closure.as_ref() {
            level0_pressure += ev.excursion_peak.max(ev.sigma);
        }
        self.add_consolidation_pressure(0, level0_pressure);
        for ev in &hierarchy_events {
            self.add_consolidation_pressure(ev.level, ev.excursion_peak.max(ev.sigma));
        }
        let consolidation_reports = self.consolidate_ready_levels(self_free_energy);

        let valence = self.prev_sfe.map_or(0.0, |p| p - self_free_energy);
        self.prev_sfe = Some(self_free_energy);

        // Update slow body state from the step's derived signals.
        // step_pressure is the architecture's activation mass for this step.
        // valence is the signed SFE delta.
        self.neuromod.update(level0_pressure, valence);

        Step {
            input: *carrier,
            field_read: s_prime,
            verification,
            cell_a_sigma: sigma(&self.cell_a),
            cell_c_sigma: sigma(&self.cell_c),
            prediction_error,
            self_free_energy,
            valence,
            // parity was incremented above; this position had parity = count-1
            fiber_nontrivial: (self.cell_a_parity - 1) % 2 == 1,
            closure,
            hierarchy_events,
            consolidation_reports,
            semantic,
            arousal_tone: self.neuromod.arousal_tone,
            coherence_tone: self.neuromod.coherence_tone,
            noise_kappa: step_kappa,
        }
    }

    /// Self-observation: Cell C reads its own genome through the Hopf field.
    ///
    /// Returns σ(cell_c, zread_at_query(cell_c, genome, buffer)) —
    /// the geodesic gap between the brain's accumulated state and the
    /// field's integrated response to that state. This is self-free-energy
    /// under the FEP: how far is the brain from being a fixed point of
    /// its own model?
    ///
    /// σ = 0 means Cell C is a fixed point of the genome's ZREAD field.
    /// σ large means the genome has structure Cell C has not yet integrated.
    pub fn self_observe(&self) -> f64 {
        sigma(&self.self_difference_carrier())
    }

    /// The current slow body state.
    /// Useful for experiments and diagnostics that want to inspect the
    /// arousal and coherence tones without reading from the Step output.
    pub fn neuromod(&self) -> &NeuromodState {
        &self.neuromod
    }

    /// The field response to the current Cell C self-model.
    pub fn self_response_carrier(&self) -> [f64; 4] {
        use crate::field::zread_at_query;
        zread_at_query(&self.cell_c, &self.hierarchy.genomes[0], &self.buffer)
    }

    /// The self-difference carrier: Cell C compared to its own field mirror.
    pub fn self_difference_carrier(&self) -> [f64; 4] {
        let field_response = self.self_response_carrier();
        compose(&self.cell_c, &inverse(&field_response))
    }

    /// Ingest a sequence of tokens in order.
    ///
    /// Tokens are stored as-is — no position baked into the carrier.
    /// Order is encoded by Cell A through non-commutative composition:
    /// "A then B" produces a different running product than "B then A."
    /// The genome stays position-agnostic so the same token at any
    /// position reinforces the same entry.
    ///
    /// For position-aware ingestion use `position::embed_sequence_with_positions`
    /// first, then pass the result here.
    pub fn ingest_sequence(&mut self, tokens: &[[f64; 4]]) -> Vec<Step> {
        tokens.iter().map(|c| self.ingest(c)).collect()
    }

    /// Run one cycle without new input. BRAIN.md §9.1: "Steps 4–7
    /// still run, but the population the field machine reads is now
    /// just the genome alone" (or whatever buffer entries have not
    /// yet expired). Idle is **the same loop** ingest runs, minus
    /// steps 1–3 (no push) and minus the Cell A composition
    /// (no new raw input to record).
    ///
    /// Order: tick → §9 round trip → chunk-boundary drain. Tick
    /// happens first so any entry that just reached lifetime is
    /// included in the round trip one last time before draining.
    /// Drain comes last so a long idle stretch eventually flushes
    /// every tagged buffer entry into the genome through chunked
    /// promotion. **Idle never tags buffer entries** — there is no
    /// just-pushed entry, and §29.2 (sleep) is the spec's
    /// reorganization mechanism, not per-cycle idle.
    ///
    /// Returns the `VerificationEvent` from this cycle's round trip,
    /// or `None` if both the genome and the buffer were empty (which
    /// only happens if the boot identity was somehow removed; in
    /// normal operation the genome always has at least one entry).
    pub fn idle(&mut self) -> Option<VerificationEvent> {
        self.buffer.tick();

        // ── §9 steps 4–7 over the current population. ────────────
        let f = zread(1.0, 0.0, &self.hierarchy.genomes[0], &self.buffer);
        let s_prime = resonate(&f, &self.hierarchy.genomes[0], &self.buffer);
        let verification = s_prime.as_ref().map(|hit| {
            let stack = self.load_genome_value_for_hit(hit);
            verify(&stack, &hit.carrier)
        });

        // ── Chunk-boundary drain. ────────────────────────────────
        let expired = self.buffer.drain_expired();
        for entry in expired {
            if entry.closures > 0 {
                self.hierarchy.genomes[0].ingest(
                    &entry.carrier,
                    entry.support,
                    entry.closure_sigma,
                    entry.excursion_peak,
                );
            }
        }

        verification
    }

    /// §9 step 6: stack-only LOAD.
    ///
    /// If the field hit came from the **genome**, the slot identity
    /// is already known: load the value at exactly `hit.index`. The
    /// genome design allows `entry.address != entry.value` (after
    /// Law 5 corrections), so re-searching by nearest-to-value would
    /// return the wrong slot once value drift accumulates.
    ///
    /// If the field hit came from the **buffer**, the carrier is not
    /// in the genome yet — fall back to a genome-only nearest-by-
    /// address scan to find the closest matching slot, and load that
    /// slot's value. The genome always has at least the seeded
    /// identity, so this branch always succeeds.
    fn load_genome_value_for_hit(&self, hit: &ResonanceHit) -> [f64; 4] {
        match hit.source {
            PopulationSource::Genome => self.hierarchy.genomes[0].entries[hit.index].value,
            PopulationSource::Buffer => match self.hierarchy.genomes[0].nearest(&hit.carrier) {
                Some((idx, _)) => self.hierarchy.genomes[0].entries[idx].value,
                None => IDENTITY,
            },
        }
    }

    // ── Native bootstrap drivers ─────────────────────────────────
    //
    // The brain's only low-level verb is `ingest`. The methods below
    // are *drivers* on top of `ingest` that script the standard
    // §19 bootstrap operations: feed a generator until Cell A
    // closes (a single orbit, CP2/CP3), feed a perpendicular pair
    // (CP4), and drive cross-orbit closing sequences (CP5). They
    // work for arbitrary unit-quaternion generators, not just the
    // hardcoded examples — any rotation that produces a finite
    // orbit on S³ will bootstrap into the genome.

    /// Drive the diabolo with a single generator until Cell A's
    /// running product closes (Law 1 fires) or `max_steps` is reached.
    /// Returns `Some(orbit_length)` if Cell A closed, `None` if it
    /// did not close within the budget.
    ///
    /// **Side effect**: every step is a normal `ingest`, so the buffer
    /// fills, the round-trip runs, and on closure the orbit-position
    /// persistence loop writes ε^1, ε^2, …, ε^(n-1) into the genome
    /// (the integer ladder of the orbit).
    pub fn bootstrap_single_orbit(
        &mut self,
        generator: &[f64; 4],
        max_steps: usize,
    ) -> Option<usize> {
        for step in 1..=max_steps {
            let s = self.ingest(generator);
            // Only terminate on an identity closure (σ → 0). Balance
            // closures (σ → π/4) fire mid-orbit and must not truncate it.
            if let Some(ev) = s.closure {
                if ev.sigma < self.cell_a_threshold {
                    return Some(step);
                }
            }
        }
        None
    }

    /// Fast orbit seeding for large moduli: skip the full §9 loop and
    /// directly compute ε^k for k = 0..n, seeding each as a DNA entry.
    ///
    /// This is O(n) instead of the O(n²) that `bootstrap_single_orbit`
    /// takes (which runs the full §9 round-trip at each step). For orbits
    /// above ~1000 entries, use this.
    ///
    /// After this call, genome slot k holds ε^k (0-indexed):
    ///   slot 0 = IDENTITY  (already seeded by `new()` — not duplicated)
    ///   slot 1 = ε^1
    ///   ...
    ///   slot n-1 = ε^(n-1)
    ///
    /// The entries are DNA-layer — immutable, survive sleep.
    pub fn seed_orbit_dna(&mut self, generator: &[f64; 4], n: usize) {
        // slot 0 = IDENTITY already seeded in new(); start from ε^1.
        let mut r = compose(&IDENTITY, generator);
        for _ in 1..n {
            self.hierarchy.genomes[0].seed_dna(r, 0, 0.0, 0.0);
            r = compose(&r, generator);
        }
    }

    /// Drive a fixed sequence of carriers through `ingest` and report
    /// how many Cell A closures fired and how much the genome grew.
    /// Useful for CP5: alternating sequences whose total Hamilton
    /// product returns to identity will fire a Cell A closure, and
    /// the closing window's partial products land in the genome —
    /// these are the **cross-orbit lattice points**.
    pub fn drive_sequence(&mut self, seq: &[[f64; 4]]) -> SequenceOutcome {
        let closures_before = self.total_closures();
        let size_before = self.hierarchy.genomes[0].len();
        for q in seq {
            self.ingest(q);
        }
        SequenceOutcome {
            closures_fired: self.total_closures() - closures_before,
            genome_growth: self.hierarchy.genomes[0].len().saturating_sub(size_before),
        }
    }

    /// Compose a sequence externally (as if it were a single query)
    /// and look it up in the genome via RESONATE. **Read-only** — no
    /// `ingest`, no buffer write, no genome growth. The brain reads
    /// what the composition resolves to, like asking it "what slot
    /// is `ε^a · δ^b`?".
    pub fn evaluate(&self, seq: &[[f64; 4]]) -> Option<ResonanceHit> {
        let mut q = IDENTITY;
        for c in seq {
            q = compose(&q, c);
        }
        crate::field::resonate(&q, &self.hierarchy.genomes[0], &self.buffer)
    }

    /// Convenience: compose a single product `compose(a, b)` and look
    /// it up. The geometric "addition along an orbit" or "cross-orbit
    /// multiplication" verb, depending on whether `a` and `b` live on
    /// the same orbit.
    pub fn evaluate_product(&self, a: &[f64; 4], b: &[f64; 4]) -> Option<ResonanceHit> {
        self.evaluate(&[*a, *b])
    }

    /// Autoregressive generation via genome edge traversal.
    ///
    /// RESONATE the `seed` against the genome to find the entry it lands
    /// on, then follow the highest-weight outgoing edge at each step.
    /// Each step emits the matched entry's `value` carrier. Returns up to
    /// `steps` carriers; terminates early when an entry has no outgoing
    /// edges.
    ///
    /// `entry.edges` records which entries followed the current one during
    /// ingestion — the brain's learned transition function. Following the
    /// highest-weight edge is the deterministic argmax; for stochastic
    /// generation, perturb the edge selection by the caller.
    ///
    /// Read-only: no `ingest`, no buffer write, no genome growth.
    pub fn generate(&self, seed: &[f64; 4], steps: usize) -> Vec<[f64; 4]> {
        let mut result = Vec::with_capacity(steps);
        let mut current_idx = match self.hierarchy.genomes[0].nearest(seed) {
            Some((idx, _)) => idx,
            None => return result,
        };

        for _ in 0..steps {
            result.push(self.hierarchy.genomes[0].entries[current_idx].value);

            // Most-frequent successor: edge with highest transition count.
            let next = self.hierarchy.genomes[0].entries[current_idx]
                .edges
                .iter()
                .max_by_key(|(_, count)| *count)
                .map(|(target, _)| *target);

            match next {
                Some(idx) => current_idx = idx,
                None => break,
            }
        }

        result
    }

    /// Force one consolidation pass over every existing memory level.
    ///
    /// Returns one runtime report per level. Normal runtime uses pressure
    /// accumulation inside `ingest`; this method is an explicit metabolic flush.
    pub fn sleep(&mut self) -> Vec<RuntimeConsolidationReport> {
        let levels = self.hierarchy.genomes.len();
        let mut reports = Vec::new();
        for level in 0..levels {
            let before = if level == 0 {
                Some(self.self_observe())
            } else {
                None
            };
            if let Some(report) = self.consolidate_level(level, before, true) {
                reports.push(report);
            }
        }
        reports
    }

    /// Equation solver — search the genome for the slot index `k` such
    /// that `make_expr(genome.entries[k].address)` RESONATEs to the same
    /// slot as `target`.
    ///
    /// The search space is the genome. The equality test is RESONATE —
    /// the same substrate primitive the brain uses for everything else.
    /// No external arithmetic, no `%`, no floats outside of quaternion
    /// operations. "Find x such that f(x) = target" is just "enumerate
    /// orbit positions until one evaluates to target."
    ///
    /// Returns the first matching slot index, or `None` if no slot
    /// satisfies the equation. For multi-solution equations (e.g.
    /// x² = 4 mod 12 has x=2 and x=10) only the first is returned.
    pub fn solve(
        &self,
        make_expr: impl Fn(&[f64; 4]) -> [f64; 4],
        target: &[f64; 4],
    ) -> Option<usize> {
        // Genome-only lookup for both target and result — the solver is
        // an evaluator, not a live ingest step. Using genome ∪ buffer
        // risks comparing a buffer index against a genome index (both
        // are usize, different namespaces) which gives wrong answers.
        let (target_idx, _) = self.hierarchy.genomes[0].nearest(target)?;
        for (k, entry) in self.hierarchy.genomes[0].entries.iter().enumerate() {
            let result = make_expr(&entry.address.geometry());
            if let Some((hit_idx, _)) = self.hierarchy.genomes[0].nearest(&result) {
                if hit_idx == target_idx {
                    return Some(k);
                }
            }
        }
        None
    }

    /// Feed a corpus of carriers through the living runtime.
    ///
    /// Consolidation is not forced here; it fires inside `ingest` when
    /// accumulated σ-pressure reaches the balance locus.
    ///
    /// Returns total Cell A closures and genome growth over the corpus.
    pub fn run_curriculum(&mut self, corpus: &[[f64; 4]]) -> SequenceOutcome {
        let closures_before = self.total_closures();
        let size_before = self.hierarchy.genomes[0].len();
        for q in corpus {
            self.ingest(q);
        }
        SequenceOutcome {
            closures_fired: self.total_closures() - closures_before,
            genome_growth: self.hierarchy.genomes[0].len().saturating_sub(size_before),
        }
    }

    // ── Explicit update contract ──────────────────────────────────────────────
    //
    // `update` and `update_sequence` are the canonical "living" verbs.
    // Every call to these methods mutates the brain's body and returns a
    // uniform `UpdateReport`. Read-only verbs (`evaluate`, `generate`,
    // `cell_a`, etc.) return nothing that satisfies this contract — the
    // type boundary IS the living/reading boundary.

    /// Single-step body-change verb. Delegates to `ingest`, returns an
    /// `UpdateReport` summarizing the mutation.
    ///
    /// Use this when you need a uniform mutation summary. Use `ingest`
    /// when you need the full per-step trace (`Step`).
    pub fn update(&mut self, carrier: &[f64; 4]) -> UpdateReport {
        let size_before = self.hierarchy.genomes[0].len();
        let closures_before = self.total_closures();
        let step = self.ingest(carrier);
        let consolidation_fired = !step.consolidation_reports.is_empty();
        UpdateReport {
            genome_delta: self.hierarchy.genomes[0]
                .len()
                .saturating_sub(size_before),
            closures_fired: self.total_closures() - closures_before,
            hierarchy_depth: self.hierarchy.depth(),
            consolidation_fired,
            consolidation_reports: step.consolidation_reports,
        }
    }

    /// Sequence body-change verb. Runs `update` for each carrier in `seq`
    /// and accumulates a single `UpdateReport` for the whole sequence.
    pub fn update_sequence(&mut self, seq: &[[f64; 4]]) -> UpdateReport {
        let size_before = self.hierarchy.genomes[0].len();
        let closures_before = self.total_closures();
        let mut all_reports = Vec::new();
        for q in seq {
            let step = self.ingest(q);
            all_reports.extend(step.consolidation_reports);
        }
        let consolidation_fired = !all_reports.is_empty();
        UpdateReport {
            genome_delta: self.hierarchy.genomes[0]
                .len()
                .saturating_sub(size_before),
            closures_fired: self.total_closures() - closures_before,
            hierarchy_depth: self.hierarchy.depth(),
            consolidation_fired,
            consolidation_reports: all_reports,
        }
    }

    // ── CP6: irreducibility detection (primes) ──────────────────
    //
    // After CP3, the brain has a single-orbit basis: slot 0 holds
    // the boot identity (= ε^0), slot k holds ε^k for k = 1..n-1.
    // The slot index *is* the integer label of the orbit position.
    //
    // BRAIN.md §19 CP6 says: "for each carrier in the genome,
    // search for a pair of smaller carriers whose composition
    // produces it. The carriers that can be factored: composites.
    // The carriers that cannot: primes in the brain's basis."
    // The example uses a 12-orbit and expects 2, 3, 5, 7, 11 to
    // be prime — that is *integer* irreducibility, the kind that
    // lives in the Hurwitz quaternion ring's multiplicative norm
    // (|q̂_a · q̂_b|² = a · b).
    //
    // The substrate has the operationalization in `crate::zeta`:
    // `is_prime_geometric(k)` enumerates Hurwitz factor pairs of
    // `k` via the integer Hamilton product (`hamilton_int`) and
    // reads the norm of the result. **Multiplication on S³, no `%`.**
    // CP6 in the brain calls that function for every orbit position
    // the genome holds.

    /// CP6: classify each orbit position in the genome as prime or
    /// composite using the geometric primality test from
    /// [`crate::zeta::is_prime_geometric`].
    ///
    /// **Precondition**: the brain is in a single-orbit state — i.e.,
    /// genome slot `k` for `k ∈ [0, n−1]` holds the carrier `ε^k`
    /// of one bootstrapped orbit. (Calling this on a richer genome
    /// — e.g., after CP4/CP5 cross-orbit drives — still works as a
    /// scan over slot indices, but the answer is then "is the slot
    /// index a prime integer", which only matches the spec on a
    /// pure single-orbit basis.)
    ///
    /// Returns a `Vec<(orbit_position, is_prime)>` for `k ∈ [2, n−1]`
    /// where `n` is the genome size. The position `k = 0` is the
    /// boot identity and `k = 1` is the orbit generator — both
    /// excluded from the "is this irreducible" question, matching
    /// the standard definition of primes in `ℤ`.
    pub fn classify_orbit_irreducibility(&self) -> Vec<(u64, bool)> {
        // DNA entries are structural anchors (identity, equatorial, prime-3).
        // Only epigenetic entries are bootstrapped orbit positions. The k-th
        // epigenetic entry (0-indexed) corresponds to orbit position k+2.
        self.hierarchy.genomes[0]
            .entries
            .iter()
            .filter(|e| e.layer == Layer::Epigenetic)
            .enumerate()
            .map(|(k, _)| {
                let orbit = (k + 2) as u64;
                (orbit, crate::zeta::is_prime_geometric(orbit))
            })
            .collect()
    }

    /// CP6 with explicit factor witnesses.
    pub fn explain_orbit_irreducibility(
        &self,
    ) -> Vec<(u64, Option<crate::zeta::GeometricFactorization>)> {
        self.hierarchy.genomes[0]
            .entries
            .iter()
            .filter(|e| e.layer == Layer::Epigenetic)
            .enumerate()
            .map(|(k, _)| {
                let orbit = (k + 2) as u64;
                (orbit, crate::zeta::find_geometric_factor(orbit))
            })
            .collect()
    }

    // ── Sensory boundary note ─────────────────────────────────────
    //
    // Domain glyph/token parsing lives outside the body loop. The runtime
    // receives carriers; sensory boundaries decide how symbols become carriers.

    // --- Observation (read the diabolo's geometric state) ---

    pub fn cell_a(&self) -> [f64; 4] {
        self.cell_a
    }

    pub fn cell_a_sigma(&self) -> f64 {
        sigma(&self.cell_a)
    }

    pub fn cell_c(&self) -> [f64; 4] {
        self.cell_c
    }

    pub fn cell_c_sigma(&self) -> f64 {
        sigma(&self.cell_c)
    }

    /// S² Hopf base of the current Cell C carrier.
    ///
    /// The base encodes the semantic type of the brain's accumulated
    /// prediction — where on the sphere the self-model currently points.
    pub fn cell_c_hopf_base(&self) -> [f64; 3] {
        hopf::decompose(&self.cell_c).0
    }

    /// W-component of Cell C: |cell_c[0]|.
    ///
    /// This crate defines `sigma(q) = acos(|q[0]|)`, so `|w| = cos(σ)`.
    /// Close to 1 = σ near 0 (near identity, high existence depth).
    /// Equal to 1/√2 ≈ 0.707 = Hopf equator (σ = π/4, balance locus, FixedPoint basin).
    /// Close to 0 = σ near π/2, deep RGB zone, far from identity.
    pub fn cell_c_w_depth(&self) -> f64 {
        self.cell_c[0].abs()
    }

    /// Minimum W-depth Cell C has ever reached across all closures.
    ///
    /// Records the deepest excursion of the self-model away from identity.
    /// Initialized to 1.0 (identity). Decreases as the brain accumulates
    /// packets further from identity. Persisted in `BrainState`.
    pub fn cell_c_min_w(&self) -> f64 {
        self.cell_c_min_w
    }

    /// The current prediction: Cell C itself.
    pub fn prediction(&self) -> [f64; 4] {
        self.cell_c
    }

    /// Raw staging verb — sets `pending_prediction` directly without collecting
    /// eligibility or recording co-resonance.
    ///
    /// **Do not use this for the learning path.**  Passing `vec![]` for
    /// `eligibility` severs `credit_response` and co-resonance accumulation,
    /// so category birth will never fire for the prediction you stage here.
    ///
    /// Use `commit_prediction()` for any prediction that should drive learning.
    /// This method exists for tests that explicitly exercise prediction
    /// lifecycle mechanics (evaluate_prediction feedback, pending_prediction
    /// persistence, ingest non-consumption) where the learning path is not
    /// the subject under test.
    pub(crate) fn stage_prediction(
        &mut self,
        predicted: [f64; 4],
        context: [f64; 4],
        source: PredictionSource,
        eligibility: Vec<(usize, f64)>,
    ) {
        // Use the perception-time salience direction set by the preceding ingest()
        // call: the X-component (R/salience axis) of compose(f, inverse(cell_c)).
        // This is antisymmetric — compose(f, inv(cell_c))[1] = -compose(cell_c, inv(f))[1] —
        // so the learning rate genuinely changes when total and known are swapped.
        //
        // Using the perception-time value (stored in self.last_salience_x, set
        // in ingest()) rather than recomputing here from (predicted, context)
        // makes Step.semantic the actual source of the learning signal.
        //
        // Default 0.0 when no preceding ingest exists (first call after init or
        // after from_brain_state) — safe: no learning amplification.
        let salience_x = self.last_salience_x;
        self.pending_prediction = Some(PendingPrediction {
            predicted,
            context,
            source,
            eligibility,
            cycle: self.cycle_count,
            salience_x,
        });
    }

    /// Commit a prediction into the learning loop — the canonical path.
    ///
    /// Encapsulates the full corrective-loop staging sequence:
    /// 1. Capture `cell_c` as the context (the accumulated self-model at this moment).
    /// 2. Collect which Response entries are active in the current field — these
    ///    are causally responsible for the prediction.
    /// 3. Record pairwise co-resonance for the active coalition (coalition evidence
    ///    for category birth).
    /// 4. Stage the prediction with effective eligibility — so `credit_response`
    ///    and co-resonance accumulation are fully wired when `evaluate_prediction`
    ///    is later called.
    ///
    /// Eligibility is split into two objects:
    ///   raw_eligibility   — full coalition (t ≥ ZREAD_T_MIN, no inhibition)
    ///                       used for co-resonance and category formation evidence
    ///   effective_eligibility — raw after lateral inhibition
    ///                       used for stage_prediction and credit_response
    ///
    /// Use this instead of calling `stage_prediction` directly.  Passing `vec![]`
    /// to `stage_prediction` would sever the co-resonance path entirely.
    pub fn commit_prediction(&mut self, predicted: [f64; 4], source: PredictionSource) {
        let context = self.cell_c;
        let raw = collect_response_eligibility_raw(&context, &self.hierarchy.genomes[0]);
        let effective = collect_response_eligibility(&context, &self.hierarchy.genomes[0]);
        self.hierarchy.genomes[0].record_co_resonance(&raw); // structural: full coalition evidence
        self.stage_prediction(predicted, context, source, effective); // causal: winner-suppressed
    }

    /// Pending output waiting for reality's next evaluation tick.
    pub fn pending_prediction(&self) -> Option<PendingPrediction> {
        self.pending_prediction.clone()
    }

    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    pub fn buffer_mut(&mut self) -> &mut Buffer {
        &mut self.buffer
    }

    pub fn genome_size(&self) -> usize {
        self.hierarchy.genome_size()
    }

    /// Access to the level-0 genome entries for inspection.
    pub fn genome_entries(&self) -> &[crate::genome::GenomeEntry] {
        &self.hierarchy.genomes[0].entries
    }

    /// The last carrier pushed into Cell A's history, if any.
    /// When noise is enabled, this is the noisy version.
    pub fn cell_a_last_carrier(&self) -> Option<[f64; 4]> {
        self.cell_a_history.last().copied()
    }

    /// Current Cell A state (alias for cell_a()).
    pub fn cell_a_state(&self) -> [f64; 4] {
        self.cell_a
    }

    pub fn hierarchy_depth(&self) -> usize {
        self.hierarchy.depth()
    }

    pub fn total_closures(&self) -> usize {
        self.hierarchy.events.len()
    }

    pub fn hierarchy(&self) -> &Hierarchy {
        &self.hierarchy
    }

    pub fn hierarchy_mut(&mut self) -> &mut Hierarchy {
        &mut self.hierarchy
    }

    // ── Full body persistence ──────────────────────────────────────────────────

    /// Snapshot the complete runtime state into a serializable `BrainState`.
    pub fn to_brain_state(&self) -> BrainState {
        BrainState {
            cell_a: self.cell_a,
            cell_a_count: self.cell_a_count,
            cell_a_threshold: self.cell_a_threshold,
            cell_a_excursion_peak: self.cell_a_excursion_peak,
            cell_a_balance_excursion_peak: self.cell_a_balance_excursion_peak,
            cell_a_history: self.cell_a_history.clone(),
            cell_a_parity: self.cell_a_parity,
            cell_c: self.cell_c,
            cell_c_min_w: self.cell_c_min_w,
            self_difference_product: self.self_difference_product,
            self_difference_count: self.self_difference_count,
            self_difference_history: self.self_difference_history.clone(),
            self_balance_excursion_peak: self.self_balance_excursion_peak,
            buffer: self.buffer.clone(),
            genomes: self.hierarchy.genomes.clone(),
            levels: self.hierarchy.levels().to_vec(),
            default_threshold: self.hierarchy.default_threshold,
            default_genome_config: self.hierarchy.default_genome_config.clone(),
            cycle_count: self.cycle_count,
            consolidation_pressure: self.consolidation_pressure.clone(),
            pending_prediction: self.pending_prediction.clone(),
            events: self.hierarchy.events.clone(),
        }
    }

    /// Restore a `ThreeCell` from a saved `BrainState`.
    pub fn from_brain_state(state: BrainState) -> Self {
        let hierarchy = Hierarchy::from_parts(
            state.levels,
            state.genomes,
            state.default_threshold,
            state.default_genome_config.clone(),
            state.events,
        );
        Self {
            cell_a: state.cell_a,
            cell_a_count: state.cell_a_count,
            cell_a_threshold: state.cell_a_threshold,
            cell_a_excursion_peak: state.cell_a_excursion_peak,
            cell_a_balance_excursion_peak: state.cell_a_balance_excursion_peak,
            cell_a_history: state.cell_a_history,
            cell_a_parity: state.cell_a_parity,
            cell_c: state.cell_c,
            cell_c_min_w: state.cell_c_min_w,
            self_difference_product: state.self_difference_product,
            self_difference_count: state.self_difference_count,
            self_difference_history: state.self_difference_history,
            self_balance_excursion_peak: state.self_balance_excursion_peak,
            neuromod: NeuromodState::new(state.buffer.lifetime()),  // session-ephemeral
            buffer: state.buffer,
            hierarchy,
            cycle_count: state.cycle_count,
            consolidation_pressure: state.consolidation_pressure,
            pending_prediction: state.pending_prediction,
            prev_sfe: None,
            last_salience_x: 0.0,  // transient — reset on next ingest
            noise_rng: None,      // noise must be re-enabled after restore
            noise_base_kappa: 0.0,
        }
    }

    /// Save the full runtime state to a JSON file.
    pub fn save_state_to_file(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let state = self.to_brain_state();
        let json = serde_json::to_string_pretty(&state)?;
        let tmp = format!("{}.tmp", path);
        std::fs::write(&tmp, &json)?;
        std::fs::rename(&tmp, path)?;
        Ok(())
    }

    /// Load a `ThreeCell` from a JSON file written by `save_state_to_file`.
    pub fn load_state_from_file(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let json = std::fs::read_to_string(path)?;
        let state: BrainState = serde_json::from_str(&json)?;
        Ok(Self::from_brain_state(state))
    }
}

/// Hopf classification: W-dominant or RGB-dominant. The geometry
/// classifies; the brain reads the classification.
fn hopf_classify(q: &[f64; 4]) -> ClosureKind {
    let w_mag = q[0].abs();
    let rgb_mag = (q[1].powi(2) + q[2].powi(2) + q[3].powi(2)).sqrt();
    if w_mag >= rgb_mag {
        ClosureKind::Completion
    } else {
        ClosureKind::Arrangement
    }
}
