//! Curriculum drivers over the living runtime.
//!
//! Learning is not a separate phase outside runtime. A curriculum is only
//! ordered experience: input carriers and target carriers are fed through
//! `ThreeCell::ingest`, and the same closure/write/consolidation loop handles
//! memory.
//!
//! A directed curriculum adds a target observation after the input stream and
//! measures the same geometric gap the runtime already uses:
//!
//! ```text
//! Error = σ(compose(predicted, inverse(target)))
//! ```
//!
//! `σ = 0` means the brain's prediction exactly matches the target. `σ = π/2`
//! means maximum gap.
//!
//! ## The teaching loop
//!
//! 1. Ingest the input sequence — one pass, step by step.
//! 2. Take the brain's prediction from the last step's `field_read`.
//!    This is the ZREAD + RESONATE output produced while processing the
//!    last input token — the same code path the genome learns through.
//! 3. Ingest the target — drives the genome toward the correct answer.
//! 4. Return σ(prediction, target) as the curriculum gap.
//!
//! The prediction and the learning step share the same code path
//! (`ingest`). There is no separate `evaluate()` call before ingestion —
//! that would use a different path (composed product → RESONATE) and
//! leave the input unprocessed during the loss measurement.
//!
//! ## Runtime convergence
//!
//! The brain converges when repeated presentation of (input, target) pairs
//! causes the genome to reinforce the target slot's weight until it
//! becomes the argmax in RESONATE. No gradient descent, no backprop —
//! convergence is driven by the geometry of repeated closure.

use crate::hierarchy::ClosureRole;
use crate::sphere::{compose, inverse, sigma, IDENTITY};
use crate::three_cell::{PredictionSource, ThreeCell};

// ── Curriculum trace harness ──────────────────────────────────────────────────
//
// A CurriculumTrace is a deterministic sequence of experience windows. Build
// it once; replay it on multiple independent BrainStates to measure convergence.
// Two brains that receive the same trace must produce the same forced closure
// structure — geometry constrains the observer, not the initialization.
//
// CurriculumReport records per-window and aggregate outcomes so callers can
// audit what each experience window produced, compare convergence rates across
// observers, and detect when a window produced zero closures (dead zone).

/// One labeled experience window in a curriculum trace.
#[derive(Clone, Debug)]
pub struct CurriculumWindow {
    /// Window index (0-based within the trace).
    pub index: usize,
    /// Human-readable label (e.g. "orbit-0", "pass-3", "bootstrap").
    pub label: String,
    /// Carriers to present in this window, in order.
    pub carriers: Vec<[f64; 4]>,
}

/// A deterministic, ordered sequence of experience windows.
///
/// Build once, replay on any number of independent `ThreeCell` instances.
/// The trace is a first-class value: it can be saved, inspected, and
/// shared between observers to measure convergence under a shared curriculum.
#[derive(Clone, Debug, Default)]
pub struct CurriculumTrace {
    pub windows: Vec<CurriculumWindow>,
}

impl CurriculumTrace {
    pub fn new() -> Self {
        Self {
            windows: Vec::new(),
        }
    }

    /// Append a labeled window. Returns the window index assigned.
    pub fn add_window(
        &mut self,
        label: impl Into<String>,
        carriers: Vec<[f64; 4]>,
    ) -> usize {
        let index = self.windows.len();
        self.windows.push(CurriculumWindow {
            index,
            label: label.into(),
            carriers,
        });
        index
    }

    /// Partition a flat carrier sequence into fixed-size windows.
    /// Trailing windows shorter than `window_size` are included.
    pub fn from_flat(corpus: &[[f64; 4]], window_size: usize) -> Self {
        let mut trace = Self::new();
        let size = window_size.max(1);
        for (i, chunk) in corpus.chunks(size).enumerate() {
            trace.add_window(format!("window-{i}"), chunk.to_vec());
        }
        trace
    }

    /// Run every window through a brain in order, recording per-window outcomes.
    ///
    /// Each window is a body-mutating update: `ingest` is called for every
    /// carrier. Returns a `CurriculumReport` with a full living audit per window.
    pub fn run(&self, brain: &mut ThreeCell) -> CurriculumReport {
        let mut window_reports = Vec::with_capacity(self.windows.len());
        let mut total_closures = 0usize;
        let mut total_genome_growth = 0usize;

        for window in &self.windows {
            let closures_before = brain.total_closures();
            let size_before = brain.genome_size();

            // Per-level genome sizes before this window.
            let level_sizes_before: Vec<usize> = brain
                .hierarchy()
                .genomes
                .iter()
                .map(|g| g.len())
                .collect();

            let self_free_energy_start = brain.self_observe();

            let mut total_error = 0.0_f64;
            let mut last_sfe = self_free_energy_start;
            let mut carry_closures = 0usize;
            let mut fixedpoint_closures = 0usize;
            let mut consolidation_count = 0usize;

            for q in &window.carriers {
                let step = brain.ingest(q);
                total_error += step.prediction_error;
                last_sfe = step.self_free_energy;

                if let Some(ev) = &step.closure {
                    match ev.role {
                        ClosureRole::Carry => carry_closures += 1,
                        ClosureRole::FixedPoint => fixedpoint_closures += 1,
                    }
                }
                for ev in &step.hierarchy_events {
                    match ev.role {
                        ClosureRole::Carry => carry_closures += 1,
                        ClosureRole::FixedPoint => fixedpoint_closures += 1,
                    }
                }
                consolidation_count += step.consolidation_reports.len();
            }

            let n = window.carriers.len().max(1);
            let closures = brain.total_closures() - closures_before;
            let growth = brain.genome_size().saturating_sub(size_before);
            total_closures += closures;
            total_genome_growth += growth;

            // Per-level genome growth over this window.
            let level_genome_growth: Vec<usize> = brain
                .hierarchy()
                .genomes
                .iter()
                .enumerate()
                .map(|(i, g)| {
                    let before = level_sizes_before.get(i).copied().unwrap_or(0);
                    g.len().saturating_sub(before)
                })
                .collect();

            window_reports.push(WindowReport {
                index: window.index,
                label: window.label.clone(),
                closures_fired: closures,
                carry_closures,
                fixedpoint_closures,
                genome_delta: growth,
                level_genome_growth,
                mean_prediction_error: total_error / n as f64,
                self_free_energy_start,
                self_free_energy_end: last_sfe,
                consolidation_count,
            });
        }

        CurriculumReport {
            windows: window_reports,
            total_closures,
            total_genome_growth,
            final_genome_size: brain.genome_size(),
        }
    }

    /// Number of windows in this trace.
    pub fn len(&self) -> usize {
        self.windows.len()
    }

    pub fn is_empty(&self) -> bool {
        self.windows.is_empty()
    }
}

/// Per-window outcome from replaying a `CurriculumTrace`.
#[derive(Clone, Debug)]
pub struct WindowReport {
    /// Window index (matches `CurriculumWindow::index`).
    pub index: usize,
    /// Window label (matches `CurriculumWindow::label`).
    pub label: String,
    /// Total closures fired during this window (Carry + FixedPoint, all levels).
    pub closures_fired: usize,
    /// Of `closures_fired`: closures with role `ClosureRole::Carry` (identity return,
    /// inter-level handoff). These are the dimensional escalation events.
    pub carry_closures: usize,
    /// Of `closures_fired`: closures with role `ClosureRole::FixedPoint` (balance,
    /// intra-level). These are the FEP fixed-point events.
    pub fixedpoint_closures: usize,
    /// Net genome growth at level 0 during this window.
    pub genome_delta: usize,
    /// Per-level genome growth: `level_genome_growth[k]` = new entries in `genomes[k]`.
    /// Length equals the number of hierarchy levels after this window.
    pub level_genome_growth: Vec<usize>,
    /// Mean prediction error (σ) across all steps in the window.
    pub mean_prediction_error: f64,
    /// Self-free-energy at the first step of this window (before ingesting).
    pub self_free_energy_start: f64,
    /// Self-free-energy at the last step of this window (after ingesting).
    pub self_free_energy_end: f64,
    /// Number of consolidation events that fired during this window.
    pub consolidation_count: usize,
}

/// Full outcome of running a `CurriculumTrace` on a brain.
#[derive(Clone, Debug)]
pub struct CurriculumReport {
    /// Per-window outcomes, in trace order.
    pub windows: Vec<WindowReport>,
    /// Total closures across all windows.
    pub total_closures: usize,
    /// Total genome growth across all windows.
    pub total_genome_growth: usize,
    /// Genome size after the last window.
    pub final_genome_size: usize,
}

// ── Single-example curriculum ─────────────────────────────────────────────────

/// Teach the brain one (input, target) example.
///
/// The architecture is one loop with a perceptual phase and a corrective phase:
///
/// 1. Ingest input (System 1 — perceptual).
/// 2. Read the brain's field prediction.
/// 3. Stage the prediction.
/// 4. Evaluate against target (System 2 — corrective). Writes to Response layer.
///
/// The target is NEVER fed through `ingest`. It is teacher feedback, not a
/// sensory observation. Feeding it through `ingest` would write it into the
/// Epigenetic (perceptual) layer, contaminating perception with labels.
///
/// For sequential tasks where the target is also the next perceptual event
/// (e.g. music, next-token prediction), the caller should additionally call
/// `brain.ingest(target)` after `teach()` returns.
///
/// Returns σ(predicted, target) — zero means exact prediction.
pub fn teach(brain: &mut ThreeCell, input: &[[f64; 4]], target: &[f64; 4]) -> f64 {
    let steps = brain.ingest_sequence(input);
    let last = steps.last();

    let predicted = last
        .and_then(|s| s.field_read.as_ref())
        .map(|h| h.carrier)
        .unwrap_or(IDENTITY);

    let source = last
        .and_then(|s| s.field_read.as_ref())
        .map(|h| PredictionSource::GenomeSlot(h.index))
        .unwrap_or(PredictionSource::GeometricFallback(predicted));

    brain.commit_prediction(predicted, source);
    brain.evaluate_prediction(target);

    sigma(&compose(&predicted, &inverse(target)))
}

/// Run a directed example without gap tracking.
pub fn teach_silent(brain: &mut ThreeCell, input: &[[f64; 4]], target: &[f64; 4]) {
    teach(brain, input, target);
}

// ── Curriculum batches ────────────────────────────────────────────────────────

/// Teach the brain a batch of (input, target) examples.
///
/// Returns the mean σ-gap across the batch.
pub fn teach_batch(brain: &mut ThreeCell, examples: &[(&[[f64; 4]], [f64; 4])]) -> f64 {
    if examples.is_empty() {
        return 0.0;
    }
    let total: f64 = examples
        .iter()
        .map(|(input, target)| teach(brain, input, target))
        .sum();
    total / examples.len() as f64
}

/// Repeat a curriculum for a fixed number of passes.
///
/// Returns the mean σ-gap history.
pub fn run_curriculum_passes(
    brain: &mut ThreeCell,
    examples: &[(&[[f64; 4]], [f64; 4])],
    passes: usize,
) -> Vec<f64> {
    let mut history = Vec::with_capacity(passes);
    for _ in 0..passes {
        let loss = teach_batch(brain, examples);
        history.push(loss);
    }
    history
}

// ── Evaluation ────────────────────────────────────────────────────────────────

/// Evaluate the brain's accuracy on a held-out test set. Read-only.
///
/// For each (input, target) pair, measures σ between the brain's RESONATE
/// output and the target. Returns (mean_gap, exact_hits) where `exact_hits`
/// counts examples where gap < `threshold`.
///
/// Uses `evaluate()` (composed product → RESONATE) rather than
/// `ingest_sequence` — inference doesn't update the genome.
pub fn evaluate_accuracy(
    brain: &ThreeCell,
    examples: &[(&[[f64; 4]], [f64; 4])],
    threshold: f64,
) -> (f64, usize) {
    if examples.is_empty() {
        return (0.0, 0);
    }
    let mut total_gap = 0.0;
    let mut exact_hits = 0usize;
    for (input, target) in examples {
        let predicted = brain.evaluate(input).map(|h| h.carrier).unwrap_or(IDENTITY);
        let gap = sigma(&compose(&predicted, &inverse(target)));
        total_gap += gap;
        if gap < threshold {
            exact_hits += 1;
        }
    }
    (total_gap / examples.len() as f64, exact_hits)
}
