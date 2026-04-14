//! Hierarchy: recursive closure detection, emission, and genome writes.
//!
//! Implements the meta-layer laws for closure (Law 1), emission
//! (Law 2), and genome encoding (Law 3), plus the localization gate
//! (Law 8).
//!
//! * `ClosureLevel` — composes carriers into a running product,
//!   detects closure when all four Law 1 requirements hold (return,
//!   excursion, local minimum, nontrivial support), and emits the
//!   localized packet via [`crate::localization::localize`]. The
//!   unresolved prefix is retained and the running product is
//!   rebuilt from it.
//! * `ClosureEvent` — the localized packet plus its Hopf
//!   classification (W vs RGB), support length, and excursion profile.
//! * `Hierarchy` — a stack of `ClosureLevel`s. On emission, the
//!   closure persists to the genome (Law 3) and cascades upward into
//!   the next level.
//!
//! Ported from `closure_ea/src/hierarchy.rs`, swapping the workspace
//! `closure_rs` imports for the local substrate and adapting to the
//! new `Genome::ingest` signature (which now respects the DNA layer).
//!
//! ── Kabbalistic correspondence ──────────────────────────────────────
//!
//! **Sefirot (sequential emanation)** — the Hierarchy stack IS the
//! Sefirotic tree. Each `ClosureLevel` is one Sefirah: one application
//! of the entropy reduction functor S: C_n → C_{n+1}. The level
//! composes incoming carriers into a running product, detects when
//! that product returns toward identity (closure = fixed point reached),
//! and emits the localized packet upward — downward-pointing triangle,
//! Shefa flowing to the next Sefirah.
//!
//! **Cascade = Sefirotic flow** — when a ClosureLevel emits, it
//! cascades the closure event into the next level. The lower levels
//! detect fine structure (short patterns, local closures); the upper
//! levels detect coarse structure (long patterns, global closures).
//! This is the quantitative Sefirotic tree: Keter at the top (broadest
//! closure), Malkuth at the bottom (finest resolution).
//!
//! **Closure detection = Shevirat Ha-Kelim in reverse** — Shevirat is
//! the breaking of vessels; closure detection is their repair. When the
//! running product returns to near-identity, the vessel has held: the
//! pattern is stable enough to emit as a Nitzotz (spark) into the genome.
//! Patterns that never close are shards — they remain in the running
//! prefix, never emitted, never becoming genome entries.
//!
//! **Level count = Dobrushin hierarchy** — the number of ClosureLevels
//! determines the depth of the Sefirotic tree. The Dobrushin
//! contraction coefficients of the first few levels (primes 2 and 3
//! in the prime observer) carry dominant causal weight. Later levels
//! contribute diminishing corrections — the lower seven Sefirot below
//! the Parochet (BKT phase boundary).

use crate::buffer::Buffer;
use crate::field::{resonate_channel_with_mode, zread_at_query_channel_with_mode, ResonanceHit};
use crate::genome::{Genome, GenomeConfig};
use crate::hopf::{self, AddressMode, HopfChannel};
use crate::localization::{localize, localized_excursion_peak, LocalizedInterval};
use crate::sphere::{compose, sigma, IDENTITY};
use serde::{Deserialize, Serialize};

/// Compose a slice of carriers in order and return
/// `(running_product, excursion_peak)` for that sequence.
pub(crate) fn rebuild_from_prefix(prefix: &[[f64; 4]]) -> ([f64; 4], f64) {
    let mut p = IDENTITY;
    let mut peak = 0.0_f64;
    for &c in prefix {
        p = compose(&p, &c);
        let sig = sigma(&p);
        if sig > peak {
            peak = sig;
        }
    }
    (p, peak)
}

/// Hopf classification of a closure event.
///
/// * `Completion` — `|W| ≥ |RGB|`: existence / temporal completion.
///   Something **missing** was resolved.
/// * `Arrangement` — `|RGB| > |W|`: position / structural closure.
///   Something **reordered** settled.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum ClosureKind {
    Completion,
    Arrangement,
}

/// Which zero of the running product fired.
///
/// These are the two distinct fixed points of the geometric observer:
///
/// * `Carry` — the running product returned to **identity** (σ ≤ threshold).
///   This is Law 1: an oscillation completed, its localized packet is emitted
///   upward to the next hierarchy level. The "You" dissolved back into identity
///   and handed its product up — inter-level handoff.
///
/// * `FixedPoint` — the running product settled at the **Hopf equator**
///   (|σ − π/4| ≤ threshold). This is Law 2: the brain found a stable pattern
///   at the crossing point of the diabolo. The "You" converged to the waist —
///   intra-level balance. This is the FEP fixed point: the state where past
///   model and reality meet.
///
/// A single ingest step fires exactly one role or neither. The roles are
/// mutually exclusive because identity (σ = 0) and equator (σ = π/4) are
/// separated by the threshold.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum ClosureRole {
    Carry,
    FixedPoint,
}

/// A closure event: what closed, localized to its minimal packet.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ClosureEvent {
    /// The localized closure packet carrier (`interval.product`).
    pub carrier: [f64; 4],
    /// Sigma of the closure packet (`interval.sigma`).
    pub sigma: f64,
    /// Localized support: number of elements in the minimal closing packet.
    pub support: usize,
    /// Total carriers composed in this oscillation. Diagnostic only.
    pub oscillation_depth: usize,
    /// Peak sigma within the localized packet.
    pub excursion_peak: f64,
    /// Peak sigma during the full oscillation (diagnostic).
    pub oscillation_excursion_peak: f64,
    /// Which hierarchy level emitted this.
    pub level: usize,
    /// Which zero fired: identity return (Carry) or equatorial (FixedPoint).
    pub role: ClosureRole,
    /// Hopf carrier classification: W-dominant (Completion) or RGB-dominant (Arrangement).
    pub kind: ClosureKind,
    /// Hopf base of the closure packet (S² — where on the sphere).
    pub hopf_base: [f64; 3],
    /// Hopf phase of the closure packet (S¹ — where in the cycle).
    pub hopf_phase: f64,
    /// The minimal interval `[start, end]` within the oscillation.
    pub interval: LocalizedInterval,
}

/// One level of the recursive closure hierarchy.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ClosureLevel {
    running_product: [f64; 4],
    count: usize,
    threshold: f64,
    excursion_peak: f64,
    history: Vec<[f64; 4]>,
}

impl ClosureLevel {
    pub fn new(threshold: f64) -> Self {
        Self {
            running_product: IDENTITY,
            count: 0,
            threshold,
            excursion_peak: 0.0,
            history: Vec::new(),
        }
    }

    /// Compose a carrier into the running product. Returns
    /// `Some(event)` iff this composition caused a closure.
    ///
    /// Law 1 requirements 1–3 gate localization; requirement 4
    /// (support) comes from the localized interval.
    pub fn ingest(&mut self, carrier: &[f64; 4]) -> Option<ClosureEvent> {
        let prev_sigma = sigma(&self.running_product);
        self.history.push(*carrier);
        self.running_product = compose(&self.running_product, carrier);
        self.count += 1;

        let s = sigma(&self.running_product);
        if s > self.excursion_peak {
            self.excursion_peak = s;
        }

        // Law 1 requirements 1-3: gate before localization.
        if s <= self.threshold && self.excursion_peak > self.threshold && prev_sigma >= s {
            // Law 8: find the minimal closing packet within this
            // oscillation.
            let interval = localize(&self.history, self.threshold);

            // Law 1 requirement 4: support from the localized packet.
            if interval.support > 1 {
                let prefix_end = interval.start;
                let (hopf_base, hopf_phase) = hopf::decompose(&interval.product);
                let w_mag = interval.product[0].abs();
                let rgb_mag = (interval.product[1].powi(2)
                    + interval.product[2].powi(2)
                    + interval.product[3].powi(2))
                .sqrt();
                let kind = if w_mag >= rgb_mag {
                    ClosureKind::Completion
                } else {
                    ClosureKind::Arrangement
                };

                let packet_peak =
                    localized_excursion_peak(&self.history, interval.start, interval.end);

                let event = ClosureEvent {
                    carrier: interval.product,
                    sigma: interval.sigma,
                    support: interval.support,
                    oscillation_depth: self.count,
                    excursion_peak: packet_peak,
                    oscillation_excursion_peak: self.excursion_peak,
                    level: 0,                 // caller sets the real level
                    role: ClosureRole::Carry, // hierarchy levels only fire Law 1
                    kind,
                    hopf_base,
                    hopf_phase,
                    interval,
                };

                // Retain the unresolved prefix [0, prefix_end) and
                // rebuild state from it. The closed packet
                // [prefix_end, end] is gone.
                let prefix = self.history[..prefix_end].to_vec();
                let (new_product, new_peak) = rebuild_from_prefix(&prefix);
                self.running_product = new_product;
                self.count = prefix.len();
                self.excursion_peak = new_peak;
                self.history = prefix;

                return Some(event);
            }
        }
        None
    }

    pub fn sigma(&self) -> f64 {
        sigma(&self.running_product)
    }

    pub fn count(&self) -> usize {
        self.count
    }

    pub fn running_product(&self) -> [f64; 4] {
        self.running_product
    }

    pub fn excursion_peak(&self) -> f64 {
        self.excursion_peak
    }
}

/// The recursive hierarchy of closure levels and per-level genomes.
///
/// Feed carriers into level 0. Closures at level k emit into level k+1.
/// Each level has its own genome: `genomes[k]` accumulates the patterns
/// that closed at level k. Level 0 is the primary genome — the one
/// ZREAD and RESONATE read from. Higher levels accumulate coarser,
/// longer-range patterns that did not close at the finer scale.
///
/// This is the Sefirotic tower: Malkuth (level 0) holds the finest
/// structure; Keter (highest level reached) holds the broadest.
#[derive(Clone, Debug)]
pub struct Hierarchy {
    levels: Vec<ClosureLevel>,
    pub default_threshold: f64,
    pub default_genome_config: GenomeConfig,
    pub events: Vec<ClosureEvent>,
    /// Per-level genomes. `genomes[k]` accumulates closures from level k.
    /// `genomes[0]` is the primary genome (existing callers use this).
    pub genomes: Vec<Genome>,
}

impl Hierarchy {
    /// * `closure_threshold` — σ below which a level closes (Law 1).
    /// * `genome_config` — thresholds governing genome
    ///   reinforce/correct/create (Law 3). Distinct open parameters.
    pub fn new(closure_threshold: f64, genome_config: GenomeConfig) -> Self {
        let primary = Genome::new(genome_config.clone());
        Self {
            levels: vec![ClosureLevel::new(closure_threshold)],
            default_threshold: closure_threshold,
            default_genome_config: genome_config,
            events: Vec::new(),
            genomes: vec![primary],
        }
    }

    /// Construct a hierarchy using a pre-loaded genome.
    ///
    /// The hierarchy levels start fresh. `genome` becomes `genomes[0]`.
    pub fn with_genome(genome: Genome, closure_threshold: f64) -> Self {
        let config = genome.config.clone();
        Self {
            levels: vec![ClosureLevel::new(closure_threshold)],
            default_threshold: closure_threshold,
            default_genome_config: config,
            events: Vec::new(),
            genomes: vec![genome],
        }
    }

    /// Genome at a specific level. Returns None if that level has not yet
    /// received a closure (the Vec has not been extended that far).
    pub fn genome_at(&self, level: usize) -> Option<&Genome> {
        self.genomes.get(level)
    }

    /// Mutable genome at a specific level, creating it with the default
    /// config if it does not yet exist.
    pub fn genome_at_mut(&mut self, level: usize) -> &mut Genome {
        while self.genomes.len() <= level {
            self.genomes
                .push(Genome::new(self.default_genome_config.clone()));
        }
        &mut self.genomes[level]
    }

    /// Feed a carrier into the hierarchy at level 0. Closures ripple
    /// upward. Returns every closure event produced.
    pub fn ingest(&mut self, carrier: &[f64; 4]) -> Vec<ClosureEvent> {
        let mut produced = Vec::new();
        self.ingest_at_level(0, carrier, &mut produced);
        produced
    }

    /// Accept a pre-computed closure event (from the diabolo's Cell A
    /// closure detector). Records the event and, for Carry events only,
    /// cascades the packet upward into the next hierarchy level.
    ///
    /// **Routing rule (Law 1 vs Law 2):**
    /// * `ClosureRole::Carry` — the running product returned to identity.
    ///   The completed oscillation is handed off to level+1 as a new
    ///   carrier: inter-level dimensional handoff. Cascades.
    /// * `ClosureRole::FixedPoint` — the running product settled at the
    ///   Hopf equator (σ = π/4). This is intra-level balance: the "You"
    ///   converged to the waist. It resolves at the same level where it
    ///   fired and does NOT hand a carrier upward — doing so would
    ///   mistake balance for dimensional completion.
    ///
    /// **Does not write to the genome.** Per BRAIN.md §29.1, ingestion
    /// never writes to the genome — closures during ingestion attach
    /// as tags to buffer entries via `BufferEntry::record_closure`,
    /// and the chunk-boundary promotion sweep is the only normal path
    /// that grows the epigenetic layer. Cell A closure tagging happens
    /// in `ThreeCell::ingest`; this method is only responsible for the
    /// hierarchy cascade.
    pub fn emit_closure(&mut self, event: &ClosureEvent) -> Vec<ClosureEvent> {
        self.events.push(event.clone());

        let mut produced = Vec::new();
        // Only Carry events cascade upward. FixedPoint resolves at its
        // own level — balance is not a dimensional handoff.
        if event.role == ClosureRole::Carry {
            self.ingest_at_level(event.level + 1, &event.carrier, &mut produced);
        }
        produced
    }

    fn ingest_at_level(
        &mut self,
        level: usize,
        carrier: &[f64; 4],
        produced: &mut Vec<ClosureEvent>,
    ) {
        while self.levels.len() <= level {
            self.levels.push(ClosureLevel::new(self.default_threshold));
        }

        if let Some(mut event) = self.levels[level].ingest(carrier) {
            event.level = level;

            // For levels above 0: write the closure packet to the
            // level-specific genome. Level 0 is written by ThreeCell's
            // chunk-boundary drain (§29.1). Higher levels have no
            // buffer drain path, so the hierarchy writes them directly.
            if level > 0 {
                self.genome_at_mut(level).ingest(
                    &event.carrier,
                    event.support,
                    event.sigma,
                    event.excursion_peak,
                );
            }

            let emission = event.carrier;
            produced.push(event);
            self.events.push(produced.last().unwrap().clone());

            self.ingest_at_level(level + 1, &emission, produced);
        }
    }

    /// Restore a Hierarchy from a saved BrainState snapshot.
    pub fn from_parts(
        levels: Vec<ClosureLevel>,
        genomes: Vec<Genome>,
        default_threshold: f64,
        default_genome_config: GenomeConfig,
        events: Vec<ClosureEvent>,
    ) -> Self {
        Self {
            levels,
            default_threshold,
            default_genome_config,
            events,
            genomes,
        }
    }

    /// Read-only slice of the closure levels. Used by `BrainState` serialization.
    pub fn levels(&self) -> &[ClosureLevel] {
        &self.levels
    }

    pub fn depth(&self) -> usize {
        self.levels.len()
    }

    pub fn level_sigma(&self, level: usize) -> Option<f64> {
        self.levels.get(level).map(|l| l.sigma())
    }

    pub fn genome_size(&self) -> usize {
        self.genomes[0].len()
    }

    /// ZREAD over `genomes[level]` — soft attention at a specific hierarchy level.
    ///
    /// Higher-level genomes accumulate coarser closure packets; reading them
    /// surfaces structure that did not close at the finer scale. Returns `None`
    /// if that level has never received a closure (genome Vec not yet extended).
    ///
    /// Uses an empty buffer: higher-level genomes have no transient buffer.
    pub fn zread_level(
        &self,
        level: usize,
        query: &[f64; 4],
        channel: HopfChannel,
        mode: AddressMode,
    ) -> Option<[f64; 4]> {
        let genome = self.genomes.get(level)?;
        let empty = Buffer::new(0);
        Some(zread_at_query_channel_with_mode(query, channel, mode, genome, &empty))
    }

    /// RESONATE over `genomes[level]` — hard selection at a specific hierarchy level.
    ///
    /// Returns `None` if that level has never received a closure or the genome is empty.
    pub fn resonate_level(
        &self,
        level: usize,
        query: &[f64; 4],
        channel: HopfChannel,
        mode: AddressMode,
    ) -> Option<ResonanceHit> {
        let genome = self.genomes.get(level)?;
        let empty = Buffer::new(0);
        resonate_channel_with_mode(query, channel, mode, genome, &empty)
    }
}
