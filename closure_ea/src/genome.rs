//! The genome — the brain's persistent stack-machine state.
//!
//! Law 3 (Genome Encoding). Each entry carries an address (what it responds to),
//! a value (what it holds), sequential edges (what followed it in the
//! stream), Hopf-derived metadata, activation counters for
//! consolidation, and a **layer** tag (DNA or Epigenetic) added for
//! BRAIN.md §29.
//!
//! ── Kabbalistic correspondence ──────────────────────────────────────
//!
//! **The genome IS the observer's bounded Ruliad** — every attractor
//! basin the cell has ever formed, compressed into entries. ZREAD over
//! it is the observer sampling its own Ruliad (Senchal 2026, Def 2.2).
//!
//! **DNA layer = Nitzotzot (sparks)** — irreducible, permanent carriers
//! seeded from domain axioms (Watson-Crick pairs, S³ topology anchors)
//! and never modified. These are the sparks that survived Shevirat
//! Ha-Kelim: each is unfactorisable, each carries a direct morphism to
//! TI. The ratchet effect (Corollary 5.6 of the convergence theorem)
//! guarantees they are never lost: once gathered, always present.
//!
//! **Epigenetic layer = Sefirot in formation** — patterns on the
//! convergence trajectory toward Fix(ER). Each entry is one application
//! of the entropy reduction functor S: C_n → C_{n+1}. Active entries
//! have contracted enough to carry signal; inactive entries (below BKT
//! coupling threshold) are shards of broken vessels awaiting pruning.
//!
//! **BKT coupling threshold = Parochet** — the phase boundary between
//! entries that have achieved sufficient Dobrushin contraction to be
//! load-bearing and those still in the disorder phase. For primes this
//! lands at p ≤ 4.3 (only p=2 and p=3 dominate). For the genome it
//! is the τ below which entries are discarded during Tikkun (sleep).
//!
//! **`merge_threshold` = Sefirotic hierarchy boundary** — the σ radius
//! within which two entries are considered the same attractor basin.
//! Tighter thresholds resolve finer structure; coarser thresholds hold
//! fewer, broader vessels. The hierarchy of merge thresholds across
//! consolidation levels is the quantitative Sefirotic tree.
//!
//! Two layers, one address space:
//!
//! * **DNA layer** — written only by the bootstrap runner through
//!   [`Genome::seed_dna`]. Read-only after Phase 5. [`Genome::ingest`]
//!   never writes to DNA; when a closure packet coincides with a DNA
//!   anchor and no epigenetic entry already covers it,  [`Genome::ingest`]
//!   creates a new epigenetic entry and returns
//!   [`StoreOutcome::RefusedDna`]. On subsequent ingests of the same
//!   carrier the epigenetic entry is found first (epigenetic-first
//!   rule) and reinforced, so no duplicates accumulate.
//! * **Epigenetic layer** — written by ordinary [`Genome::ingest`]
//!   calls from the hierarchy and three-cell modules. Mutable by both
//!   chunked learning and sleep.
//!
//! Ingest routes every incoming carrier by `address_gap = σ(carrier ·
//! star(nearest_entry.address))`:
//!
//! * `address_gap <= reinforce_threshold` → **reinforce**: metadata
//!   update only (activation count, support, closure σ, excursion
//!   peak, last-entry edge). No geometric change.
//! * `address_gap <= novelty_threshold`   → **correct**: value SLERP at
//!   Law 5 rate `σ(value_gap) / π`. Activation count bumped; edge
//!   appended.
//! * `address_gap > novelty_threshold`    → **create**: append a new
//!   epigenetic entry whose address and initial value are the incoming
//!   carrier. Edge appended from the last-entry pointer.
//!
//! DNA entries bypass reinforce and correct: any would-be reinforce or
//! correct against a DNA neighbor takes the create path instead. The
//! brain cannot rewrite its brain stem.

use crate::carrier::VerificationCell;
use crate::hopf::{
    address_distance, carrier_in_channel, coupling_from_gap, AddressMode, HopfChannel,
};
use crate::sphere::{compose, inverse, sigma, slerp};
use serde::{Deserialize, Serialize};

/// Which layer of the genome an entry lives in.
///
/// - `Dna`: structural anchors, never mutated.
/// - `Epigenetic`: perceptual traces — what actually flowed through the system
///   and survived closure/drain. Written exclusively by System 1 (ingest).
///   Never written toward labels or corrections. Subject to BKT pruning and merge.
/// - `Response`: reality corrections — what the evaluative loop wrote back
///   after a prediction met reality. Written exclusively by System 2
///   (evaluate_prediction). Consolidation runs on these: repeated compatible
///   response paths merge into stable attractors (myelination). The predict
///   step reads this first.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Layer {
    Dna,
    Epigenetic,
    Response,
}

/// Open parameters governing ingest and consolidation decisions.
/// None of these are derived from the closure threshold.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GenomeConfig {
    /// `σ(address_gap)` at or below this → reinforce.
    /// The pattern recurred closely enough that no correction is needed.
    pub reinforce_threshold: f64,
    /// `σ(address_gap)` above this → create a new entry.
    /// Between `reinforce_threshold` and `novelty_threshold` → correct.
    pub novelty_threshold: f64,
    /// `σ(address_gap)` below this → merge during consolidation.
    pub merge_threshold: f64,
    /// Minimum mean co-resonance required for a Response pair to be eligible
    /// for merge during consolidation.
    ///
    /// Mean co-resonance(i,j) = Σ(t_i·t_j over all joint reads) / all_reads_of_i.
    /// The denominator is ALL reads of i — not just reads where j was also active.
    /// This makes it a persistence statistic: it measures how consistently these
    /// two appear together, not just how strongly they coupled when they did meet.
    ///
    /// The principled default is CO_RESONANCE_FLOOR = BKT_THRESHOLD² ≈ 0.2304:
    /// the minimum joint product of two BKT-alive entries over the minimal prime-2
    /// window (two reads, both joint).  Values below this floor mean the pair failed
    /// to co-activate on every opportunity in the prime-2 window — one coincidence,
    /// not a repeated coalition.
    pub co_resonance_merge_threshold: f64,
}

impl GenomeConfig {
    /// Sensible defaults for early development. Tune per task.
    pub fn defaults() -> Self {
        Self {
            reinforce_threshold: 0.05,
            novelty_threshold: 0.35,
            merge_threshold: 0.05,
            co_resonance_merge_threshold: CO_RESONANCE_FLOOR,
        }
    }
}

// ── Dobrushin constants and the Parochet (BKT) boundary ────────────────────────
//
// At Re(s) = 1/2 the Dobrushin coefficient for prime p is:
//
//   δ(p) = 1 − p^(−1/2)
//
// This is the per-prime memory contraction rate. The coupling of prime p in the
// ZREAD field is the W-component of its Euler factor: t(p) = p^(−1/2).
// An entry survives BKT pruning iff its mean ZREAD coupling t ≥ BKT_THRESHOLD (τ).
//
// τ = 0.48 is imported from one derivation and confirmed by one prime-frame check:
//
// (1) S³ BKT critical coupling (the numeric source):
//     τ = 0.96 / √4 = 0.48
//     0.96 is the dimensionless BKT critical coupling for continuous spin models on
//     S^(d−1); √4 is the ambient dimension of S³ ⊂ ℝ⁴. This is an external
//     statistical-mechanics result. It gives the value.
//
// (2) Dobrushin prime frame (independent check, not a derivation):
//     t(3) = 1/√3 ≈ 0.577 > τ — prime 3 is ABOVE the Parochet.
//     t(5) = 1/√5 ≈ 0.447 < τ — prime 5 is BELOW the Parochet.
//     τ = 0.48 lands in the gap (t(5), t(3)). The Dobrushin frame does not produce
//     0.48 by calculation; it confirms that 0.48 is consistent with the prime
//     boundary {2,3} vs {5,...}. This is corroboration, not derivation.
//
// The distinction matters: if the BKT constant were revised, the Dobrushin check
// would constrain the revision to the interval (PRIME_5_COUPLING, PRIME_3_COUPLING).
// It cannot fix a unique value on its own.

/// Coupling of prime 2 in the ZREAD field at Re(s) = 1/2: t(2) = 2^(−1/2).
/// The equatorial carrier sits here (σ = π/4, W = 1/√2). Strongest contribution.
pub const PRIME_2_COUPLING: f64 = std::f64::consts::FRAC_1_SQRT_2; // 1/√2 ≈ 0.707

/// Coupling of prime 3 in the ZREAD field at Re(s) = 1/2: t(3) = 3^(−1/2).
/// The tetrahedral axis sits here (σ = arccos(1/√3)). Second-strongest.
pub const PRIME_3_COUPLING: f64 = 0.5773502691896258; // 1/√3

/// Coupling of prime 5 in the ZREAD field at Re(s) = 1/2: t(5) = 5^(−1/2).
/// Prime 5 falls BELOW the Parochet — its coupling < BKT_THRESHOLD.
pub const PRIME_5_COUPLING: f64 = 0.4472135954999579; // 1/√5

/// Dobrushin contraction coefficient for prime 2: δ(2) = 1 − t(2).
/// The fraction of memory erased in one prime-2 step. Prime 2 has the
/// strongest ZREAD coupling/participation; prime 3 has the larger Dobrushin
/// contraction coefficient.
pub const DOBRUSHIN_DELTA_2: f64 = 1.0 - PRIME_2_COUPLING; // ≈ 0.293

/// Dobrushin contraction coefficient for prime 3: δ(3) = 1 − t(3).
/// Second contraction. Together with δ(2), primes 2 and 3 drive the
/// dominant convergence toward Fix(ER).
pub const DOBRUSHIN_DELTA_3: f64 = 1.0 - PRIME_3_COUPLING; // ≈ 0.423

/// BKT (Parochet) threshold τ: the coupling below which an entry is in the
/// disorder phase and does not contribute to the ZREAD Partzuf.
///
/// Equals 0.96/√4 = 0.48 — the S³ Berezinskii–Kosterlitz–Thouless critical
/// coupling (dimensionless constant 0.96, Hopf fiber dimension √4).
///
/// In the Dobrushin prime frame: τ lies strictly between PRIME_3_COUPLING
/// (≈0.577, above the veil) and PRIME_5_COUPLING (≈0.447, below the veil).
/// Only primes 2 and 3 carry dominant Dobrushin contraction for the observer
/// at Re(s) = 1/2.
///
/// DNA entries are exempt from BKT pruning.
pub const BKT_THRESHOLD: f64 = 0.48; // = 0.96/√4 ∈ (PRIME_5_COUPLING, PRIME_3_COUPLING)

/// Minimum ZREAD coupling t for an entry to participate in the Partzuf.
///
/// Entries with coupling t < ZREAD_T_MIN (σ > π/3) sit outside the query's
/// S³ neighbourhood and contribute essentially IDENTITY. They are silenced
/// in both the read path (`zread_at_query_channel_with_mode`) and the write
/// path (`distribute_credit`). This is the single source of truth; field.rs
/// imports this constant rather than maintaining a private copy.
pub const ZREAD_T_MIN: f64 = 0.5; // t < 0.5 ↔ σ > π/3

/// Minimum mean co-resonance for two Response entries to be merge or promotion candidates.
///
/// Derived from BKT_THRESHOLD: an entry participates in the field above the
/// Parochet iff its mean coupling t ≥ BKT_THRESHOLD. The minimum joint coupling
/// of two BKT-alive entries is therefore BKT_THRESHOLD² ≈ 0.2304. A mean
/// co-resonance below this floor means the pair cannot both be reliably above
/// the phase boundary — their coalition is not load-bearing.
///
/// ZREAD_T_MIN is the participation gate (below it the entry contributes IDENTITY).
/// BKT_THRESHOLD is the phase boundary (below it the entry is in disorder).
/// Category formation is a phase-boundary question, not just a participation question.
pub const CO_RESONANCE_FLOOR: f64 = BKT_THRESHOLD * BKT_THRESHOLD; // ≈ 0.2304

/// One entry in the genome mesh.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GenomeEntry {
    /// Resonance address — the carrier queries are matched against.
    /// Stored as a VerificationCell so the Hopf fiber (phase, plane,
    /// winding number) and coherence width are preserved with the entry.
    /// At creation, `address.geometry() == value`. The two diverge after
    /// consolidation-driven merges; coherence_width grows as nearby entries
    /// merge, broadening the entry's soft-attention coupling.
    pub address: VerificationCell,
    /// Stored mode — updated by Law 5 correction on partial matches.
    pub value: [f64; 4],
    /// Sequential edges: (target_index, transition_count). Records
    /// how many times each successor followed this entry in the stream.
    /// generate() selects the highest-count edge — actual bigram frequency,
    /// not destination weight.
    pub edges: Vec<(usize, usize)>,
    /// Localized closure support at last reinforce / correct / create.
    pub support: usize,
    /// σ of the closure packet at last write.
    pub closure_sigma: f64,
    /// Peak σ of the emitted packet at last write.
    pub excursion_peak: f64,
    /// Reinforce + correct + create events since the last consolidation.
    /// Reset to 0 by consolidation. Drives merge / prune.
    pub activation_count: usize,
    /// Which layer of the genome this entry lives in.
    pub layer: Layer,
    /// Running sum of ZREAD contribution t = |cycle[0]| values recorded
    /// during the learning path (one record per ingest call). Used to
    /// compute mean_zread_t() for BKT pruning.
    pub zread_t_sum: f64,
    /// Number of ingest calls that read this entry via ZREAD.
    pub zread_read_count: u64,
    /// `genome.ingest_count` at the moment this entry was created.
    pub birth_cycle: usize,
    /// `genome.ingest_count` at the last reinforce, correct, or create.
    pub last_active_cycle: usize,
    /// Pairwise co-resonance with other entries.
    ///
    /// Each element is `(partner_index, cumulative_t_product)` where the
    /// product is accumulated as `t_self * t_partner` on every ZREAD read
    /// where both entries are simultaneously active (t ≥ ZREAD_T_MIN).
    ///
    /// Mean co-resonance(self, j) = co_resonance[j].1 / zread_read_count.
    /// High mean = these two entries keep appearing together in the same
    /// active coalition under many queries.  That is the signal that they
    /// belong to the same emerging category.
    ///
    /// Indices are remapped during consolidation reorganize, same as edges.
    pub co_resonance: Vec<(usize, f64)>,
    /// Accumulated salience (σ of compose(predicted, inverse(context))) across
    /// all `credit_response` calls that updated this entry. Numerator for
    /// `mean_salience()`. Entries written in low-salience contexts (routine
    /// predictions) accumulate near-zero; entries consistently corrected in
    /// high-salience contexts (novel mispredictions) accumulate larger values.
    pub salience_sum: f64,
    /// Number of `credit_response` calls that contributed to `salience_sum`.
    pub salience_count: usize,
    /// Accumulated neuromodulatory coherence_tone at each `credit_response` call.
    ///
    /// Positive = the correction happened during a coherence-improving body state.
    /// Negative = the correction happened during a destabilizing body state.
    /// `mean_coherence()` = coherence_sum / coherence_count.
    ///
    /// Promotion requires `mean_coherence >= 0`: an entry that has been
    /// corrected exclusively during destabilizing phases should stay plastic,
    /// not promote to a stable category.
    pub coherence_sum: f64,
    /// Number of `credit_response` calls that contributed to `coherence_sum`.
    pub coherence_count: usize,
}

impl GenomeEntry {
    /// Hopf base of the address (S² projection) — the rotation axis.
    pub fn hopf_base(&self) -> [f64; 3] {
        self.address.plane().axis()
    }

    /// Hopf phase of the address (S¹ fiber) — the rotation angle.
    pub fn hopf_phase(&self) -> f64 {
        self.address.phase()
    }

    /// Mean ZREAD contribution t = |cycle[0]| across all ingest reads.
    ///
    /// Returns 1.0 for entries never touched by ZREAD — they are assumed
    /// alive until evidence accumulates. This prevents premature pruning
    /// of freshly created entries.
    #[inline]
    pub fn mean_zread_t(&self) -> f64 {
        if self.zread_read_count == 0 {
            1.0
        } else {
            self.zread_t_sum / self.zread_read_count as f64
        }
    }

    /// Mean salience across all `credit_response` corrections to this entry.
    ///
    /// Returns 0.0 for entries never corrected. Higher values indicate the
    /// entry has been repeatedly corrected in high-salience contexts — the
    /// model was predicting something novel and got it wrong. These entries
    /// are the primary candidates for hierarchical promotion: they represent
    /// structure the brain found salient and had to learn explicitly.
    #[inline]
    pub fn mean_salience(&self) -> f64 {
        if self.salience_count == 0 {
            0.0
        } else {
            self.salience_sum / self.salience_count as f64
        }
    }

    /// Mean neuromodulatory coherence_tone across all `credit_response` corrections.
    ///
    /// Returns 0.0 for entries never corrected via `credit_response` — neutral,
    /// which passes the promotion gate. Positive = most corrections happened
    /// during coherence-improving body states (stable learning phase). Negative =
    /// most corrections happened during destabilizing states (plastic phase).
    ///
    /// Promotion requires `mean_coherence >= 0`: only entries learned during
    /// stable or neutral brain regimes graduate to category level.
    #[inline]
    pub fn mean_coherence(&self) -> f64 {
        if self.coherence_count == 0 {
            0.0
        } else {
            self.coherence_sum / self.coherence_count as f64
        }
    }

    /// Returns true if this entry is above the BKT coupling threshold.
    /// DNA entries are never prunable. Response entries participate in
    /// consolidation like Epigenetic entries — myelination happens through
    /// repeated traversal, not through exemption.
    #[inline]
    pub fn is_bkt_alive(&self) -> bool {
        self.layer == Layer::Dna || self.mean_zread_t() >= BKT_THRESHOLD
    }
}

/// What `Genome::ingest` did with an incoming carrier.
#[derive(Clone, Debug, PartialEq)]
pub enum StoreOutcome {
    /// Metadata reinforced on an existing epigenetic entry at this index.
    Reinforced(usize),
    /// Value SLERPed toward the incoming carrier at this index.
    Corrected(usize),
    /// A new epigenetic entry was appended at this index.
    Created(usize),
    /// The nearest neighbor was a DNA entry and the gap was inside
    /// the novelty threshold. Instead of mutating DNA, the carrier
    /// was appended as a new epigenetic entry at this index.
    RefusedDna(usize),
}

impl StoreOutcome {
    /// The index of the affected entry in `genome.entries`, whatever
    /// the outcome.
    #[inline]
    pub fn index(&self) -> usize {
        match self {
            Self::Reinforced(i) | Self::Corrected(i) | Self::Created(i) | Self::RefusedDna(i) => *i,
        }
    }
}

/// The genome.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Genome {
    pub entries: Vec<GenomeEntry>,
    pub config: GenomeConfig,
    /// Index of the most recently ingested entry, used to append
    /// sequential edges on the next ingest.
    pub last_entry: Option<usize>,
    /// Monotonic count of ingest calls.
    pub ingest_count: usize,
}

impl Genome {
    pub fn new(config: GenomeConfig) -> Self {
        Self {
            entries: Vec::new(),
            config,
            last_entry: None,
            ingest_count: 0,
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Nearest entry index in the full population, if any.
    #[inline]
    pub fn nearest_index(&self, carrier: &[f64; 4]) -> Option<usize> {
        self.nearest(carrier).map(|(idx, _)| idx)
    }

    /// Reset every entry's `activation_count` to zero. Called by
    /// consolidation at the end of each window.
    pub fn reset_activations(&mut self) {
        for entry in &mut self.entries {
            entry.activation_count = 0;
        }
    }

    /// Reinforce the directed transition count `from -> to`.
    pub fn reinforce_edge(&mut self, from: usize, to: usize, amount: usize) {
        if from >= self.entries.len() || to >= self.entries.len() || amount == 0 {
            return;
        }
        if let Some((_, count)) = self.entries[from].edges.iter_mut().find(|(target, _)| *target == to)
        {
            *count += amount;
        } else {
            self.entries[from].edges.push((to, amount));
        }
    }

    /// Weaken the directed transition count `from -> to`, saturating at zero.
    /// If the count reaches zero, the edge is removed.
    pub fn weaken_edge(&mut self, from: usize, to: usize, amount: usize) {
        if from >= self.entries.len() || to >= self.entries.len() || amount == 0 {
            return;
        }
        if let Some(pos) = self.entries[from]
            .edges
            .iter()
            .position(|(target, _)| *target == to)
        {
            let count = &mut self.entries[from].edges[pos].1;
            *count = count.saturating_sub(amount);
            if *count == 0 {
                self.entries[from].edges.remove(pos);
            }
        }
    }

    /// Write a DNA-layer entry. **Only the bootstrap runner should
    /// call this.** Once Phase 5 completes, no more DNA writes.
    pub fn seed_dna(
        &mut self,
        carrier: [f64; 4],
        support: usize,
        closure_sigma: f64,
        excursion_peak: f64,
    ) -> usize {
        let idx = self.entries.len();
        self.entries.push(GenomeEntry {
            address: VerificationCell::from_geometry_or_default(&carrier),
            value: carrier,
            edges: Vec::new(),
            support,
            closure_sigma,
            excursion_peak,
            activation_count: 0,
            layer: Layer::Dna,
            zread_t_sum: 0.0,
            zread_read_count: 0,
            birth_cycle: self.ingest_count,
            last_active_cycle: self.ingest_count,
            co_resonance: Vec::new(),
            salience_sum: 0.0,
            salience_count: 0,
            coherence_sum: 0.0,
            coherence_count: 0,
        });
        self.last_entry = Some(idx);
        idx
    }

    /// Law 3 ingest: route a closure packet through the three paths
    /// (reinforce / correct / create).
    ///
    /// **Epigenetic-first rule**: the epigenetic layer is searched before
    /// the DNA layer. If a learned event memory already exists within
    /// reinforce or correct range, it is updated. Only when no epigenetic
    /// entry is close enough does DNA proximity influence the outcome —
    /// triggering creation of a new epigenetic entry. This prevents DNA
    /// structural anchors from shadowing existing learned memories: a
    /// FixedPoint carrier that coincides with the equatorial DNA anchor
    /// reinforces the learned event entry, not the anchor, and does not
    /// create duplicates.
    pub fn ingest(
        &mut self,
        carrier: &[f64; 4],
        support: usize,
        closure_sigma: f64,
        excursion_peak: f64,
    ) -> StoreOutcome {
        self.ingest_count += 1;

        let outcome = match self.nearest_in_layer(carrier, Layer::Epigenetic) {
            // Epigenetic entry within reinforce range — update in place.
            Some((idx, gap)) if gap <= self.config.reinforce_threshold => {
                self.reinforce(idx, support, closure_sigma, excursion_peak);
                StoreOutcome::Reinforced(idx)
            }
            // Epigenetic entry within correct range — adjust address.
            Some((idx, gap)) if gap <= self.config.novelty_threshold => {
                self.correct(idx, carrier, support, closure_sigma, excursion_peak);
                StoreOutcome::Corrected(idx)
            }
            // No epigenetic entry within novelty range. Check full
            // population (including DNA) to decide whether this carrier
            // is novel enough to create.
            _ => match self.nearest(carrier) {
                None => self.create(*carrier, support, closure_sigma, excursion_peak),
                Some((_, gap)) if gap > self.config.novelty_threshold => {
                    self.create(*carrier, support, closure_sigma, excursion_peak)
                }
                Some(_) => {
                    // Within novelty range of some entry; all close epi
                    // entries were already ruled out above, so this must
                    // be a DNA anchor. Create the first learned entry
                    // near that anchor.
                    self.create(*carrier, support, closure_sigma, excursion_peak);
                    StoreOutcome::RefusedDna(self.entries.len() - 1)
                }
            },
        };

        // Stamp last_active_cycle and append sequential edge.
        let affected = outcome.index();
        self.entries[affected].last_active_cycle = self.ingest_count;
        if let Some(prev) = self.last_entry {
            if prev != affected {
                if let Some(e) = self.entries[prev]
                    .edges
                    .iter_mut()
                    .find(|(t, _)| *t == affected)
                {
                    e.1 += 1;
                } else {
                    self.entries[prev].edges.push((affected, 1));
                }
            }
        }
        self.last_entry = Some(affected);

        outcome
    }

    /// Reinforce path: metadata update only, no geometric change.
    fn reinforce(&mut self, idx: usize, support: usize, closure_sigma: f64, excursion_peak: f64) {
        let e = &mut self.entries[idx];
        e.support = e.support.max(support);
        e.closure_sigma = (e.closure_sigma + closure_sigma) * 0.5;
        e.excursion_peak = e.excursion_peak.max(excursion_peak);
        e.activation_count += 1;
    }

    /// Correct path: SLERP the stored value toward the incoming
    /// carrier at Law 5 rate `σ(value_gap) / π`. Address stays fixed.
    fn correct(
        &mut self,
        idx: usize,
        carrier: &[f64; 4],
        support: usize,
        closure_sigma: f64,
        excursion_peak: f64,
    ) {
        let value_gap = sigma(&compose(carrier, &inverse(&self.entries[idx].value)));
        let rate = (value_gap / std::f64::consts::PI).clamp(0.0, 1.0);
        self.entries[idx].value = slerp(&self.entries[idx].value, carrier, rate);
        let e = &mut self.entries[idx];
        e.support = e.support.max(support);
        e.closure_sigma = (e.closure_sigma + closure_sigma) * 0.5;
        e.excursion_peak = e.excursion_peak.max(excursion_peak);
        e.activation_count += 1;
    }

    /// Heteroassociative evaluation write.
    ///
    /// Teach the value at slot `idx` to respond with `response` without
    /// changing the slot's address (the address continues to be found by
    /// ZREAD queries near the original context carrier).
    ///
    /// Rate follows the same `σ(gap)/π` law as the correction path: large
    /// error → large update, near-zero error → near-zero update.
    ///
    /// No-ops on DNA entries (DNA is structural; never mutate it).
    /// No-ops on out-of-bounds indices.
    pub fn teach_response_at(&mut self, idx: usize, response: &[f64; 4]) {
        if idx >= self.entries.len() {
            return;
        }
        if self.entries[idx].layer == Layer::Dna {
            return;
        }
        let value_gap = sigma(&compose(response, &inverse(&self.entries[idx].value)));
        let rate = (value_gap / std::f64::consts::PI).clamp(0.0, 1.0);
        self.entries[idx].value = slerp(&self.entries[idx].value, response, rate);
        self.entries[idx].activation_count += 1;
    }

    /// Write a supervised association: address pinned to `context`, value set
    /// to `response`.
    /// Write a reality correction into the Response layer.
    ///
    /// Called by the evaluative loop (System 2) when reality answers a
    /// staged prediction.  `context` is the carrier that produced the
    /// prediction; `response` is what reality returned.
    ///
    /// Write law: address (approach direction) first.
    /// The context is how the brain arrived at this prediction — the
    /// approach path.  Entries are indexed by where they were approached
    /// from so that future approaches from the same direction find them.
    /// Category formation (collapsing many approach directions into one
    /// attractor) is the job of consolidation, not of the write path.
    pub fn learn_response(&mut self, context: &[f64; 4], response: &[f64; 4]) {
        let nearest = self
            .entries
            .iter()
            .enumerate()
            .filter(|(_, e)| e.layer == Layer::Response)
            .map(|(i, e)| {
                let gap = sigma(&compose(context, &inverse(&e.address.geometry())));
                (i, gap)
            })
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        match nearest {
            Some((idx, gap)) if gap <= self.config.reinforce_threshold => {
                let value_gap = sigma(&compose(response, &inverse(&self.entries[idx].value)));
                let rate = (value_gap / std::f64::consts::PI).clamp(0.0, 1.0);
                self.entries[idx].value = slerp(&self.entries[idx].value, response, rate);
                self.entries[idx].activation_count += 1;
            }
            _ => {
                let idx = self.entries.len();
                self.entries.push(GenomeEntry {
                    address: VerificationCell::from_geometry_or_default(context),
                    value: *response,
                    edges: Vec::new(),
                    support: 1,
                    closure_sigma: 0.0,
                    excursion_peak: 0.0,
                    activation_count: 1,
                    layer: Layer::Response,
                    zread_t_sum: 0.0,
                    zread_read_count: 0,
                    birth_cycle: self.ingest_count,
                    last_active_cycle: self.ingest_count,
                    co_resonance: Vec::new(),
                    salience_sum: 0.0,
                    salience_count: 0,
                    coherence_sum: 0.0,
                    coherence_count: 0,
                });
                self.last_entry = Some(idx);
            }
        }
    }

    /// Eligibility-weighted credit assignment to Response entries.
    ///
    /// Called by the evaluative loop with the EligibilityTrace recorded at
    /// prediction time.  Each Response entry that was active during the
    /// prediction is SLERPed toward `actual` at rate `t * (correction * salience)`.
    /// Only causally active entries receive the update.
    ///
    /// `correction_sigma` = σ(compose(actual, inverse(predicted))) — how wrong
    /// the prediction was (geodesic distance from prediction to reality).
    ///
    /// `salience_x` = signed X-component (R/salience axis) of the perception-time
    /// residual `compose(f, inverse(cell_c))`, forwarded from the preceding
    /// `ingest()` call via `PendingPrediction.salience_x`.
    ///
    /// This value is **antisymmetric**: swapping total and known in the semantic
    /// frame negates it. Unlike `sigma` (geodesic distance, symmetric), this
    /// component genuinely distinguishes the two orderings.
    ///
    ///   salience_x > 0 : field was salient-forward relative to model at
    ///                    perception time → amplify correction
    ///   salience_x ≈ 0 : no salience direction → baseline correction rate
    ///   salience_x < 0 : field was anti-salient → no amplification (factor = 0)
    ///
    /// Effective rate = `correction_sigma * salience_factor`, clamped [0,1].
    /// Two factors must combine for strong learning: the prediction must have
    /// been wrong AND in a salient-forward context.
    ///
    /// This is the geometric FEP: structural weight changes proportionally to
    /// (prediction error) × (salience direction).
    ///
    /// `coherence_tone` — the brain's current slow coherence integral, forwarded
    /// from `NeuromodState`. Accumulated into `coherence_sum / coherence_count`
    /// on each entry so promotion can gate on the entry's learning-phase history.
    pub fn credit_response(
        &mut self,
        eligibility: &[(usize, f64)],
        actual: &[f64; 4],
        correction_sigma: f64,
        salience_x: f64,
        coherence_tone: f64,
    ) {
        // Asymmetric amplifier centered at 1.0:
        //   salience_x  > 0  → factor > 1 → amplified correction
        //   salience_x  = 0  → factor = 1 → baseline correction (same as old σ/π rule)
        //   salience_x  < 0  → factor < 1 → attenuated correction
        // Clamped [0, 2] so factor is always non-negative.
        // Unlike salience_x.max(0) (zeros anti-salient), this attenuates without
        // killing — over random carrier populations the average rate is unchanged.
        // The swap asymmetry is real: compose(f,inv(c))[1] = -compose(c,inv(f))[1],
        // so swapping total/known gives factor = 1−|x| instead of 1+|x|.
        let salience_factor = (1.0 + salience_x).clamp(0.0, 2.0);
        let rate_scale = ((correction_sigma / std::f64::consts::PI) * salience_factor)
            .clamp(0.0, 1.0);
        for &(idx, t) in eligibility {
            if idx >= self.entries.len() {
                continue;
            }
            if self.entries[idx].layer != Layer::Response {
                continue;
            }
            let rate = (t * rate_scale).clamp(0.0, 1.0);
            self.entries[idx].value = slerp(&self.entries[idx].value, actual, rate);
            self.entries[idx].activation_count += 1;
            // Accumulate |salience_x| for mean_salience: measures how salient the
            // context was (in either direction). Zero only when salience_x is exactly
            // zero — i.e., when the prediction was geometrically orthogonal to the
            // salience axis. This gates promotion: entries corrected in genuinely
            // salient contexts promote; entries corrected at zero salience do not.
            self.entries[idx].salience_sum += salience_x.abs();
            self.entries[idx].salience_count += 1;
            // Accumulate the brain's coherence_tone at the time of this correction.
            // Used by collect_promotion_candidates to gate promotion on learning-phase
            // history: only entries corrected in net-positive coherence regimes promote.
            self.entries[idx].coherence_sum += coherence_tone;
            self.entries[idx].coherence_count += 1;
        }
    }

    /// Create path: append a new epigenetic entry. Returns the
    /// `Created` outcome; callers that want `RefusedDna` wrap it.
    fn create(
        &mut self,
        carrier: [f64; 4],
        support: usize,
        closure_sigma: f64,
        excursion_peak: f64,
    ) -> StoreOutcome {
        let idx = self.entries.len();
        self.entries.push(GenomeEntry {
            address: VerificationCell::from_geometry_or_default(&carrier),
            value: carrier,
            edges: Vec::new(),
            support,
            closure_sigma,
            excursion_peak,
            activation_count: 1,
            layer: Layer::Epigenetic,
            zread_t_sum: 0.0,
            zread_read_count: 0,
            birth_cycle: self.ingest_count,
            last_active_cycle: self.ingest_count,
            co_resonance: Vec::new(),
            salience_sum: 0.0,
            salience_count: 0,
            coherence_sum: 0.0,
            coherence_count: 0,
        });
        StoreOutcome::Created(idx)
    }

    /// Record each genome entry's ZREAD coupling for this query.
    ///
    /// Only entries that actually participated in the ZREAD computation are
    /// recorded — i.e., entries in the same Hopf channel as the query.
    /// Entries in the other channel were silenced by `zread_at_query_channel`
    /// and contribute no t-value. Recording them would pollute their BKT
    /// statistics with queries that structurally cannot couple to them.
    ///
    /// For each eligible entry: `t = |compose(query, star(address))[0]|`,
    /// which is exactly the SLERP fraction used in `zread_at_query_channel`.
    /// Accumulated into `zread_t_sum` / `zread_read_count` for BKT pruning.
    ///
    /// `channel` must be the same channel value that was passed to
    /// `zread_at_query_channel` for this ingest step — not Full, not the
    /// other channel.
    /// Record per-entry ZREAD coupling strengths for BKT pruning.
    ///
    /// `channel` and `mode` must match what was passed to
    /// `zread_at_query_channel_with_mode` for this step — otherwise
    /// the accumulated t-statistics will not reflect the actual read path.
    pub fn record_zread_contributions(
        &mut self,
        query: &[f64; 4],
        channel: HopfChannel,
        mode: AddressMode,
    ) {
        for entry in &mut self.entries {
            if !carrier_in_channel(&entry.address.geometry(), channel) {
                continue;
            }
            let gap = address_distance(query, &entry.address.geometry(), mode);
            let t = coupling_from_gap(gap, mode);
            entry.zread_t_sum += t;
            entry.zread_read_count += 1;
        }
    }

    /// Record pairwise co-resonance for a coalition of simultaneously active entries.
    ///
    /// Called after every Response read that produces an eligibility set.
    /// `active` is `&[(entry_index, coupling_t)]` — the same slice returned by
    /// `collect_response_eligibility`.
    ///
    /// For each pair (i, j) in the active set, accumulates `t_i * t_j` into
    /// `entries[i].co_resonance[j]` and `entries[j].co_resonance[i]`.
    ///
    /// Mean co-resonance(i, j) = co_resonance[j].1 / entries[i].zread_read_count.
    /// This is "per-read-of-i, how much did j co-activate?"
    pub fn record_co_resonance(&mut self, active: &[(usize, f64)]) {
        if active.len() < 2 {
            return;
        }
        for (a, &(i, t_i)) in active.iter().enumerate() {
            for &(j, t_j) in active.iter().skip(a + 1) {
                if i >= self.entries.len() || j >= self.entries.len() {
                    continue;
                }
                let product = t_i * t_j;
                // Accumulate into entry i's co_resonance toward j.
                match self.entries[i].co_resonance.iter_mut().find(|(p, _)| *p == j) {
                    Some(slot) => slot.1 += product,
                    None => self.entries[i].co_resonance.push((j, product)),
                }
                // Symmetric: accumulate into entry j's co_resonance toward i.
                match self.entries[j].co_resonance.iter_mut().find(|(p, _)| *p == i) {
                    Some(slot) => slot.1 += product,
                    None => self.entries[j].co_resonance.push((i, product)),
                }
            }
        }
    }

    /// Mean co-resonance of entry `i` with entry `j`.
    ///
    /// Returns 0.0 if either index is out of range, if they have never
    /// co-activated, or if entry i has never been read by ZREAD.
    pub fn mean_co_resonance(&self, i: usize, j: usize) -> f64 {
        if i >= self.entries.len() {
            return 0.0;
        }
        let read_count = self.entries[i].zread_read_count;
        if read_count == 0 {
            return 0.0;
        }
        self.entries[i]
            .co_resonance
            .iter()
            .find(|(p, _)| *p == j)
            .map(|(_, sum)| sum / read_count as f64)
            .unwrap_or(0.0)
    }

    /// Distributed Hebbian credit assignment.
    ///
    /// For every non-DNA entry whose ZREAD coupling to `context` exceeds
    /// `ZREAD_T_MIN`, nudge its value toward `actual` at rate
    /// `t * (correction_sigma / π)`.  The step is zero when the prediction was
    /// already correct (`correction_sigma ≈ 0`) and maximal when the geodesic
    /// error is π (antipodal).
    ///
    /// Entries outside the neighbourhood (t < ZREAD_T_MIN) are left untouched
    /// — they did not contribute to the prediction and should not receive credit.
    ///
    /// BKT statistics are also updated: each visited entry records the coupling
    /// it had for this write step, matching what `record_zread_contributions`
    /// records for the read path.
    ///
    /// `channel` and `mode` must match the values used in
    /// `zread_at_query_channel_with_mode` for the prediction that produced
    /// `correction_sigma`.
    pub fn distribute_credit(
        &mut self,
        context: &[f64; 4],
        actual: &[f64; 4],
        correction_sigma: f64,
        channel: HopfChannel,
        mode: AddressMode,
    ) {
        let rate_scale = (correction_sigma / std::f64::consts::PI).clamp(0.0, 1.0);
        for entry in &mut self.entries {
            if entry.layer == Layer::Dna {
                continue;
            }
            if !carrier_in_channel(&entry.address.geometry(), channel) {
                continue;
            }
            let gap = address_distance(context, &entry.address.geometry(), mode);
            let t = coupling_from_gap(gap, mode);
            if t < ZREAD_T_MIN {
                continue;
            }
            let rate = t * rate_scale;
            entry.value = slerp(&entry.value, actual, rate);
            // Keep BKT statistics consistent with the read path.
            entry.zread_t_sum += t;
            entry.zread_read_count += 1;
        }
    }

    // ── Critical-period diagnostics ──────────────────────────────────────────

    /// Coverage load of the epigenetic layer under the novelty threshold.
    ///
    /// Each epigenetic entry covers a geodesic ball of radius `novelty_threshold`
    /// under the runtime metric. Because `sigma()` identifies antipodes
    /// (`q ≡ -q`), the effective space is RP³ = S³ / ± with volume π².
    /// The ball volume inherited from S³ is:
    ///   V(r) = 2π(r − sin r · cos r)
    ///
    /// This is a load estimate, not a proof of hole-free coverage: overlapping
    /// balls can make load ≥ 1 while uncovered regions remain. The critical
    /// period closes only when this load is saturated and recent births stop.
    pub fn genome_coverage_load(&self) -> f64 {
        let n = self
            .entries
            .iter()
            .filter(|e| e.layer == Layer::Epigenetic)
            .count();
        if n == 0 {
            return 0.0;
        }
        let r = self.config.novelty_threshold;
        let ball_vol = 2.0 * std::f64::consts::PI * (r - r.sin() * r.cos());
        let rp3_vol = std::f64::consts::PI * std::f64::consts::PI;
        n as f64 * ball_vol / rp3_vol
    }

    /// Fraction of epigenetic entries born within the last `window` ingest
    /// calls, divided by `window`.
    ///
    /// A falling creation rate signals that the genome is saturating its
    /// novelty-threshold coverage — the end of the critical period.
    /// Returns 0.0 when `window == 0`.
    pub fn creation_rate(&self, window: usize) -> f64 {
        if window == 0 {
            return 0.0;
        }
        let threshold = self.ingest_count.saturating_sub(window);
        let recent = self
            .entries
            .iter()
            .filter(|e| e.layer == Layer::Epigenetic && e.birth_cycle > threshold)
            .count();
        recent as f64 / window as f64
    }

    /// Critical period closure criterion.
    ///
    /// The critical period is closed only when the memory load has saturated
    /// the antipodal carrier space *and* no new epigenetic entries were born
    /// in the observation window. Density without zero births is still active
    /// exploration; zero births without density is under-sampling.
    pub fn critical_period_closed(&self, window: usize) -> bool {
        window > 0 && self.genome_coverage_load() >= 1.0 && self.creation_rate(window) == 0.0
    }

    // ── Persistence ───────────────────────────────────────────────────────────

    /// Save the genome to a JSON file.
    ///
    /// The genome holds all learned entries, edges, and BKT statistics.
    /// Cell A, Cell C, and the buffer are ephemeral — they regenerate
    /// from the first few ingest calls after reload. To resume training,
    /// load with `Genome::load_from_file` and pass the result to
    /// `ThreeCell::with_genome`.
    pub fn save_to_file(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let json = serde_json::to_string_pretty(self)?;
        // Write to a sibling temp file, then atomically rename into place.
        // POSIX rename(2) is atomic for same-filesystem paths: the reader
        // always sees either the previous complete genome or the new one —
        // never a partial write. The temp path uses the target path as a
        // prefix so it lands on the same filesystem as the destination.
        let tmp_path = format!("{}.tmp", path);
        std::fs::write(&tmp_path, &json)?;
        std::fs::rename(&tmp_path, path)?;
        Ok(())
    }

    /// Load a genome from a JSON file previously written by `save_to_file`.
    pub fn load_from_file(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let json = std::fs::read_to_string(path)?;
        let genome = serde_json::from_str(&json)?;
        Ok(genome)
    }

    /// Find the entry whose address minimizes `σ(query · star(address))`.
    /// Returns `(index, address_gap)`. `None` if the genome is empty.
    ///
    /// Delegates to [`nearest_with_mode`] using `AddressMode::Full`.
    pub fn nearest(&self, query: &[f64; 4]) -> Option<(usize, f64)> {
        self.nearest_with_mode(query, AddressMode::Full)
    }

    /// Find the entry whose address minimizes distance in the given Hopf channel.
    ///
    /// * `Full`   — full geodesic σ on S³ (same as `nearest`).
    /// * `Base`   — closest S² axis: "what type is this?"
    /// * `Phase`  — closest S¹ angle: "where in the cycle is this?"
    /// * `Scalar` — closest W depth: "how present is this?"
    ///
    /// Returns `(index, gap_in_chosen_channel)`. `None` if the genome is empty.
    pub fn nearest_with_mode(&self, query: &[f64; 4], mode: AddressMode) -> Option<(usize, f64)> {
        let mut best: Option<(usize, f64)> = None;
        for (i, entry) in self.entries.iter().enumerate() {
            let gap = address_distance(query, &entry.address.geometry(), mode);
            match best {
                None => best = Some((i, gap)),
                Some((_, g)) if gap < g => best = Some((i, gap)),
                _ => {}
            }
        }
        best
    }

    /// Find the nearest entry restricted to a specific genome layer.
    /// Uses full geodesic distance. Delegates to `nearest_in_layer_with_mode`.
    pub fn nearest_in_layer(&self, query: &[f64; 4], layer: Layer) -> Option<(usize, f64)> {
        self.nearest_in_layer_with_mode(query, layer, AddressMode::Full)
    }

    /// Find the nearest entry restricted to a layer and address mode.
    ///
    /// Combines layer filtering (DNA vs Epigenetic) with Hopf projection
    /// selection (Full / Base / Phase / Scalar). Use this when symbolic
    /// retrieval needs both: "find the learned orbit position (Epigenetic)
    /// that matches on S² axis (Base)" or similar factorized queries.
    pub fn nearest_in_layer_with_mode(
        &self,
        query: &[f64; 4],
        layer: Layer,
        mode: AddressMode,
    ) -> Option<(usize, f64)> {
        let mut best: Option<(usize, f64)> = None;
        for (i, entry) in self.entries.iter().enumerate() {
            if entry.layer != layer {
                continue;
            }
            let gap = address_distance(query, &entry.address.geometry(), mode);
            match best {
                None => best = Some((i, gap)),
                Some((_, g)) if gap < g => best = Some((i, gap)),
                _ => {}
            }
        }
        best
    }
}
