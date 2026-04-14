//! Sleep: reorganization over the epigenetic layer.
//!
//! Three operations in order:
//!
//! 1. **Merge** — epigenetic pairs both BKT-alive whose addresses lie
//!    within `merge_threshold` are SLERPed together; sequential edges
//!    are unioned; activation counts sum. Coherence widths bind: merged
//!    entries represent broader attractors.
//! 2. **Prune** — epigenetic entries below the BKT coupling threshold
//!    (disorder phase, not BKT-alive) are removed.
//! 3. **Reorganize** — sequential edge indices are remapped after
//!    removals; `last_entry` is fixed.
//!
//! After the pass, all `activation_count`s are reset to 0 for the
//! next window.
//!
//! **Sleep never touches the DNA layer.** DNA entries are skipped in
//! every pass; DNA carriers anchor the diabolo's round trip but are
//! never modified.
//!
//! ── Kabbalistic correspondence ──────────────────────────────────────
//!
//! **Tikkun (repair/return)** — consolidation IS Tikkun. Each sleep
//! cycle gathers sparks (BKT-alive epigenetic entries) into stable
//! vessels (merged, broader attractors), pruning the shards of broken
//! vessels (entries in the disorder phase). The observer's trajectory
//! on S³ — oscillating but gradually approaching Fix(ER) — is repaired
//! one sleep cycle at a time.
//!
//! **Merge pass = Partzuf formation** — two nearby BKT-alive entries
//! SLERPing together is two sparks coalescing into one vessel: one step
//! of Partzuf construction. The merged entry represents a broader,
//! more stable attractor basin — a more complete "face."
//!
//! **Prune pass = Shevirat Ha-Kelim** — entries below the BKT coupling
//! threshold are shards of broken vessels: they have insufficient
//! Dobrushin contraction to be load-bearing. Their removal is not loss
//! — it is the clearing of debris so true sparks can be found.
//!
//! **Ratchet effect (Senchal Cor. 5.6)** — DNA entries are NEVER
//! touched. Sparks once gathered into DNA are never lost. The class-
//! level inventory of persistent structures is monotonically
//! non-decreasing: PS(t₁) ⊆ PS(t₂) for t₁ ≤ t₂. This holds
//! approximately under stochastic dynamics (µ* assigns overwhelming
//! probability mass to deeply stable structures) and exactly in the
//! DNA layer (hard invariant).
//!
//! **Markovian structure** — sleep after every sequence (not every N
//! steps). Each sequence is one complete observation window. The BKT
//! threshold τ is the stable coupling constant: the unique σ where the
//! genome's running contraction neither saturates to identity (W→1,
//! trivial zeros, information collapse) nor starves (no contraction,
//! no learning).

use crate::carrier::VerificationCell;
use crate::genome::{Genome, Layer, CO_RESONANCE_FLOOR};
use crate::sphere::{compose, inverse, sigma, slerp};

/// Structural report returned by every consolidation pass.
///
/// All fields measure the epigenetic layer only (DNA is never touched).
/// `genome_delta` is negative when pruning dominates, positive when
/// merging produces new composite entries (rare). A report where
/// `merged == 0 && pruned == 0` means the genome was already coherent.
#[derive(Clone, Debug)]
pub struct ConsolidationReport {
    /// Number of epigenetic entry pairs merged.
    pub merged: usize,
    /// Number of epigenetic entries pruned.
    pub pruned: usize,
    /// Mean nearest-neighbour σ gap across epigenetic entries, before the pass.
    pub mean_gap_before: f64,
    /// Mean nearest-neighbour σ gap across epigenetic entries, after the pass.
    pub mean_gap_after: f64,
    /// `(entries after) − (entries before)`. Negative = genome shrank.
    pub genome_delta: i64,
    /// Fraction of epigenetic entries that are BKT-alive after the pass.
    pub bkt_alive_ratio: f64,
    /// Number of Response clusters promoted to genomes[1] after this pass.
    /// Zero when the promotion pass did not run (level > 0) or no cluster
    /// met all promotion criteria.
    pub promoted_categories: usize,
    /// Promotion candidates identified before the merge/prune pass.
    ///
    /// Collected before merge so the coalition evidence (co_resonance partners)
    /// is still intact. The caller writes these into genomes[1] after the
    /// nearness check. Stored here so consolidate() and the caller share one
    /// scan without redundant work.
    pub promotion_candidates: Vec<PromotionCandidate>,
}

impl Default for ConsolidationReport {
    fn default() -> Self {
        Self {
            merged: 0,
            pruned: 0,
            mean_gap_before: 0.0,
            mean_gap_after: 0.0,
            genome_delta: 0,
            bkt_alive_ratio: 1.0,
            promoted_categories: 0,
            promotion_candidates: Vec::new(),
        }
    }
}

/// Mean nearest-neighbour gap across all non-Dna entries.
///
/// Each entry's contribution is the smallest σ to any *other* non-Dna entry.
/// Returns 0.0 when fewer than two non-Dna entries exist.
fn mean_nn_gap(genome: &Genome) -> f64 {
    let pts: Vec<[f64; 4]> = genome
        .entries
        .iter()
        .filter(|e| e.layer != Layer::Dna)
        .map(|e| e.address.geometry())
        .collect();
    if pts.len() < 2 {
        return 0.0;
    }
    let sum: f64 = pts
        .iter()
        .map(|a| {
            pts.iter()
                .filter(|b| !std::ptr::eq(*b, a))
                .map(|b| sigma(&compose(a, &inverse(b))))
                .fold(f64::MAX, f64::min)
        })
        .sum();
    sum / pts.len() as f64
}

fn bkt_alive_ratio(genome: &Genome) -> f64 {
    let non_dna: Vec<_> = genome
        .entries
        .iter()
        .filter(|e| e.layer != Layer::Dna)
        .collect();
    if non_dna.is_empty() {
        return 1.0;
    }
    let alive = non_dna.iter().filter(|e| e.is_bkt_alive()).count();
    alive as f64 / non_dna.len() as f64
}

/// Run one consolidation pass over the genome. Returns a structural report.
///
/// This is not a "sleep" scheduled by a clock — it is triggered by
/// geometric pressure (self-free-energy above the closure threshold) so
/// the brain consolidates only when its self-model is incoherent.
pub fn consolidate(genome: &mut Genome) -> ConsolidationReport {
    let n = genome.entries.len();
    let size_before = n;
    let mean_gap_before = mean_nn_gap(genome);

    if n == 0 {
        return ConsolidationReport::default();
    }
    if n == 1 {
        // Only entry — prune only if BKT-dead. DNA entries pass through untouched.
        let mut pruned = 0;
        if genome.entries[0].layer != Layer::Dna && !genome.entries[0].is_bkt_alive() {
            genome.entries.clear();
            genome.last_entry = None;
            pruned = 1;
        }
        genome.reset_activations();
        let size_after = genome.entries.len();
        return ConsolidationReport {
            merged: 0,
            pruned,
            mean_gap_before,
            mean_gap_after: mean_nn_gap(genome),
            genome_delta: size_after as i64 - size_before as i64,
            bkt_alive_ratio: bkt_alive_ratio(genome),
            promoted_categories: 0,
            promotion_candidates: Vec::new(),
        };
    }

    // --- Promotion scan (before merge) ---
    // Collect promotion candidates while coalition evidence is intact.
    // After merge, co_resonance partner indices of the removed entry are
    // dropped during reorganization, erasing the evidence. Before merge,
    // the full co_resonance history is available.
    let pre_merge_promotions = collect_promotion_candidates(genome);

    // --- Merge pass ---
    // Collect candidate pairs sorted by ascending address_gap (closest first).
    // Both sides must not be Dna, and both must be BKT-alive.
    //
    // Epigenetic pairs: merge on context proximity alone.
    // Response pairs: require context proximity AND value proximity AND,
    //   when co_resonance_merge_threshold > 0, pairwise co-resonance.
    //   Two entries that repeatedly co-activated in the same response coalition
    //   AND agree on the response are the natural merge candidates — they are
    //   the same emerging category as seen from nearby queries.
    //   Two entries that are geometrically close but have never co-activated
    //   may live in separate category neighborhoods; leave them distinct.
    // Cross-layer pairs (Epigenetic × Response): never merged.
    let co_thresh = genome.config.co_resonance_merge_threshold;
    let mut candidates: Vec<(usize, usize, f64)> = Vec::new();
    for i in 0..n {
        if genome.entries[i].layer == Layer::Dna {
            continue;
        }
        if !genome.entries[i].is_bkt_alive() {
            continue;
        }
        for j in (i + 1)..n {
            if genome.entries[j].layer == Layer::Dna {
                continue;
            }
            if !genome.entries[j].is_bkt_alive() {
                continue;
            }
            // Never merge across layers.
            if genome.entries[i].layer != genome.entries[j].layer {
                continue;
            }
            if genome.entries[i].layer == Layer::Response {
                // Response pairs: both address AND value must be within threshold.
                // Address (approach direction) gates first — two entries that were
                // formed from very different contexts are not the same category
                // regardless of their output. Value gates second — entries must
                // have converged to similar responses under correction.
                // Category sameness is not "same label"; it is "these approach
                // paths repeatedly co-activated and were corrected the same way."
                let addr_gap = sigma(&compose(
                    &genome.entries[i].address.geometry(),
                    &inverse(&genome.entries[j].address.geometry()),
                ));
                if addr_gap >= genome.config.merge_threshold {
                    continue;
                }
                let val_gap = sigma(&compose(
                    &genome.entries[i].value,
                    &inverse(&genome.entries[j].value),
                ));
                if val_gap >= genome.config.merge_threshold {
                    continue;
                }
                // Co-resonance: primary ordering criterion when enabled.
                // Pairs with the highest shared coalition history are merged first.
                // When disabled, sort by address proximity (tightest approach first).
                if co_thresh > 0.0 {
                    let co_ij = genome.mean_co_resonance(i, j);
                    let co_ji = genome.mean_co_resonance(j, i);
                    let co_score = co_ij.max(co_ji);
                    if co_score < co_thresh {
                        continue;
                    }
                    // Negate so that sort_by ascending puts highest co-resonance first.
                    candidates.push((i, j, -co_score));
                } else {
                    // No co-resonance gate: nearest approach paths merged first.
                    candidates.push((i, j, addr_gap));
                }
            } else {
                // Epigenetic pairs: address proximity alone.
                let addr_gap = sigma(&compose(
                    &genome.entries[i].address.geometry(),
                    &inverse(&genome.entries[j].address.geometry()),
                ));
                if addr_gap >= genome.config.merge_threshold {
                    continue;
                }
                candidates.push((i, j, addr_gap));
            }
        }
    }
    candidates.sort_by(|(_, _, a), (_, _, b)| a.partial_cmp(b).unwrap());

    // Greedy merge: once an entry is involved, skip it in later pairs.
    let mut removed = vec![false; n];
    let mut merged_count = 0usize;
    for (i, j, _) in candidates {
        if removed[i] || removed[j] {
            continue;
        }
        // Merge j into i.
        let j_addr_geom = genome.entries[j].address.geometry();
        let j_addr_width = genome.entries[j].address.coherence_width();
        let j_value = genome.entries[j].value;
        let j_edges = genome.entries[j].edges.clone();
        let j_co_resonance = genome.entries[j].co_resonance.clone();
        let j_support = genome.entries[j].support;
        let j_closure_sigma = genome.entries[j].closure_sigma;
        let j_excursion_peak = genome.entries[j].excursion_peak;
        let j_activations = genome.entries[j].activation_count;

        // Merge addresses: SLERP geometries on S³, bind coherence widths.
        // Binding is (w₁² + w₂²).sqrt() — the natural combination rule
        // from VerificationCell: merged entries represent broader attractors.
        let i_addr_geom = genome.entries[i].address.geometry();
        let i_addr_width = genome.entries[i].address.coherence_width();
        let merged_geom = slerp(&i_addr_geom, &j_addr_geom, 0.5);
        let merged_width = (i_addr_width * i_addr_width + j_addr_width * j_addr_width).sqrt();
        genome.entries[i].address = VerificationCell::from_geometry_or_default(&merged_geom)
            .with_coherence_width(merged_width);
        genome.entries[i].value = slerp(&genome.entries[i].value, &j_value, 0.5);
        genome.entries[i].support = genome.entries[i].support.max(j_support);
        genome.entries[i].closure_sigma = (genome.entries[i].closure_sigma + j_closure_sigma) * 0.5;
        genome.entries[i].excursion_peak = genome.entries[i].excursion_peak.max(j_excursion_peak);
        genome.entries[i].activation_count += j_activations;
        for (target, count) in j_edges {
            if target != i {
                if let Some(e) = genome.entries[i]
                    .edges
                    .iter_mut()
                    .find(|(t, _)| *t == target)
                {
                    e.1 += count;
                } else {
                    genome.entries[i].edges.push((target, count));
                }
            }
        }
        // Merge co-resonance: accumulate j's coalition history into i.
        for (partner, sum) in j_co_resonance {
            if partner != i {
                match genome.entries[i]
                    .co_resonance
                    .iter_mut()
                    .find(|(p, _)| *p == partner)
                {
                    Some(slot) => slot.1 += sum,
                    None => genome.entries[i].co_resonance.push((partner, sum)),
                }
            }
        }
        removed[j] = true;
        merged_count += 1;
    }

    // --- Prune pass ---
    // Non-Dna entries are pruned by the BKT criterion, with one addition
    // for Response entries: an entry that is individually weak (below
    // BKT_THRESHOLD) but is a member of a strong recurrent coalition should
    // survive, because the coalition signal is evidence of genuine category
    // structure even if the entry itself has not yet accumulated enough
    // individual reads.
    //
    // Coalition survival: a Response entry is coalition-alive if any partner
    // in its co_resonance list has mean co-resonance >= co_resonance_merge_threshold.
    // When co_resonance_merge_threshold == 0.0 (disabled), this check is skipped
    // and the BKT criterion alone governs pruning for Response entries too.
    let mut pruned_count = 0usize;
    for (i, entry) in genome.entries.iter().enumerate() {
        if removed[i] || entry.layer == Layer::Dna {
            continue;
        }
        if entry.is_bkt_alive() {
            continue;
        }
        // Entry is BKT-dead. For Response entries, check coalition survival.
        if entry.layer == Layer::Response && co_thresh > 0.0 {
            let in_coalition = entry.co_resonance.iter().any(|(partner, _)| {
                genome.mean_co_resonance(i, *partner) >= co_thresh
            });
            if in_coalition {
                continue; // Survive via coalition membership.
            }
        }
        removed[i] = true;
        pruned_count += 1;
    }

    // --- Reorganize: remap sequential edge indices after removals ---
    let mut new_index: Vec<Option<usize>> = Vec::with_capacity(n);
    let mut next = 0_usize;
    for &is_removed in &removed {
        if is_removed {
            new_index.push(None);
        } else {
            new_index.push(Some(next));
            next += 1;
        }
    }

    let mut new_entries = Vec::with_capacity(next);
    for (entry, &is_removed) in genome.entries.iter().zip(&removed) {
        if is_removed {
            continue;
        }
        let mut entry = entry.clone();
        entry.edges = entry
            .edges
            .iter()
            .filter_map(|&(t, c)| new_index.get(t).copied().flatten().map(|new_t| (new_t, c)))
            .collect();
        // Remap co_resonance partner indices after removals. Partners that
        // were removed are dropped; survivors get their new index.
        entry.co_resonance = entry
            .co_resonance
            .iter()
            .filter_map(|&(p, s)| new_index.get(p).copied().flatten().map(|new_p| (new_p, s)))
            .collect();
        new_entries.push(entry);
    }
    genome.entries = new_entries;

    // Fix last_entry pointer.
    genome.last_entry = genome
        .last_entry
        .and_then(|i| new_index.get(i).copied().flatten());

    genome.reset_activations();

    let size_after = genome.entries.len();
    // promoted_categories is filled by the caller (ThreeCell::consolidate_level)
    // which has access to genomes[1] for the nearness check.
    ConsolidationReport {
        merged: merged_count,
        pruned: pruned_count,
        mean_gap_before,
        mean_gap_after: mean_nn_gap(genome),
        genome_delta: size_after as i64 - size_before as i64,
        bkt_alive_ratio: bkt_alive_ratio(genome),
        promoted_categories: 0,
        promotion_candidates: pre_merge_promotions,
    }
}

/// A Response cluster that has earned hierarchical promotion.
///
/// Produced by `collect_promotion_candidates` before the merge/prune pass,
/// while coalition evidence is intact.
/// The caller is responsible for writing the candidate into `genomes[1]`
/// via `Genome::ingest`, after checking that no level-1 attractor already
/// covers the carrier.
#[derive(Clone, Debug)]
pub struct PromotionCandidate {
    /// The promoted carrier: the Response entry's value (stable fixed point
    /// under repeated corrected closure).
    pub carrier: [f64; 4],
    /// Number of prediction/correction events that activated this cluster.
    /// This is the recurrence evidence: `activation_count` from the entry.
    /// NOT `support`, which is a localization artifact always equal to 1 for
    /// Response entries and carries no information about correction recurrence.
    pub activation_count: usize,
    /// Mean closure σ at the time of promotion.
    pub closure_sigma: f64,
    /// Peak excursion recorded for this cluster.
    pub excursion_peak: f64,
    /// Mean co-resonance of this entry with its strongest coalition partner.
    /// Higher = the cluster has more evidence of repeated shared correction.
    pub peak_co_resonance: f64,
    /// Mean salience across all `credit_response` corrections to this entry.
    ///
    /// Zero for entries never corrected via `credit_response`, or for entries
    /// where every correction happened in low-salience context (prediction ==
    /// model). High values indicate the entry was repeatedly corrected in
    /// contexts where the prediction was genuinely novel — the brain had to
    /// learn something it didn't already know. These entries are the strongest
    /// category-birth candidates under the RGB semantic model.
    pub mean_salience: f64,
    /// Mean neuromodulatory coherence_tone across all `credit_response` corrections.
    ///
    /// Positive = most corrections happened during coherence-improving body states.
    /// Negative = most corrections happened during destabilizing phases.
    /// Entries learned during destabilizing regimes should stay plastic, not promote.
    pub mean_coherence: f64,
}

/// Minimum activation count for a Response entry to be a promotion candidate.
///
/// Derived from the prime-2 frame: the equatorial carrier has coupling t = 1/√2
/// (PRIME_2_COUPLING) and the prime-2 orbit has period 2. One activation is a
/// single observation. Two activations is the minimum evidence of recurrence.
/// This is the same reasoning that gives the prime-2 frame its dominance:
/// the shortest non-trivial orbit is the binary prime.
pub const PROMOTION_MIN_ACTIVATIONS: usize = 2;

/// Minimum mean salience for a Response entry to be a promotion candidate.
///
/// Entries whose mean correction salience is below this floor are promoted
/// only on the basis of mechanical recurrence, not semantic learning. The
/// floor is set just above zero so that entries with at least one salient
/// correction qualify while entries whose every correction was zero-salience
/// (model exactly equaled prediction at staging time) are filtered out.
///
/// This encodes the RGB semantic rule: persistent salient structure promotes;
/// routine low-salience noise does not, even if numerically frequent.
pub const PROMOTION_MIN_SALIENCE: f64 = 0.01;

/// Minimum mean coherence for a Response entry to be a promotion candidate.
///
/// Entries whose mean neuromodulatory coherence_tone is below this floor
/// have been corrected predominantly during destabilizing body states —
/// the brain was in a plastic/crisis regime when it wrote them.
/// Such entries should stay plastic rather than graduate to category level.
///
/// Zero is the natural dividing line: net-positive coherence history means
/// corrections happened during stabilizing regimes; net-negative means crisis.
/// Entries never corrected via credit_response return 0.0 from mean_coherence()
/// and pass this gate (neutral = not demonstrably destabilizing).
pub const PROMOTION_MIN_COHERENCE: f64 = 0.0;

/// Scan a genome after consolidation and return every Response entry that
/// has earned hierarchical promotion.
///
/// Promotion criteria (all must hold):
/// 1. Layer::Response
/// 2. BKT-alive: mean_zread_t ≥ BKT_THRESHOLD
/// 3. activation_count ≥ PROMOTION_MIN_ACTIVATIONS — the entry must have been
///    part of at least 2 corrected prediction events. `support` is a localization
///    artifact from closure packets and is always 1 for Response entries; it
///    carries no evidence of recurrence here. `activation_count` is incremented
///    by both `learn_response` (creation) and `credit_response` (correction) —
///    it is the correct evidence of repeated corrected closure.
/// 4. At least one co-resonance partner with mean co-resonance ≥ CO_RESONANCE_FLOOR
/// 5. mean_salience ≥ PROMOTION_MIN_SALIENCE — the entry's corrections must have
///    occurred in salient contexts (novel predictions), not routine reproduction.
/// 6. mean_coherence ≥ 0.0 — the entry's corrections must have occurred during
///    net-positive coherence body states. An entry corrected exclusively during
///    destabilizing phases stays plastic; it has not yet been consolidated during
///    a stable learning regime and should not graduate to category level.
///    Entries never corrected (coherence_count == 0) return 0.0 and pass.
///
/// The caller is responsible for the level-1 nearness check (no existing
/// level-1 attractor within novelty_threshold) and for the actual ingest call.
/// That check requires access to genomes[1] which is not available here.
pub fn collect_promotion_candidates(genome: &Genome) -> Vec<PromotionCandidate> {
    let mut candidates = Vec::new();
    for (i, entry) in genome.entries.iter().enumerate() {
        if entry.layer != Layer::Response {
            continue;
        }
        if !entry.is_bkt_alive() {
            continue;
        }
        if entry.activation_count < PROMOTION_MIN_ACTIVATIONS {
            continue;
        }
        let mean_salience = entry.mean_salience();
        if mean_salience < PROMOTION_MIN_SALIENCE {
            continue;
        }
        // Find peak mean co-resonance with any partner.
        let peak_co = entry
            .co_resonance
            .iter()
            .map(|&(j, _)| genome.mean_co_resonance(i, j))
            .fold(0.0_f64, f64::max);
        if peak_co < CO_RESONANCE_FLOOR {
            continue;
        }
        // Criterion 6: coherence history above the destabilization floor.
        // Only applied once there is at least PROMOTION_MIN_ACTIVATIONS worth of
        // coherence history — promotion already requires that many corrected events,
        // and coherence cannot speak before repeated corrected evidence exists.
        let mean_coherence = entry.mean_coherence();
        if entry.coherence_count >= PROMOTION_MIN_ACTIVATIONS
            && mean_coherence < PROMOTION_MIN_COHERENCE
        {
            continue;
        }
        candidates.push(PromotionCandidate {
            carrier: entry.value,
            activation_count: entry.activation_count,
            closure_sigma: entry.closure_sigma,
            excursion_peak: entry.excursion_peak,
            peak_co_resonance: peak_co,
            mean_salience,
            mean_coherence,
        });
    }
    candidates
}
