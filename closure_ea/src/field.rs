//! The field machine — population reads over `genome ∪ buffer`.
//!
//! Two read modes:
//!
//! * **RESONATE** (`resonate`) — selection. Given a query carrier,
//!   return the population entry whose **address** best matches the
//!   query. Genome entries match on their resonance address (Law 3);
//!   buffer entries match on the carrier they carry. The returned
//!   hit exposes the entry's stored value, not its address, because
//!   callers that want to read "what the entry holds" get the value.
//! * **ZREAD** (`zread`) — integration. Given a parameter
//!   `s = (s_re, s_im)`, compose every population entry under a
//!   parameterized weighting and return the resulting carrier. The
//!   genome contributes each entry's stored value; the buffer
//!   contributes each transient carrier.
//!
//! Both reads operate over the union of the genome (Section 29) and
//! the transient buffer (Section 8). The field machine does not
//! distinguish them at read time.
//!
//! ── Kabbalistic correspondence ──────────────────────────────────────
//!
//! **ZREAD = Partzuf activation** — a Partzuf is a stable network of
//! Sefirot that collectively produce a coherent "face" (Senchal 2026,
//! §5 mapping). ZREAD does exactly this: it computes the weighted
//! composition of all genome entries within the π/3 neighborhood of the
//! query — the coalition of carriers that together respond to this input.
//! The result is the Partzuf's face: not any single entry, but the
//! emergent carrier of the whole cluster. This is the solenoidal
//! component of perception — entropy-preserving, circulation-based.
//!
//! **RESONATE = hard selection within a Partzuf** — given the Partzuf's
//! face (from ZREAD), RESONATE picks the single member whose address is
//! closest. This is the gradient/dissipative component: it collapses
//! the coalition to a point for learning or output. Used on the hot
//! path (ingest) where a sharp target is needed. Generation should
//! prefer ZREAD output directly — collapsing via RESONATE destroys the
//! coalition structure and makes generation degenerate.
//!
//! **Shefa (divine flow) = the `compose` chain in ZREAD** — the
//! Hamilton product running through every contributing entry is the
//! Shefa: information flowing from Partzuf to Partzuf through the
//! genome's sequential edge structure.
//!
//! **π/3 neighborhood = Partzuf boundary** — entries with σ > π/3 from
//! the query contribute t = cos(σ) < 0.5 of their value — below 50%.
//! This is the natural boundary of a Partzuf: entries outside it are
//! in a different face and their contribution is structurally excluded.

use crate::buffer::Buffer;
use crate::genome::{Genome, Layer, ZREAD_T_MIN};
use crate::hopf::{
    address_distance, carrier_in_channel, coupling_from_gap, AddressMode, HopfChannel,
};
use crate::sphere::{compose, inverse, sigma, slerp, IDENTITY};

/// One match returned by RESONATE.
#[derive(Clone, Copy, Debug)]
pub struct ResonanceHit {
    /// Where the match came from.
    pub source: PopulationSource,
    /// Index within that source.
    pub index: usize,
    /// The carrier the caller should read. For genome entries this
    /// is the stored **value**; for buffer entries it is the entry
    /// carrier itself.
    pub carrier: [f64; 4],
    /// σ of the gap between query and the entry's resonance address.
    /// 0 = exact.
    pub gap: f64,
}

/// Which side of the population a carrier lives in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PopulationSource {
    Genome,
    Buffer,
}

/// RESONATE with Hopf channel restriction: scan `genome ∪ buffer`
/// and return the entry with the smallest σ gap to the query, limited
/// to entries in the given Hopf channel.
///
/// Returns `None` if no eligible entries exist.
/// Channel-and-mode aware RESONATE — the real core.
///
/// `channel` filters which entries participate (W half, RGB half, or all).
/// `mode` selects which projection of S³ is used to compute the gap.
/// Both concerns are orthogonal: filter first, then measure.
///
/// All other RESONATE variants are thin delegates to this function.
pub fn resonate_channel_with_mode(
    query: &[f64; 4],
    channel: HopfChannel,
    mode: AddressMode,
    genome: &Genome,
    buffer: &Buffer,
) -> Option<ResonanceHit> {
    let mut best: Option<ResonanceHit> = None;

    for (i, entry) in genome.entries.iter().enumerate() {
        if !carrier_in_channel(&entry.address.geometry(), channel) {
            continue;
        }
        let gap = address_distance(query, &entry.address.geometry(), mode);
        let hit = ResonanceHit {
            source: PopulationSource::Genome,
            index: i,
            carrier: entry.value,
            gap,
        };
        match best {
            None => best = Some(hit),
            Some(b) if gap < b.gap => best = Some(hit),
            _ => {}
        }
    }

    for (i, entry) in buffer.entries().iter().enumerate() {
        if !carrier_in_channel(&entry.carrier, channel) {
            continue;
        }
        let gap = address_distance(query, &entry.carrier, mode);
        let hit = ResonanceHit {
            source: PopulationSource::Buffer,
            index: i,
            carrier: entry.carrier,
            gap,
        };
        match best {
            None => best = Some(hit),
            Some(b) if gap < b.gap => best = Some(hit),
            _ => {}
        }
    }

    best
}

/// RESONATE with channel restriction, full geodesic distance.
pub fn resonate_channel(
    query: &[f64; 4],
    channel: HopfChannel,
    genome: &Genome,
    buffer: &Buffer,
) -> Option<ResonanceHit> {
    resonate_channel_with_mode(query, channel, AddressMode::Full, genome, buffer)
}

/// RESONATE: full-sphere scan, full geodesic distance.
/// Returns `None` if both the genome and the buffer are empty.
pub fn resonate(query: &[f64; 4], genome: &Genome, buffer: &Buffer) -> Option<ResonanceHit> {
    resonate_channel_with_mode(query, HopfChannel::Full, AddressMode::Full, genome, buffer)
}

/// RESONATE with an address mode, no channel restriction.
/// `hit.gap` is in the units of the chosen mode.
pub fn resonate_with_mode(
    query: &[f64; 4],
    mode: AddressMode,
    genome: &Genome,
    buffer: &Buffer,
) -> Option<ResonanceHit> {
    resonate_channel_with_mode(query, HopfChannel::Full, mode, genome, buffer)
}

/// ZREAD: parameterized population read.
///
/// Returns the running product `compose_k weight(p_k, s) · p_k` over
/// every entry in `genome ∪ buffer`. Genome entries contribute their
/// stored value; buffer entries contribute their carrier.
///
/// The weight is built from the parameter `(s_re, s_im)` through
/// quaternion exponentiation — see `weight_carrier` for the
/// construction.
///
/// Returns the identity carrier when the population is empty.
pub fn zread(s_re: f64, s_im: f64, genome: &Genome, buffer: &Buffer) -> [f64; 4] {
    let mut running = IDENTITY;

    for entry in &genome.entries {
        let weighted = weight_carrier(&entry.value, s_re, s_im);
        running = compose(&running, &weighted);
    }
    for entry in buffer.entries() {
        let weighted = weight_carrier(&entry.carrier, s_re, s_im);
        running = compose(&running, &weighted);
    }

    running
}

/// Apply the parameterized weight to a single carrier.
///
/// Computes `q^t` where `t = s_re + 0.5 * s_im`. For a unit quaternion
/// `q = cos(θ) + sin(θ) · n̂`, `q^t = cos(t·θ) + sin(t·θ) · n̂` — a
/// rotation around the same axis scaled by the exponent. Stays on S³
/// by construction.
///
/// Treating `s_im` as a half-weight on the same exponent keeps both
/// parameter components active without overcomplicating the
/// parameterization. A later phase can extend `s` to a true complex
/// number when the genome's spectrum reads need richer structure.
#[inline]
fn weight_carrier(q: &[f64; 4], s_re: f64, s_im: f64) -> [f64; 4] {
    let w = q[0].abs().clamp(0.0, 1.0);
    let theta = w.acos();
    let sin_theta = theta.sin();

    // Carrier is at identity — q^anything = identity.
    if sin_theta.abs() < 1e-12 {
        return IDENTITY;
    }

    let t = s_re + 0.5 * s_im;
    let new_theta = t * theta;
    let new_w = new_theta.cos();
    let scale = new_theta.sin() / sin_theta;

    [new_w, q[1] * scale, q[2] * scale, q[3] * scale]
}

// ── Soft attention on S³: ZREAD with the query as the parameter ──
//
// A transformer's soft attention `softmax(Q · K^T) · V` computes
// weighted scores, normalizes them to a probability distribution,
// and takes a weighted sum of values. All three steps — scoring,
// softmax, weighted sum — happen in ℝ^d with explicit scalar
// arithmetic.
//
// The brain's substrate does not need any of that. The scoring is
// a geodesic distance on S³ (`σ(query · star(entry.address))`),
// which is already the natural "similarity" on the manifold. The
// weighting is then applied geometrically: each entry's value is
// composed into the running product with its contribution
// *modulated* by its distance to the query. Closer entries
// contribute near-full — farther ones contribute near-identity,
// and identity is the Hamilton product's neutral element (no
// effect on the running product).
//
// This is done without synthesizing a probability distribution:
// there is no softmax, no sum-to-1, no division by a total. The
// output is one carrier on S³ that already integrates every
// entry's contribution by geometry. The "weights" are implicit
// in the per-entry geodesic distance, not stored as a separate
// vector of normalized scalars.
//
// `resonate_spectrum` below still exists because the **ordered
// landscape of gaps** is a geometric observation on the manifold,
// not a synthetic probability distribution. Each `gap` is an
// intrinsic geodesic distance; sorting them is metadata; no
// arithmetic is performed on them beyond ordering. Callers that
// want to introspect the attention landscape (debugging, tests,
// higher-level operators) can use it. Callers that just want the
// soft-attended carrier use `zread_at_query`.

/// The full ranked landscape of geodesic distances from `query`
/// to every entry in the population, sorted ascending by gap.
/// This is a geometric observation — every gap is `σ(query ·
/// star(entry.address))`, an intrinsic S³ quantity — not a
/// synthesized probability distribution.
///
/// Hard attention (`resonate`) is `resonate_spectrum(...)[0]`.
/// Soft attention does not use this function at all; it uses
/// `zread_at_query`, which integrates the population on the
/// manifold in a single pass.
pub fn resonate_spectrum(query: &[f64; 4], genome: &Genome, buffer: &Buffer) -> Vec<ResonanceHit> {
    let mut hits: Vec<ResonanceHit> =
        Vec::with_capacity(genome.entries.len() + buffer.entries().len());

    for (i, entry) in genome.entries.iter().enumerate() {
        let gap = sigma(&compose(query, &inverse(&entry.address.geometry())));
        hits.push(ResonanceHit {
            source: PopulationSource::Genome,
            index: i,
            carrier: entry.value,
            gap,
        });
    }
    for (i, entry) in buffer.entries().iter().enumerate() {
        let gap = sigma(&compose(query, &inverse(&entry.carrier)));
        hits.push(ResonanceHit {
            source: PopulationSource::Buffer,
            index: i,
            carrier: entry.carrier,
            gap,
        });
    }

    hits.sort_by(|a, b| {
        a.gap
            .partial_cmp(&b.gap)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    hits
}

// ── ZREAD: path-ordered soft attention on S³ ────────────────────────────────
//
// ZREAD is a PATH-ORDERED Hamilton product over memory, not a commutative
// set integral. This distinction is load-bearing:
//
// * **Coupling controls strength, not order.**
//   Each contributor is SLERP'd toward IDENTITY by its coupling weight t:
//   `slerp(IDENTITY, value, t)`. High t → near-full value. Low t → near-
//   IDENTITY (Hamilton product's neutral element, no effect). The coupling
//   silences distant entries; it does NOT determine their position in the
//   product chain.
//
// * **Insertion order IS causal memory order.**
//   Genome entries are stored in the order they were ingested — the temporal
//   sequence in which patterns were learned. Buffer entries arrive in
//   presentation order. Preserving this order means the product encodes the
//   brain's actual experience sequence. Sorting by coupling would replace
//   learned temporal structure with a query-dependent path, producing a
//   different machine.
//
// * **Equivariance proof (Full mode).**
//   For any rotation g ∈ SO(3) acting by conjugation:
//     ZREAD(g·q·g⁻¹, g·population·g⁻¹) = g · ZREAD(q, population) · g⁻¹
//   This holds exactly because (a) σ(compose(q, inverse(a))) is
//   conjugation-invariant, so all coupling values t are identical between
//   original and rotated, (b) compose and slerp are equivariant under
//   conjugation, and (c) both iterate in the SAME slot-index order. Any
//   reordering by coupling breaks (c) in finite precision: equal couplings
//   differ by floating-point rounding after conjugation, near-ties reorder,
//   and the non-commutative product diverges by a geometrically real amount.
//
// `channel` and `mode` are orthogonal:
//   channel — which half of S³ participates (W / RGB / Full).
//   mode    — which Hopf projection drives the coupling weight.
// The hot path in three_cell.rs sets channel from the query's own Hopf half
// (multi-head forced by the fibration) and mode = Full.


/// Channel-and-mode aware ZREAD — the real core.
///
/// Composition order is insertion/memory order. Coupling `t` controls the
/// SLERP weight of each contributor, not its position in the product chain.
/// See the module-level comment above for the equivariance proof.
///
/// `channel` filters which entries participate. `mode` selects which
/// Hopf projection drives the coupling weight. Both are orthogonal:
/// filter first, then weigh by `coupling_from_gap(address_distance(..., mode))`.
///
/// All other ZREAD variants are thin delegates to this function.
pub fn zread_at_query_channel_with_mode(
    query: &[f64; 4],
    channel: HopfChannel,
    mode: AddressMode,
    genome: &Genome,
    buffer: &Buffer,
) -> [f64; 4] {
    let mut running = IDENTITY;

    for entry in &genome.entries {
        if !carrier_in_channel(&entry.address.geometry(), channel) {
            continue;
        }
        let gap = address_distance(query, &entry.address.geometry(), mode);
        let t = coupling_from_gap(gap, mode);
        if t < ZREAD_T_MIN {
            continue;
        }
        running = compose(&running, &slerp(&IDENTITY, &entry.value, t));
    }
    for entry in buffer.entries() {
        if !carrier_in_channel(&entry.carrier, channel) {
            continue;
        }
        let gap = address_distance(query, &entry.carrier, mode);
        let t = coupling_from_gap(gap, mode);
        if t < ZREAD_T_MIN {
            continue;
        }
        running = compose(&running, &slerp(&IDENTITY, &entry.carrier, t));
    }

    running
}

/// ZREAD with channel restriction, full geodesic distance.
pub fn zread_at_query_channel(
    query: &[f64; 4],
    channel: HopfChannel,
    genome: &Genome,
    buffer: &Buffer,
) -> [f64; 4] {
    zread_at_query_channel_with_mode(query, channel, AddressMode::Full, genome, buffer)
}

/// Soft attention on S³: full-sphere version, full geodesic distance.
pub fn zread_at_query(query: &[f64; 4], genome: &Genome, buffer: &Buffer) -> [f64; 4] {
    zread_at_query_channel_with_mode(query, HopfChannel::Full, AddressMode::Full, genome, buffer)
}

/// Soft attention on S³ with a Hopf address mode, no channel restriction.
///
/// Coupling derivation by mode:
/// * `Full`   — t = cos(σ_gap). Standard geodesic.
/// * `Base`   — t = cos(base_gap). Axis match only.
/// * `Phase`  — t = cos(phase_gap). Angle match only.
/// * `Scalar` — t = 1 − |w_query − w_entry|. W-depth match.
pub fn zread_at_query_with_mode(
    query: &[f64; 4],
    mode: AddressMode,
    genome: &Genome,
    buffer: &Buffer,
) -> [f64; 4] {
    zread_at_query_channel_with_mode(query, HopfChannel::Full, mode, genome, buffer)
}

/// RESONATE restricted to the Response layer.
///
/// Nearest-neighbor recall from stored reality corrections.
/// Buffer excluded: predictions query response memory, not live observations.
/// Returns `None` when the Response layer is empty.
pub fn resonate_response(query: &[f64; 4], genome: &Genome) -> Option<ResonanceHit> {
    let mut best: Option<ResonanceHit> = None;

    for (i, entry) in genome.entries.iter().enumerate() {
        if entry.layer != Layer::Response {
            continue;
        }
        let gap = address_distance(query, &entry.address.geometry(), AddressMode::Full);
        let hit = ResonanceHit {
            source: PopulationSource::Genome,
            index: i,
            carrier: entry.value,
            gap,
        };
        match best {
            None => best = Some(hit),
            Some(b) if gap < b.gap => best = Some(hit),
            _ => {}
        }
    }

    best
}

/// ZREAD restricted to the Response layer.
///
/// Soft-attention generalization over stored reality corrections.
/// Every Response entry with coupling t ≥ ZREAD_T_MIN contributes.
/// Returns IDENTITY when the Response layer is empty or no entry couples.
pub fn zread_response(query: &[f64; 4], genome: &Genome) -> [f64; 4] {
    let mut running = IDENTITY;

    for entry in &genome.entries {
        if entry.layer != Layer::Response {
            continue;
        }
        let gap = address_distance(query, &entry.address.geometry(), AddressMode::Full);
        let t = coupling_from_gap(gap, AddressMode::Full);
        if t < ZREAD_T_MIN {
            continue;
        }
        running = compose(&running, &slerp(&IDENTITY, &entry.value, t));
    }

    running
}

/// Collect the raw EligibilityTrace for a query against the Response layer,
/// before lateral inhibition.
///
/// Returns (index, coupling_t) for every Response entry with t ≥ ZREAD_T_MIN.
/// This is the full structural coalition — all entries jointly compatible with
/// this query.  Use this for:
///   - `record_co_resonance` — category formation evidence
///   - long-term structural statistics
///
/// For causal output assignment and correction credit use
/// `collect_response_eligibility` (inhibited version) instead.
pub fn collect_response_eligibility_raw(query: &[f64; 4], genome: &Genome) -> Vec<(usize, f64)> {
    genome
        .entries
        .iter()
        .enumerate()
        .filter(|(_, e)| e.layer == Layer::Response)
        .map(|(i, e)| {
            let gap = address_distance(query, &e.address.geometry(), AddressMode::Full);
            let t = coupling_from_gap(gap, AddressMode::Full);
            (i, t)
        })
        .filter(|(_, t)| *t >= ZREAD_T_MIN)
        .collect()
}

/// Collect the inhibited EligibilityTrace for a query against the Response layer.
///
/// Collects the raw coalition (all entries with t ≥ ZREAD_T_MIN) then applies
/// lateral inhibition: nearby losers are suppressed before credit assignment.
/// Use this for:
///   - `stage_prediction` — effective causal output
///   - `credit_response` — immediate correction credit
///
/// For co-resonance and category formation use `collect_response_eligibility_raw`.
pub fn collect_response_eligibility(query: &[f64; 4], genome: &Genome) -> Vec<(usize, f64)> {
    let raw = collect_response_eligibility_raw(query, genome);
    apply_lateral_inhibition(&raw, genome)
}

/// Apply lateral inhibition to a raw response eligibility coalition.
///
/// Law (derived from existing S³ geometry — no new constants):
///
///   t_winner = max t_i in the raw coalition.
///   Winner set W = { i | t_i ≥ t_winner − ε }.   (ε = 1e-12 for fp ties)
///   For each loser j ∉ W:
///     o_j = max_{w ∈ W} coupling_from_gap(address_distance(addr_j, addr_w, Full), Full)
///     t′_j = max(0, t_j − o_j × t_winner)
///   A′ = { (i, t′_i) | t′_i ≥ ZREAD_T_MIN }
///
/// Why this is the right law:
///   o_j is large only when j's address is geometrically close to the winner —
///   i.e., j is a near-duplicate, not a separate category.  Entries far from
///   the winner have o_j ≈ 0 and are untouched.  The winner itself keeps its
///   original t.  No new thresholds, no new state: the same coupling function
///   used everywhere else.
fn apply_lateral_inhibition(raw: &[(usize, f64)], genome: &Genome) -> Vec<(usize, f64)> {
    if raw.len() <= 1 {
        return raw.to_vec();
    }

    let t_winner = raw.iter().map(|(_, t)| *t).fold(f64::NEG_INFINITY, f64::max);

    raw.iter()
        .map(|&(i, t_i)| {
            if t_i >= t_winner - 1e-12 {
                // Winner: unchanged.
                (i, t_i)
            } else {
                // Loser: suppress by overlap with the winner coalition.
                let o_j = raw
                    .iter()
                    .filter(|(_, tw)| *tw >= t_winner - 1e-12)
                    .map(|&(w, _)| {
                        let gap = address_distance(
                            &genome.entries[i].address.geometry(),
                            &genome.entries[w].address.geometry(),
                            AddressMode::Full,
                        );
                        coupling_from_gap(gap, AddressMode::Full)
                    })
                    .fold(0.0_f64, f64::max);

                let t_prime = (t_i - o_j * t_winner).max(0.0);
                (i, t_prime)
            }
        })
        .filter(|(_, t)| *t >= ZREAD_T_MIN)
        .collect()
}
