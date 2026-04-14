//! Hopf decomposition of S³ → S² × S¹, and the two-channel split.
//!
//! Every quaternion on the unit sphere splits into a base direction
//! (S² — the RGB / vector channel) and a phase angle (S¹ — the W /
//! scalar channel). The brain reads errors through this split: a
//! W-dominant gap means something is *missing*; an RGB-dominant gap
//! means something is *reordered*.
//!
//! `HopfChannel` and `carrier_in_channel` live here — not in the field
//! machine — because the two-channel structure is a property of the
//! fibration, not of the population read. Both `field.rs` and `genome.rs`
//! need channel classification, and both already import `hopf`.
//!
//! ── RGB semantic channels ────────────────────────────────────────────
//!
//! The S² base is a 3D direction on the unit sphere. Its three canonical
//! axes carry distinct semantic roles, derived from the opponent-channel
//! structure of human vision and from the information geometry of the
//! closed loop:
//!
//! * **R (X / index 1) = SALIENCE** — what demands attention. Extracted
//!   from the field by contrast between total and known.
//! * **G (Y / index 2) = TOTAL** — the whole field; the prior. In paint
//!   mixing, Green = Blue + Yellow, so G = known + unknown = everything.
//!   G = 1 in the limit.
//! * **B (Z / index 3) = UNKNOWN** — what has not yet been integrated
//!   into the model; the novel input; the error. Yellow = G − B = the known.
//!
//! The key relationship: in the Hamilton product the full X (salience)
//! update is
//!
//! ```text
//! X_out = w1·x2 + x1·w2 + (y1·z2 − z1·y2)
//!                           ───────────────
//!                           cross term = [G, B]
//! ```
//!
//! The cross term `y1·z2 − z1·y2` is the commutator of the G (totality)
//! and B (unknown) components. It is one of four terms, not the whole update.
//! The other two terms couple the existing salience (x) with the scalar (w)
//! of the other carrier. Salience accumulates through interaction with both
//! the scalar and the G/B commutator. The semantic emerges from committing
//! to the index assignment — no new arithmetic is needed.
//!
//! This is why the human visual system evolved three opponent channels in
//! exactly this arrangement (S-(L+M) blue-yellow = known/unknown; L-M
//! red-green = salience against the total): the visual system is running
//! closure computation on photon input with the same geometric split.
//! The Hopf equator condition |W| = |RGB| is the balance between scalar
//! integration and the full opponent decomposition — the brain "resolves"
//! when known, unknown, and salient have collectively integrated to match
//! the scalar closure signal.
//!
//! ── Kabbalistic correspondence ──────────────────────────────────────
//!
//! **Kav (the channel)** — the Hopf fibration S³ → S² is the Kav: the
//! bandwidth-limited channel through which Ein Sof's light enters the
//! Vacated Space. S² is the base (the accessible projection); S¹ is the
//! fiber (the phase, the cycle). The projection is what every bounded
//! observer *sees*; the fiber is what determines *where in the cycle*
//! that perception occurs.
//!
//! **Parochet (the veil) = W/RGB channel boundary** — the split between
//! the W-dominant channel (scalar, existence, |w| ≥ |rgb|) and the
//! RGB-dominant channel (vector, position, |rgb| > |w|) is the Parochet
//! dividing the supernal triad from the lower seven Sefirot. W-channel
//! queries resolve existence questions (is this present? identity? yes/no).
//! RGB-channel queries resolve positional questions (orbit slot, arithmetic
//! result, angle in sequence).
//!
//! **The Hopf equator (σ = π/4) = the Parochet locus** — carriers
//! exactly on the equator (W = 1/√2) sit precisely at the veil. The prime
//! 2 lives here. Watson-Crick closures resolve here. Every domain's
//! fundamental unit is an orbit of the equatorial carrier.
//!
//! **Hairy ball theorem** — any continuous field projected from S³ to S²
//! must vanish somewhere (χ(S²) = 2). These forced zeros are the
//! observer's invariants: the Riemann zeros for the prime observer, the
//! codon boundaries for the RNA observer, the bar-lines for the music
//! observer. Perception cannot avoid them regardless of architecture.

use crate::sphere::{compose, inverse, sigma};

// ── RGB semantic axes ────────────────────────────────────────────────────────

/// Salience axis — R channel, X component (index 1).
///
/// Carriers whose S² base aligns with this axis are pure salience signals.
/// In Hamilton composition, the X output receives `G×B − B×G`: the commutator
/// of totality and unknown. Salience is the non-commutativity of those two.
pub const SALIENCE_AXIS: [f64; 3] = [1.0, 0.0, 0.0];

/// Totality axis — G channel, Y component (index 2).
///
/// Carriers on this axis represent the whole field: known + unknown = everything.
/// In paint mixing, Green = Blue + Yellow, so G = B + (G−B) = 1 in the limit.
/// The prior. The denominator against which salience and knowledge are measured.
pub const TOTAL_AXIS: [f64; 3] = [0.0, 1.0, 0.0];

/// Unknown axis — B channel, Z component (index 3).
///
/// Carriers on this axis encode what has not yet been integrated — the novel
/// input, the error, the unfamiliar. Yellow = G − B is the known: everything
/// minus what is unknown.
pub const UNKNOWN_AXIS: [f64; 3] = [0.0, 0.0, 1.0];

/// The semantic role of a carrier's dominant S² component.
///
/// Every carrier on S³ has a base direction on S². When that direction lies
/// within π/6 (30°) of one of the three canonical opponent axes, the carrier
/// is classified into that channel. Otherwise it is `Mixed`.
///
/// The opponent structure:
/// * `Salience` (R / X): emerges from `compose(Total, Unknown)` cross terms.
/// * `Total`    (G / Y): the full field; the prior.
/// * `Unknown`  (B / Z): the novel input; the error; not yet integrated.
/// * `Mixed`           : no single axis dominates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VectorChannel {
    /// R channel (X / index 1): what demands attention.
    Salience,
    /// G channel (Y / index 2): the whole field; the prior.
    Total,
    /// B channel (Z / index 3): the novel input; what is unknown.
    Unknown,
    /// No single canonical axis dominates.
    Mixed,
}

/// Classify a carrier's dominant S² component into a semantic vector channel.
///
/// Decomposes the carrier via the Hopf fibration, then measures the geodesic
/// distance from the S² base to each of the three canonical axes. The closest
/// axis within π/6 (30°) is returned; if none qualifies, returns `Mixed`.
///
/// The π/6 threshold is natural: it is half the π/3 Partzuf boundary used
/// throughout the field machine. A carrier within π/6 of an axis "belongs to"
/// that channel in the same sense that entries within π/3 of a query belong to
/// its Partzuf.
pub fn dominant_vector_channel(q: &[f64; 4]) -> VectorChannel {
    let (base, _) = decompose(q);
    const THRESHOLD: f64 = std::f64::consts::FRAC_PI_6;

    // Distance to each axis, folded to [0, π/2] to handle antipodal alignment.
    let ds = base_distance(&base, &SALIENCE_AXIS).min(base_distance(&[-base[0], -base[1], -base[2]], &SALIENCE_AXIS));
    let dt = base_distance(&base, &TOTAL_AXIS   ).min(base_distance(&[-base[0], -base[1], -base[2]], &TOTAL_AXIS));
    let dk = base_distance(&base, &UNKNOWN_AXIS   ).min(base_distance(&[-base[0], -base[1], -base[2]], &UNKNOWN_AXIS));

    let min = ds.min(dt).min(dk);
    if min >= THRESHOLD {
        return VectorChannel::Mixed;
    }
    if ds == min { VectorChannel::Salience }
    else if dt == min { VectorChannel::Total }
    else { VectorChannel::Unknown }
}

/// The runtime semantic triple at one computation step.
///
/// At every step of the three-cell loop three objects are in play:
///
/// - **total**: what the field says is there — the ZREAD aggregate over the
///   genome and buffer, queried by the incoming carrier. This is the prior:
///   the collective belief of the whole memory structure about the input.
/// - **known**: what the brain already predicts — Cell C, the slow accumulator
///   of every closure packet the brain has emitted. This is the model.
/// - **residual**: `compose(total, inverse(known))` — the rotation on S³ that
///   takes `known` to `total`. When total == known this collapses to identity.
///   When they diverge it encodes *how* and *how much* they differ.
///
/// `salience_sigma = sigma(residual)` is the geodesic distance between total
/// and known. It is zero when the model perfectly predicts the field, and
/// reaches π/2 at maximum mismatch.
///
/// `w_gap` and `rgb_gap` decompose the residual along the Hopf W/RGB boundary:
/// - `w_gap` large → existence mismatch (something should be there that isn't)
/// - `rgb_gap` large → type/position mismatch (something is there, wrong kind)
///
/// This struct is the carrier-level analog of the Free Energy Principle:
/// salience = the prediction error that the brain must reduce by updating
/// its model (Cell C) toward the field (total).
#[derive(Clone, Copy, Debug)]
pub struct SemanticFrame {
    /// Field aggregate / prior. ZREAD output for the incoming carrier.
    pub total: [f64; 4],
    /// Accumulated model. Cell C before this step's update, or staged output.
    pub known: [f64; 4],
    /// `compose(total, inverse(known))` — rotation from known to total.
    pub residual: [f64; 4],
    /// `sigma(residual)` — geodesic distance between total and known.
    /// Zero = perfect prediction. Approaches π/2 at maximum mismatch.
    pub salience_sigma: f64,
    /// `|residual[0]|` — W-component of the residual (existence gap).
    pub w_gap: f64,
    /// `sqrt(x²+y²+z²)` of the residual — RGB-component magnitude (type gap).
    pub rgb_gap: f64,
}

/// Compute the semantic triple from a field-total and model-known carrier pair.
///
/// `total` is the field's reading of the current input (ZREAD output, neighbor
/// coalition, or any field aggregate). `known` is the accumulated model (Cell C,
/// staged prediction, etc.).
///
/// The residual `compose(total, inverse(known))` is the unique rotation on S³
/// that maps `known` to `total`. Its magnitude `salience_sigma = sigma(residual)`
/// is the geodesic prediction error. Zero means the model already accounts for
/// the field; π/2 is maximum geometric mismatch.
///
/// No new arithmetic: every call is one Hamilton product and one arccos, both
/// already used throughout the runtime.
pub fn semantic_frame(total: &[f64; 4], known: &[f64; 4]) -> SemanticFrame {
    let residual = compose(total, &inverse(known));
    let salience_sigma = sigma(&residual);
    let w_gap = residual[0].abs();
    let rgb_gap = (residual[1] * residual[1]
        + residual[2] * residual[2]
        + residual[3] * residual[3])
        .sqrt();
    SemanticFrame {
        total: *total,
        known: *known,
        residual,
        salience_sigma,
        w_gap,
        rgb_gap,
    }
}

/// Which Hopf channel to restrict population reads to.
///
/// Every carrier on S³ decomposes into a W (scalar/existence) component
/// and an RGB (vector/position) component. The brain's multi-head attention
/// is forced by this split — it is not synthesized on top of the geometry.
///
/// * `Full` — no restriction.
/// * `W` — entries where |w| ≥ |rgb|. Existence questions: equality,
///   identity, yes/no, primality.
/// * `Rgb` — entries where |rgb| > |w|. Position questions: orbit slot,
///   arithmetic result, angle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HopfChannel {
    Full,
    W,
    Rgb,
}

/// Returns true if `addr` lives in the given Hopf channel.
///
/// W channel: the scalar part dominates (|w| ≥ |rgb| — carrier is near
/// identity or its antipode). RGB channel: the vector part dominates
/// (carrier is near the equatorial Hopf locus). Full: always true.
#[inline]
pub fn carrier_in_channel(addr: &[f64; 4], ch: HopfChannel) -> bool {
    match ch {
        HopfChannel::Full => true,
        HopfChannel::W => {
            let w = addr[0].abs();
            let rgb = (addr[1].powi(2) + addr[2].powi(2) + addr[3].powi(2)).sqrt();
            w >= rgb
        }
        HopfChannel::Rgb => {
            let w = addr[0].abs();
            let rgb = (addr[1].powi(2) + addr[2].powi(2) + addr[3].powi(2)).sqrt();
            rgb > w
        }
    }
}

pub const IDENTITY_BASE: [f64; 3] = [0.0, 0.0, 1.0];
pub const IDENTITY_PHASE: f64 = 0.0;

/// Normalize an S¹ phase into `[0, 2π)`.
#[inline(always)]
pub fn wrap_phase(phase: f64) -> f64 {
    phase.rem_euclid(std::f64::consts::TAU)
}

/// Normalize a base direction onto S². The zero vector is the identity base.
#[inline(always)]
pub fn normalize_base(base: [f64; 3]) -> [f64; 3] {
    let norm = (base[0] * base[0] + base[1] * base[1] + base[2] * base[2]).sqrt();
    if norm < 1e-15 {
        IDENTITY_BASE
    } else {
        [base[0] / norm, base[1] / norm, base[2] / norm]
    }
}

/// Hopf decomposition: q ∈ S³ → (base ∈ S², fiber phase ∈ S¹).
///
/// We identify `S³` with `(z1, z2) ∈ C²`, `|z1|² + |z2|² = 1`, using
/// `z1 = w + i z` and `z2 = y + i x`. The Hopf base is
/// `(2 Re(z1 conj(z2)), 2 Im(z1 conj(z2)), |z1|² - |z2|²)`.
///
/// The S¹ coordinate is the common fiber phase. Under
/// `(z1, z2) -> e^{i chi}(z1, z2)`, the base is unchanged and the phase
/// shifts by `2 chi`. This is the cyclic/positional coordinate used by
/// factorized addressing.
#[inline(always)]
pub fn decompose(q: &[f64; 4]) -> ([f64; 3], f64) {
    let (w, x, y, z) = (q[0], q[1], q[2], q[3]);
    let base = [
        2.0 * (x * z + w * y),
        2.0 * (y * z - w * x),
        w * w + z * z - x * x - y * y,
    ];
    let alpha = z.atan2(w);
    let beta = x.atan2(y);
    (normalize_base(base), wrap_phase(alpha + beta))
}

/// Canonical Hopf lift: (base ∈ S², fiber phase ∈ S¹) → carrier ∈ S³.
///
/// This is the write-side inverse of [`decompose`]. It is the canonical
/// constructor domain encoders must use when they already know the semantic
/// type (`S² base`) and cyclic position (`S¹ phase`) of an object.
///
/// The lift uses the same complex coordinates as [`decompose`]:
///
/// ```text
/// z1 = cos(theta/2) * exp(i alpha)
/// z2 = sin(theta/2) * exp(i beta)
/// base = (sin(theta) cos(delta), sin(theta) sin(delta), cos(theta))
/// delta = alpha - beta
/// phase = alpha + beta
/// ```
///
/// At the Hopf poles one complex coordinate vanishes, so the missing angle is
/// fixed canonically to zero. This preserves the requested fiber phase under
/// this crate's `decompose` convention.
#[inline]
pub fn carrier_from_hopf(base: [f64; 3], phase: f64) -> [f64; 4] {
    let base = normalize_base(base);
    let phase = wrap_phase(phase);
    let bz = base[2].clamp(-1.0, 1.0);
    let sin_theta = (1.0 - bz * bz).max(0.0).sqrt();

    if sin_theta < 1e-12 {
        if bz >= 0.0 {
            return [phase.cos(), 0.0, 0.0, phase.sin()];
        }
        return [0.0, phase.sin(), phase.cos(), 0.0];
    }

    let theta = bz.acos();
    let delta = base[1].atan2(base[0]);
    let alpha = 0.5 * (phase + delta);
    let beta = 0.5 * (phase - delta);
    let ct = (0.5 * theta).cos();
    let st = (0.5 * theta).sin();

    let q = [
        ct * alpha.cos(),
        st * beta.sin(),
        st * beta.cos(),
        ct * alpha.sin(),
    ];
    let norm = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    [q[0] / norm, q[1] / norm, q[2] / norm, q[3] / norm]
}

/// Geodesic distance on S² between two base directions.
#[inline(always)]
pub fn base_distance(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    let dot = (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]).clamp(-1.0, 1.0);
    dot.acos()
}

/// Wrapped distance on S¹ between two phases.
#[inline(always)]
pub fn circular_distance(a: f64, b: f64) -> f64 {
    let tau = std::f64::consts::PI * 2.0;
    let mut d = (a - b).abs();
    while d > tau {
        d -= tau;
    }
    if d > std::f64::consts::PI {
        tau - d
    } else {
        d
    }
}

/// Combined identity distance through both Hopf channels.
#[inline(always)]
pub fn identity_distance(q: &[f64; 4]) -> f64 {
    let (base, phase) = decompose(q);
    base_distance(&base, &IDENTITY_BASE) + circular_distance(phase, IDENTITY_PHASE)
}

/// Which projection of S³ to use when computing distance between carriers.
///
/// The Hopf fibration exposes three independent geometric channels. A query
/// carrier can ask about semantic type (S² base, the rotation axis), position
/// in cycle (S¹ phase, the rotation angle), or existence depth (W scalar).
/// Querying all three simultaneously with `Full` is the default and gives the
/// standard geodesic σ. Querying one channel independently lets the genome
/// distinguish "same type, different position" from "same position, different
/// type" — the factorized address semantics required for arithmetic, grammar,
/// and any other domain where slot identity and value identity are independent.
///
/// * `Full`   — σ(q · address⁻¹). Full geodesic distance on S³. Default.
/// * `Base`   — arc distance on S² between the rotation axes. Two carriers
///   match on Base iff they rotate around the same axis, regardless
///   of how far they rotate.
/// * `Phase`  — wrapped angular distance on S¹ between rotation angles. Two
///   carriers match on Phase iff they rotate by the same amount,
///   regardless of which axis.
/// * `Scalar` — |w(q)| − |w(address)|. Two carriers match on Scalar iff they
///   sit at the same depth below the Hopf equator, regardless of
///   axis or angle. This is the existence/depth channel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AddressMode {
    Full,
    Base,
    Phase,
    Scalar,
}

/// Distance between `query` and `address` in the given Hopf channel.
///
/// All four modes are dimensionally homogeneous (radians or |W| difference
/// on [0, 1]). `Full` is exactly `sigma(&compose(query, &inverse(address)))`,
/// which is the existing behavior everywhere in the codebase. The other three
/// modes project onto independent sub-manifolds of S³ before measuring.
#[inline]
pub fn address_distance(query: &[f64; 4], address: &[f64; 4], mode: AddressMode) -> f64 {
    match mode {
        AddressMode::Full => sigma(&compose(query, &inverse(address))),
        AddressMode::Base => {
            let (qa, _) = decompose(query);
            let (aa, _) = decompose(address);
            base_distance(&qa, &aa)
        }
        AddressMode::Phase => {
            let (_, qp) = decompose(query);
            let (_, ap) = decompose(address);
            circular_distance(qp, ap)
        }
        AddressMode::Scalar => (query[0].abs() - address[0].abs()).abs(),
    }
}

/// Convert a mode-specific gap to a SLERP coupling weight t ∈ [0, 1].
///
/// This is the single definition of "how close = how much coupling." Both
/// the ZREAD accumulator and the BKT statistics recorder must call this
/// function with the same mode so that the read path and its accounting
/// stay synchronized.
///
/// * Angular modes (Full / Base / Phase): gap is an angle in [0, π].
///   t = cos(gap) — 1 for exact match, 0 at π/2, negative beyond (clamped to 0).
/// * Scalar mode: gap = |w_q − w_a| ∈ [0, 1]. t = 1 − gap — 1 for identical
///   W depth, 0 for maximally different.
///
/// The threshold `ZREAD_T_MIN = cos(π/3) = 0.5` (defined in `field`) is
/// applied by the caller after this function returns.
#[inline]
pub fn coupling_from_gap(gap: f64, mode: AddressMode) -> f64 {
    match mode {
        AddressMode::Full | AddressMode::Base | AddressMode::Phase => gap.cos().clamp(0.0, 1.0),
        AddressMode::Scalar => (1.0 - gap).clamp(0.0, 1.0),
    }
}
