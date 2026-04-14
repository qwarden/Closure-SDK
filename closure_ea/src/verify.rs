//! VERIFY — the brain's only verb.
//!
//! Take two carriers, ask whether they are still the same, and answer
//! through a structured verification cycle. The result is **not a
//! boolean** — it is a `VerificationEvent` containing the closure
//! quaternion, its σ, its Hopf decomposition, and which closure type
//! fired (if any).
//!
//! This is the operation Section 6 of BRAIN.md describes. Every other
//! comparison/closure/branch in the brain is a named composition built
//! around it.

use crate::hopf::decompose;
use crate::sphere::{compose, inverse, sigma, IDENTITY};

/// The two privileged σ values on S³ (Section 4 of BRAIN.md).
///
/// These are not parameters. They are forced by the manifold:
/// * `σ = 0` is the rest state at the north pole.
/// * `σ = π/4` is the equatorial Hopf-balance locus where the W
///   channel and the RGB channel each carry exactly half the unit
///   norm. Forced by the conjunction "unit sphere + 1=3 condition".
pub const SIGMA_IDENTITY: f64 = 0.0;
pub const SIGMA_BALANCE: f64 = std::f64::consts::FRAC_PI_4;

/// Default tolerance for closure detection.
/// Tunable per call but the spec gives no smaller forced value.
pub const DEFAULT_TOLERANCE: f64 = 1e-6;

/// Which closure (if any) the verification event fired.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClosureKind {
    /// σ → 0. The cycle returned to the north pole.
    Identity,
    /// σ → π/4. The cycle reached the balance locus.
    Balance,
    /// Neither closure type fired. The diabolo is between the two
    /// privileged values; the verification verb has registered a gap
    /// but no closure event.
    Open,
}

/// Which Hopf channel dominates the gap.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HopfDominance {
    /// |W|² > |RGB|² — closer to identity than to balance.
    /// In error-classification language: something is **missing**.
    W,
    /// |RGB|² > |W|² — past the balance locus.
    /// In error-classification language: something is **reordered**.
    Rgb,
    /// |W|² == |RGB|² (within tolerance) — sitting on the balance
    /// locus itself.
    Balanced,
}

/// The result of one verification cycle.
///
/// This is what `VERIFY(a, b)` returns. It is the geometric trace of
/// the question "are these the same?" — magnitude (σ), direction
/// (Hopf base), phase, and a discrete classification of the closure
/// type and the dominant channel.
#[derive(Clone, Copy, Debug)]
pub struct VerificationEvent {
    /// `cycle = a · star(b)`. The carrier whose σ measures the gap.
    pub cycle: [f64; 4],
    /// Geodesic distance from identity. 0 = exact match.
    pub sigma: f64,
    /// Hopf base direction (S² — what kind of relationship).
    pub base: [f64; 3],
    /// Hopf phase (S¹ — where in the cycle).
    pub phase: f64,
    /// Which closure type fired.
    pub kind: ClosureKind,
    /// Which Hopf channel dominates.
    pub hopf: HopfDominance,
}

impl VerificationEvent {
    /// `true` iff identity closure fired (`a = b` in the strong sense).
    #[inline]
    pub fn closes(self) -> bool {
        matches!(self.kind, ClosureKind::Identity)
    }

    /// `true` iff balance closure fired (the cycle hit σ = π/4).
    #[inline]
    pub fn balances(self) -> bool {
        matches!(self.kind, ClosureKind::Balance)
    }
}

/// Run the A?=A verification cycle on two carriers with the default
/// tolerance.
#[inline]
pub fn verify(a: &[f64; 4], b: &[f64; 4]) -> VerificationEvent {
    verify_with_tolerance(a, b, DEFAULT_TOLERANCE)
}

/// Run the verification cycle with a caller-supplied tolerance for
/// closure detection.
pub fn verify_with_tolerance(a: &[f64; 4], b: &[f64; 4], tolerance: f64) -> VerificationEvent {
    let cycle = compose(a, &inverse(b));
    let s = sigma(&cycle);
    let (base, phase) = decompose(&cycle);
    let kind = closure_kind(s, tolerance);
    let hopf = hopf_dominance(&cycle, tolerance);
    VerificationEvent {
        cycle,
        sigma: s,
        base,
        phase,
        kind,
        hopf,
    }
}

/// Classify a σ value into a closure kind given a tolerance.
#[inline]
pub fn closure_kind(s: f64, tolerance: f64) -> ClosureKind {
    if (s - SIGMA_IDENTITY).abs() <= tolerance {
        ClosureKind::Identity
    } else if (s - SIGMA_BALANCE).abs() <= tolerance {
        ClosureKind::Balance
    } else {
        ClosureKind::Open
    }
}

/// Read which Hopf channel dominates a carrier.
///
/// `|W|² = q.re² ` versus `|RGB|² = q.x² + q.y² + q.z²`. The
/// classification flips at the balance locus where they are equal.
#[inline]
pub fn hopf_dominance(q: &[f64; 4], tolerance: f64) -> HopfDominance {
    let w_sq = q[0] * q[0];
    let rgb_sq = q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
    if (w_sq - rgb_sq).abs() <= tolerance {
        HopfDominance::Balanced
    } else if w_sq > rgb_sq {
        HopfDominance::W
    } else {
        HopfDominance::Rgb
    }
}

/// Convenience: VERIFY against the identity carrier.
/// Equivalent to `verify(q, &IDENTITY)` but skips one inverse.
#[inline]
pub fn verify_against_identity(q: &[f64; 4]) -> VerificationEvent {
    verify(q, &IDENTITY)
}
