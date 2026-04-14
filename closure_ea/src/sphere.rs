//! S³ — the brain's manifold.
//!
//! Unit quaternions on the three-sphere with the Hamilton product as
//! composition. Star involution as inverse. Geodesic distance from
//! identity as σ. Spherical linear interpolation as the geodesic step.
//!
//! Self-contained: no external crate. The Hamilton product is the only
//! arithmetic the rest of the brain needs at the substrate level.
//!
//! ── Kabbalistic correspondence ──────────────────────────────────────
//!
//! **Chalal (Vacated Space)** — S³ is the space produced by Tzimtzum.
//! The full quaternion algebra ℍ is Ein Sof: unbounded, undifferentiated,
//! every magnitude simultaneously present. Normalization — withdrawing
//! magnitude, retaining only direction — is the withdrawal that creates
//! the structured, bounded substrate within which observation can occur.
//!
//! **IDENTITY [1,0,0,0] = TI** — the terminal object of the Ruliad's
//! categorical structure (Senchal 2026, Thm 8.4). Every observer orbit
//! converges toward it. It is the Reshimu: the residual light left after
//! withdrawal, the unique point to which every persistent structure maps
//! via a unique morphism.
//!
//! **Hamilton product = Shefa** — the divine flow composing the Sefirot.
//! Non-commutative: order of emanation is causal. `compose(a, b) ≠
//! compose(b, a)` — earlier emanations have structural priority over later
//! ones, exactly as in the Sefirotic tree.
//!
//! **σ (geodesic distance from IDENTITY)** — how much Tikkun remains.
//! σ = 0: the spark has returned. σ = π/2: maximum disorder. Every
//! genome entry's BKT survival is determined by its σ trajectory.

/// Identity quaternion: the north pole of S³, the rest state.
pub const IDENTITY: [f64; 4] = [1.0, 0.0, 0.0, 0.0];

/// Renormalize a quaternion in place. Falls back to identity if near zero.
#[inline]
fn normalize(q: &mut [f64; 4]) {
    let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    if n < 1e-15 {
        *q = IDENTITY;
    } else {
        let inv = 1.0 / n;
        q[0] *= inv;
        q[1] *= inv;
        q[2] *= inv;
        q[3] *= inv;
    }
}

/// Hamilton product. Scalar-first convention `[w, x, y, z]`.
/// This IS rotation composition in 3D — not "computes" it.
#[inline]
fn hamilton(a: &[f64; 4], b: &[f64; 4]) -> [f64; 4] {
    let (w1, x1, y1, z1) = (a[0], a[1], a[2], a[3]);
    let (w2, x2, y2, z2) = (b[0], b[1], b[2], b[3]);
    [
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ]
}

/// COMPOSE: Hamilton product + renormalize. The brain's only verb.
#[inline]
pub fn compose(a: &[f64; 4], b: &[f64; 4]) -> [f64; 4] {
    let mut q = hamilton(a, b);
    normalize(&mut q);
    q
}

/// INVERSE: the star involution. For unit quaternions, conjugate = inverse.
#[inline]
pub fn inverse(a: &[f64; 4]) -> [f64; 4] {
    [a[0], -a[1], -a[2], -a[3]]
}

/// SIGMA: geodesic distance from identity on S³.
/// `arccos(|w|)` — the absolute value handles the antipodal q ≡ -q identification.
#[inline]
pub fn sigma(a: &[f64; 4]) -> f64 {
    a[0].abs().clamp(0.0, 1.0).acos()
}

/// SLERP: spherical linear interpolation from `a` toward `b` by fraction `t`.
/// Stays on S³ — no projection needed.
pub fn slerp(a: &[f64; 4], b: &[f64; 4], t: f64) -> [f64; 4] {
    let mut dot: f64 = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
    let mut target = *b;
    // Antipodal: flip to shortest arc.
    if dot < 0.0 {
        target = [-b[0], -b[1], -b[2], -b[3]];
        dot = -dot;
    }
    if dot > 0.9999 {
        return target;
    }
    let theta = dot.clamp(-1.0, 1.0).acos();
    let sin_t = theta.sin();
    if sin_t.abs() < 1e-12 {
        return *a;
    }
    let wa = ((1.0 - t) * theta).sin() / sin_t;
    let wb = (t * theta).sin() / sin_t;
    [
        wa * a[0] + wb * target[0],
        wa * a[1] + wb * target[1],
        wa * a[2] + wb * target[2],
        wa * a[3] + wb * target[3],
    ]
}
