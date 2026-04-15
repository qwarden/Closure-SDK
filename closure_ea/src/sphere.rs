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

// ── Von Mises-Fisher sampling on S³ ──────────────────────────────────────────
//
// The natural noise distribution on S³ = SU(2). Concentration parameter κ:
//   κ = 0  → Haar measure (uniform on S³, maximum disorder)
//   κ → ∞  → delta at center (deterministic, no noise)
//   κ ≈ 1/BKT_THRESHOLD ≈ 2.08 → critical noise: perturbations reach the
//     closure threshold σ ≈ π/4 with meaningful probability.
//
// Sampling uses the rejection-free method for S³: generate a 4D Gaussian
// centered on the north pole with tangent-space variance ~1/κ, then rotate
// the result to the desired center via Hamilton product.
//
// The RNG is a simple xoshiro256** seeded from the caller. No external
// crate dependency — the substrate stays self-contained.

/// Xoshiro256** PRNG — fast, high-quality, no-dependency.
/// State is 256 bits; period is 2^256 − 1.
#[derive(Clone, Debug)]
pub struct Rng {
    s: [u64; 4],
}

impl Rng {
    /// Seed from a single u64. Splitmix64 expands to full state.
    pub fn new(seed: u64) -> Self {
        let mut z = seed;
        let mut s = [0u64; 4];
        for slot in &mut s {
            z = z.wrapping_add(0x9e3779b97f4a7c15);
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            *slot = z ^ (z >> 31);
        }
        Self { s }
    }

    /// Next u64.
    #[inline]
    fn next_u64(&mut self) -> u64 {
        let result = (self.s[1].wrapping_mul(5)).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Uniform f64 in [0, 1).
    #[inline]
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Standard normal via Box-Muller.
    pub fn normal(&mut self) -> f64 {
        let u1 = self.uniform().max(1e-300);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

/// Sample a unit quaternion from the von Mises-Fisher distribution on S³.
///
/// `center`: the mean direction (unit quaternion).
/// `kappa`: concentration. 0 = uniform/Haar; large = tight around center.
///
/// Method: sample a tangent-space perturbation at the north pole [1,0,0,0],
/// then rotate to `center` via Hamilton product. The tangent vector has
/// Gaussian components with std = 1/√κ, giving geodesic spread ~1/√κ.
pub fn sample_vmf_s3(center: &[f64; 4], kappa: f64, rng: &mut Rng) -> [f64; 4] {
    if kappa < 1e-9 {
        // κ ≈ 0: uniform on S³ (Haar measure).
        let mut q = [rng.normal(), rng.normal(), rng.normal(), rng.normal()];
        normalize(&mut q);
        return q;
    }

    // Tangent-space perturbation at the north pole.
    // The imaginary components are Gaussian with std = 1/√κ.
    let std_dev = 1.0 / kappa.sqrt();
    let dx = rng.normal() * std_dev;
    let dy = rng.normal() * std_dev;
    let dz = rng.normal() * std_dev;

    // Lift to S³: w = √(1 − |v|²), or normalize if |v| > 1.
    let v_sq = dx * dx + dy * dy + dz * dz;
    let perturbation = if v_sq < 1.0 {
        [(1.0 - v_sq).sqrt(), dx, dy, dz]
    } else {
        let mut q = [0.0, dx, dy, dz];
        normalize(&mut q);
        q
    };

    // Rotate from north pole to center.
    compose(center, &perturbation)
}
