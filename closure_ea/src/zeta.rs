//! Geometric zero on S³: the Hurwitz–s lift of the Euler product.
//!
//! This is Walter's `geometric-zero-rh` construction, ported to Rust
//! against the local S³ substrate. It is the exact statement of
//! `Q : ℂ → ℍ[ℝ]` from `ClosureRH.lean` axiom Q (line 241), in
//! computable form.
//!
//! ## The 1 = 3 condition (the geometric zero)
//!
//! The star involution on S³ fixes the scalar channel (W, real part,
//! +1 eigenspace) and negates the imaginary channel (RGB, +i, +j, +k
//! components, −1 eigenspace). Writing `1 = 3` is the assertion that
//! the 1-dimensional W channel and the 3-dimensional RGB channel are
//! bound under that intrinsic mirror — they are each other's inverse
//! under conjugation. The equation "closes" (in the verification-verb
//! sense, `a · star(b) → identity`) precisely when both channels
//! contribute equally to the unit norm:
//!
//! ```text
//! w² = x² + y² + z²
//! ```
//!
//! which on the unit sphere `w² + x² + y² + z² = 1` forces `w² = 1/2`,
//! equivalently `σ = arccos(|w|) = π/4`. This is the **Hopf balance
//! locus**, an equatorial 2-sphere inside S³, and it is the geometric
//! zero. Lean theorem `hopf_balance_iff_half` (zero axioms,
//! `ClosureRH.lean` line 152) proves the value `1/2` is forced — not
//! a tunable threshold.
//!
//! Operationally, the detector for the geometric zero is just:
//!
//! ```text
//! |σ(Q(s)) − π/4|
//! ```
//!
//! local minima in that quantity, swept over `s`, are the closure
//! events. For the prime basis (this module) the closures are the
//! imaginary parts of the Riemann zeros.
//!
//! ## The Hurwitz–s construction
//!
//! For each prime `p` with Lagrange decomposition `p = a²+b²+c²+d²`,
//! the **Hurwitz carrier** is the unit quaternion
//!
//! ```text
//! q̂_p = [a, b, c, d] / √p
//! ```
//!
//! the canonical position of the prime on S³. The **s-encoding** of
//! the prime at `s = σ + it` is
//!
//! ```text
//! enc(p, s) = ( p^(-σ),
//!               √(1 - p^(-2σ)) · sin(-t·ln p),
//!               √(1 - p^(-2σ)) · cos(-t·ln p),
//!               0 )
//! ```
//!
//! a unit quaternion by construction: `r² + (1−r²) = 1`. The
//! **Hurwitz–s Euler factor** is the quaternion product
//!
//! ```text
//! F(p, s) = enc(p, s) · q̂_p
//! ```
//!
//! and the **quaternionic Euler running product** is
//!
//! ```text
//! Q(s) = F(2, s) · F(3, s) · F(5, s) · F(7, s) · …
//! ```
//!
//! `Q(s)` lives on S³ at every finite stage. The classical `ζ(s)` is
//! its projection onto `ℂ ⊂ ℍ`; the geometric content is in the full
//! four-dimensional product.
//!
//! The functions in this module port [GeometricZero.tex line 211]
//! and [experiments/primes_on_s3/core.py line 172] verbatim, against
//! the local `crate::sphere` substrate.

use crate::sphere::{compose, sigma, IDENTITY};
use std::f64::consts::FRAC_PI_4;

// ── Prime generation ─────────────────────────────────────────────

/// Sieve of Eratosthenes — every prime up to and including `limit`.
pub fn sieve_of_eratosthenes(limit: u64) -> Vec<u64> {
    if limit < 2 {
        return Vec::new();
    }
    let n = limit as usize;
    let mut is_prime = vec![true; n + 1];
    is_prime[0] = false;
    is_prime[1] = false;
    let bound = (n as f64).sqrt() as usize;
    for i in 2..=bound {
        if is_prime[i] {
            let mut j = i * i;
            while j <= n {
                is_prime[j] = false;
                j += i;
            }
        }
    }
    is_prime
        .iter()
        .enumerate()
        .filter_map(|(i, &p)| if p { Some(i as u64) } else { None })
        .collect()
}

/// First `n` primes.
pub fn first_n_primes(n: usize) -> Vec<u64> {
    if n == 0 {
        return Vec::new();
    }
    // Upper bound from the prime counting function: the n-th prime
    // is at most n·(ln n + ln ln n) for n ≥ 6. Add slack for n < 6
    // and a safety factor.
    let limit = if n < 6 {
        20
    } else {
        let nf = n as f64;
        ((nf * (nf.ln() + nf.ln().ln())) * 1.3) as u64 + 50
    };
    let mut primes = sieve_of_eratosthenes(limit);
    primes.truncate(n);
    primes
}

// ── Lagrange four-square decomposition ──────────────────────────

/// Find `(a, b, c, d)` with `a² + b² + c² + d² = n` and
/// `a ≥ b ≥ c ≥ d ≥ 0` (canonical lex-largest representative).
///
/// Brute force; fine for the small primes this module typically uses.
pub fn find_four_squares(n: u64) -> (u64, u64, u64, u64) {
    let a_max = (n as f64).sqrt() as u64;
    for a in (0..=a_max).rev() {
        let r1 = n - a * a;
        let b_max = a.min((r1 as f64).sqrt() as u64);
        for b in (0..=b_max).rev() {
            let r2 = r1 - b * b;
            let c_max = b.min((r2 as f64).sqrt() as u64);
            for c in (0..=c_max).rev() {
                let r3 = r2 - c * c;
                let d = (r3 as f64).sqrt() as u64;
                if d * d == r3 && d <= c {
                    return (a, b, c, d);
                }
            }
        }
    }
    panic!("no four-square representation for {}", n);
}

// ── Hurwitz carrier and s-encoding ──────────────────────────────

/// Canonical Hurwitz unit quaternion `q̂_n = [a, b, c, d] / √n`.
/// For `n = 1` returns the identity (Lagrange decomposition is
/// `(1, 0, 0, 0)`).
pub fn hurwitz_carrier(n: u64) -> [f64; 4] {
    if n == 0 {
        return IDENTITY;
    }
    let (a, b, c, d) = find_four_squares(n);
    let inv_sqrt = 1.0 / (n as f64).sqrt();
    [
        a as f64 * inv_sqrt,
        b as f64 * inv_sqrt,
        c as f64 * inv_sqrt,
        d as f64 * inv_sqrt,
    ]
}

/// Hurwitz–s encoding of a prime at parameter `s = σ + it`:
///
/// ```text
/// enc(p, s) = ( p^(-σ),
///               √(1 - p^(-2σ)) · sin(-t·ln p),
///               √(1 - p^(-2σ)) · cos(-t·ln p),
///               0 )
/// ```
///
/// The W component carries `p^(-σ)`; the RGB phase is `−t·ln p`.
/// Unit by construction.
pub fn enc_hurwitz_s(p: u64, sigma_re: f64, t: f64) -> [f64; 4] {
    let r = (p as f64).powf(-sigma_re);
    let r_clamped = r.clamp(0.0, 1.0);
    let rgb_mag = (1.0 - r_clamped * r_clamped).max(0.0).sqrt();
    let phi = -t * (p as f64).ln();
    [r_clamped, rgb_mag * phi.sin(), rgb_mag * phi.cos(), 0.0]
}

/// Hurwitz–s Euler factor `F(p, s) = enc(p, s) · q̂_p`.
pub fn euler_factor(p: u64, sigma_re: f64, t: f64) -> [f64; 4] {
    let enc = enc_hurwitz_s(p, sigma_re, t);
    let qp = hurwitz_carrier(p);
    compose(&enc, &qp)
}

// ── Q(s) and the geometric-zero detector ────────────────────────

/// Quaternionic Euler running product over a prime list.
///
/// ```text
/// Q(s) = ∏_p F(p, s)
/// ```
///
/// Composed in order; the result lives on S³ at every finite stage.
pub fn running_product(primes: &[u64], sigma_re: f64, t: f64) -> [f64; 4] {
    let mut q = IDENTITY;
    for &p in primes {
        q = compose(&q, &euler_factor(p, sigma_re, t));
    }
    q
}

/// Hopf balance error: `|σ(q) − π/4|`. Zero iff `q` sits exactly on
/// the geometric zero manifold (the equatorial 2-sphere where
/// `w² = 1/2`).
#[inline]
pub fn hopf_balance_error(q: &[f64; 4]) -> f64 {
    (sigma(q) - FRAC_PI_4).abs()
}

/// Geometric zero predicate: is `q` within `tol` of the Hopf balance
/// locus?
#[inline]
pub fn is_geometric_zero(q: &[f64; 4], tol: f64) -> bool {
    hopf_balance_error(q) < tol
}

// ── Spectrum scan ───────────────────────────────────────────────

/// One sample from a spectrum scan: a t-value, `σ(Q(s))`, and the
/// Hopf balance error at that t.
#[derive(Clone, Copy, Debug)]
pub struct SpectrumSample {
    pub t: f64,
    pub sigma_q: f64,
    pub balance_error: f64,
}

/// Sample `Q(σ_re + it)` at every `t` in `[t_min, t_max]` stepping
/// by `t_step`, and return the full sequence of samples.
pub fn spectrum_samples(
    primes: &[u64],
    sigma_re: f64,
    t_min: f64,
    t_max: f64,
    t_step: f64,
) -> Vec<SpectrumSample> {
    let mut out = Vec::new();
    let n = ((t_max - t_min) / t_step).floor() as usize + 1;
    for i in 0..n {
        let t = t_min + i as f64 * t_step;
        if t > t_max + 1e-12 { break; }
        let q = running_product(primes, sigma_re, t);
        let s_q = sigma(&q);
        out.push(SpectrumSample {
            t,
            sigma_q: s_q,
            balance_error: (s_q - FRAC_PI_4).abs(),
        });
    }
    out
}

/// Local-minima scan: find `t` values where `|σ(Q(s)) − π/4|` is
/// strictly less than its immediate neighbours. These are the
/// candidate **geometric zeros** of the basis. For the prime basis
/// they should land on the imaginary parts of the Riemann zeros.
pub fn spectrum_local_minima(samples: &[SpectrumSample]) -> Vec<SpectrumSample> {
    let mut zeros = Vec::new();
    if samples.len() < 3 {
        return zeros;
    }
    for i in 1..samples.len() - 1 {
        if samples[i].balance_error < samples[i - 1].balance_error
            && samples[i].balance_error < samples[i + 1].balance_error
        {
            zeros.push(samples[i]);
        }
    }
    zeros
}

/// Convenience: scan and return only the local minima.
pub fn spectrum_scan(
    primes: &[u64],
    sigma_re: f64,
    t_min: f64,
    t_max: f64,
    t_step: f64,
) -> Vec<SpectrumSample> {
    let samples = spectrum_samples(primes, sigma_re, t_min, t_max, t_step);
    spectrum_local_minima(&samples)
}

// ── Multiplication on S³: integer Hamilton product ──────────────
//
// The Hurwitz quaternion ring has a **multiplicative norm**: for
// any two integer quaternions q₁ = [a,b,c,d] and q₂ = [e,f,g,h]
// with |q₁|² = m and |q₂|² = n, the Hamilton product q₁ · q₂ has
// |q₁ · q₂|² = m · n. This is the geometric statement of integer
// multiplication on S³ — multiplying two carriers multiplies their
// norms, and Lagrange's theorem maps every positive integer to a
// quaternion of the corresponding norm.
//
// The unnormalized integer form is what makes this work; the
// `crate::sphere::compose` operation normalizes after every step
// (necessary for the brain's S³ substrate, where carriers must be
// unit), so we keep a separate integer Hamilton product here for
// the multiplicative-norm path.

/// Unnormalized integer Hurwitz quaternion for `n`: the canonical
/// lex-largest four-square decomposition `(a, b, c, d)` packed as
/// `[a, b, c, d]`. Norm² = a² + b² + c² + d² = n.
pub fn hurwitz_int(n: u64) -> [i64; 4] {
    if n == 0 {
        return [0, 0, 0, 0];
    }
    let (a, b, c, d) = find_four_squares(n);
    [a as i64, b as i64, c as i64, d as i64]
}

/// Integer Hamilton product on Hurwitz quaternions. Same formula as
/// `crate::sphere::compose`'s Hamilton product, but on integers and
/// without renormalization, so the multiplicative-norm property is
/// preserved exactly:
///
/// ```text
/// |hamilton_int(q₁, q₂)|² = |q₁|² · |q₂|²
/// ```
#[inline]
pub fn hamilton_int(a: &[i64; 4], b: &[i64; 4]) -> [i64; 4] {
    let (w1, x1, y1, z1) = (a[0], a[1], a[2], a[3]);
    let (w2, x2, y2, z2) = (b[0], b[1], b[2], b[3]);
    [
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ]
}

/// Norm² of an integer Hurwitz quaternion: `w² + x² + y² + z²`.
#[inline]
pub fn norm_sq_int(q: &[i64; 4]) -> u64 {
    (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]) as u64
}

// ── Primality from geometric multiplication ─────────────────────

/// A geometric factorization of `n`: two Hurwitz quaternions whose
/// Hamilton product has norm² exactly `n`, with both factors having
/// norm² strictly between `1` and `n`.
#[derive(Clone, Copy, Debug)]
pub struct GeometricFactorization {
    pub a: u64,
    pub b: u64,
    pub q_a: [i64; 4],
    pub q_b: [i64; 4],
    pub product: [i64; 4],
}

/// Search for a geometric factorization of `n` using only Hamilton
/// multiplication and norm reads — no `%`, no trial division by
/// remainder. For every candidate `a` in `[2, √n]` and every `b` in
/// `[a, n/2]`, multiply the Hurwitz carriers and check whether the
/// product's norm² is exactly `n`. The first hit is the
/// factorization.
pub fn find_geometric_factor(n: u64) -> Option<GeometricFactorization> {
    if n < 4 {
        return None;
    }
    let sqrt_n = (n as f64).sqrt() as u64;
    for a in 2..=sqrt_n {
        let q_a = hurwitz_int(a);
        // The smallest possible `b` is `a` (so we cover a² ≤ n);
        // the largest is whatever keeps norm² ≤ n. The norm of the
        // product is exactly a·b by the multiplicative norm, so we
        // can stop as soon as a·b > n.
        for b in a..=n {
            if a.checked_mul(b).is_none_or(|ab| ab > n) {
                break;
            }
            let q_b = hurwitz_int(b);
            let q_ab = hamilton_int(&q_a, &q_b);
            if norm_sq_int(&q_ab) == n {
                return Some(GeometricFactorization {
                    a,
                    b,
                    q_a,
                    q_b,
                    product: q_ab,
                });
            }
        }
    }
    None
}

/// Primality test using only geometric multiplication: `n` is prime
/// iff its Hurwitz carrier admits no factorization as a Hamilton
/// product of two smaller Hurwitz carriers whose norms multiply to
/// `n`. The test is `find_geometric_factor(n).is_none()`, with the
/// usual edge cases for `n < 2`.
///
/// This is the geometric statement of primality: the multiplicative
/// norm of the Hurwitz quaternion ring is the only thing being read.
/// `n % a` never appears.
pub fn is_prime_geometric(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    if n < 4 {
        return true; // 2, 3 are prime
    }
    find_geometric_factor(n).is_none()
}

/// Geometrically enumerate the primes up to `limit` by running
/// `is_prime_geometric` on every integer in `[2, limit]`. The result
/// is the same set of primes as `sieve_of_eratosthenes(limit)` but
/// produced by Hamilton multiplication of Hurwitz carriers, not by
/// sieving.
pub fn primes_by_geometric_multiplication(limit: u64) -> Vec<u64> {
    (2..=limit).filter(|&n| is_prime_geometric(n)).collect()
}
