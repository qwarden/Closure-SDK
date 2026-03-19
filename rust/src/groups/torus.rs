//! Torus — the per-channel diagnostics mode.
//!
//! k independent phase channels, each in [0, 2π). Compose = componentwise
//! addition mod 2π. Distance = L2 norm of wrapped residuals.
//!
//! Commutative — order doesn't matter.
//!
//! The unique feature: `channel_residuals()` returns a signed vector in
//! [-π, π]^k telling you WHICH channel is off and BY HOW MUCH.
//! One call, per-account diagnostics, O(1).

use super::LieGroup;
use rand::RngCore;
use std::f64::consts::{PI, TAU};

#[derive(Clone)]
pub struct Torus {
    k: usize,
}

impl Torus {
    pub fn new(k: usize) -> Self {
        Self { k }
    }

    /// Signed residual per channel. Positive = excess, negative = deficit.
    pub fn channel_residuals(&self, a: &[f64]) -> Vec<f64> {
        a.iter()
            .map(|&v| {
                let r = v.rem_euclid(TAU);
                if r > PI {
                    r - TAU
                } else {
                    r
                }
            })
            .collect()
    }
}

impl LieGroup for Torus {
    fn compose(&self, a: &[f64], b: &[f64]) -> Vec<f64> {
        a.iter()
            .zip(b)
            .map(|(&x, &y)| (x + y).rem_euclid(TAU))
            .collect()
    }

    fn compose_into(&self, a: &[f64], b: &[f64], out: &mut [f64]) {
        for i in 0..self.k {
            out[i] = (a[i] + b[i]).rem_euclid(TAU);
        }
    }

    fn inverse(&self, a: &[f64]) -> Vec<f64> {
        a.iter().map(|&x| (TAU - x).rem_euclid(TAU)).collect()
    }

    fn inverse_into(&self, a: &[f64], out: &mut [f64]) {
        for i in 0..self.k {
            out[i] = (TAU - a[i]).rem_euclid(TAU);
        }
    }

    fn identity(&self) -> Vec<f64> {
        vec![0.0; self.k]
    }

    fn distance_from_identity(&self, a: &[f64]) -> f64 {
        let sum_sq: f64 = a
            .iter()
            .map(|&v| {
                let r = v.rem_euclid(TAU);
                let d = r.min(TAU - r);
                d * d
            })
            .sum();
        sum_sq.sqrt()
    }

    fn random(&self, rng: &mut dyn RngCore) -> Vec<f64> {
        (0..self.k)
            .map(|_| {
                let u = (rng.next_u64() as f64) / (u64::MAX as f64);
                u * TAU
            })
            .collect()
    }

    fn dim(&self) -> usize {
        self.k
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn identity_laws() {
        let g = Torus::new(5);
        let mut rng = StdRng::seed_from_u64(42);
        for _ in 0..100 {
            let a = g.random(&mut rng);
            let id = g.identity();
            let la = g.compose(&id, &a);
            let ra = g.compose(&a, &id);
            for i in 0..5 {
                assert!((la[i] - a[i]).abs() < 1e-12);
                assert!((ra[i] - a[i]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn inverse_law() {
        let g = Torus::new(5);
        let mut rng = StdRng::seed_from_u64(42);
        for _ in 0..100 {
            let a = g.random(&mut rng);
            let a_inv = g.inverse(&a);
            let prod = g.compose(&a, &a_inv);
            assert!(g.distance_from_identity(&prod) < 1e-12);
        }
    }

    #[test]
    fn commutativity() {
        let g = Torus::new(5);
        let mut rng = StdRng::seed_from_u64(42);
        for _ in 0..100 {
            let a = g.random(&mut rng);
            let b = g.random(&mut rng);
            let ab = g.compose(&a, &b);
            let ba = g.compose(&b, &a);
            let diff: f64 = ab.iter().zip(&ba).map(|(x, y)| (x - y).abs()).sum();
            assert!(diff < 1e-12, "torus should be commutative");
        }
    }

    #[test]
    fn channel_independence() {
        let g = Torus::new(5);
        let mut a = g.identity();
        a[2] = 0.5;
        let res = g.channel_residuals(&a);
        assert!((res[2] - 0.5).abs() < 1e-12);
        for i in [0, 1, 3, 4] {
            assert!(res[i].abs() < 1e-12, "channel {i} should be unaffected");
        }
    }
}
