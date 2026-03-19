//! Circle — the simplest mode.
//!
//! Each element is a single angle θ ∈ [0, 2π). Compose = add mod 2π.
//! Distance = shorter arc length: min(θ, 2π − θ).
//!
//! Commutative — order doesn't matter, only content.
//! Use when you just need "did anything change?"

use super::LieGroup;
use rand::RngCore;
use std::f64::consts::TAU;

#[derive(Clone)]
pub struct CircleGroup;

impl LieGroup for CircleGroup {
    fn compose(&self, a: &[f64], b: &[f64]) -> Vec<f64> {
        vec![(a[0] + b[0]).rem_euclid(TAU)]
    }

    fn compose_into(&self, a: &[f64], b: &[f64], out: &mut [f64]) {
        out[0] = (a[0] + b[0]).rem_euclid(TAU);
    }

    fn inverse(&self, a: &[f64]) -> Vec<f64> {
        vec![(TAU - a[0]).rem_euclid(TAU)]
    }

    fn inverse_into(&self, a: &[f64], out: &mut [f64]) {
        out[0] = (TAU - a[0]).rem_euclid(TAU);
    }

    fn identity(&self) -> Vec<f64> {
        vec![0.0]
    }

    fn distance_from_identity(&self, a: &[f64]) -> f64 {
        let theta = a[0].rem_euclid(TAU);
        theta.min(TAU - theta)
    }

    fn random(&self, rng: &mut dyn RngCore) -> Vec<f64> {
        let u = (rng.next_u64() as f64) / (u64::MAX as f64);
        vec![u * TAU]
    }

    fn dim(&self) -> usize {
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn identity_laws() {
        let g = CircleGroup;
        let mut rng = StdRng::seed_from_u64(42);
        for _ in 0..100 {
            let a = g.random(&mut rng);
            let id = g.identity();
            let la = g.compose(&id, &a);
            let ra = g.compose(&a, &id);
            assert!((la[0] - a[0]).abs() < 1e-12);
            assert!((ra[0] - a[0]).abs() < 1e-12);
        }
    }

    #[test]
    fn inverse_law() {
        let g = CircleGroup;
        let mut rng = StdRng::seed_from_u64(42);
        for _ in 0..100 {
            let a = g.random(&mut rng);
            let a_inv = g.inverse(&a);
            let prod = g.compose(&a, &a_inv);
            assert!(g.distance_from_identity(&prod) < 1e-12);
        }
    }

    #[test]
    fn distance_nonneg_and_identity_zero() {
        let g = CircleGroup;
        let id = g.identity();
        assert!(g.distance_from_identity(&id) < 1e-15);

        let mut rng = StdRng::seed_from_u64(42);
        for _ in 0..100 {
            let a = g.random(&mut rng);
            assert!(g.distance_from_identity(&a) >= 0.0);
        }
    }
}
