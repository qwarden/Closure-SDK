//! Hybrid — combine two groups into one.
//!
//! Elements are concatenated: [g₁ components | g₂ components].
//! Operations split at the dimension boundary and delegate to each factor.
//! Distance = sqrt(d₁² + d₂²).
//!
//! Example: Hybrid(Torus(8), Sphere) checks that 8 accounts balance
//! AND that events arrived in order — simultaneously, from one summary.

use super::LieGroup;
use rand::RngCore;

pub struct HybridGroup {
    g1: Box<dyn LieGroup>,
    g2: Box<dyn LieGroup>,
    split: usize, // index where g1 ends and g2 begins
}

impl HybridGroup {
    pub fn new(g1: Box<dyn LieGroup>, g2: Box<dyn LieGroup>) -> Self {
        let split = g1.dim();
        Self { g1, g2, split }
    }

    #[inline]
    fn parts<'a>(&self, a: &'a [f64]) -> (&'a [f64], &'a [f64]) {
        (&a[..self.split], &a[self.split..])
    }
}

impl LieGroup for HybridGroup {
    fn compose(&self, a: &[f64], b: &[f64]) -> Vec<f64> {
        let (a1, a2) = self.parts(a);
        let (b1, b2) = self.parts(b);
        let mut result = self.g1.compose(a1, b1);
        result.extend(self.g2.compose(a2, b2));
        result
    }

    fn compose_into(&self, a: &[f64], b: &[f64], out: &mut [f64]) {
        let (a1, a2) = self.parts(a);
        let (b1, b2) = self.parts(b);
        self.g1.compose_into(a1, b1, &mut out[..self.split]);
        self.g2.compose_into(a2, b2, &mut out[self.split..]);
    }

    fn inverse(&self, a: &[f64]) -> Vec<f64> {
        let (a1, a2) = self.parts(a);
        let mut result = self.g1.inverse(a1);
        result.extend(self.g2.inverse(a2));
        result
    }

    fn inverse_into(&self, a: &[f64], out: &mut [f64]) {
        let (a1, a2) = self.parts(a);
        self.g1.inverse_into(a1, &mut out[..self.split]);
        self.g2.inverse_into(a2, &mut out[self.split..]);
    }

    fn identity(&self) -> Vec<f64> {
        let mut result = self.g1.identity();
        result.extend(self.g2.identity());
        result
    }

    fn distance_from_identity(&self, a: &[f64]) -> f64 {
        let (a1, a2) = self.parts(a);
        let d1 = self.g1.distance_from_identity(a1);
        let d2 = self.g2.distance_from_identity(a2);
        (d1 * d1 + d2 * d2).sqrt()
    }

    fn random(&self, rng: &mut dyn RngCore) -> Vec<f64> {
        let mut result = self.g1.random(rng);
        result.extend(self.g2.random(rng));
        result
    }

    fn dim(&self) -> usize {
        self.g1.dim() + self.g2.dim()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::groups::sphere::SphereGroup;
    use crate::groups::torus::Torus;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn identity_laws() {
        let g = HybridGroup::new(Box::new(Torus::new(5)), Box::new(SphereGroup));
        let mut rng = StdRng::seed_from_u64(42);
        for _ in 0..50 {
            let a = g.random(&mut rng);
            let id = g.identity();
            let la = g.compose(&id, &a);
            let ra = g.compose(&a, &id);
            for i in 0..9 {
                assert!((la[i] - a[i]).abs() < 1e-10);
                assert!((ra[i] - a[i]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn inverse_law() {
        let g = HybridGroup::new(Box::new(Torus::new(5)), Box::new(SphereGroup));
        let mut rng = StdRng::seed_from_u64(42);
        for _ in 0..50 {
            let a = g.random(&mut rng);
            let a_inv = g.inverse(&a);
            let prod = g.compose(&a, &a_inv);
            assert!(g.distance_from_identity(&prod) < 1e-10);
        }
    }
}
