//! Geometric surfaces that summaries live on.
//!
//! Each group defines: compose (multiply two elements), inverse, identity,
//! and distance_from_identity (how far a summary is from "perfectly clean").
//!
//! All groups are compact with bi-invariant metrics — this is what makes
//! Theorems 1 and 2 work. Non-compact groups (GL(n)) or groups with only
//! left-invariant metrics (SE(3)) would break the guarantees.

pub mod circle;
pub mod hybrid;
pub mod sphere;
pub mod torus;

use rand::RngCore;

/// The interface every geometric surface implements.
///
/// Elements are `&[f64]` slices — dimension depends on the group:
/// Circle = 1, Sphere = 4, Torus(k) = k, Hybrid(A, B) = dim(A) + dim(B).
pub trait LieGroup: Send + Sync {
    /// Multiply two elements: a · b.
    fn compose(&self, a: &[f64], b: &[f64]) -> Vec<f64>;

    /// Multiply a · b, writing the result into `out` (no allocation).
    fn compose_into(&self, a: &[f64], b: &[f64], out: &mut [f64]) {
        let result = self.compose(a, b);
        out.copy_from_slice(&result);
    }

    /// Compute a⁻¹ such that a · a⁻¹ = identity.
    fn inverse(&self, a: &[f64]) -> Vec<f64>;

    /// Compute a⁻¹, writing the result into `out` (no allocation).
    fn inverse_into(&self, a: &[f64], out: &mut [f64]) {
        let result = self.inverse(a);
        out.copy_from_slice(&result);
    }

    /// The neutral element — the "zero" summary.
    fn identity(&self) -> Vec<f64>;

    /// How far is this element from identity? This is the σ (sigma) value.
    /// σ = 0 → perfectly clean. σ > 0 → something drifted by exactly σ.
    fn distance_from_identity(&self, a: &[f64]) -> f64;

    /// Random element for testing. Not used in production pipeline.
    fn random(&self, rng: &mut dyn RngCore) -> Vec<f64>;

    /// Number of f64 values per element.
    fn dim(&self) -> usize;
}
