//! Embedding: raw bytes → group elements via SHA-256.
//!
//! This is step 1 of the pipeline. User data (any bytes) gets hashed
//! into a group element that the rest of the system can compose.
//!
//! The embedding is deterministic (same input → same element) but not
//! cryptographic. SHA-256 is used for uniform distribution, not security.
//!
//! When embedding is NOT needed: torus accounting data is already a group
//! element (debit = negative phase, credit = positive phase).

use crate::groups::LieGroup;
use sha2::{Digest, Sha256};
use std::f64::consts::TAU;

/// bytes → Circle element (one phase in [0, 2π)).
pub fn bytes_to_phase(data: &[u8]) -> Vec<f64> {
    let hash = Sha256::digest(data);
    let h = u64::from_le_bytes(hash[..8].try_into().unwrap());
    let angle = (h as f64 / u64::MAX as f64) * TAU;
    vec![angle]
}

/// bytes → Sphere element (unit quaternion on S³).
/// Uses Box-Muller on SHA-256 output for uniform distribution on S³,
/// matching the approach in SphereGroup::random().
pub fn bytes_to_sphere(data: &[u8]) -> Vec<f64> {
    let hash = Sha256::digest(data);
    // Extract 4 uniform values in (0, 1) from the hash
    let mut u = [0.0f64; 4];
    for i in 0..4 {
        let v = u64::from_le_bytes(hash[i * 8..(i + 1) * 8].try_into().unwrap());
        // Map to (0, 1) — avoid exact 0 for ln()
        u[i] = (v as f64 + 1.0) / (u64::MAX as f64 + 2.0);
    }
    // Box-Muller: two pairs of uniform → two pairs of Gaussian
    let r1 = (-2.0 * u[0].ln()).sqrt();
    let theta1 = TAU * u[1];
    let r2 = (-2.0 * u[2].ln()).sqrt();
    let theta2 = TAU * u[3];
    let vals = [
        r1 * theta1.cos(),
        r1 * theta1.sin(),
        r2 * theta2.cos(),
        r2 * theta2.sin(),
    ];
    // Normalize to S³
    let norm =
        (vals[0] * vals[0] + vals[1] * vals[1] + vals[2] * vals[2] + vals[3] * vals[3]).sqrt();
    if norm < 1e-10 {
        return vec![1.0, 0.0, 0.0, 0.0];
    }
    let inv = 1.0 / norm;
    vec![vals[0] * inv, vals[1] * inv, vals[2] * inv, vals[3] * inv]
}

/// bytes → Torus element (k phases in [0, 2π), domain-separated).
pub fn bytes_to_torus(data: &[u8], k: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(k);
    for i in 0..k {
        let h = hash_u64_with_domain(data, 0x544F5255, i as u32);
        let angle = (h as f64 / u64::MAX as f64) * TAU;
        out.push(angle);
    }
    out
}

/// Domain-separated SHA-256: hash(data || domain || idx) → u64.
fn hash_u64_with_domain(data: &[u8], domain: u32, idx: u32) -> u64 {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hasher.update(domain.to_le_bytes());
    hasher.update(idx.to_le_bytes());
    let hash = hasher.finalize();
    u64::from_le_bytes(hash[..8].try_into().unwrap())
}

/// Compute closure element from pre-embedded elements without storing
/// intermediate products. O(n) time, O(1) memory.
/// Used by closure_element_from_elements() in the Python API.
pub fn closure_element_from_elements(group: &dyn LieGroup, data: &[f64], dim: usize) -> Vec<f64> {
    let n = data.len() / dim;
    let mut running = group.identity();
    let mut buf = vec![0.0; dim];
    for i in 0..n {
        let g = &data[i * dim..(i + 1) * dim];
        group.compose_into(&running, g, &mut buf);
        running.copy_from_slice(&buf);
    }
    running
}
