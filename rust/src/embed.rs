//! Embedding: raw bytes -> group elements on S^3.
//!
//! Two modes, same Sphere function:
//!
//!   embed(data, hashed=false)  -> geometric. Each byte composes as a
//!                                 rotation on S^3. Similar bytes -> nearby
//!                                 quaternions.
//!
//!   embed(data, hashed=true)   -> cryptographic. SHA-256 first, then
//!                                 Box-Muller -> S^3. Destroys similarity.
//!
//! The adapter chooses the embedding mode. The group operations are the same.

use crate::groups::sphere::SphereGroup;
use crate::groups::LieGroup;
use sha2::{Digest, Sha256};
use std::f64::consts::TAU;
use std::sync::OnceLock;

static BYTE_TABLE: OnceLock<[[f64; 4]; 256]> = OnceLock::new();

fn byte_quaternions() -> &'static [[f64; 4]; 256] {
    BYTE_TABLE.get_or_init(|| {
        let mut table = [[0.0f64; 4]; 256];
        for i in 0..256u16 {
            table[i as usize] = bytes_to_sphere_hashed(&[i as u8]);
        }
        table
    })
}

pub fn bytes_to_phase(data: &[u8]) -> Vec<f64> {
    let hash = Sha256::digest(data);
    let h = u64::from_le_bytes(hash[..8].try_into().unwrap());
    let angle = (h as f64 / u64::MAX as f64) * TAU;
    vec![angle]
}

pub fn bytes_to_sphere(data: &[u8], hashed: bool) -> Vec<f64> {
    bytes_to_sphere4(data, hashed).to_vec()
}

pub fn bytes_to_sphere4(data: &[u8], hashed: bool) -> [f64; 4] {
    if hashed {
        bytes_to_sphere_hashed(data)
    } else {
        bytes_to_sphere_geometric(data)
    }
}

fn bytes_to_sphere_geometric(data: &[u8]) -> [f64; 4] {
    if data.is_empty() {
        return [1.0, 0.0, 0.0, 0.0];
    }
    let table = byte_quaternions();
    let g = SphereGroup;
    let mut running = [1.0f64, 0.0, 0.0, 0.0];
    let mut buf = [0.0f64; 4];
    for &byte in data {
        g.compose_into(&running, &table[byte as usize], &mut buf);
        running = buf;
    }
    running
}

fn bytes_to_sphere_hashed(data: &[u8]) -> [f64; 4] {
    let hash = Sha256::digest(data);
    let mut u = [0.0f64; 4];
    for i in 0..4 {
        let v = u64::from_le_bytes(hash[i * 8..(i + 1) * 8].try_into().unwrap());
        u[i] = (v as f64 + 1.0) / (u64::MAX as f64 + 2.0);
    }
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
    let norm =
        (vals[0] * vals[0] + vals[1] * vals[1] + vals[2] * vals[2] + vals[3] * vals[3]).sqrt();
    if norm < 1e-10 {
        return [1.0, 0.0, 0.0, 0.0];
    }
    let inv = 1.0 / norm;
    [vals[0] * inv, vals[1] * inv, vals[2] * inv, vals[3] * inv]
}

pub fn bytes_to_torus(data: &[u8], k: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(k);
    for i in 0..k {
        let h = hash_u64_with_domain(data, 0x544F5255, i as u32);
        let angle = (h as f64 / u64::MAX as f64) * TAU;
        out.push(angle);
    }
    out
}

fn hash_u64_with_domain(data: &[u8], domain: u32, idx: u32) -> u64 {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hasher.update(domain.to_le_bytes());
    hasher.update(idx.to_le_bytes());
    let hash = hasher.finalize();
    u64::from_le_bytes(hash[..8].try_into().unwrap())
}

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
