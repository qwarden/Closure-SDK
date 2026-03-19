//! GeometricPath — the core data structure.
//!
//! Stores running products: C_0 = identity, C_t = g_1 · g_2 · ... · g_t.
//! This is a prefix-sum on the group — like a running total, but with
//! group multiplication instead of addition.
//!
//! What you can do with it (all O(1) after O(n) build):
//! - check_global()  → σ: is the whole sequence clean?
//! - check_range()   → σ for any sub-sequence
//! - recover(t)      → reconstruct original element g_t
//! - closure_element → the final summary C_n (constant-size)
//!
//! Storage: flat contiguous Vec<f64> with stride = dim.
//! products[t] lives at data[t*dim .. (t+1)*dim].

use crate::groups::LieGroup;

pub struct GeometricPath {
    group: Box<dyn LieGroup>,
    dim: usize,
    /// Flat storage: (n+1) * dim floats. products[t] = &data[t*dim..(t+1)*dim]
    data: Vec<f64>,
    /// Reusable scratch buffer for compose_into (avoids per-append allocation).
    buf: Vec<f64>,
}

impl GeometricPath {
    /// Empty path — just the identity.
    pub fn new(group: Box<dyn LieGroup>) -> Self {
        let dim = group.dim();
        let id = group.identity();
        let buf = vec![0.0; dim];
        Self {
            group,
            dim,
            data: id,
            buf,
        }
    }

    /// Batch build from flat data (n elements × dim floats).
    pub fn from_elements(group: Box<dyn LieGroup>, elements: &[f64], dim: usize) -> Self {
        assert!(dim > 0, "dim must be > 0");
        assert!(
            elements.len() % dim == 0,
            "flat data length {} is not divisible by dim {}",
            elements.len(),
            dim
        );
        let n = elements.len() / dim;
        let id = group.identity();
        let mut data = Vec::with_capacity((n + 1) * dim);
        data.extend_from_slice(&id);
        let mut buf = vec![0.0; dim];
        for i in 0..n {
            let g = &elements[i * dim..(i + 1) * dim];
            let last = &data[i * dim..(i + 1) * dim];
            group.compose_into(last, g, &mut buf);
            data.extend_from_slice(&buf);
        }
        Self { group, dim, data, buf }
    }

    pub fn len(&self) -> usize {
        self.data.len() / self.dim - 1
    }

    /// Append one element. O(1): C_{n+1} = C_n · g. Zero allocations.
    pub fn append(&mut self, g: &[f64]) {
        let start = self.data.len() - self.dim;
        let last = &self.data[start..start + self.dim];
        self.group.compose_into(last, g, &mut self.buf);
        self.data.extend_from_slice(&self.buf);
    }

    /// Running product at position t.
    pub fn running_product(&self, t: usize) -> &[f64] {
        &self.data[t * self.dim..(t + 1) * self.dim]
    }

    /// Recover original element: g_t = C_{t-1}⁻¹ · C_t. 1-indexed, O(1).
    pub fn recover(&self, t: usize) -> Vec<f64> {
        assert!(
            t >= 1 && t <= self.len(),
            "t={} out of range [1, {}]",
            t,
            self.len()
        );
        let inv = self.group.inverse(self.running_product(t - 1));
        self.group.compose(&inv, self.running_product(t))
    }

    /// σ for sub-sequence [i+1..j]. O(1).
    /// σ ≈ 0 → that range is clean. σ > 0 → drifted by σ.
    pub fn check_range(&self, i: usize, j: usize) -> f64 {
        assert!(j <= self.len() && i < j, "invalid range [{}, {}]", i, j);
        let inv = self.group.inverse(self.running_product(i));
        let relative = self.group.compose(&inv, self.running_product(j));
        self.group.distance_from_identity(&relative)
    }

    /// σ for the whole sequence. O(1). The single-number health check.
    pub fn check_global(&self) -> f64 {
        let n = self.len();
        self.group
            .distance_from_identity(self.running_product(n))
    }

    /// The final running product — constant-size summary of everything.
    pub fn closure_element(&self) -> Vec<f64> {
        self.running_product(self.len()).to_vec()
    }

    pub fn group(&self) -> &dyn LieGroup {
        self.group.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::groups::circle::CircleGroup;
    use crate::groups::sphere::SphereGroup;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn recovery_circle() {
        let g = CircleGroup;
        let mut rng = StdRng::seed_from_u64(42);
        let elements: Vec<Vec<f64>> = (0..100).map(|_| g.random(&mut rng)).collect();

        let mut path = GeometricPath::new(Box::new(CircleGroup));
        for e in &elements {
            path.append(e);
        }

        for t in 1..=100 {
            let recovered = path.recover(t);
            assert!(
                (recovered[0] - elements[t - 1][0]).abs() < 1e-9,
                "recovery failed at t={}",
                t
            );
        }
    }

    #[test]
    fn recovery_sphere() {
        let g = SphereGroup;
        let mut rng = StdRng::seed_from_u64(42);
        let elements: Vec<Vec<f64>> = (0..100).map(|_| g.random(&mut rng)).collect();

        let mut path = GeometricPath::new(Box::new(SphereGroup));
        for e in &elements {
            path.append(e);
        }

        for t in 1..=100 {
            let recovered = path.recover(t);
            let err: f64 = recovered
                .iter()
                .zip(&elements[t - 1])
                .map(|(a, b)| (a - b).abs())
                .sum();
            assert!(err < 1e-9, "recovery failed at t={}, err={}", t, err);
        }
    }

    /// A closed sequence (product = identity) should have σ ≈ 0.
    #[test]
    fn closed_sequence_circle() {
        let g = CircleGroup;
        let mut rng = StdRng::seed_from_u64(42);
        let mut path = GeometricPath::new(Box::new(CircleGroup));

        let mut elements: Vec<Vec<f64>> = (0..99).map(|_| g.random(&mut rng)).collect();
        for e in &elements {
            path.append(e);
        }
        let n = path.len();
        let closing = g.inverse(path.running_product(n));
        path.append(&closing);
        elements.push(closing);

        assert!(
            path.check_global() < 1e-9,
            "closed sequence should have sigma ≈ 0"
        );
    }
}
