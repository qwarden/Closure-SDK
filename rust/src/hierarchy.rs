//! HierarchicalClosure — first-divergence localization.
//!
//! Given a reference (known-good) and test (maybe-corrupted) sequence:
//!
//! 1. Compare final summaries. Match → clean, done.
//! 2. Mismatch → binary search for the first divergence point.
//!    At each step, compare running products at the midpoint.
//!    Converges in ≤ log₂(n) comparisons once both running-product paths exist.
//!
//! Why this works: running products are prefix sums. If elements 1..k
//! are identical in both sequences, their running products match at k.
//! The first corrupted element causes all subsequent products to diverge.
//! Binary search finds this transition point.
//!
//! Note: localization returns the FIRST divergence point only. If multiple
//! elements are corrupted, fix the first one, then re-check to find the next.
//!
//! Storage: flat contiguous Vec<f64> with stride = dim.

use crate::groups::LieGroup;

pub struct LocalizationResult {
    /// 0-indexed position of the first corrupted element, or None if clean.
    pub index: Option<usize>,
    /// Number of comparisons performed (≤ log₂(n) + 1).
    pub checks: usize,
    /// Binary search depth.
    pub depth: usize,
}

/// Shared binary search for first divergence point.
/// `compare_at(t)` returns the distance between reference and test at position t.
/// Returns (first_divergence_index, checks, depth).
pub fn binary_search_divergence(
    shared_len: usize,
    total_ref: usize,
    total_test: usize,
    threshold: f64,
    compare_at: impl Fn(usize) -> f64,
) -> LocalizationResult {
    let mut checks = 1;
    let d_shared = compare_at(shared_len);

    if d_shared < threshold {
        if total_ref == total_test {
            return LocalizationResult {
                index: None,
                checks,
                depth: 0,
            };
        }
        return LocalizationResult {
            index: Some(shared_len),
            checks,
            depth: 0,
        };
    }

    let mut lo = 0usize;
    let mut hi = shared_len;
    let mut depth = 0;
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        checks += 1;
        depth += 1;
        if compare_at(mid) > threshold {
            hi = mid;
        } else {
            lo = mid;
        }
    }

    LocalizationResult {
        index: Some(lo),
        checks,
        depth,
    }
}

/// Stores reference running products. Compare test data against them.
pub struct HierarchicalClosure {
    group: Box<dyn LieGroup>,
    dim: usize,
    n: usize,
    /// Flat storage: (n+1) * dim floats
    ref_data: Vec<f64>,
}

impl HierarchicalClosure {
    /// Build reference from known-good elements. O(n).
    pub fn new(group: Box<dyn LieGroup>, elements: &[&[f64]]) -> Self {
        let n = elements.len();
        let dim = group.dim();
        let id = group.identity();
        let mut ref_data = Vec::with_capacity((n + 1) * dim);
        ref_data.extend_from_slice(&id);
        let mut buf = vec![0.0; dim];
        for (i, &g) in elements.iter().enumerate() {
            let last = &ref_data[i * dim..(i + 1) * dim];
            group.compose_into(last, g, &mut buf);
            ref_data.extend_from_slice(&buf);
        }
        Self {
            group,
            dim,
            n,
            ref_data,
        }
    }

    fn ref_product(&self, t: usize) -> &[f64] {
        &self.ref_data[t * self.dim..(t + 1) * self.dim]
    }

    /// Quick detection: distance between final running products.
    /// σ ≈ 0 → clean. σ > 0 → corrupted by σ.
    pub fn check(&self, test_elements: &[&[f64]]) -> f64 {
        let test_data = self.build_test_products(test_elements);
        let shared = self.n.min(test_elements.len());
        let d_shared = self.compare_at_flat(&test_data, shared);
        if test_elements.len() != self.n && d_shared < 1e-9 {
            f64::INFINITY
        } else {
            d_shared
        }
    }

    /// Find the first corrupted element.
    /// O(log n) comparisons × O(n) compose per comparison in worst case,
    /// but caches partial products so the total work is bounded by O(n)
    /// compose operations rather than O(n) upfront + O(log n) lookups.
    pub fn localize(&self, test_elements: &[&[f64]], threshold: f64) -> LocalizationResult {
        let shared = self.n.min(test_elements.len());
        let dim = self.dim;

        // Lazy cache: only compute running products at positions we visit.
        // Cache maps position → running product (flat dim-sized slice).
        let mut cache: std::collections::HashMap<usize, Vec<f64>> =
            std::collections::HashMap::new();
        cache.insert(0, self.group.identity());

        let compute_at = |cache: &mut std::collections::HashMap<usize, Vec<f64>>, t: usize| -> f64 {
            if !cache.contains_key(&t) {
                // Find the nearest cached position before t
                let mut start = 0;
                for &k in cache.keys() {
                    if k <= t && k > start {
                        start = k;
                    }
                }
                // Build forward from start to t
                let mut running = cache[&start].clone();
                let mut buf = vec![0.0; dim];
                for i in start..t {
                    self.group.compose_into(&running, test_elements[i], &mut buf);
                    running.copy_from_slice(&buf);
                }
                cache.insert(t, running);
            }
            let test_prod = &cache[&t];
            let ref_prod = self.ref_product(t);
            let inv = self.group.inverse(ref_prod);
            let rel = self.group.compose(&inv, test_prod);
            self.group.distance_from_identity(&rel)
        };

        // Initial check at shared endpoint
        let mut checks = 1;
        let d_shared = compute_at(&mut cache, shared);

        if d_shared < threshold {
            if self.n == test_elements.len() {
                return LocalizationResult { index: None, checks, depth: 0 };
            }
            return LocalizationResult { index: Some(shared), checks, depth: 0 };
        }

        // Binary search
        let mut lo = 0usize;
        let mut hi = shared;
        let mut depth = 0;
        while hi - lo > 1 {
            let mid = (lo + hi) / 2;
            checks += 1;
            depth += 1;
            if compute_at(&mut cache, mid) > threshold {
                hi = mid;
            } else {
                lo = mid;
            }
        }

        LocalizationResult { index: Some(lo), checks, depth }
    }

    pub fn len(&self) -> usize {
        self.n
    }

    /// Build flat running products for test data. O(n).
    fn build_test_products(&self, elements: &[&[f64]]) -> Vec<f64> {
        let n = elements.len();
        let mut data = Vec::with_capacity((n + 1) * self.dim);
        let id = self.group.identity();
        data.extend_from_slice(&id);
        let mut buf = vec![0.0; self.dim];
        for (i, &g) in elements.iter().enumerate() {
            let last = &data[i * self.dim..(i + 1) * self.dim];
            self.group.compose_into(last, g, &mut buf);
            data.extend_from_slice(&buf);
        }
        data
    }

    /// Distance between reference and test running products at position t.
    fn compare_at_flat(&self, test_data: &[f64], t: usize) -> f64 {
        let ref_prod = self.ref_product(t);
        let test_prod = &test_data[t * self.dim..(t + 1) * self.dim];
        let inv = self.group.inverse(ref_prod);
        let relative = self.group.compose(&inv, test_prod);
        self.group.distance_from_identity(&relative)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::groups::circle::CircleGroup;
    use crate::groups::sphere::SphereGroup;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    /// Build n elements whose product = identity.
    fn make_closed_sequence(group: &dyn LieGroup, n: usize, seed: u64) -> Vec<Vec<f64>> {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut elements: Vec<Vec<f64>> = (0..n - 1).map(|_| group.random(&mut rng)).collect();
        let mut prod = group.identity();
        for e in &elements {
            prod = group.compose(&prod, e);
        }
        elements.push(group.inverse(&prod));
        elements
    }

    #[test]
    fn localize_circle() {
        let g = CircleGroup;
        let elements = make_closed_sequence(&g, 200, 42);
        let refs: Vec<&[f64]> = elements.iter().map(|v| v.as_slice()).collect();
        let hc = HierarchicalClosure::new(Box::new(CircleGroup), &refs);

        for corrupt_idx in [0, 50, 99, 150, 199] {
            let mut test = elements.clone();
            test[corrupt_idx] = vec![test[corrupt_idx][0] + 0.3];
            let test_refs: Vec<&[f64]> = test.iter().map(|v| v.as_slice()).collect();
            let result = hc.localize(&test_refs, 1e-6);
            assert_eq!(
                result.index,
                Some(corrupt_idx),
                "failed to localize at {corrupt_idx}"
            );
            assert!(result.checks <= 20, "too many checks: {}", result.checks);
        }
    }

    #[test]
    fn localize_sphere() {
        let g = SphereGroup;
        let elements = make_closed_sequence(&g, 1000, 42);
        let refs: Vec<&[f64]> = elements.iter().map(|v| v.as_slice()).collect();
        let hc = HierarchicalClosure::new(Box::new(SphereGroup), &refs);

        let corrupt_idx = 500;
        let mut test = elements.clone();
        let eps: f64 = 0.1;
        let perturbation = vec![eps.cos(), eps.sin(), 0.0, 0.0];
        test[corrupt_idx] = g.compose(&test[corrupt_idx], &perturbation);
        let test_refs: Vec<&[f64]> = test.iter().map(|v| v.as_slice()).collect();
        let result = hc.localize(&test_refs, 1e-6);
        assert_eq!(result.index, Some(corrupt_idx));
        assert!(
            result.checks <= 15,
            "expected ~11 checks, got {}",
            result.checks
        );
    }

    #[test]
    fn clean_passes() {
        let g = CircleGroup;
        let elements = make_closed_sequence(&g, 100, 42);
        let refs: Vec<&[f64]> = elements.iter().map(|v| v.as_slice()).collect();
        let hc = HierarchicalClosure::new(Box::new(CircleGroup), &refs);
        let sigma = hc.check(&refs);
        assert!(
            sigma < 1e-9,
            "clean sequence should pass, got sigma={sigma}"
        );
    }
}
