//! Law 8: Hierarchy Localization.
//!
//! A closure is not merely "sigma got small at time t." It is the
//! identification of the specific interval `[s, t]` whose running
//! product closes — the oscillatory packet that completed.
//!
//! Law 8 statement:
//! * Given a running product that has entered the closure basin, find
//!   the minimal interval `[s, t]` such that:
//!   * `C(s-1)` was NOT in the closure basin
//!   * `C(t)` IS in the closure basin
//!   * `[s, t]` is the minimal interval producing this closure
//!
//! Algebraic form:
//! * `C_{s→t} = history[s] · history[s+1] · ... · history[t]`
//! * Closure: `σ(C_{s→t}) ≤ threshold`
//! * The minimal interval has the latest start `s` satisfying this.
//!
//! Algorithm: exact backward scan. Build the suffix product from the
//! right; the first `s` (scanning from `t` down to `0`) where the
//! suffix closes is the answer.
//!
//! Why not binary search: `σ(interval_product(history, s, t))` is not
//! monotone in `s` on S³. Adding a carrier to the left of a closing
//! interval may or may not keep it closed. Binary search requires
//! monotonicity; the backward scan is exact regardless of composition
//! structure.
//!
//! Ported from `closure_ea/src/localization.rs`, swapping the
//! workspace `closure_rs` import for the local substrate.

use crate::sphere::{compose, sigma, IDENTITY};
use crate::verify::SIGMA_BALANCE;
use serde::{Deserialize, Serialize};

/// The localized closure interval. Result of Law 8.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LocalizedInterval {
    /// Index of the first element (0-based within the oscillation history).
    pub start: usize,
    /// Index of the last element (inclusive). This is the closure step.
    pub end: usize,
    /// Number of elements: `end - start + 1`.
    pub support: usize,
    /// Running product of `history[start..=end]`.
    pub product: [f64; 4],
    /// Sigma of the interval product.
    pub sigma: f64,
}

/// Compute the peak sigma over the sub-interval `history[start..=end]`.
///
/// The running product begins at identity at `start` and accumulates
/// each carrier in order. The peak is the maximum sigma reached during
/// that trajectory — the excursion profile of the emitted packet
/// itself, not of the surrounding oscillation.
pub fn localized_excursion_peak(history: &[[f64; 4]], start: usize, end: usize) -> f64 {
    let mut p = IDENTITY;
    let mut peak = 0.0_f64;
    for &c in &history[start..=end] {
        p = compose(&p, &c);
        let sig = sigma(&p);
        if sig > peak {
            peak = sig;
        }
    }
    peak
}

/// Find the minimal closure interval for a **balance** closure: the shortest
/// suffix whose running product's σ is within `threshold` of `SIGMA_BALANCE`
/// (π/4). Mirrors [`localize`] exactly but uses the balance basin.
pub fn localize_balance(history: &[[f64; 4]], threshold: f64) -> LocalizedInterval {
    if history.is_empty() {
        return LocalizedInterval {
            start: 0,
            end: 0,
            support: 0,
            product: IDENTITY,
            sigma: 0.0,
        };
    }

    let end = history.len() - 1;
    let mut p = IDENTITY;

    for s in (0..=end).rev() {
        p = compose(&history[s], &p);
        let sig = sigma(&p);
        if (sig - SIGMA_BALANCE).abs() <= threshold {
            return LocalizedInterval {
                start: s,
                end,
                support: end - s + 1,
                product: p,
                sigma: sig,
            };
        }
    }

    LocalizedInterval {
        start: 0,
        end,
        support: end + 1,
        product: p,
        sigma: sigma(&p),
    }
}

/// Find the minimal closure interval by exact backward scan (Law 8).
///
/// * `history` — ordered carrier sequence for this oscillation.
/// * `threshold` — identity basin bound (`σ ≤ threshold` means closed).
///
/// Precondition: the caller has confirmed the full interval closes.
/// Returns the shortest suffix of `history` that still closes.
pub fn localize(history: &[[f64; 4]], threshold: f64) -> LocalizedInterval {
    if history.is_empty() {
        return LocalizedInterval {
            start: 0,
            end: 0,
            support: 0,
            product: IDENTITY,
            sigma: 0.0,
        };
    }

    let end = history.len() - 1;

    // Build suffix product from right to left.
    // At each step s, p = history[s] · history[s+1] · ... · history[end].
    // Return the first s (latest start) where this suffix closes.
    let mut p = IDENTITY;

    for s in (0..=end).rev() {
        p = compose(&history[s], &p);
        let sig = sigma(&p);
        if sig <= threshold {
            return LocalizedInterval {
                start: s,
                end,
                support: end - s + 1,
                product: p,
                sigma: sig,
            };
        }
    }

    // Precondition was violated: no suffix closes. Return the full
    // interval. This should not happen when the caller has verified
    // full closure.
    LocalizedInterval {
        start: 0,
        end,
        support: end + 1,
        product: p,
        sigma: sigma(&p),
    }
}
