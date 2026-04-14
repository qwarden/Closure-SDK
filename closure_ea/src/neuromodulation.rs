//! Neuromodulation — the brain's slow body state.
//!
//! Three fast signals (step_pressure, valence) are computed every ingest step
//! in ThreeCell. They are precise but instantaneous: each tells you about
//! this single step, not about the regime the brain has been in for the last N steps.
//!
//! `NeuromodState` integrates those signals over time using a low-pass filter
//! and exposes two tones that record recent experience.
//!
//! ## The two tones
//!
//! **`arousal_tone`** — how strongly activated the brain has been recently.
//! Derived from `step_pressure` (the architecture's own per-step activation mass)
//! normalized by `SIGMA_BALANCE` — the topological consolidation threshold on S³.
//!
//! **`coherence_tone`** — whether recent updates have been making the brain more
//! or less internally consistent. Derived from `valence` (signed SFE change)
//! normalized by π/2 (the equatorial sigma).
//! Positive when SFE is decreasing (model improving).
//! Negative when SFE is increasing (model destabilizing).
//!
//! ## Architectural position
//!
//! `NeuromodState` is observational: it records the character of recent experience
//! but does not directly control write rate or consolidation threshold.
//! The promotion gate (criterion 6) reads `coherence_tone` to block entries
//! learned during net-negative coherence regimes from advancing to category level.
//!
//! `NeuromodState` is session-ephemeral: it is NOT persisted in `BrainState`
//! and resets to (0, 0) when a brain is restored from disk. This matches
//! the treatment of `prev_sfe` — both are running integrals that re-converge
//! from zero within a few dozen steps. The genome (the brain's permanent
//! memory) is unaffected by the reset.

use crate::verify::SIGMA_BALANCE;
use std::f64::consts::FRAC_PI_2;

/// The brain's slow body state.
///
/// Both tones live in bounded ranges:
/// - `arousal_tone` ∈ [0, 1]
/// - `coherence_tone` ∈ [−1, 1]
///
/// The integration time constant `alpha` is derived from `buffer_lifetime`:
///   `alpha = 1 − 1/buffer_lifetime`
/// so the EMA window matches the buffer's observation window — the only
/// local timescale already in the architecture.
#[derive(Clone, Copy, Debug)]
pub struct NeuromodState {
    /// Low-pass integral of input activation intensity.
    pub arousal_tone: f64,
    /// Low-pass integral of signed coherence change.
    /// Positive = recently improving self-consistency, negative = destabilizing.
    pub coherence_tone: f64,
    /// EMA time constant, derived from buffer_lifetime at construction.
    alpha: f64,
}

impl NeuromodState {
    /// Construct with integration window matched to the buffer lifetime.
    ///
    /// `alpha = 1 − 1/buffer_lifetime`. Requires `buffer_lifetime >= 2`.
    pub fn new(buffer_lifetime: usize) -> Self {
        assert!(buffer_lifetime >= 2, "buffer_lifetime must be >= 2 (alpha = 1 − 1/n requires n ≥ 2 for a non-degenerate EMA)");
        let alpha = 1.0 - 1.0 / buffer_lifetime as f64;
        Self {
            arousal_tone: 0.0,
            coherence_tone: 0.0,
            alpha,
        }
    }

    /// Update the body state from one ingest step's derived signals.
    ///
    /// Called once per `ThreeCell::ingest()`, after valence is computed.
    ///
    /// - `step_pressure` — the level-0 pressure increment computed by `ingest()`.
    ///   This is the architecture's own per-step activation mass: the sum of
    ///   prediction error above threshold, SFE above threshold, genome growth,
    ///   and closure excursion. Normalized by `SIGMA_BALANCE` (the topological
    ///   consolidation threshold) to [0, 1].
    ///
    /// - `valence` = `prev_sfe − sfe` — signed change in self-free-energy.
    ///   Positive = brain became more self-consistent this step.
    ///   Normalized by π/2 and clamped to [−1, 1].
    pub fn update(&mut self, step_pressure: f64, valence: f64) {
        // Arousal: step pressure normalized by SIGMA_BALANCE.
        // SIGMA_BALANCE = π/4 is the topological consolidation threshold on S³.
        // A step that adds one full equatorial sigma of pressure saturates arousal.
        let arousal_input = (step_pressure / SIGMA_BALANCE).clamp(0.0, 1.0);

        // Coherence: normalized signed valence.
        // A full-equatorial SFE drop (π/2) saturates the signal.
        let coherence_input = (valence / FRAC_PI_2).clamp(-1.0, 1.0);

        // Low-pass update using the per-brain derived alpha.
        self.arousal_tone = self.alpha * self.arousal_tone + (1.0 - self.alpha) * arousal_input;
        self.coherence_tone = self.alpha * self.coherence_tone + (1.0 - self.alpha) * coherence_input;
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    /// Alpha is exactly 1 − 1/buffer_lifetime for the values used in the architecture.
    #[test]
    fn alpha_derived_from_buffer_lifetime() {
        for n in [2usize, 4, 8, 16, 32] {
            let nm = NeuromodState::new(n);
            let expected = 1.0 - 1.0 / n as f64;
            assert!(
                (nm.alpha - expected).abs() < 1e-15,
                "buffer_lifetime={n}: alpha={:.16} expected={:.16}",
                nm.alpha, expected
            );
        }
    }

    /// Starting from zero state, one update with saturating pressure gives
    /// arousal = (1 − α) × 1.0 and coherence = (1 − α) × 1.0 exactly.
    #[test]
    fn one_step_update_recurrence() {
        let mut nm = NeuromodState::new(4); // alpha = 0.75
        let alpha = nm.alpha;

        // Saturating inputs: pressure = SIGMA_BALANCE → arousal_input = 1.0.
        // valence = FRAC_PI_2 → coherence_input = 1.0.
        nm.update(SIGMA_BALANCE, FRAC_PI_2);

        let expected_arousal   = (1.0 - alpha) * 1.0;
        let expected_coherence = (1.0 - alpha) * 1.0;

        assert!(
            (nm.arousal_tone - expected_arousal).abs() < 1e-15,
            "arousal after one saturating step: got {:.16} expected {:.16}",
            nm.arousal_tone, expected_arousal
        );
        assert!(
            (nm.coherence_tone - expected_coherence).abs() < 1e-15,
            "coherence after one saturating step: got {:.16} expected {:.16}",
            nm.coherence_tone, expected_coherence
        );
    }

    /// Two-step EMA recurrence: a(t+1) = α·a(t) + (1−α)·x.
    #[test]
    fn two_step_update_recurrence() {
        let mut nm = NeuromodState::new(4); // alpha = 0.75
        let alpha = nm.alpha;

        // Step 1: pressure = SIGMA_BALANCE (saturating), valence = 0.
        nm.update(SIGMA_BALANCE, 0.0);
        let a1 = nm.arousal_tone;
        assert!((a1 - (1.0 - alpha)).abs() < 1e-15, "step 1 arousal wrong");

        // Step 2: zero pressure, zero valence.
        nm.update(0.0, 0.0);
        let a2 = nm.arousal_tone;
        let expected_a2 = alpha * a1;
        assert!(
            (a2 - expected_a2).abs() < 1e-15,
            "step 2 arousal: got {:.16} expected {:.16}",
            a2, expected_a2
        );
    }
}
