//! The transient buffer — the brain's working memory.
//!
//! Holds carriers received from outside in the current input phase.
//! EMBED writes to it. The field machine reads `genome ∪ buffer` as
//! one population. Buffer entries decay autonomously over a
//! configurable lifetime unless they are promoted into the genome
//! through chunked learning.
//!
//! There is no separate primitive for the buffer — it is part of the
//! field machine's input field. This module exposes the data
//! structure and the lifetime / decay mechanics.

use serde::{Deserialize, Serialize};

/// One entry in the transient buffer.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BufferEntry {
    /// The carrier that arrived through EMBED.
    pub carrier: [f64; 4],
    /// How many cycles the entry has been in the buffer.
    pub age: usize,
    /// How many closures the entry has earned during its lifetime.
    /// Non-zero at lifetime expiry → eligible for promotion through
    /// the chunked-learning sweep (BRAIN.md §29.1).
    pub closures: usize,
    /// Strongest closure metadata accumulated against this entry
    /// during its lifetime. The chunk-boundary promotion sweep feeds
    /// these straight into `Genome::ingest`. "Strongest" is defined
    /// as the closure with the largest support; ties go to the one
    /// with the smallest σ.
    pub support: usize,
    pub closure_sigma: f64,
    pub excursion_peak: f64,
}

impl BufferEntry {
    pub fn new(carrier: [f64; 4]) -> Self {
        Self {
            carrier,
            age: 0,
            closures: 0,
            support: 0,
            closure_sigma: 0.0,
            excursion_peak: 0.0,
        }
    }

    /// Record one closure event against this entry. Increments the
    /// closure count and updates the strongest-closure metadata so the
    /// chunk-boundary promotion sweep has everything it needs to call
    /// `Genome::ingest`.
    #[inline]
    pub fn record_closure(&mut self, support: usize, closure_sigma: f64, excursion_peak: f64) {
        self.closures += 1;
        let stronger = support > self.support
            || (support == self.support && closure_sigma < self.closure_sigma);
        if stronger || self.support == 0 {
            self.support = support;
            self.closure_sigma = closure_sigma;
        }
        if excursion_peak > self.excursion_peak {
            self.excursion_peak = excursion_peak;
        }
    }
}

/// The transient buffer itself.
///
/// Append-only during a chunk; aged-out entries are surfaced through
/// `expired` and removed by `prune_expired`. The lifetime is the
/// chunk boundary the spec describes — when an entry's `age` reaches
/// `lifetime`, the chunked-learning sweep evaluates it for promotion.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Buffer {
    entries: Vec<BufferEntry>,
    lifetime: usize,
}

impl Buffer {
    /// Create an empty buffer with the given entry lifetime.
    pub fn new(lifetime: usize) -> Self {
        Self {
            entries: Vec::new(),
            lifetime,
        }
    }

    /// Number of entries currently in the buffer.
    #[inline]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Configured entry lifetime in cycles.
    #[inline]
    pub fn lifetime(&self) -> usize {
        self.lifetime
    }

    /// Update the lifetime. Useful for tuning between bootstrap
    /// (long lifetime) and ordinary input phase (short lifetime).
    pub fn set_lifetime(&mut self, lifetime: usize) {
        self.lifetime = lifetime;
    }

    /// Add a carrier to the buffer. The brain's only write path for
    /// external input.
    pub fn push(&mut self, carrier: [f64; 4]) {
        self.entries.push(BufferEntry::new(carrier));
    }

    /// Borrow the entries for reading. Used by the field machine to
    /// compose `genome ∪ buffer` populations.
    #[inline]
    pub fn entries(&self) -> &[BufferEntry] {
        &self.entries
    }

    /// Mutable borrow — used by the gradient path to record closures
    /// against entries that participated in successful round trips.
    #[inline]
    pub fn entries_mut(&mut self) -> &mut [BufferEntry] {
        &mut self.entries
    }

    /// Advance every entry by one cycle. Should be called once per
    /// Cell B tick.
    pub fn tick(&mut self) {
        for entry in &mut self.entries {
            entry.age += 1;
        }
    }

    /// Remove and return every entry whose lifetime has expired.
    /// The chunked-learning sweep evaluates each one for promotion.
    pub fn drain_expired(&mut self) -> Vec<BufferEntry> {
        let lifetime = self.lifetime;
        let mut keep = Vec::with_capacity(self.entries.len());
        let mut expired = Vec::new();
        for entry in self.entries.drain(..) {
            if entry.age >= lifetime {
                expired.push(entry);
            } else {
                keep.push(entry);
            }
        }
        self.entries = keep;
        expired
    }

    /// Clear every entry, regardless of age. Used at consolidation
    /// boundaries when the brain moves cleanly into idle phase.
    pub fn clear(&mut self) {
        self.entries.clear();
    }
}
