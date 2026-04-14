//! Closure ea — the digital brain.
//!
//! Self-contained Rust crate. No external workspace dependencies.
//! The brain is S³, six irreducible primitives, two machines, one
//! verb (VERIFY), and a transient buffer. Phase 4 wires them through
//! the three-cell diabolo on the critical line — see
//! [`three_cell::ThreeCell`].
//!
//! ## Modules
//!
//! **Substrate (Phase 1).**
//!
//! - [`sphere`] — S³ primitives: COMPOSE (Hamilton product), INVERSE
//!   (star involution), SIGMA (geodesic distance from identity), SLERP, IDENTITY.
//! - [`hopf`] — Hopf decomposition of S³ → S² × S¹.
//! - [`embed`] — EMBED: bytes / f64 / i64 → carrier on S³.
//! - [`verify`] — VERIFY: the A?=A verb. Returns a `VerificationEvent`
//!   with σ, Hopf decomposition, and the closure kind (identity / balance / open).
//!
//! **Carriers and words (Phase 2).**
//!
//! - [`carrier`] — verification cells with plane, phase, turns, sheet,
//!   coherence width, and coupling state.
//! - [`word`] — ordered sequences of carriers (numbers, programs, queries,
//!   memory node identities).
//!
//! **Field machine (Phase 3).**
//!
//! - [`buffer`] — transient input buffer. EMBED writes here.
//! - [`genome`] — DNA + epigenetic layers, Law 3 ingest, nearest neighbor.
//!   Bootstrap seeds DNA; chunked learning writes epigenetic.
//! - [`field`] — RESONATE and ZREAD over `genome ∪ buffer`.
//!
//! **Hierarchy and consolidation (Phases 7–8).**
//!
//! - [`localization`] — Law 8: minimal closure interval via backward scan.
//! - [`hierarchy`] — recursive closure detection, emission, genome writes.
//! - [`consolidation`] — runtime reorganization over the epigenetic layer
//!   (merge, prune, reorganize). Never touches DNA.
//!
//! **Diabolo (Phase 4).**
//!
//! - [`three_cell`] — the diabolo on the critical line. One `ingest` verb
//!   that runs the seven-step loop: buffer → ZREAD → RESONATE → VERIFY →
//!   Cell A composition → closure detection → Cell C integration →
//!   hierarchy emission → genome ingest.
//!
//! **Curriculum and generation (Phase 5).**
//!
//! - [`teach`] — directed curriculum drivers. (input, target) pairs are
//!   fed through the same `ingest` runtime and measured by σ-gap.

pub mod buffer;
pub mod carrier;
pub mod consolidation;
pub mod embed;
pub mod execution;
pub mod field;
pub mod genome;
pub mod hierarchy;
pub mod hopf;
pub mod localization;
pub mod neuromodulation;
pub mod sphere;
pub mod teach;
pub mod three_cell;
pub mod verify;
pub mod zeta;

// Re-exports for the crate's public surface.
pub use buffer::{Buffer, BufferEntry};
pub use carrier::{
    identity_geometry, CarrierObservation, CouplingState, EulerPlane, NeighborCoupling,
    PlaneRelation, SheetRelation, TwistSheet, VerificationCell, VerificationLandmark,
};
pub use consolidation::{
    collect_promotion_candidates, consolidate, ConsolidationReport, PromotionCandidate,
    PROMOTION_MIN_ACTIVATIONS, PROMOTION_MIN_COHERENCE, PROMOTION_MIN_SALIENCE,
};
pub use embed::{
    bytes_to_sphere4, bytes_to_sphere4_with_parity, domain_embed, domain_embed_with_parity,
    f64_to_sphere4, f64_to_sphere4_with_parity, i64_to_sphere4, i64_to_sphere4_with_parity,
    parity_gate, parity_phase_gate, semantic_base_from_bytes, MusicEncoder, Vocabulary,
    PARITY_CARRIER,
};
pub use execution::{
    orbit_generator, FractranMachine, FractranState, Fraction, MinskyInstr, MinskyMachine,
    MinskyState, OrbitRuntime, OrbitSeed,
};
pub use field::{
    resonate, resonate_channel, resonate_channel_with_mode, resonate_response, resonate_spectrum,
    resonate_with_mode, zread, zread_at_query, zread_at_query_channel,
    zread_at_query_channel_with_mode, zread_at_query_with_mode, zread_response, PopulationSource,
    ResonanceHit,
};
pub use genome::{
    Genome, GenomeConfig, GenomeEntry, Layer, StoreOutcome, BKT_THRESHOLD, CO_RESONANCE_FLOOR,
    DOBRUSHIN_DELTA_2, DOBRUSHIN_DELTA_3, PRIME_2_COUPLING, PRIME_3_COUPLING, PRIME_5_COUPLING,
    ZREAD_T_MIN,
};
pub use hierarchy::{ClosureEvent, ClosureKind, ClosureLevel, ClosureRole, Hierarchy};
pub use hopf::{
    address_distance, carrier_from_hopf, circular_distance, coupling_from_gap,
    decompose as hopf_decompose, dominant_vector_channel, semantic_frame, AddressMode,
    HopfChannel, SemanticFrame, VectorChannel, UNKNOWN_AXIS, SALIENCE_AXIS, TOTAL_AXIS,
};
pub use localization::{localize, localized_excursion_peak, LocalizedInterval};
pub use neuromodulation::NeuromodState;
pub use sphere::{compose, inverse, sigma, slerp, IDENTITY};
pub use teach::{
    evaluate_accuracy, run_curriculum_passes, teach, teach_batch, teach_silent, CurriculumReport,
    CurriculumTrace, CurriculumWindow, WindowReport,
};
pub use three_cell::{
    BrainState, EvaluationReport, PendingPrediction, PredictionFeedback, PredictionSource,
    RuntimeConsolidationReport, SequenceOutcome, Step, ThreeCell, UpdateReport,
};
pub use verify::{
    closure_kind, hopf_dominance, verify, verify_against_identity, verify_with_tolerance,
    ClosureKind as VerifyClosureKind, HopfDominance, VerificationEvent, DEFAULT_TOLERANCE,
    SIGMA_BALANCE, SIGMA_IDENTITY,
};

#[cfg(test)]
mod tests {
    //! Native basic-operations tests, driven through the brain's
    //! own bootstrap API (`ThreeCell::bootstrap_single_orbit`,
    //! `drive_sequence`, `evaluate_product`). No hardcoded examples.
    //!
    //! For each (ε generator, axis) test case the brain:
    //!
    //!   1. Bootstraps the single ε orbit (CP2/CP3) — Cell A closes,
    //!      orbit positions land in the genome.
    //!   2. Verifies counting: every ε^k slot is RESONATE-reachable
    //!      using the brain's own `evaluate` (compose + lookup).
    //!   3. Verifies addition for every (a, b) pair: composing two
    //!      orbit positions externally and looking the result up
    //!      via `evaluate_product` lands on the (a+b mod n) slot.
    //!   4. Verifies subtraction the same way using the star
    //!      involution (`compose(a, star(b))`).
    //!   5. Asserts persistence: the genome size and the boot
    //!      identity are unchanged after every read.
    //!
    //! For perpendicular-pair test cases the brain additionally:
    //!
    //!   6. Bootstraps a second perpendicular generator δ (CP4) —
    //!      a second orbit lands in the genome.
    //!   7. Drives a cross-orbit closing sequence (CP5) — Cell A
    //!      catches the (ε·δ)^k = -identity cycle and persists the
    //!      closing window's partial products as cross-orbit lattice
    //!      points in the genome.
    //!   8. Verifies that ε^a · δ^b for several (a, b) pairs
    //!      RESONATEs in the lattice, and that ε·δ ≠ δ·ε
    //!      (multiplication on S³ is non-commutative).
    //!
    //! All test cases share one parameterized helper. The brain's
    //! API is exercised, not the example layer.

    use super::*;
    use std::f64::consts::PI;

    fn axis_rotation(angle: f64, axis: usize) -> [f64; 4] {
        let mut q = [0.0; 4];
        q[0] = (angle / 2.0).cos();
        q[1 + axis] = (angle / 2.0).sin();
        q
    }

    fn make_brain(closure_threshold: f64, novelty: f64) -> ThreeCell {
        ThreeCell::new(
            closure_threshold,
            closure_threshold,
            128,
            GenomeConfig {
                reinforce_threshold: novelty * 0.1,
                novelty_threshold: novelty,
                merge_threshold: novelty * 0.1,
                co_resonance_merge_threshold: 0.0,
            },
        )
    }

    #[test]
    fn consolidation_preserves_fresh_single_epigenetic_entry() {
        let mut genome = Genome::new(GenomeConfig::defaults());
        let q = axis_rotation(0.7, 0);
        genome.ingest(&q, 1, sigma(&q), sigma(&q));

        let report = consolidate(&mut genome);

        assert_eq!(
            genome.len(),
            1,
            "a fresh single epigenetic memory is alive until BKT evidence exists"
        );
        assert_eq!(genome.entries[0].layer, Layer::Epigenetic);
        assert_eq!(report.pruned, 0);
        assert_eq!(report.bkt_alive_ratio, 1.0);
    }

    #[test]
    fn genome_coverage_load_uses_antipodal_runtime_volume() {
        let r = 0.5;
        let mut genome = Genome::new(GenomeConfig {
            reinforce_threshold: 0.01,
            novelty_threshold: r,
            merge_threshold: 0.01,
            co_resonance_merge_threshold: 0.0,
        });
        let q = axis_rotation(1.7, 0);
        genome.ingest(&q, 1, sigma(&q), sigma(&q));

        let ball_vol = 2.0 * std::f64::consts::PI * (r - r.sin() * r.cos());
        let rp3_vol = std::f64::consts::PI * std::f64::consts::PI;
        let expected = ball_vol / rp3_vol;

        assert!(
            (genome.genome_coverage_load() - expected).abs() < 1e-12,
            "coverage load must use RP3 volume because sigma identifies q with -q"
        );
    }

    #[test]
    fn critical_period_closes_only_after_saturated_load_and_zero_births() {
        let mut genome = Genome::new(GenomeConfig {
            reinforce_threshold: 0.1,
            novelty_threshold: std::f64::consts::FRAC_PI_2,
            merge_threshold: 0.1,
            co_resonance_merge_threshold: 0.0,
        });
        let q = axis_rotation(0.7, 0);
        genome.ingest(&q, 1, sigma(&q), sigma(&q));

        assert!(genome.genome_coverage_load() >= 1.0);
        assert!(!genome.critical_period_closed(3));

        for _ in 0..3 {
            genome.ingest(&q, 1, sigma(&q), sigma(&q));
        }

        assert_eq!(genome.creation_rate(3), 0.0);
        assert!(genome.critical_period_closed(3));
        assert!(!genome.critical_period_closed(0));
    }

    /// CP2 + CP3 + addition + subtraction over a single orbit, driven
    /// through `ThreeCell::bootstrap_single_orbit` and `evaluate_product`.
    fn assert_counting_and_arithmetic(eps_angle: f64, axis: usize) {
        let cell_a_threshold = 0.05_f64.min(eps_angle / 8.0);
        let novelty = (eps_angle * 0.45).min(0.10);
        let mut brain = make_brain(cell_a_threshold, novelty);

        let epsilon = axis_rotation(eps_angle, axis);

        // ── Native bootstrap of one orbit. ──────────────────────
        let orbit_len = brain
            .bootstrap_single_orbit(&epsilon, 64)
            .unwrap_or_else(|| {
                panic!(
                    "ε orbit failed to close in 64 steps for angle {}",
                    eps_angle
                )
            });

        let post_bootstrap = brain.hierarchy().genomes[0].len();
        assert!(
            post_bootstrap >= orbit_len - 1,
            "expected ≥ {} entries after bootstrap, got {}",
            orbit_len - 1,
            post_bootstrap
        );

        // ── Counting: every ε^k must RESONATE through evaluate. ─
        for k in 1..orbit_len {
            let target = axis_rotation(k as f64 * eps_angle, axis);
            if sigma(&target) < 1e-4 {
                continue; // skipped near-identity wrap
            }
            let hit = brain
                .evaluate(&[target])
                .expect("every ε^k should RESONATE");
            assert!(
                hit.gap < 1e-4,
                "RESONATE on ε^{} should land at gap 0 (saw {}, ε angle = {})",
                k,
                hit.gap,
                eps_angle
            );
        }

        // ── Addition: ε^a · ε^b = ε^((a+b) mod n) for every pair. ─
        for a in 1..orbit_len {
            for b in 1..orbit_len {
                let eps_a = axis_rotation(a as f64 * eps_angle, axis);
                let eps_b = axis_rotation(b as f64 * eps_angle, axis);
                let sum = compose(&eps_a, &eps_b);
                if sigma(&sum) < 1e-4 {
                    continue;
                }
                let hit = brain
                    .evaluate_product(&eps_a, &eps_b)
                    .expect("addition should produce a hit");
                assert!(
                    hit.gap < 1e-4,
                    "ε^{} · ε^{} → gap {} (ε angle {})",
                    a,
                    b,
                    hit.gap,
                    eps_angle
                );

                // Cross-check the geometric label.
                let expected = axis_rotation(((a + b) % orbit_len) as f64 * eps_angle, axis);
                if sigma(&expected) > 1e-4 {
                    let label_gap = sigma(&compose(&hit.carrier, &inverse(&expected)));
                    assert!(
                        label_gap < 1e-4,
                        "ε^{} · ε^{} should land at ε^{} (label gap = {})",
                        a,
                        b,
                        (a + b) % orbit_len,
                        label_gap
                    );
                }
            }
        }

        // ── Subtraction via the star involution. ────────────────
        for a in 1..orbit_len {
            for b in 1..orbit_len {
                let eps_a = axis_rotation(a as f64 * eps_angle, axis);
                let eps_b = axis_rotation(b as f64 * eps_angle, axis);
                let diff = compose(&eps_a, &inverse(&eps_b));
                if sigma(&diff) < 1e-4 {
                    continue;
                }
                let hit = brain
                    .evaluate(&[eps_a, inverse(&eps_b)])
                    .expect("subtraction should produce a hit");
                assert!(
                    hit.gap < 1e-4,
                    "ε^{} · ε^(-{}) → gap {} (ε angle {})",
                    a,
                    b,
                    hit.gap,
                    eps_angle
                );
            }
        }

        // ── Persistence + DNA invariance. ───────────────────────
        let final_size = brain.hierarchy().genomes[0].len();
        assert_eq!(
            final_size, post_bootstrap,
            "evaluate / RESONATE must not mutate the genome"
        );
        assert_eq!(
            brain.hierarchy().genomes[0].entries[0].address.geometry(),
            IDENTITY
        );
        assert_eq!(brain.hierarchy().genomes[0].entries[0].layer, Layer::Dna);
    }

    /// CP4 + CP5 across a perpendicular pair, driven through the
    /// native API. The closing sequence for the cross-orbit Cell A
    /// closure is computed for the input pair: (ε·δ)^order = -identity,
    /// and the alternation length is `2 * order`.
    fn assert_perpendicular_pair_lattice(
        eps_angle: f64,
        delta_angle: f64,
        eps_axis: usize,
        delta_axis: usize,
    ) {
        assert_ne!(eps_axis, delta_axis, "axes must be perpendicular");

        let smaller = eps_angle.min(delta_angle);
        let cell_a_threshold = 0.05_f64.min(smaller / 8.0);
        let novelty = 0.10;
        let mut brain = make_brain(cell_a_threshold, novelty);

        let epsilon = axis_rotation(eps_angle, eps_axis);
        let delta = axis_rotation(delta_angle, delta_axis);

        // ── CP3: bootstrap ε orbit. ─────────────────────────────
        let n_eps = brain
            .bootstrap_single_orbit(&epsilon, 64)
            .expect("ε orbit must close");

        // ── CP4: bootstrap perpendicular δ orbit. ───────────────
        let n_del = brain
            .bootstrap_single_orbit(&delta, 64)
            .expect("δ orbit must close");

        // Both single orbits live in the genome.
        for k in 1..n_eps {
            let q = axis_rotation(k as f64 * eps_angle, eps_axis);
            if sigma(&q) < 1e-4 {
                continue;
            }
            let hit = brain.evaluate(&[q]).expect("ε^k must hit");
            assert!(hit.gap < 1e-4);
        }
        for l in 1..n_del {
            let q = axis_rotation(l as f64 * delta_angle, delta_axis);
            if sigma(&q) < 1e-4 {
                continue;
            }
            let hit = brain.evaluate(&[q]).expect("δ^l must hit");
            assert!(hit.gap < 1e-4);
        }

        // ── CP5: drive cross-orbit closing sequences through Cell A.
        //
        // The brain has the same Cell A closure detector that built
        // the single orbits in CP3. To populate cross-orbit lattice
        // points we need closing sequences whose **partial products**
        // include the lattice points we care about. For an arbitrary
        // perpendicular pair (ε, δ) we walk the standard alternating
        // patterns and compute their orders in SU(2); each order n
        // gives a length-2n sequence whose Cell A closure deposits
        // that pattern's partial products into the genome.
        //
        // Helper: compute the SU(2) order of a quaternion (smallest
        // k such that q^k has σ ≈ 0). Returns None if no finite
        // order is found within `max_order`.
        fn su2_order(q: &[f64; 4], max_order: usize) -> Option<usize> {
            let mut power = *q;
            for k in 1..=max_order {
                if sigma(&power) < 1e-6 {
                    return Some(k);
                }
                power = compose(&power, q);
            }
            None
        }

        // Drive a periodic alternation [a, b, a, b, …] of length
        // 2 · order(a · b). Returns whether Cell A fired a closure.
        let mut drive_alternation = |a: [f64; 4], b: [f64; 4]| -> bool {
            let prod = compose(&a, &b);
            match su2_order(&prod, 32) {
                Some(order) => {
                    let mut seq = Vec::with_capacity(2 * order);
                    for _ in 0..order {
                        seq.push(a);
                        seq.push(b);
                    }
                    brain.drive_sequence(&seq).closures_fired > 0
                }
                None => false,
            }
        };

        // (a) [ε, δ] alternation — closes at (ε·δ)^order = -id.
        //     Partial products include ε·δ, ε·δ·ε, (ε·δ)², …
        assert!(
            drive_alternation(epsilon, delta),
            "ε·δ must have finite order"
        );

        // (b) [δ, ε] alternation — closes at (δ·ε)^order = -id.
        //     Different partial products because ε·δ ≠ δ·ε.
        drive_alternation(delta, epsilon);

        // (c) [ε², δ²] alternation — when ε² and δ² have a finite-
        //     order product, this lifts ε², δ², ε²·δ, ε²·δ², etc.
        //     into the genome. For (π/2, π/2) perpendicular it
        //     closes at (ε²·δ²)² = -id.
        let eps_sq = compose(&epsilon, &epsilon);
        let del_sq = compose(&delta, &delta);
        drive_alternation(eps_sq, del_sq);

        // ── Cross-orbit lookups: every probe must land on a
        //    stored slot. Probes are exactly the partial products
        //    that the alternation drivers above guarantee. ──────
        let cross_probes: Vec<(&str, [f64; 4])> = vec![
            ("ε · δ", compose(&epsilon, &delta)),
            ("δ · ε", compose(&delta, &epsilon)),
            ("ε · δ · ε", compose(&compose(&epsilon, &delta), &epsilon)),
            ("(ε·δ)²", {
                let ed = compose(&epsilon, &delta);
                compose(&ed, &ed)
            }),
            ("δ · ε · δ", compose(&compose(&delta, &epsilon), &delta)),
            ("(δ·ε)²", {
                let de = compose(&delta, &epsilon);
                compose(&de, &de)
            }),
        ];
        for (label, q) in cross_probes {
            let hit = brain
                .evaluate(&[q])
                .unwrap_or_else(|| panic!("{} must produce a hit", label));
            assert!(
                hit.gap < 1e-4,
                "{} should land on a stored slot (gap = {}, ε={}, δ={})",
                label,
                hit.gap,
                eps_angle,
                delta_angle
            );
        }

        // Non-commutativity sanity: ε·δ and δ·ε live at different
        // points and therefore at different slots.
        let hit_ed = brain.evaluate(&[epsilon, delta]).unwrap();
        let hit_de = brain.evaluate(&[delta, epsilon]).unwrap();
        assert_ne!(
            hit_ed.index, hit_de.index,
            "ε·δ and δ·ε must occupy different slots (non-commutative algebra)"
        );

        // DNA invariance.
        assert_eq!(
            brain.hierarchy().genomes[0].entries[0].address.geometry(),
            IDENTITY
        );
        assert_eq!(brain.hierarchy().genomes[0].entries[0].layer, Layer::Dna);
    }

    #[test]
    fn brain_counts_and_does_arithmetic_for_arbitrary_orbits() {
        // Multiple generator angles and axes — the brain's bootstrap
        // path handles each through the same native API.
        let single_orbit_cases: &[(f64, usize)] = &[
            (PI / 2.0, 0), //  4-orbit on x
            (PI / 3.0, 0), //  6-orbit on x
            (PI / 4.0, 0), //  8-orbit on x
            (PI / 4.0, 1), //  8-orbit on y
            (PI / 6.0, 0), // 12-orbit on x
            (PI / 4.0, 2), //  8-orbit on z
        ];
        for &(angle, axis) in single_orbit_cases {
            assert_counting_and_arithmetic(angle, axis);
        }
    }

    #[test]
    fn brain_handles_perpendicular_generators_and_cross_orbit_lattice() {
        // Pairs that produce finite SU(2) groups (so cross-orbit
        // Cell A closures fire deterministically).
        //
        //   π/2 around x  +  π/2 around y  → binary octahedral
        //   π/2 around x  +  π/2 around z  → binary octahedral (different axis)
        //   π/2 around y  +  π/2 around z  → binary octahedral (different axis)
        let pair_cases: &[(f64, f64, usize, usize)] = &[
            (PI / 2.0, PI / 2.0, 0, 1),
            (PI / 2.0, PI / 2.0, 0, 2),
            (PI / 2.0, PI / 2.0, 1, 2),
        ];
        for &(ea, da, ex, dx) in pair_cases {
            assert_perpendicular_pair_lattice(ea, da, ex, dx);
        }
    }

    /// CP6: bootstrap a 12-orbit through the brain's native API,
    /// then call the brain's `classify_orbit_irreducibility`
    /// method and verify the classification matches BRAIN.md §19's
    /// example exactly: in a length-12 orbit, positions 2, 3, 5,
    /// 7, 11 are prime; 4, 6, 8, 9, 10 are composite.
    ///
    /// The classification is the brain asking the geometric
    /// primality test (`zeta::is_prime_geometric`) for every orbit
    /// position it holds. The test uses Hurwitz integer
    /// quaternions and the integer Hamilton product — multiplication
    /// on S³, no `%`, no trial division by remainder.
    #[test]
    fn brain_classifies_orbit_irreducibility_cp6() {
        // ε = 2π/12 = π/6 around x → orbit length 12.
        let theta = PI / 6.0;
        let epsilon = axis_rotation(theta, 0);

        let cell_a_threshold = 0.05_f64.min(theta / 8.0);
        let novelty = (theta * 0.45).min(0.10);
        let mut brain = make_brain(cell_a_threshold, novelty);

        let orbit_length = brain
            .bootstrap_single_orbit(&epsilon, 64)
            .expect("12-orbit must close");
        assert_eq!(orbit_length, 12, "ε = π/6 must close after 12 ingestions");

        // Genome should hold 14 entries: 3 DNA anchors (identity,
        // equatorial, prime-3) plus ε^1..ε^11 from Cell A persistence
        // (ε^12 = −identity is skipped, σ ≈ 0).
        let genome_size = brain.hierarchy().genomes[0].len();
        assert_eq!(
            genome_size, 14,
            "12-orbit bootstrap should populate 14 slots (3 DNA + 11 orbit), got {}",
            genome_size
        );

        // CP6: classify each orbit position 2..11 as prime/composite
        // through the brain's native API.
        let classification = brain.classify_orbit_irreducibility();

        // 11 epigenetic entries (ε^1..ε^11) → orbit positions 2..12.
        let expected = vec![
            (2u64, true),   // 2 — prime
            (3u64, true),   // 3 — prime
            (4u64, false),  // 4 = 2 · 2
            (5u64, true),   // 5 — prime
            (6u64, false),  // 6 = 2 · 3
            (7u64, true),   // 7 — prime
            (8u64, false),  // 8 = 2³
            (9u64, false),  // 9 = 3²
            (10u64, false), // 10 = 2 · 5
            (11u64, true),  // 11 — prime
            (12u64, false), // 12 = 2² · 3
        ];
        assert_eq!(
            classification, expected,
            "CP6 classification must match BRAIN.md §19 example"
        );

        // Witness check: every composite must come with a concrete
        // Hurwitz factorization that the brain can produce on demand
        // through `explain_orbit_irreducibility`.
        let witnesses = brain.explain_orbit_irreducibility();
        for (k, factor_opt) in &witnesses {
            let is_prime_in_classification = classification
                .iter()
                .find(|(kk, _)| kk == k)
                .map(|(_, p)| *p)
                .unwrap();
            if is_prime_in_classification {
                assert!(
                    factor_opt.is_none(),
                    "{} is classified as prime but explain produced a factor",
                    k
                );
            } else {
                let f = factor_opt.expect("composite must have a witness");
                assert_eq!(f.a * f.b, *k, "factorization must multiply to k");
                assert!(f.a >= 2 && f.b >= 2, "factors must be > 1");
                assert!(
                    f.a < *k && f.b < *k,
                    "factors must be smaller than the carrier"
                );
            }
        }

        // The brain must have learned each orbit position as an
        // Epigenetic entry. We check the Epigenetic layer directly
        // so DNA anchors (which may share a carrier with some ε^k)
        // do not shadow the learned orbit positions.
        for k in 1..12 {
            let target = axis_rotation(k as f64 * theta, 0);
            if sigma(&target) < 1e-4 {
                continue; // ε^12 = identity, already at σ=0
            }
            let (_, gap) = brain.hierarchy().genomes[0]
                .nearest_in_layer(&target, Layer::Epigenetic)
                .unwrap_or_else(|| panic!("ε^{} must have an Epigenetic entry", k));
            assert!(
                gap < 1e-4,
                "ε^{} Epigenetic entry must be within σ < 1e-4 of its orbit carrier (gap = {})",
                k,
                gap
            );
        }

        // Persistence + DNA invariance after CP6.
        assert_eq!(brain.hierarchy().genomes[0].len(), 14); // 3 DNA + 11 orbit
        assert_eq!(
            brain.hierarchy().genomes[0].entries[0].address.geometry(),
            IDENTITY
        );
        assert_eq!(brain.hierarchy().genomes[0].entries[0].layer, Layer::Dna);
    }

    // ── Manifold-native soft attention (no probability wrapper) ──
    //
    // Hard attention is `resonate` — argmax over the geodesic
    // distance landscape. The ranked landscape itself (without any
    // synthetic weights vector) is `resonate_spectrum`. Soft
    // attention is `zread_at_query` — a population-wide ZREAD
    // where each entry's contribution is SLERPed from IDENTITY
    // toward its value by a fraction equal to the real part of
    // the gap quaternion. Every step is a substrate primitive:
    // compose, inverse, slerp. No softmax, no sum-to-1, no
    // ℝ-valued probability distribution.

    #[test]
    fn soft_attention_is_manifold_native() {
        // Bootstrap a 12-orbit so the population has structure.
        let theta = PI / 6.0;
        let epsilon = axis_rotation(theta, 0);
        let cell_a_threshold = 0.05_f64.min(theta / 8.0);
        let novelty = (theta * 0.45).min(0.10);
        let mut brain = make_brain(cell_a_threshold, novelty);
        brain
            .bootstrap_single_orbit(&epsilon, 64)
            .expect("orbit must close");

        // Drain the buffer so the population is the genome alone.
        // The bootstrap leaves 12 raw ε ingestions in the buffer
        // which would pull the soft-attention integration toward
        // ε^1; this test isolates the algorithm from that bias.
        // (The "with buffer" case is exercised separately.)
        brain.buffer_mut().clear();

        // ── Hard attention: argmax over the gap landscape. ──────
        let query = axis_rotation(5.0 * theta, 0);
        let hard = resonate(&query, &brain.hierarchy().genomes[0], brain.buffer())
            .expect("resonate must produce a hit");
        assert_eq!(hard.source, PopulationSource::Genome);
        // Slots 0..2 are DNA anchors (identity, equatorial, prime-3).
        // ε^k lands at slot 2 + k. ε^5 → slot 7.
        assert_eq!(hard.index, 7);
        assert!(hard.gap < 1e-6);

        // ── Ranked landscape — pure geometric observation. ──────
        let spectrum = resonate_spectrum(&query, &brain.hierarchy().genomes[0], brain.buffer());
        assert_eq!(
            spectrum.len(),
            brain.hierarchy().genomes[0].len(),
            "spectrum reads the population (genome only after drain)"
        );
        assert_eq!(spectrum[0].index, 7); // ε^5 at slot 7 (3 DNA + 4 orbit steps)
        assert!(spectrum[0].gap < 1e-6);
        for w in spectrum.windows(2) {
            assert!(w[0].gap <= w[1].gap, "spectrum must be sorted by gap");
        }

        // ── Soft attention: manifold-native ZREAD at the query. ──
        //
        // Every step inside `zread_at_query` is a substrate op:
        //   - compose / inverse for the gap quaternion
        //   - reading `cycle[0]` (the real part of the cycle) as
        //     the closeness measure (= cos(½·θ_so3), no synthesis)
        //   - slerp(IDENTITY, entry.value, |cycle[0]|) for the
        //     per-entry contribution
        //   - compose for the population integration
        //
        // No softmax, no sum-to-1, no division. The result is a
        // single carrier on S³ that integrates the whole population
        // by proximity to the query.
        let attended = zread_at_query(&query, &brain.hierarchy().genomes[0], brain.buffer());

        // ── Invariant 1: result lives on S³. ────────────────────
        // Every COMPOSE renormalizes; the manifold enforces
        // unit norm without any explicit projection step.
        let norm = (attended[0] * attended[0]
            + attended[1] * attended[1]
            + attended[2] * attended[2]
            + attended[3] * attended[3])
            .sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-6,
            "soft attention output must lie on S³ (norm = {})",
            norm
        );

        // ── Invariant 2: geometric monotonicity. ────────────────
        // The integration is proximity-weighted, so the result
        // must drift toward the populated region. Specifically,
        // the soft-attention output must be closer to the query
        // than to the farthest entry in the spectrum.
        let query_gap_to_attended = sigma(&compose(&query, &inverse(&attended)));
        let farthest = spectrum.last().expect("spectrum is non-empty");
        let farthest_carrier = brain.hierarchy().genomes[0].entries[farthest.index]
            .address
            .geometry();
        let query_gap_to_farthest = sigma(&compose(&query, &inverse(&farthest_carrier)));
        assert!(
            query_gap_to_attended < query_gap_to_farthest,
            "soft-attention output must sit strictly nearer the query than the farthest \
             population entry (attended gap = {}, farthest gap = {})",
            query_gap_to_attended,
            query_gap_to_farthest
        );

        // ── Invariant 3: determinism. ───────────────────────────
        // The substrate is pure: same query, same population →
        // bitwise-identical output.
        let attended_again = zread_at_query(&query, &brain.hierarchy().genomes[0], brain.buffer());
        assert_eq!(
            attended, attended_again,
            "soft attention must be deterministic"
        );

        // ── Invariant 4: query sensitivity. ─────────────────────
        // Different queries must produce different outputs. The
        // function is not collapsing the query parameter — it
        // really integrates the population at the query.
        let other_query = axis_rotation(2.0 * theta, 0);
        let other_attended =
            zread_at_query(&other_query, &brain.hierarchy().genomes[0], brain.buffer());
        let cross_gap = sigma(&compose(&attended, &inverse(&other_attended)));
        assert!(
            cross_gap > 1e-3,
            "soft attention must depend on the query (cross gap = {})",
            cross_gap
        );

        // ── Invariant 5: rotational equivariance. ───────────────
        // Any global rotation g applied to BOTH the query and every
        // entry's address must produce a rotated output. Build the
        // rotated genome as a pure rotation of the original — not
        // through make_brain() (which seeds its own DNA anchors),
        // but from a blank Genome with every original entry rotated.
        let g = axis_rotation(0.4, 1);
        let g_inv = inverse(&g);
        let mut rotated_genome = Genome::new(GenomeConfig::defaults());
        // Remove the auto-seeded identity; we'll rotate all entries
        // from the original brain, including its identity (which maps
        // to itself under conjugation, so the slot is preserved).
        rotated_genome.entries.clear();
        for entry in brain.hierarchy().genomes[0].entries.iter() {
            let rotated = compose(&compose(&g, &entry.address.geometry()), &g_inv);
            rotated_genome.seed_dna(rotated, 0, 0.0, 0.0);
        }
        let empty_buffer = Buffer::new(1);
        let rotated_query = compose(&compose(&g, &query), &g_inv);
        let rotated_attended = zread_at_query(&rotated_query, &rotated_genome, &empty_buffer);
        let conjugated_attended = compose(&compose(&g, &attended), &g_inv);
        let equivariance_gap = sigma(&compose(&conjugated_attended, &inverse(&rotated_attended)));
        assert!(
            equivariance_gap < 1e-4,
            "soft attention must be equivariant under rotation (gap = {})",
            equivariance_gap
        );
    }

    // ── AddressMode tests ────────────────────────────────────────────
    //
    // The tests below prove the factorized address semantics.
    // Carriers are chosen so the Hopf decomposition is known analytically:
    //
    // q_id = [1, 0, 0, 0]                — identity; base=(0,0,1), phase=0
    // q_z  = [√2/2, 0, 0, √2/2]          — z-fiber turn; base=(0,0,1), phase=π/4
    // q_y  = [√2/2, 0, √2/2, 0]          — π/2 rot around y; base=(1,0,0), phase=0
    // q_x  = [√2/2, √2/2, 0, 0]          — π/2 rot around x; base=(0,-1,0), phase=π/2
    //
    // q_id and q_z share the same S² base but differ in S¹ phase and Full
    // geodesic: σ(q_id, q_z) = π/4.
    //
    // q_y and q_z share phase=0 but have orthogonal bases:
    //   base_distance((1,0,0),(0,0,1)) = π/2.
    //
    // q_x and q_z have same Full distance as q_y to q_z (both π/3) but
    // different phase: phase_x = π/2 vs phase_z = 0.

    #[test]
    fn address_mode_full_equals_direct_sigma() {
        // Full mode must be bit-compatible with the direct σ(q·p⁻¹) computation.
        let s2 = 2.0_f64.sqrt() / 2.0;
        let pairs: &[([f64; 4], [f64; 4])] = &[
            ([1.0, 0.0, 0.0, 0.0], [s2, 0.0, 0.0, s2]),
            ([s2, s2, 0.0, 0.0], [s2, 0.0, s2, 0.0]),
            ([s2, 0.0, 0.0, s2], [s2, 0.0, s2, 0.0]),
        ];
        for &(q, p) in pairs {
            let expected = sigma(&compose(&q, &inverse(&p)));
            let got = address_distance(&q, &p, AddressMode::Full);
            assert!(
                (expected - got).abs() < 1e-12,
                "Full mode diverges from sigma: expected={expected}, got={got}"
            );
        }
    }

    #[test]
    fn carrier_from_hopf_round_trips_base_and_phase() {
        let inv_sqrt_3 = 1.0_f64 / 3.0_f64.sqrt();
        let cases = [
            ([1.0, 0.0, 0.0], std::f64::consts::FRAC_PI_3),
            ([0.0, -1.0, 0.0], std::f64::consts::FRAC_PI_2),
            ([inv_sqrt_3, inv_sqrt_3, inv_sqrt_3], 1.7),
            ([0.0, 0.0, 1.0], std::f64::consts::FRAC_PI_4),
            ([0.0, 0.0, -1.0], 2.1),
        ];

        for (base, phase) in cases {
            let q = carrier_from_hopf(base, phase);
            let (round_base, round_phase) = hopf_decompose(&q);
            assert!(
                base_distance_for_test(base, round_base) < 1e-10,
                "Hopf lift failed base round trip: base={base:?}, got={round_base:?}"
            );
            assert!(
                circular_distance(phase, round_phase) < 1e-10,
                "Hopf lift failed phase round trip: phase={phase}, got={round_phase}"
            );
        }
    }

    fn base_distance_for_test(a: [f64; 3], b: [f64; 3]) -> f64 {
        let an = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
        let a = [a[0] / an, a[1] / an, a[2] / an];
        let dot = (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]).clamp(-1.0, 1.0);
        dot.acos()
    }

    #[test]
    fn vocabulary_writes_type_to_base_and_position_to_phase() {
        let mut vocab = Vocabulary::new();
        let alpha_0 = vocab.embed_at_phase("alpha", 0.0);
        let alpha_1 = vocab.embed_at_phase("alpha", std::f64::consts::FRAC_PI_2);
        let beta_0 = vocab.embed_at_phase("beta", 0.0);

        let (alpha_0_base, alpha_0_phase) = hopf_decompose(&alpha_0);
        let (alpha_1_base, alpha_1_phase) = hopf_decompose(&alpha_1);
        let (beta_0_base, beta_0_phase) = hopf_decompose(&beta_0);

        assert!(
            base_distance_for_test(alpha_0_base, alpha_1_base) < 1e-10,
            "same token at different positions must keep the same S² type"
        );
        assert!(
            circular_distance(alpha_0_phase, alpha_1_phase) > 1e-6,
            "same token at different positions must differ in S¹ phase"
        );
        assert!(
            circular_distance(alpha_0_phase, beta_0_phase) < 1e-10,
            "different tokens at same position must share S¹ phase"
        );
        assert!(
            base_distance_for_test(alpha_0_base, beta_0_base) > 1e-6,
            "different tokens must differ in S² type"
        );
        assert_eq!(
            vocab.decode_nearest(&alpha_1),
            Some("alpha"),
            "decode must recover token by S² type and ignore S¹ position"
        );
    }

    #[test]
    fn address_mode_base_detects_same_fiber() {
        // q_id and q_z sit over the same S² base: both have base=(0,0,1).
        // Base distance should be 0; Full distance should be π/4.
        //
        // In a genome containing both q_z and q_x, a Full-mode query for q_id
        // cannot distinguish them (both at π/4). Base mode resolves the tie
        // decisively: q_z is at distance 0, q_x is at π/2.
        let s2 = 2.0_f64.sqrt() / 2.0;
        let q_id = [1.0_f64, 0.0, 0.0, 0.0];
        let q_z = [s2, 0.0, 0.0, s2];
        let q_x = [s2, s2, 0.0, 0.0];

        let base_to_z = address_distance(&q_id, &q_z, AddressMode::Base);
        let base_to_x = address_distance(&q_id, &q_x, AddressMode::Base);
        let full_to_z = address_distance(&q_id, &q_z, AddressMode::Full);
        let full_to_x = address_distance(&q_id, &q_x, AddressMode::Full);

        // Same fiber → Base distance zero.
        assert!(
            base_to_z < 1e-10,
            "q_id and q_z share fiber; Base dist={base_to_z}"
        );
        // Different base → Base distance π/2.
        assert!(
            (base_to_x - std::f64::consts::FRAC_PI_2).abs() < 1e-10,
            "q_id and q_x have orthogonal bases; expected π/2, got {base_to_x}"
        );
        // Full mode: both q_z and q_x are at π/4 from q_id — it can't distinguish.
        assert!(
            (full_to_z - std::f64::consts::FRAC_PI_4).abs() < 1e-10,
            "Full dist(q_id, q_z) should be π/4, got {full_to_z}"
        );
        assert!(
            (full_to_x - std::f64::consts::FRAC_PI_4).abs() < 1e-10,
            "Full dist(q_id, q_x) should be π/4, got {full_to_x}"
        );
        // Base mode: q_z wins decisively over q_x.
        assert!(
            base_to_z < base_to_x,
            "Base mode must prefer q_z (same fiber) over q_x (different base)"
        );
    }

    #[test]
    fn address_mode_phase_detects_same_cycle_position() {
        // q_id and q_y both have phase=0; q_x has phase=π/2.
        // Full distance from q_id: both q_y and q_x are at π/4 — indistinguishable.
        // Phase mode: q_y wins (same phase 0), q_x loses (phase π/2).
        let s2 = 2.0_f64.sqrt() / 2.0;
        let q_id = [1.0_f64, 0.0, 0.0, 0.0];
        let q_y = [s2, 0.0, s2, 0.0];
        let q_x = [s2, s2, 0.0, 0.0];

        let phase_to_y = address_distance(&q_id, &q_y, AddressMode::Phase);
        let phase_to_x = address_distance(&q_id, &q_x, AddressMode::Phase);
        let full_to_y = address_distance(&q_id, &q_y, AddressMode::Full);
        let full_to_x = address_distance(&q_id, &q_x, AddressMode::Full);

        // Same phase → Phase distance zero.
        assert!(
            phase_to_y < 1e-10,
            "q_id and q_y share phase=0; Phase dist={phase_to_y}"
        );
        // Different phase → Phase distance π/2.
        assert!(
            (phase_to_x - std::f64::consts::FRAC_PI_2).abs() < 1e-10,
            "q_id and q_x differ in phase by π/2; got {phase_to_x}"
        );
        // Full mode: q_y and q_x both at π/4 from q_id — indistinguishable.
        assert!(
            (full_to_y - std::f64::consts::FRAC_PI_4).abs() < 1e-10,
            "Full dist(q_id, q_y) should be π/4, got {full_to_y}"
        );
        assert!(
            (full_to_x - std::f64::consts::FRAC_PI_4).abs() < 1e-10,
            "Full dist(q_id, q_x) should be π/4, got {full_to_x}"
        );
        // Phase mode: q_y wins decisively over q_x.
        assert!(
            phase_to_y < phase_to_x,
            "Phase mode must prefer q_y (same phase) over q_x (different phase)"
        );
    }

    #[test]
    fn nearest_with_mode_full_matches_nearest() {
        // nearest_with_mode(Full) must return the same index and gap as nearest().
        let genome_config = GenomeConfig::defaults();
        let mut genome = Genome::new(genome_config);
        let s2 = 2.0_f64.sqrt() / 2.0;
        // Store three carriers.
        genome.ingest(&[s2, 0.0, 0.0, s2], 1, 0.0, 0.0);
        genome.ingest(&[s2, s2, 0.0, 0.0], 1, 0.0, 0.0);
        genome.ingest(&[s2, 0.0, s2, 0.0], 1, 0.0, 0.0);

        let query = [1.0_f64, 0.0, 0.0, 0.0];
        let (idx_old, gap_old) = genome.nearest(&query).expect("genome non-empty");
        let (idx_new, gap_new) = genome
            .nearest_with_mode(&query, AddressMode::Full)
            .expect("genome non-empty");

        assert_eq!(
            idx_old, idx_new,
            "nearest and nearest_with_mode(Full) must return same index"
        );
        assert!(
            (gap_old - gap_new).abs() < 1e-12,
            "nearest and nearest_with_mode(Full) must return same gap"
        );
    }

    #[test]
    fn resonate_channel_with_mode_returns_correct_entry() {
        // Prove that resonate_channel_with_mode selects by the chosen
        // projection, not by full geodesic.
        //
        // Setup: genome has three entries.
        //   slot 0: address = q_z = [√2/2, 0, 0, √2/2]
        //             value = [0, 1, 0, 0]  (arbitrary distinct value)
        //   slot 1: address = q_y = [√2/2, 0, √2/2, 0]
        //             value = [0, 0, 1, 0]
        //   slot 2: address = q_x = [√2/2, √2/2, 0, 0]
        //             value = [0, 0, 0, 1]
        //
        // Query = q_id = [1, 0, 0, 0].
        //   Full geodesic: q_z, q_y, and q_x are at σ = π/4 from q_id — tie.
        //   Base mode: q_id and q_z share base=(0,0,1); q_x has base=(0,-1,0).
        //     → slot 0 wins.
        //   Phase mode: q_id phase=0, q_y phase=0, q_x phase=π/2.
        //     → slot 1 wins.
        //
        // This proves retrieval behavior, not just address_distance().
        let s2 = 2.0_f64.sqrt() / 2.0;
        let q_z = [s2, 0.0, 0.0, s2];
        let q_y = [s2, 0.0, s2, 0.0];
        let q_x = [s2, s2, 0.0, 0.0];
        let val_z = [0.0_f64, 1.0, 0.0, 0.0];
        let val_y = [0.0_f64, 0.0, 1.0, 0.0];
        let val_x = [0.0_f64, 0.0, 0.0, 1.0];
        let query = [1.0_f64, 0.0, 0.0, 0.0];

        let genome_config = GenomeConfig::defaults();
        let mut genome = Genome::new(genome_config);
        genome.ingest(&q_z, 1, 0.0, 0.0);
        genome.ingest(&q_y, 1, 0.0, 0.0);
        genome.ingest(&q_x, 1, 0.0, 0.0);
        // After ingest the stored values equal the addresses (no prior genome).
        // Override with distinct values so the test can check which was returned.
        genome.entries[0].value = val_z;
        genome.entries[1].value = val_y;
        genome.entries[2].value = val_x;

        let buffer = Buffer::new(64);

        // Base mode: q_id shares the S² axis with q_z → must return val_z.
        let hit_base = resonate_channel_with_mode(
            &query,
            HopfChannel::Full,
            AddressMode::Base,
            &genome,
            &buffer,
        )
        .expect("genome non-empty");
        assert_eq!(
            hit_base.carrier, val_z,
            "Base mode must return the entry with the same S² axis as the query"
        );

        // Phase mode: q_id and q_y share phase=0 → must return val_y.
        let hit_phase = resonate_channel_with_mode(
            &query,
            HopfChannel::Full,
            AddressMode::Phase,
            &genome,
            &buffer,
        )
        .expect("genome non-empty");
        assert_eq!(
            hit_phase.carrier, val_y,
            "Phase mode must return the entry with the same S¹ phase as the query"
        );
    }

    #[test]
    fn zread_channel_with_mode_weights_by_projection() {
        // Prove that zread_at_query_channel_with_mode produces different results
        // for different modes when the query is deliberately closer on one Hopf
        // projection than on another.
        //
        // Setup: genome has two entries with the same Full distance to the query.
        //   q_id  = [1, 0, 0, 0] — query
        //   q_z   = [√2/2, 0, 0, √2/2] — same S² base as q_id (base dist = 0)
        //   q_x   = [√2/2, √2/2, 0, 0]  — different S² base (base dist = π/2)
        //
        // Base mode: q_z is at base_gap=0, q_x is at π/2.
        //   coupling(0) = cos(0) = 1, coupling(π/2) = cos(π/2) = 0.
        //   Only q_z contributes; the result must equal slerp(IDENTITY, q_z, 1) = q_z.
        //
        // Full mode: both at σ=π/4, t=cos(π/4)=√2/2 ≈ 0.707 for both.
        //   The result is a mixture of q_z and q_x; not equal to q_z alone.
        let s2 = 2.0_f64.sqrt() / 2.0;
        let q_z = [s2, 0.0, 0.0, s2];
        let q_x = [s2, s2, 0.0, 0.0];
        let query = [1.0_f64, 0.0, 0.0, 0.0];

        let genome_config = GenomeConfig::defaults();
        let mut genome = Genome::new(genome_config);
        genome.ingest(&q_z, 1, 0.0, 0.0);
        genome.ingest(&q_x, 1, 0.0, 0.0);
        // Values equal addresses — the ZREAD result is the weighted composition
        // of the addresses themselves.
        let buffer = Buffer::new(64);

        let result_base = zread_at_query_channel_with_mode(
            &query,
            HopfChannel::Full,
            AddressMode::Base,
            &genome,
            &buffer,
        );
        let result_full = zread_at_query_channel_with_mode(
            &query,
            HopfChannel::Full,
            AddressMode::Full,
            &genome,
            &buffer,
        );

        // Base mode excludes q_x (coupling = 0), so result_base ≈ q_z.
        let gap_base_to_qz = sigma(&compose(&result_base, &inverse(&q_z)));
        assert!(
            gap_base_to_qz < 1e-10,
            "Base mode ZREAD must be dominated by q_z (same S² axis); gap={gap_base_to_qz}"
        );

        // Full mode includes both; result differs from pure q_z.
        let gap_full_to_qz = sigma(&compose(&result_full, &inverse(&q_z)));
        assert!(
            gap_full_to_qz > 1e-6,
            "Full mode ZREAD must mix both entries; result should not equal q_z alone"
        );
    }

    #[test]
    fn zread_is_path_ordered_not_commutative_integral() {
        // ZREAD is a path-ordered Hamilton product. Inserting two entries in
        // opposite orders must produce different outputs when both have identical
        // coupling to the query — proving that coupling controls strength, not
        // order. The brain's memory sequence is preserved by insertion order.
        //
        // Both entries are equidistant from the identity query (σ = π/4).
        // The coupling weight t is identical for both in either order.
        // The composition A·B ≠ B·A because Hamilton multiplication is
        // non-commutative on S³.
        let s2 = 2.0_f64.sqrt() / 2.0;
        let a = [s2, s2, 0.0, 0.0]; // π/2 rotation around x
        let b = [s2, 0.0, s2, 0.0]; // π/2 rotation around y
        let query = [1.0_f64, 0.0, 0.0, 0.0]; // identity

        // Genome AB: a ingested first, then b.
        let config = GenomeConfig::defaults();
        let mut genome_ab = Genome::new(config.clone());
        genome_ab.ingest(&a, 1, 0.0, 0.0);
        genome_ab.ingest(&b, 1, 0.0, 0.0);

        // Genome BA: b ingested first, then a.
        let mut genome_ba = Genome::new(config);
        genome_ba.ingest(&b, 1, 0.0, 0.0);
        genome_ba.ingest(&a, 1, 0.0, 0.0);

        let buf = Buffer::new(0);
        let result_ab = zread_at_query(&query, &genome_ab, &buf);
        let result_ba = zread_at_query(&query, &genome_ba, &buf);

        let gap = sigma(&compose(&result_ab, &inverse(&result_ba)));
        assert!(
            gap > 1e-6,
            "ZREAD must be path-ordered: AB != BA (gap={gap})"
        );
    }

    #[test]
    fn hierarchy_level_reads_are_isolated() {
        // Step 4 test: zread_level and resonate_level read the correct level
        // genome, and level boundaries are hard — a carrier in genomes[1]
        // is invisible from a level-0 read.
        let config = GenomeConfig::defaults();
        let mut h = Hierarchy::new(0.1, config);

        // genomes[1] starts non-existent. zread_level returns None.
        assert!(
            h.zread_level(1, &IDENTITY, HopfChannel::Full, AddressMode::Full)
                .is_none(),
            "zread_level must return None for a level that has never received a closure"
        );
        assert!(
            h.resonate_level(1, &IDENTITY, HopfChannel::Full, AddressMode::Full)
                .is_none(),
            "resonate_level must return None for a non-existent level"
        );

        // Plant a known carrier directly into genomes[1].
        let s2 = 2.0_f64.sqrt() / 2.0;
        let known = [s2, s2, 0.0, 0.0]; // π/2 rotation around x — far from identity
        h.genome_at_mut(1).ingest(&known, 2, 0.05, 0.1);

        // resonate_level(1) must find the planted carrier.
        let hit1 = h
            .resonate_level(1, &known, HopfChannel::Full, AddressMode::Full)
            .expect("resonate_level(1) must see the carrier ingested into genomes[1]");
        let gap1 = sigma(&compose(&hit1.carrier, &inverse(&known)));
        assert!(
            gap1 < 1e-10,
            "resonate_level(1) must return the exact planted carrier (gap={gap1})"
        );

        // Level-0 genome is empty (Hierarchy::new seeds no entries).
        // resonate_level(0) must return None, not the level-1 carrier.
        assert!(
            h.resonate_level(0, &known, HopfChannel::Full, AddressMode::Full)
                .is_none(),
            "resonate_level(0) must not see a carrier that only exists in genomes[1]"
        );

        // zread_level(1) returns Some — level exists and has an entry.
        let zread1 = h
            .zread_level(1, &known, HopfChannel::Full, AddressMode::Full)
            .expect("zread_level(1) must return Some for an existing level");
        // The result is a weighted product; at minimum it must not be exactly IDENTITY
        // because the planted carrier has t = coupling_from_gap(0) = 1.0 at the query.
        let zread_gap = sigma(&zread1);
        assert!(
            zread_gap > 1e-6,
            "zread_level(1) with a full-coupling entry must not return IDENTITY (gap={zread_gap})"
        );

        // Non-existent level 99 still returns None.
        assert!(
            h.zread_level(99, &known, HopfChannel::Full, AddressMode::Full)
                .is_none(),
            "zread_level must return None for any level beyond the genomes Vec"
        );
    }

    #[test]
    fn music_encoder_applies_hopf_write_contract() {
        // MusicEncoder must obey the Hopf write contract:
        //   - Same role at different beats → same S² base, different S¹ phase.
        //   - Different roles at same beat → different S² base, same S¹ phase.
        // This mirrors the vocabulary_writes_type_to_base_and_position_to_phase test.

        let mut enc = MusicEncoder::new();

        // Tonic at beat 0 and beat 2 of 4/4.
        let tonic_0 = enc.embed("tonic", 0, 4);
        let tonic_2 = enc.embed("tonic", 2, 4);
        // Dominant at beat 0.
        let dominant_0 = enc.embed("dominant", 0, 4);

        let (base_t0, phase_t0) = hopf_decompose(&tonic_0);
        let (base_t2, phase_t2) = hopf_decompose(&tonic_2);
        let (base_d0, phase_d0) = hopf_decompose(&dominant_0);

        // Same role → same S² base.
        let base_gap_same_role = {
            let dot = (base_t0[0] * base_t2[0]
                + base_t0[1] * base_t2[1]
                + base_t0[2] * base_t2[2])
                .clamp(-1.0, 1.0);
            dot.acos()
        };
        assert!(
            base_gap_same_role < 1e-10,
            "tonic at different beats must share S² base (gap={base_gap_same_role})"
        );

        // Same role at different beats → different S¹ phase.
        let phase_gap_same_role = circular_distance(phase_t0, phase_t2);
        assert!(
            phase_gap_same_role > 1e-6,
            "tonic at beats 0 and 2 must differ in S¹ phase (gap={phase_gap_same_role})"
        );

        // Different roles at same beat → different S² base.
        let base_gap_diff_role = {
            let dot = (base_t0[0] * base_d0[0]
                + base_t0[1] * base_d0[1]
                + base_t0[2] * base_d0[2])
                .clamp(-1.0, 1.0);
            dot.acos()
        };
        assert!(
            base_gap_diff_role > 1e-6,
            "tonic and dominant must differ in S² base (gap={base_gap_diff_role})"
        );

        // Different roles at same beat → same S¹ phase.
        let phase_gap_diff_role = circular_distance(phase_t0, phase_d0);
        assert!(
            phase_gap_diff_role < 1e-10,
            "tonic and dominant at beat 0 must share S¹ phase (gap={phase_gap_diff_role})"
        );

        // role_base() must return the same axis as what embed() used.
        let base_from_accessor = enc.role_base("tonic");
        let base_gap_accessor = {
            let dot = (base_t0[0] * base_from_accessor[0]
                + base_t0[1] * base_from_accessor[1]
                + base_t0[2] * base_from_accessor[2])
                .clamp(-1.0, 1.0);
            dot.acos()
        };
        assert!(
            base_gap_accessor < 1e-10,
            "role_base() must return the same S² axis used by embed() (gap={base_gap_accessor})"
        );

        // Sub-beat resolution: same role at same beat but different sub-beat
        // → same base, different phase.
        let tonic_beat1_sub0 = enc.embed_sub_beat("tonic", 1, 0, 4, 4);
        let tonic_beat1_sub2 = enc.embed_sub_beat("tonic", 1, 2, 4, 4);
        let (base_sb0, phase_sb0) = hopf_decompose(&tonic_beat1_sub0);
        let (base_sb2, phase_sb2) = hopf_decompose(&tonic_beat1_sub2);
        let sub_base_gap = {
            let dot = (base_sb0[0] * base_sb2[0]
                + base_sb0[1] * base_sb2[1]
                + base_sb0[2] * base_sb2[2])
                .clamp(-1.0, 1.0);
            dot.acos()
        };
        assert!(
            sub_base_gap < 1e-10,
            "sub-beat variants must share S² base (gap={sub_base_gap})"
        );
        assert!(
            circular_distance(phase_sb0, phase_sb2) > 1e-6,
            "sub-beat variants must differ in S¹ phase"
        );
    }

    #[test]
    fn bkt_threshold_is_dobrushin_parochet_boundary() {
        // BKT_THRESHOLD must sit strictly between PRIME_3_COUPLING and PRIME_5_COUPLING.
        // This is the Parochet: primes 2 and 3 are above; prime 5 and up are below.
        assert!(
            BKT_THRESHOLD < PRIME_3_COUPLING,
            "BKT_THRESHOLD must be below prime-3 coupling (prime 3 is above the Parochet): \
             {BKT_THRESHOLD} vs {PRIME_3_COUPLING}"
        );
        assert!(
            BKT_THRESHOLD > PRIME_5_COUPLING,
            "BKT_THRESHOLD must be above prime-5 coupling (prime 5 is below the Parochet): \
             {BKT_THRESHOLD} vs {PRIME_5_COUPLING}"
        );
        // Primes 2 and 3 are above the Parochet (dominant contraction zone).
        assert!(
            PRIME_2_COUPLING >= BKT_THRESHOLD,
            "prime-2 coupling must exceed BKT_THRESHOLD"
        );
        assert!(
            PRIME_3_COUPLING >= BKT_THRESHOLD,
            "prime-3 coupling must exceed BKT_THRESHOLD"
        );
        // The Dobrushin coefficients are the complement: δ(p) = 1 - p^(-1/2).
        assert!(
            (DOBRUSHIN_DELTA_2 - (1.0 - PRIME_2_COUPLING)).abs() < 1e-15,
            "DOBRUSHIN_DELTA_2 must equal 1 - PRIME_2_COUPLING"
        );
        assert!(
            (DOBRUSHIN_DELTA_3 - (1.0 - PRIME_3_COUPLING)).abs() < 1e-15,
            "DOBRUSHIN_DELTA_3 must equal 1 - PRIME_3_COUPLING"
        );
        // Prime 2 contracts more than prime 3: δ(2) < δ(3) because δ(p) = 1 - p^(-1/2)
        // grows with p, but the coupling t(2) > t(3), meaning prime 2 retains more
        // and contracts less? Wait — δ(p) = 1 - p^(-1/2): for p=2, δ=0.293;
        // for p=3, δ=0.423. Larger δ = stronger contraction. Prime 3 contracts MORE.
        // But prime 2's coupling is higher, meaning it PARTICIPATES more strongly in ZREAD.
        // These are two different things: coupling (participation) vs contraction (forgetting).
        assert!(
            DOBRUSHIN_DELTA_2 < DOBRUSHIN_DELTA_3,
            "prime 3 has stronger Dobrushin contraction than prime 2 (δ(2)<δ(3)): \
             {DOBRUSHIN_DELTA_2} vs {DOBRUSHIN_DELTA_3}"
        );
    }

    #[test]
    fn curriculum_trace_produces_deterministic_reproducible_report() {
        // A CurriculumTrace must:
        //   1. Record per-window outcomes (closures, growth, errors).
        //   2. Produce the same totals as run_curriculum on the same flat sequence.
        //   3. Be deterministic: same trace on a fresh brain gives identical report.

        let eps = axis_rotation(PI / 3.0, 0);
        let corpus: Vec<[f64; 4]> = (0..24).map(|_| eps).collect();

        // Run the trace on one brain.
        let mut brain_a = make_brain(0.05, 0.10);
        let trace = CurriculumTrace::from_flat(&corpus, 6); // 4 windows of 6 carriers
        assert_eq!(trace.len(), 4, "24 carriers / 6 window = 4 windows");
        let report_a = trace.run(&mut brain_a);

        // Run the same flat corpus on an equivalent brain using run_curriculum.
        let mut brain_b = make_brain(0.05, 0.10);
        let outcome = brain_b.run_curriculum(&corpus);

        // Totals must match.
        assert_eq!(
            brain_a.total_closures(),
            brain_b.total_closures(),
            "trace and run_curriculum must produce the same closure count"
        );
        assert_eq!(
            brain_a.genome_size(),
            brain_b.genome_size(),
            "trace and run_curriculum must produce the same genome size"
        );
        assert_eq!(
            report_a.total_closures,
            outcome.closures_fired,
            "CurriculumReport.total_closures must match SequenceOutcome.closures_fired"
        );
        assert_eq!(
            report_a.total_genome_growth,
            outcome.genome_growth,
            "CurriculumReport.total_genome_growth must match SequenceOutcome.genome_growth"
        );
        assert_eq!(
            report_a.final_genome_size,
            brain_a.genome_size(),
            "CurriculumReport.final_genome_size must match the brain's actual genome size"
        );

        // Window reports: there must be 4, and closures must sum to total.
        assert_eq!(report_a.windows.len(), 4);
        let sum_closures: usize = report_a.windows.iter().map(|w| w.closures_fired).sum();
        assert_eq!(
            sum_closures,
            report_a.total_closures,
            "per-window closures must sum to total"
        );

        // Carry + FixedPoint == closures_fired per window.
        for w in &report_a.windows {
            assert_eq!(
                w.carry_closures + w.fixedpoint_closures,
                w.closures_fired,
                "window {}: carry + fixedpoint must equal closures_fired ({} + {} != {})",
                w.index, w.carry_closures, w.fixedpoint_closures, w.closures_fired
            );
        }

        // level_genome_growth[0] == genome_delta (level 0 is the primary genome).
        for w in &report_a.windows {
            let level0_growth = w.level_genome_growth.first().copied().unwrap_or(0);
            assert_eq!(
                level0_growth,
                w.genome_delta,
                "window {}: level_genome_growth[0] must equal genome_delta",
                w.index
            );
        }

        // self_free_energy_start/end are finite.
        for w in &report_a.windows {
            assert!(w.self_free_energy_start.is_finite(), "window {}: sfe_start must be finite", w.index);
            assert!(w.self_free_energy_end.is_finite(), "window {}: sfe_end must be finite", w.index);
        }

        // Determinism: same trace on a fresh brain must produce the same report.
        let mut brain_c = make_brain(0.05, 0.10);
        let report_c = trace.run(&mut brain_c);
        for (w_a, w_c) in report_a.windows.iter().zip(report_c.windows.iter()) {
            assert_eq!(w_a.closures_fired, w_c.closures_fired, "window {}: closures must be deterministic", w_a.index);
            assert_eq!(w_a.carry_closures, w_c.carry_closures, "window {}: carry must be deterministic", w_a.index);
            assert_eq!(w_a.fixedpoint_closures, w_c.fixedpoint_closures, "window {}: fixedpoint must be deterministic", w_a.index);
            assert_eq!(w_a.genome_delta, w_c.genome_delta, "window {}: growth must be deterministic", w_a.index);
            assert_eq!(w_a.mean_prediction_error, w_c.mean_prediction_error, "window {}: prediction error must be deterministic", w_a.index);
            assert_eq!(w_a.self_free_energy_start, w_c.self_free_energy_start, "window {}: sfe_start must be deterministic", w_a.index);
            assert_eq!(w_a.self_free_energy_end, w_c.self_free_energy_end, "window {}: sfe_end must be deterministic", w_a.index);
        }
    }

    #[test]
    fn update_contract_is_explicit_living_verb() {
        // update() and update_sequence() must:
        //   1. Return an UpdateReport (the body was changed).
        //   2. Produce the same genome delta and closure count as the
        //      equivalent ingest / run_curriculum call.
        //   3. Not fire for read-only verbs (evaluate, generate).
        let mut brain_via_update = make_brain(0.05, 0.10);
        let mut brain_via_ingest = make_brain(0.05, 0.10);
        let eps = axis_rotation(PI / 3.0, 0);

        // Drive both brains with the same 12-step sequence.
        let seq: Vec<[f64; 4]> = (0..12).map(|_| eps).collect();

        let report = brain_via_update.update_sequence(&seq);
        for q in &seq {
            brain_via_ingest.ingest(q);
        }

        // UpdateReport must track genome growth and closures accurately.
        assert_eq!(
            brain_via_update.genome_size(),
            brain_via_ingest.genome_size(),
            "update_sequence must produce the same genome as equivalent ingest calls"
        );
        assert_eq!(
            brain_via_update.total_closures(),
            brain_via_ingest.total_closures(),
            "closure counts must match"
        );
        assert_eq!(
            report.closures_fired,
            brain_via_update.total_closures(),
            "UpdateReport.closures_fired must equal total closures when starting fresh"
        );
        assert_eq!(
            report.genome_delta,
            brain_via_update
                .genome_size()
                .saturating_sub(3), // 3 DNA anchors were there at boot
            "UpdateReport.genome_delta must equal net genome growth above the seed"
        );
        assert_eq!(
            report.hierarchy_depth,
            brain_via_update.hierarchy_depth(),
            "UpdateReport.hierarchy_depth must match actual depth"
        );

        // Single-step update must not change genome when input matches an
        // existing entry exactly (identity is already seeded).
        let size_before = brain_via_update.genome_size();
        let single = brain_via_update.update(&IDENTITY);
        // genome_delta can be 0 if identity reinforces and doesn't grow.
        assert_eq!(
            single.genome_delta,
            brain_via_update.genome_size().saturating_sub(size_before),
            "update.genome_delta must reflect actual growth for single step"
        );
    }

    #[test]
    fn domain_embed_enforces_hopf_write_contract() {
        // The Hopf write contract for domain_embed(type_bytes, position_phase):
        //   - Same type at different positions → same S² base, different S¹ phase.
        //   - Different types at same position → different S² base, same S¹ phase.
        // This is the condition that makes ZREAD/RESONATE work correctly for
        // structured domain data — type-identity and position-identity are orthogonal.

        let alpha = b"alpha".as_slice();
        let beta = b"beta".as_slice();
        let phase_0 = 0.0;
        let phase_1 = std::f64::consts::FRAC_PI_2;

        let alpha_at_0 = domain_embed(alpha, phase_0);
        let alpha_at_1 = domain_embed(alpha, phase_1);
        let beta_at_0 = domain_embed(beta, phase_0);

        let (base_a0, phase_a0) = hopf_decompose(&alpha_at_0);
        let (base_a1, phase_a1) = hopf_decompose(&alpha_at_1);
        let (base_b0, phase_b0) = hopf_decompose(&beta_at_0);

        // Same type at different positions → same S² base.
        let base_gap_same_type = {
            let dot = (base_a0[0] * base_a1[0]
                + base_a0[1] * base_a1[1]
                + base_a0[2] * base_a1[2])
                .clamp(-1.0, 1.0);
            dot.acos()
        };
        assert!(
            base_gap_same_type < 1e-10,
            "same type at different positions must share S² base (gap={base_gap_same_type})"
        );

        // Same type at different positions → different S¹ phase.
        let phase_gap_same_type = circular_distance(phase_a0, phase_a1);
        assert!(
            phase_gap_same_type > 1e-6,
            "same type at different positions must differ in S¹ phase (gap={phase_gap_same_type})"
        );

        // Different types at same position → different S² base.
        let base_gap_diff_type = {
            let dot = (base_a0[0] * base_b0[0]
                + base_a0[1] * base_b0[1]
                + base_a0[2] * base_b0[2])
                .clamp(-1.0, 1.0);
            dot.acos()
        };
        assert!(
            base_gap_diff_type > 1e-6,
            "different types must differ in S² base (gap={base_gap_diff_type})"
        );

        // Different types at same position → same S¹ phase.
        let phase_gap_diff_type = circular_distance(phase_a0, phase_b0);
        assert!(
            phase_gap_diff_type < 1e-10,
            "different types at same position must share S¹ phase (gap={phase_gap_diff_type})"
        );

        // Parity version: parity=false must equal the non-parity version.
        let alpha_no_parity = domain_embed_with_parity(alpha, phase_0, false);
        assert_eq!(
            alpha_at_0, alpha_no_parity,
            "domain_embed_with_parity(false) must equal domain_embed"
        );

        // Parity version: parity=true must differ from parity=false.
        let alpha_parity = domain_embed_with_parity(alpha, phase_0, true);
        let parity_gap = sigma(&compose(&alpha_at_0, &inverse(&alpha_parity)));
        assert!(
            parity_gap > 1e-6,
            "domain_embed_with_parity(true) must differ from parity=false (gap={parity_gap})"
        );
    }

    #[test]
    fn zero_semantics_distinguish_empty_sequence_from_empty_payload() {
        // SDK canon rule:
        //   no records / empty sequence -> algebraic identity
        //   empty payload as a record -> named content, not absence
        //
        // closure_ea preserves that split by using geometric empty bytes
        // as the sequence identity, while content-addressed and structured
        // domain encoders still assign empty/missing payloads their own
        // carriers.
        let empty_sequence = bytes_to_sphere4(&[], false);
        assert_eq!(
            empty_sequence, IDENTITY,
            "geometric empty bytes represent no composition: the sequence identity"
        );

        let empty_payload = bytes_to_sphere4(&[], true);
        let empty_payload_gap = sigma(&empty_payload);
        assert!(
            empty_payload_gap > 1e-6,
            "hashed empty payload must be named data, not collapsed to IDENTITY"
        );

        let semantic_missing = domain_embed(b"<missing>", 0.0);
        let semantic_missing_gap = sigma(&semantic_missing);
        assert!(
            semantic_missing_gap > 1e-6,
            "semantic missing/zero token must be a named carrier, not geometric absence"
        );

        let payload_vs_missing = sigma(&compose(&empty_payload, &inverse(&semantic_missing)));
        assert!(
            payload_vs_missing > 1e-6,
            "empty payload and semantic <missing> name different things"
        );
    }

    #[test]
    fn cell_c_ego_signals_are_inspectable_and_persisted() {
        // After a closure, cell_c_w_depth() must be ≤ 1.0 (moved from identity),
        // cell_c_hopf_base() must be a unit vector on S², and cell_c_min_w must be
        // ≤ cell_c_w_depth() at every moment (it is a running minimum).
        // All three signals must survive a BrainState round trip exactly.
        let mut brain = make_brain(0.05, 0.10);
        let eps = axis_rotation(PI / 3.0, 0);

        // Run until at least one closure fires.
        for _ in 0..12 {
            brain.ingest(&eps);
        }
        assert!(
            brain.total_closures() > 0,
            "precondition: must have a closure"
        );

        // W depth must have moved below 1.0 once Cell C accumulated a packet.
        let w = brain.cell_c_w_depth();
        assert!(
            w <= 1.0,
            "cell_c_w_depth must be ≤ 1.0 after accumulation (got {w})"
        );

        // Hopf base must be a unit vector.
        let base = brain.cell_c_hopf_base();
        let base_norm = (base[0] * base[0] + base[1] * base[1] + base[2] * base[2]).sqrt();
        assert!(
            (base_norm - 1.0).abs() < 1e-10,
            "cell_c_hopf_base must be a unit S² vector (norm={base_norm})"
        );

        // min_w ≤ current w (it records the minimum).
        let min_w = brain.cell_c_min_w();
        assert!(
            min_w <= w + 1e-12,
            "cell_c_min_w must be ≤ current w ({min_w} vs {w})"
        );

        // Round trip: all three values survive exactly.
        let state = brain.to_brain_state();
        let restored = ThreeCell::from_brain_state(state);
        assert_eq!(
            brain.cell_c_w_depth(),
            restored.cell_c_w_depth(),
            "cell_c_w_depth must survive round trip"
        );
        assert_eq!(
            brain.cell_c_hopf_base(),
            restored.cell_c_hopf_base(),
            "cell_c_hopf_base must survive round trip"
        );
        assert_eq!(
            brain.cell_c_min_w(),
            restored.cell_c_min_w(),
            "cell_c_min_w must survive round trip"
        );
    }

    #[test]
    fn self_critical_crossing_writes_level1_primestate() {
        // Autobiographical learning is a level-1 FixedPoint:
        // the self-difference stream compose(cell_c, inverse(ZREAD(cell_c)))
        // reaches the critical line and writes its localized packet to genomes[1].
        let mut brain = make_brain(0.05, 0.10);
        let eps = axis_rotation(PI / 3.0, 0);

        let mut saw_self_prime = false;
        for _ in 0..96 {
            let step = brain.ingest(&eps);
            if step.hierarchy_events.iter().any(|ev| {
                ev.level == 1 && ev.role == ClosureRole::FixedPoint && ev.support > 1
            }) {
                saw_self_prime = true;
                break;
            }
        }

        assert!(
            saw_self_prime,
            "self-critical crossings must emit a level-1 FixedPoint PrimeState"
        );

        let level1 = brain
            .hierarchy()
            .genome_at(1)
            .expect("level-1 autobiographical genome must exist");
        assert!(
            level1.entries.iter().any(|entry| {
                entry.layer == Layer::Epigenetic
                    && (sigma(&entry.address.geometry()) - SIGMA_BALANCE).abs() <= 0.05
            }),
            "genomes[1] must contain an epigenetic critical-line PrimeState"
        );
    }

    #[test]
    fn fixedpoint_does_not_cascade_to_next_level() {
        // ClosureRole::FixedPoint fires at the Hopf equator (intra-level balance).
        // It must NOT cascade a carrier into the next hierarchy level.
        // ClosureRole::Carry fires at identity return (inter-level handoff).
        // It MUST cascade.
        //
        // We test this by constructing ClosureEvents directly and calling
        // `Hierarchy::emit_closure`, then counting the returned Vec length
        // and the total event count.
        use crate::localization::LocalizedInterval;

        let config = GenomeConfig::defaults();
        let mut h = Hierarchy::new(0.1, config);

        let dummy_interval = LocalizedInterval {
            start: 0,
            end: 1,
            support: 2,
            product: IDENTITY,
            sigma: 0.0,
        };

        // A FixedPoint event at level 0.
        let fp_event = ClosureEvent {
            carrier: IDENTITY,
            sigma: 0.0,
            support: 2,
            oscillation_depth: 2,
            excursion_peak: 0.5,
            oscillation_excursion_peak: 0.5,
            level: 0,
            role: ClosureRole::FixedPoint,
            kind: ClosureKind::Arrangement,
            hopf_base: [0.0, 0.0, 1.0],
            hopf_phase: 0.0,
            interval: dummy_interval.clone(),
        };

        let cascaded = h.emit_closure(&fp_event);
        assert_eq!(
            cascaded.len(),
            0,
            "FixedPoint must not cascade to level+1 (got {} cascaded events)",
            cascaded.len()
        );
        // The event itself was recorded.
        assert_eq!(h.events.len(), 1, "FixedPoint event must be recorded");

        // A Carry event at level 0 targeting IDENTITY will try to ingest
        // into level 1. With IDENTITY and a threshold of 0.1, level 1 may
        // or may not close immediately — but the attempt is made.
        // What we care about is that with a non-trivial carrier (which
        // would cross the threshold on first ingest at a fresh level),
        // Carry does cascade while FixedPoint does not.
        //
        // Use a carrier far from identity so level-1 ingest does not close.
        let s2 = 2.0_f64.sqrt() / 2.0;
        let carry_carrier = [s2, s2, 0.0, 0.0]; // σ = π/4 — above threshold 0.1
        let carry_event = ClosureEvent {
            carrier: carry_carrier,
            sigma: 0.01,
            support: 3,
            oscillation_depth: 3,
            excursion_peak: 0.8,
            oscillation_excursion_peak: 0.8,
            level: 0,
            role: ClosureRole::Carry,
            kind: ClosureKind::Completion,
            hopf_base: [0.0, 0.0, 1.0],
            hopf_phase: 0.0,
            interval: dummy_interval,
        };

        let events_before = h.events.len();
        h.emit_closure(&carry_event);
        // Level 1 was created by the cascade attempt (even if it didn't close).
        assert_eq!(
            h.depth(),
            2,
            "Carry emission must cause hierarchy to reach level 1"
        );
        assert_eq!(
            h.events.len(),
            events_before + 1,
            "Carry event must be recorded in the event log"
        );
    }

    #[test]
    fn brain_state_round_trip_is_exact() {
        // Drive a brain to a non-trivial state: closures fired, buffer active,
        // cell_a and cell_c non-identity, cycle count > 0.
        // Snapshot, restore, assert all specified fields match, then ingest
        // the same next carrier into both and assert matching Step fields.
        let threshold = 0.15;
        let novelty = 0.35;
        let mut brain = ThreeCell::new(
            threshold,
            threshold,
            8,
            GenomeConfig {
                reinforce_threshold: novelty * 0.1,
                novelty_threshold: novelty,
                merge_threshold: novelty * 0.1,
                co_resonance_merge_threshold: 0.0,
            },
        );

        // π/3 rotation × 6 = identity → closure. Run two full cycles.
        let eps = axis_rotation(PI / 3.0, 0);
        for _ in 0..12 {
            brain.ingest(&eps);
        }
        assert!(
            brain.total_closures() > 0,
            "precondition: need at least one closure before snapshot"
        );

        // Snapshot and restore.
        let state = brain.to_brain_state();
        let saved_self_difference_count = state.self_difference_count;
        let saved_self_difference_history_len = state.self_difference_history.len();
        let saved_self_difference_product = state.self_difference_product;
        let saved_self_balance_excursion_peak = state.self_balance_excursion_peak;
        let mut restored = ThreeCell::from_brain_state(state);

        // All specified fields must match exactly.
        assert_eq!(brain.cell_a(), restored.cell_a(), "cell_a");
        assert_eq!(brain.cell_c(), restored.cell_c(), "cell_c");
        assert_eq!(
            brain.total_closures(),
            restored.total_closures(),
            "total_closures (event log length)"
        );
        assert_eq!(brain.genome_size(), restored.genome_size(), "genome_size");
        assert_eq!(
            brain.buffer().entries().len(),
            restored.buffer().entries().len(),
            "buffer entry count"
        );
        assert_eq!(
            brain.hierarchy().genomes.len(),
            restored.hierarchy().genomes.len(),
            "genomes Vec length"
        );
        assert_eq!(
            brain.hierarchy().levels().len(),
            restored.hierarchy().levels().len(),
            "levels Vec length"
        );
        let restored_state = restored.to_brain_state();
        assert_eq!(
            saved_self_difference_count,
            restored_state.self_difference_count,
            "self_difference_count"
        );
        assert_eq!(
            saved_self_difference_history_len,
            restored_state.self_difference_history.len(),
            "self_difference_history length"
        );
        assert_eq!(
            saved_self_difference_product,
            restored_state.self_difference_product,
            "self_difference_product"
        );
        assert_eq!(
            saved_self_balance_excursion_peak,
            restored_state.self_balance_excursion_peak,
            "self_balance_excursion_peak"
        );

        // Ingest the same next carrier into both; all deterministic Step fields
        // must be bit-identical.
        let next = axis_rotation(PI / 3.0, 1);
        let s_orig = brain.ingest(&next);
        let s_rest = restored.ingest(&next);
        assert_eq!(s_orig.cell_a_sigma, s_rest.cell_a_sigma, "step.cell_a_sigma");
        assert_eq!(s_orig.cell_c_sigma, s_rest.cell_c_sigma, "step.cell_c_sigma");
        assert_eq!(
            s_orig.prediction_error, s_rest.prediction_error,
            "step.prediction_error"
        );
        assert_eq!(
            s_orig.self_free_energy, s_rest.self_free_energy,
            "step.self_free_energy"
        );
        assert_eq!(
            s_orig.closure.is_some(),
            s_rest.closure.is_some(),
            "step.closure presence"
        );
    }

    #[test]
    fn delayed_reality_feedback_writes_correction_and_reinforces_truth() {
        let mut brain = ThreeCell::new(
            0.15,
            0.15,
            4,
            GenomeConfig {
                reinforce_threshold: 0.02,
                novelty_threshold: 0.20,
                merge_threshold: 0.01,
                co_resonance_merge_threshold: 0.0,
            },
        );

        let context = domain_embed(b"titanic:row=ctx", 0.0);
        let dead = domain_embed(b"titanic:label=died", 0.0);
        let survived = domain_embed(b"titanic:label=survived", 0.0);

        brain.hierarchy_mut().genomes[0].seed_dna(context, 1, 0.0, 0.0);
        brain.hierarchy_mut().genomes[0].seed_dna(dead, 1, 0.0, 0.0);
        brain.hierarchy_mut().genomes[0].seed_dna(survived, 1, 0.0, 0.0);

        let source = brain
            .hierarchy()
            .genomes[0]
            .nearest_index(&context)
            .map(PredictionSource::GenomeSlot)
            .unwrap_or(PredictionSource::GeometricFallback(context));
        brain.stage_prediction(dead, context, source, vec![]);
        let report = brain
            .evaluate_prediction(&survived)
            .expect("evaluation must fire when a prediction is pending");
        let feedback = report.feedback;

        assert!(!feedback.correct, "dead vs survived should be wrong");
        assert_eq!(feedback.predicted, dead, "predicted label");
        assert_eq!(feedback.actual, survived, "actual label");
        assert!(
            sigma(&feedback.correction) > 0.0,
            "wrong prediction must produce non-zero correction"
        );
        assert!(
            brain.pending_prediction().is_none(),
            "evaluation must clear pending prediction"
        );

        let genome = &brain.hierarchy().genomes[0];
        let context_idx = genome.nearest_index(&context).expect("context slot");
        let survived_idx = genome.nearest_index(&survived).expect("true label slot");
        assert!(
            genome.entries[context_idx]
                .edges
                .iter()
                .any(|(target, count)| *target == survived_idx && *count > 0),
            "context must reinforce the true label after reality feedback"
        );
    }

    #[test]
    fn ingest_does_not_consume_pending_prediction() {
        let mut brain = ThreeCell::new(
            0.15,
            0.15,
            4,
            GenomeConfig {
                reinforce_threshold: 0.02,
                novelty_threshold: 0.20,
                merge_threshold: 0.01,
                co_resonance_merge_threshold: 0.0,
            },
        );

        let context = domain_embed(b"ctx", 0.0);
        let predicted = domain_embed(b"pred", 0.1);
        let source = brain
            .hierarchy()
            .genomes[0]
            .nearest_index(&context)
            .map(PredictionSource::GenomeSlot)
            .unwrap_or(PredictionSource::GeometricFallback(context));
        brain.stage_prediction(predicted, context, source, vec![]);

        let _ = brain.ingest(&domain_embed(b"plain_input", 0.2));
        assert!(
            brain.pending_prediction().is_some(),
            "perception must not consume staged evaluation"
        );
    }

    #[test]
    fn repeated_corrected_resonance_births_level1_category() {
        // Architecture law: a Response cluster that has been activated at least
        // PROMOTION_MIN_ACTIVATIONS times AND has co-resonance evidence with a
        // coalition partner MUST be promoted to genomes[1] after consolidation.
        //
        // This test directly exercises the full path:
        //   learn_response (two distinct contexts) →
        //   record_co_resonance (both active together) →
        //   accumulate zread_read_count (denominator for mean_co_resonance) →
        //   consolidate →
        //   collect_promotion_candidates →
        //   ingest into genomes[1]
        //
        // Geometry: two contexts σ = 0.04 apart (within merge_threshold 0.05).
        // Response: same value for both.  After consolidation, both entries
        // survive BKT pruning (mean_zread_t starts at 1.0 for fresh entries).
        // Co-resonance is injected directly to simulate two shared-coalition reads.

        let config = GenomeConfig {
            reinforce_threshold: 0.01,
            novelty_threshold: 0.15,
            merge_threshold: 0.05,
            co_resonance_merge_threshold: CO_RESONANCE_FLOOR,
        };
        let mut genome = Genome::new(config.clone());

        // Two contexts separated by σ ≈ 0.03 — within merge_threshold (0.05)
        // but well above reinforce_threshold (0.01), so two distinct entries are created.
        let ctx_a = axis_rotation(0.05, 0); // σ ≈ 0.025
        let ctx_b = axis_rotation(0.11, 0); // σ ≈ 0.055; gap to ctx_a ≈ 0.03
        let response = axis_rotation(PI / 2.0, 1); // arbitrary stable response

        // Write two Response entries via the normal write law.
        genome.learn_response(&ctx_a, &response);
        genome.learn_response(&ctx_b, &response);
        assert_eq!(
            genome.entries.iter().filter(|e| e.layer == Layer::Response).count(),
            2,
            "two distinct Response entries must exist before consolidation"
        );

        // Find the indices.
        let idx_a = genome.entries.iter().position(|e| e.layer == Layer::Response).unwrap();
        let idx_b = idx_a + 1;

        // Simulate two joint prediction/correction events:
        // - both entries active (ZREAD T ≥ ZREAD_T_MIN) in the same coalition
        // - both accumulate activation_count and zread statistics
        let t_val = ZREAD_T_MIN + 0.1; // above the floor
        // Positive salience_x > PROMOTION_MIN_SALIENCE — salient-forward context.
        let sim_salience_x = 0.5_f64;
        for _ in 0..2 {
            // record_co_resonance expects (index, coupling_t) pairs.
            genome.record_co_resonance(&[(idx_a, t_val), (idx_b, t_val)]);
            // Increment zread_read_count for the mean_co_resonance denominator.
            genome.entries[idx_a].zread_t_sum += t_val;
            genome.entries[idx_a].zread_read_count += 1;
            genome.entries[idx_b].zread_t_sum += t_val;
            genome.entries[idx_b].zread_read_count += 1;
            // Simulate credit_response: activation_count + salient-forward accumulation.
            genome.entries[idx_a].activation_count += 1;
            genome.entries[idx_b].activation_count += 1;
            genome.entries[idx_a].salience_sum += sim_salience_x.abs();
            genome.entries[idx_a].salience_count += 1;
            genome.entries[idx_b].salience_sum += sim_salience_x.abs();
            genome.entries[idx_b].salience_count += 1;
        }

        // Both entries must now meet the promotion criteria individually.
        assert!(
            genome.entries[idx_a].activation_count >= PROMOTION_MIN_ACTIVATIONS,
            "entry A must have enough activations"
        );
        let co_ab = genome.mean_co_resonance(idx_a, idx_b);
        assert!(
            co_ab >= CO_RESONANCE_FLOOR,
            "co-resonance between A and B must meet the floor: got {co_ab:.4} vs {CO_RESONANCE_FLOOR:.4}"
        );

        // Consolidate — candidates are collected INSIDE consolidate() before the
        // merge/prune pass, while coalition evidence is intact.
        let report = consolidate(&mut genome);

        // Promotion candidates are in the report, collected pre-merge.
        let candidates = report.promotion_candidates;
        assert!(
            !candidates.is_empty(),
            "at least one promotion candidate must exist after repeated corrected co-resonance"
        );

        // The candidate's carrier must be the stable response value (or near it).
        let sigma_to_response = candidates
            .iter()
            .map(|c| sigma(&compose(&c.carrier, &inverse(&response))))
            .fold(f64::MAX, f64::min);
        assert!(
            sigma_to_response < 0.15,
            "promoted carrier must be near the stable response value: σ = {sigma_to_response:.4}"
        );

        // Wire the promotion into a level-1 genome (simulating ThreeCell::consolidate_level).
        let mut level1 = Genome::new(config);
        for candidate in &candidates {
            level1.ingest(
                &candidate.carrier,
                candidate.activation_count,
                candidate.closure_sigma,
                candidate.excursion_peak,
            );
        }
        assert!(
            !level1.entries.is_empty(),
            "genomes[1] must receive the promoted category"
        );
    }

    #[test]
    fn single_activation_does_not_promote() {
        // Architecture law: a Response entry created by a single prediction
        // event must NOT be promoted. Category birth requires recurrence.
        let config = GenomeConfig {
            reinforce_threshold: 0.01,
            novelty_threshold: 0.15,
            merge_threshold: 0.05,
            co_resonance_merge_threshold: CO_RESONANCE_FLOOR,
        };
        let mut genome = Genome::new(config);
        let ctx = axis_rotation(0.08, 0);
        let response = axis_rotation(PI / 2.0, 1);

        genome.learn_response(&ctx, &response);
        // activation_count = 1 (set at creation). No co-resonance recorded.
        // Do NOT simulate additional activations.

        consolidate(&mut genome);
        let candidates = collect_promotion_candidates(&genome);
        assert!(
            candidates.is_empty(),
            "a single-activation Response entry must not produce a promotion candidate"
        );
    }

    #[test]
    fn category_birth_accumulates_through_real_genome_api() {
        // Full-loop test: drives co-resonance and activation accumulation
        // through actual genome methods — learn_response, credit_response,
        // record_co_resonance, record_zread_contributions — not raw field writes.
        //
        // Architecture law verified here:
        //   Two Response entries in nearby contexts repeatedly participate in
        //   the same ZREAD coalition and receive correction toward the same
        //   fixed point. After two such events they meet ALL promotion criteria:
        //   BKT-alive, activation_count >= PROMOTION_MIN_ACTIVATIONS, and
        //   mean_co_resonance >= CO_RESONANCE_FLOOR. Consolidation must
        //   return them as promotion candidates.
        let config = GenomeConfig {
            reinforce_threshold: 0.01,
            novelty_threshold: 0.15,
            merge_threshold: 0.05,
            co_resonance_merge_threshold: CO_RESONANCE_FLOOR,
        };
        let mut genome = Genome::new(config);

        // Two contexts separated by σ ≈ 0.03 — distinct entries, within merge_threshold.
        let ctx_a = axis_rotation(0.05, 0);
        let ctx_b = axis_rotation(0.11, 0);
        let true_response = axis_rotation(PI / 2.0, 1);

        // Create two distinct Response entries via the write path.
        genome.learn_response(&ctx_a, &true_response);
        genome.learn_response(&ctx_b, &true_response);
        assert_eq!(
            genome.entries.iter().filter(|e| e.layer == Layer::Response).count(),
            2
        );

        let idx_a = genome
            .entries
            .iter()
            .position(|e| e.layer == Layer::Response)
            .unwrap();
        let idx_b = idx_a + 1;

        // Query midpoint — close enough to both entries that both exceed ZREAD_T_MIN.
        let query = axis_rotation(0.08, 0);

        // Two full correction events via real API:
        //   record_zread_contributions — increments zread_read_count (mean_co_resonance denominator)
        //   eligibility from address_distance/coupling_from_gap
        //   record_co_resonance — accumulates t_a * t_b into co_resonance lists
        //   credit_response — SLERPs value, increments activation_count
        let correction_sigma = PI / 4.0; // non-trivial correction

        for _ in 0..2 {
            // Increment zread_read_count for both entries (denominator of mean_co_resonance).
            genome.record_zread_contributions(&query, HopfChannel::W, AddressMode::Full);

            // Build eligibility inline (address_distance + coupling_from_gap + ZREAD_T_MIN gate).
            let eligibility: Vec<(usize, f64)> = genome
                .entries
                .iter()
                .enumerate()
                .filter(|(_, e)| e.layer == Layer::Response)
                .filter_map(|(i, e)| {
                    let gap = address_distance(&query, &e.address.geometry(), AddressMode::Full);
                    let t = coupling_from_gap(gap, AddressMode::Full);
                    if t >= ZREAD_T_MIN { Some((i, t)) } else { None }
                })
                .collect();

            // Both entries must be in the coalition.
            assert!(
                eligibility.iter().any(|(i, _)| *i == idx_a),
                "entry A must be in the ZREAD coalition"
            );
            assert!(
                eligibility.iter().any(|(i, _)| *i == idx_b),
                "entry B must be in the ZREAD coalition"
            );

            // Accumulate pairwise coalition evidence.
            genome.record_co_resonance(&eligibility);

            // Apply correction — increments activation_count.
            // Use a positive salience_x so the test verifies the full path
            // (salience_x = 0.5 → salient-forward, above PROMOTION_MIN_SALIENCE).
            let test_salience_x = 0.5_f64;
            genome.credit_response(&eligibility, &true_response, correction_sigma, test_salience_x, 0.0);
        }

        // Verify: both entries have real recurrence evidence from real API calls.
        assert!(
            genome.entries[idx_a].activation_count >= PROMOTION_MIN_ACTIVATIONS,
            "activation_count must reflect real correction events: got {}",
            genome.entries[idx_a].activation_count
        );
        let co = genome
            .mean_co_resonance(idx_a, idx_b)
            .max(genome.mean_co_resonance(idx_b, idx_a));
        assert!(
            co >= CO_RESONANCE_FLOOR,
            "mean co-resonance must meet floor via real API: got {co:.4}"
        );

        // Consolidation must return promotion candidates (pre-merge, evidence intact).
        let report = consolidate(&mut genome);
        assert!(
            !report.promotion_candidates.is_empty(),
            "real API-driven co-resonance must produce promotion candidates"
        );

        // Promoted carrier must be near the true response.
        let best_gap = report
            .promotion_candidates
            .iter()
            .map(|c| sigma(&compose(&c.carrier, &inverse(&true_response))))
            .fold(f64::MAX, f64::min);
        assert!(
            best_gap < 0.2,
            "promoted carrier must be near the true response: σ = {best_gap:.4}"
        );

        // activation_count on the candidate is the recurrence strength, not support.
        assert!(
            report.promotion_candidates.iter().all(|c| c.activation_count >= PROMOTION_MIN_ACTIVATIONS),
            "all promoted candidates must carry activation_count as recurrence strength"
        );
    }

    #[test]
    fn living_loop_drives_category_birth_into_level1() {
        // Full ThreeCell end-to-end test.
        //
        // Architecture law: repeated predictions on nearby contexts, corrected
        // toward the same response, must eventually fire category birth in the
        // living runtime loop — not via direct genome API manipulation.
        //
        // Path: ingest_sequence -> commit_prediction -> evaluate_prediction
        //       -> consolidate_level -> genomes[1] promotion.
        // After enough corrections, consolidation pressure exceeds SIGMA_BALANCE
        // and promoted_categories > 0.
        let mut brain = ThreeCell::new(
            0.15,
            0.15,
            4,
            GenomeConfig {
                reinforce_threshold: 0.02,
                novelty_threshold: 0.20,
                merge_threshold: 0.05,
                co_resonance_merge_threshold: CO_RESONANCE_FLOOR,
            },
        );

        let input_a = domain_embed(b"feature:a", 0.0);
        let input_b = domain_embed(b"feature:b", 0.1);
        let true_label = domain_embed(b"label:true", 0.0);

        // Run commit_prediction() + evaluate_prediction() repeatedly.
        // Drive the loop directly to observe EvaluationReport and accumulate
        // promoted_categories across all automatic consolidation events.
        // Full path: ingest_sequence -> commit_prediction -> evaluate_prediction
        //            -> consolidate_level -> promote
        let mut total_promoted = 0usize;
        for _ in 0..12 {
            for input in &[input_a, input_b] {
                let steps = brain.ingest_sequence(&[*input]);
                let last = steps.last();
                let predicted = last
                    .and_then(|s| s.field_read.as_ref())
                    .map(|h| h.carrier)
                    .unwrap_or(IDENTITY);
                let source = last
                    .and_then(|s| s.field_read.as_ref())
                    .map(|h| PredictionSource::GenomeSlot(h.index))
                    .unwrap_or(PredictionSource::GeometricFallback(predicted));
                brain.commit_prediction(predicted, source);
                if let Some(eval) = brain.evaluate_prediction(&true_label) {
                    total_promoted += eval
                        .consolidation_reports
                        .iter()
                        .map(|r| r.structural.promoted_categories)
                        .sum::<usize>();
                }
            }
        }
        // Force a final pass to catch candidates accumulated since last auto-consolidation.
        let final_reports = brain.force_consolidate();
        total_promoted += final_reports
            .iter()
            .map(|r| r.structural.promoted_categories)
            .sum::<usize>();

        let promoted = total_promoted;

        assert!(
            promoted > 0,
            "repeated corrected predictions through the living loop must birth a level-1 category"
        );

        // The level-1 entry must be near the true response (stable fixed point).
        let level1 = brain.hierarchy().genome_at(1).expect("level-1 genome must exist");
        let best_gap = level1
            .entries
            .iter()
            .map(|e| sigma(&compose(&e.value, &inverse(&true_label))))
            .fold(f64::MAX, f64::min);
        assert!(
            best_gap < 0.3,
            "level-1 category must be near the stable response: σ = {best_gap:.4}"
        );
    }

    /// Proves: mean_co_resonance = sum / ALL reads of i (persistence denominator).
    ///
    /// Two BKT-alive entries.  Entry i is read twice; j is only present for
    /// the second read (one joint activation).  Joint t-product = BKT_THRESHOLD².
    ///
    ///   mean_co_resonance(i, j) = BKT² / 2   (< CO_RESONANCE_FLOOR = BKT²)
    ///
    /// A pair that coincidentally co-activated once in two reads must fail
    /// the promotion gate.  One coincidence is not a category.
    #[test]
    fn solo_read_counts_in_denominator_so_half_coalition_fails_promotion() {
        let t = BKT_THRESHOLD;
        let joint_product = t * t; // exactly CO_RESONANCE_FLOOR when both reads are joint

        let mut genome = Genome::new(GenomeConfig {
            reinforce_threshold: 0.02,
            novelty_threshold: 0.30,
            merge_threshold: 0.01,
            co_resonance_merge_threshold: CO_RESONANCE_FLOOR,
        });

        // Distinct contexts AND distinct response carriers so we can identify
        // which entry appears in the promotion candidate list.
        let ctx_a = axis_rotation(0.30, 0);
        let ctx_b = axis_rotation(0.60, 0);
        let response_a = axis_rotation(0.45, 0);
        let response_b = axis_rotation(0.45, 1);

        genome.learn_response(&ctx_a, &response_a);
        let idx_a = genome.last_entry.unwrap();
        genome.learn_response(&ctx_b, &response_b);
        let idx_b = genome.last_entry.unwrap();

        // Gate: both entries need activation_count >= 2.
        genome.entries[idx_a].activation_count = PROMOTION_MIN_ACTIVATIONS;
        genome.entries[idx_b].activation_count = PROMOTION_MIN_ACTIVATIONS;

        // Read i alone (first read — no j): i.zread_read_count = 1, no co-resonance.
        genome.entries[idx_a].zread_t_sum += t;
        genome.entries[idx_a].zread_read_count += 1;

        // Read both together (second read): both read, co-resonance += BKT².
        genome.entries[idx_a].zread_t_sum += t;
        genome.entries[idx_a].zread_read_count += 1;
        genome.entries[idx_b].zread_t_sum += t;
        genome.entries[idx_b].zread_read_count += 1;
        genome.record_co_resonance(&[(idx_a, t), (idx_b, t)]);

        // mean_co_resonance(a, b) = joint_sum / all_reads_of_a = BKT² / 2.
        let mean = genome.mean_co_resonance(idx_a, idx_b);
        assert!(
            (mean - joint_product / 2.0).abs() < 1e-12,
            "mean must equal joint_sum / all_reads = BKT²/2: got {mean:.8}"
        );

        // BKT²/2 must be below the floor.
        assert!(
            mean < CO_RESONANCE_FLOOR,
            "half-coalition mean={mean:.4} must be below floor={CO_RESONANCE_FLOOR:.4}"
        );

        // idx_a had 2 reads, only 1 joint → mean < floor → must NOT be promoted.
        // idx_b had 1 read, 1 joint → mean = floor → may be promoted (perfect record).
        // The law we are testing: idx_a specifically must fail.
        let carrier_a = response_a;
        let candidates = collect_promotion_candidates(&genome);
        let a_promoted = candidates.iter().any(|c| {
            sigma(&compose(&c.carrier, &inverse(&carrier_a))) < 1e-9
        });
        assert!(
            !a_promoted,
            "entry with 2 reads / 1 joint must not be promoted (mean = BKT²/2 < floor); candidates: {candidates:?}"
        );
    }

    fn make_valence_brain() -> ThreeCell {
        ThreeCell::new(
            0.15,
            0.15,
            4,
            GenomeConfig {
                reinforce_threshold: 0.02,
                novelty_threshold: 0.20,
                merge_threshold: 0.01,
                co_resonance_merge_threshold: CO_RESONANCE_FLOOR,
            },
        )
    }

    /// Valence is defined as prev_sfe − sfe.  This test verifies the arithmetic
    /// is wired correctly before testing the direction.
    #[test]
    fn valence_arithmetic_is_correctly_wired() {
        let mut brain = make_valence_brain();
        let q1 = domain_embed(b"first", 0.0);
        let q2 = domain_embed(b"second", 0.0);

        let step1 = brain.ingest(&q1);
        let step2 = brain.ingest(&q2);

        // step1: no previous measurement exists, so valence = 0.0 (no delta without two points).
        assert!(
            step1.valence.abs() < 1e-12,
            "first step valence must be 0.0 (no previous sfe): got {:.6}",
            step1.valence
        );

        // step2: prev_sfe is step1.self_free_energy
        let expected_v2 = step1.self_free_energy - step2.self_free_energy;
        assert!(
            (step2.valence - expected_v2).abs() < 1e-12,
            "valence must equal prev_sfe - self_free_energy: got {:.6}, expected {:.6}",
            step2.valence,
            expected_v2
        );
    }

    /// After repeated exposure to the same carrier the genome reinforces it.
    /// At least one step in the sequence must be coherence-driving (valence > 0):
    /// the brain moved toward self-consistency, not away from it.  A sequence
    /// where every step is decoherence-driving contradicts the FEP attractor.
    #[test]
    fn coherence_driving_step_has_positive_valence() {
        let mut brain = make_valence_brain();
        let q = domain_embed(b"repeated", 0.0);

        let steps: Vec<_> = (0..12).map(|_| brain.ingest(&q)).collect();

        let has_coherence = steps.iter().any(|s| s.valence > 0.0);
        assert!(
            has_coherence,
            "repeated familiar carrier must produce at least one coherence-driving step; valences: {:?}",
            steps.iter().map(|s| s.valence).collect::<Vec<_>>()
        );
    }

    // ── Lateral inhibition tests ────────────────────────────────────────────

    /// Near loser must be suppressed.
    ///
    /// Two Response entries on the same rotation axis, PI/8 apart in rotation
    /// angle.  Query = winner address (t_winner = 1.0).  Loser coupling to
    /// query = cos(PI/16) ≈ 0.981; loser overlap with winner = cos(PI/16) ≈ 0.981.
    /// t′_loser = 0.981 − 0.981 × 1.0 = 0 < ZREAD_T_MIN → suppressed.
    #[test]
    fn lateral_inhibition_suppresses_near_loser() {
        let mut genome = Genome::new(GenomeConfig {
            reinforce_threshold: 0.01,
            novelty_threshold: 0.3,
            merge_threshold: 0.01,
            co_resonance_merge_threshold: 0.0,
        });
        let winner_ctx = axis_rotation(PI / 4.0, 0);
        let loser_ctx = axis_rotation(3.0 * PI / 8.0, 0); // PI/8 further along same axis
        let value = axis_rotation(PI / 6.0, 1);
        genome.learn_response(&winner_ctx, &value);
        genome.learn_response(&loser_ctx, &value);
        assert_eq!(
            genome.entries.iter().filter(|e| e.layer == Layer::Response).count(),
            2,
        );

        let inhibited = crate::field::collect_response_eligibility(&winner_ctx, &genome);

        assert_eq!(
            inhibited.len(),
            1,
            "near loser must be suppressed; got {} survivors",
            inhibited.len()
        );
        // Survivor is the winner
        let survivor_gap = address_distance(
            &winner_ctx,
            &genome.entries[inhibited[0].0].address.geometry(),
            AddressMode::Full,
        );
        assert!(
            survivor_gap < 1e-6,
            "survivor must be the winner entry; σ = {survivor_gap:.6}"
        );
    }

    /// Far loser must NOT be suppressed.
    ///
    /// Winner address ≈ IDENTITY; loser address = [0,1,0,0] (orthogonal, σ = π/2).
    /// Query = [0.8, 0.6, 0, 0]: t_winner = 0.8, t_loser = 0.6.
    /// o_j = cos(π/2) = 0 → t′_loser = 0.6 ≥ ZREAD_T_MIN → survives.
    #[test]
    fn lateral_inhibition_preserves_far_loser() {
        let mut genome = Genome::new(GenomeConfig {
            reinforce_threshold: 0.01,
            novelty_threshold: 0.3,
            merge_threshold: 0.01,
            co_resonance_merge_threshold: 0.0,
        });
        let winner_ctx = IDENTITY;
        let loser_ctx = [0.0f64, 1.0, 0.0, 0.0]; // pure-i quaternion, σ = π/2 from IDENTITY
        let value = axis_rotation(PI / 6.0, 1);
        genome.learn_response(&winner_ctx, &value);
        genome.learn_response(&loser_ctx, &value);
        assert_eq!(
            genome.entries.iter().filter(|e| e.layer == Layer::Response).count(),
            2,
        );

        // query: |w|=0.8 (winner coupling), |x|=0.6 (loser coupling)
        let query = [0.8f64, 0.6, 0.0, 0.0];
        let inhibited = crate::field::collect_response_eligibility(&query, &genome);

        assert_eq!(
            inhibited.len(),
            2,
            "far loser must survive inhibition; got {} survivors",
            inhibited.len()
        );
    }

    /// Winner's coupling t must be unchanged by inhibition.
    #[test]
    fn lateral_inhibition_winner_t_unchanged() {
        let mut genome = Genome::new(GenomeConfig {
            reinforce_threshold: 0.01,
            novelty_threshold: 0.3,
            merge_threshold: 0.01,
            co_resonance_merge_threshold: 0.0,
        });
        let winner_ctx = axis_rotation(PI / 4.0, 0);
        let loser_ctx = axis_rotation(3.0 * PI / 8.0, 0);
        let value = axis_rotation(PI / 6.0, 1);
        genome.learn_response(&winner_ctx, &value);
        genome.learn_response(&loser_ctx, &value);

        let inhibited = crate::field::collect_response_eligibility(&winner_ctx, &genome);

        let expected_t = coupling_from_gap(
            address_distance(&winner_ctx, &winner_ctx, AddressMode::Full),
            AddressMode::Full,
        ); // = cos(0) = 1.0

        let winner_entry = inhibited.iter().find(|&&(i, _)| {
            address_distance(
                &winner_ctx,
                &genome.entries[i].address.geometry(),
                AddressMode::Full,
            ) < 1e-6
        });
        let winner_t = winner_entry
            .map(|&(_, t)| t)
            .expect("winner must appear in inhibited set");
        assert!(
            (winner_t - expected_t).abs() < 1e-12,
            "winner t must be unchanged: got {winner_t:.6}, expected {expected_t:.6}"
        );
    }

    #[test]
    fn pending_prediction_persists_in_brain_state() {
        let mut brain = ThreeCell::new(
            0.15,
            0.15,
            4,
            GenomeConfig {
                reinforce_threshold: 0.02,
                novelty_threshold: 0.20,
                merge_threshold: 0.01,
                co_resonance_merge_threshold: 0.0,
            },
        );

        let context = domain_embed(b"ctx", 0.0);
        let predicted = domain_embed(b"pred", 0.1);
        let source = PredictionSource::GeometricFallback(context);
        brain.stage_prediction(predicted, context, source, vec![]);

        let state = brain.to_brain_state();
        let restored = ThreeCell::from_brain_state(state);
        assert_eq!(
            brain.pending_prediction(),
            restored.pending_prediction(),
            "pending prediction must survive BrainState round trip"
        );
    }

    // ── RGB semantic tests ────────────────────────────────────────────────────

    #[test]
    fn semantic_frame_is_identity_when_total_equals_known() {
        // If the field reads exactly what the model predicts, the residual
        // is identity and salience is zero.
        let q = axis_rotation(PI / 4.0, 0);
        let frame = semantic_frame(&q, &q);
        assert!(
            frame.salience_sigma < 1e-10,
            "salience_sigma must be zero when total == known, got {:.2e}",
            frame.salience_sigma
        );
        assert!(
            frame.rgb_gap < 1e-10,
            "rgb_gap must be zero when total == known, got {:.2e}",
            frame.rgb_gap
        );
    }

    #[test]
    fn semantic_frame_residual_x_is_antisymmetric_under_swap() {
        // sigma (geodesic distance) is symmetric — the audit's key observation.
        // residual[1] (X-component, salience axis) is antisymmetric.
        // Swapping total and known must negate residual[1].
        let total = axis_rotation(PI / 4.0, 0);
        let known = axis_rotation(PI / 6.0, 1);
        let forward  = semantic_frame(&total, &known);
        let backward = semantic_frame(&known, &total);

        // Sigma must be symmetric (same both ways).
        assert!(
            (forward.salience_sigma - backward.salience_sigma).abs() < 1e-10,
            "salience_sigma must be symmetric: {:.6} vs {:.6}",
            forward.salience_sigma, backward.salience_sigma
        );
        // residual[1] must flip sign exactly.
        assert!(
            (forward.residual[1] + backward.residual[1]).abs() < 1e-10,
            "residual[1] must negate when total/known are swapped: {:.6} + {:.6} = {:.2e}",
            forward.residual[1], backward.residual[1],
            forward.residual[1] + backward.residual[1]
        );
        // The two must not both be zero (otherwise the antisymmetry claim is vacuous).
        assert!(
            forward.residual[1].abs() > 1e-6,
            "residual[1] must be non-trivial for this test to be meaningful"
        );
    }

    #[test]
    fn positive_salience_x_amplifies_credit_over_negative() {
        // Forward salience (salience_x > 0) gives factor > 1.
        // Anti-salience (salience_x < 0) gives factor < 1.
        // Both factors are positive so learning always occurs, but the rates differ —
        // this is the genuine asymmetry that makes swapping total/known detectable.
        let config = GenomeConfig {
            reinforce_threshold: 0.01,
            novelty_threshold: 0.5,
            merge_threshold: 0.5,
            co_resonance_merge_threshold: 0.0,
        };
        let context  = axis_rotation(0.1, 0);
        let initial  = axis_rotation(0.2, 1);
        let target   = axis_rotation(PI / 2.0, 0);
        let correction = sigma(&compose(&target, &inverse(&initial)));
        assert!(correction > 0.1, "test requires non-trivial correction");

        // Salient-forward: factor = 1 + 0.8 = 1.8 → amplified rate.
        let mut g_fwd = Genome::new(config.clone());
        g_fwd.learn_response(&context, &initial);
        let idx_fwd = g_fwd.entries.iter().position(|e| e.layer == Layer::Response).unwrap();
        g_fwd.credit_response(&[(idx_fwd, 1.0)], &target, correction, 0.8, 0.0);
        let dist_fwd = sigma(&compose(&g_fwd.entries[idx_fwd].value, &inverse(&target)));

        // Anti-salient: factor = 1 − 0.8 = 0.2 → attenuated rate.
        let mut g_rev = Genome::new(config);
        g_rev.learn_response(&context, &initial);
        let idx_rev = g_rev.entries.iter().position(|e| e.layer == Layer::Response).unwrap();
        g_rev.credit_response(&[(idx_rev, 1.0)], &target, correction, -0.8, 0.0);
        let dist_rev = sigma(&compose(&g_rev.entries[idx_rev].value, &inverse(&target)));

        assert!(
            dist_fwd < dist_rev,
            "forward salience must move entry closer to target than anti-salience: \
             fwd σ={:.4} rev σ={:.4}",
            dist_fwd, dist_rev
        );
        // Anti-salient must still produce some movement (not zero).
        let initial_dist = sigma(&compose(&initial, &inverse(&target)));
        assert!(
            dist_rev < initial_dist,
            "anti-salient correction must still move the entry: initial={:.4} after={:.4}",
            initial_dist, dist_rev
        );
    }

    #[test]
    fn end_to_end_ingest_prediction_evaluation_uses_salience_x() {
        // Full runtime path:
        //   ingest() sets semantic.residual[1] and stores it in last_salience_x
        //   commit_prediction() snapshots that into PendingPrediction.salience_x
        //   evaluate_prediction() forwards it into credit_response()
        //
        // Two otherwise identical brains ingest probes with opposite X-sign.
        // The positive-salience brain should write the shared Response entry
        // farther toward `actual` than the negative-salience brain.
        let config = GenomeConfig {
            reinforce_threshold: 0.01,
            novelty_threshold: 0.5,
            merge_threshold: 0.5,
            co_resonance_merge_threshold: 0.0,
        };

        let mut brain_pos = ThreeCell::new(0.05, 0.05, 4, config.clone());
        let mut brain_neg = ThreeCell::new(0.05, 0.05, 4, config);

        let context = IDENTITY;
        let predicted = axis_rotation(0.2, 1);
        let actual = axis_rotation(PI / 2.0, 0);

        // Seed the same learned response entry in both brains so the only
        // difference comes from the salience_x captured during ingest().
        brain_pos
            .hierarchy_mut()
            .genome_at_mut(0)
            .learn_response(&context, &predicted);
        brain_neg
            .hierarchy_mut()
            .genome_at_mut(0)
            .learn_response(&context, &predicted);

        // Strong probes with opposite X-sign. The positive probe aligns with
        // the default equatorial DNA anchor; the negative probe opposes it.
        let probe_pos = axis_rotation(PI / 2.0, 0);
        let probe_neg = axis_rotation(-PI / 2.0, 0);

        let step_pos = brain_pos.ingest(&probe_pos);
        let step_neg = brain_neg.ingest(&probe_neg);

        assert!(
            step_pos.semantic.residual[1] > 0.05,
            "positive probe must produce positive residual X, got {:.4}",
            step_pos.semantic.residual[1]
        );
        assert!(
            step_neg.semantic.residual[1] < -0.05,
            "negative probe must produce negative residual X, got {:.4}",
            step_neg.semantic.residual[1]
        );

        brain_pos.commit_prediction(predicted, PredictionSource::ZreadAggregate);
        brain_neg.commit_prediction(predicted, PredictionSource::ZreadAggregate);

        let eval_pos = brain_pos
            .evaluate_prediction(&actual)
            .expect("positive-salience brain must have a pending prediction");
        let eval_neg = brain_neg
            .evaluate_prediction(&actual)
            .expect("negative-salience brain must have a pending prediction");

        assert!(
            eval_pos.feedback.sigma > 0.1 && eval_neg.feedback.sigma > 0.1,
            "test requires non-trivial correction in both brains"
        );

        let idx_pos = brain_pos
            .hierarchy()
            .genomes[0]
            .entries
            .iter()
            .position(|e| e.layer == Layer::Response)
            .expect("positive-salience brain must keep a response entry");
        let idx_neg = brain_neg
            .hierarchy()
            .genomes[0]
            .entries
            .iter()
            .position(|e| e.layer == Layer::Response)
            .expect("negative-salience brain must keep a response entry");

        let value_pos = brain_pos.hierarchy().genomes[0].entries[idx_pos].value;
        let value_neg = brain_neg.hierarchy().genomes[0].entries[idx_neg].value;
        let dist_pos = sigma(&compose(&value_pos, &inverse(&actual)));
        let dist_neg = sigma(&compose(&value_neg, &inverse(&actual)));

        assert!(
            dist_pos < dist_neg,
            "positive ingest salience must yield a stronger end-to-end write: \
             pos σ={:.4} neg σ={:.4}",
            dist_pos, dist_neg
        );
    }

    #[test]
    fn full_anti_salience_produces_no_update() {
        // salience_x = 0 → factor = (1 + 0).clamp(0,2) = 1 → rate = correction/π.
        // Wait, that's normal. The TRULY zero-update case is salience_x = -1.0:
        // factor = (1 - 1).clamp(0, 2) = 0 → rate = 0 → no movement.
        let config = GenomeConfig {
            reinforce_threshold: 0.01,
            novelty_threshold: 0.5,
            merge_threshold: 0.5,
            co_resonance_merge_threshold: 0.0,
        };
        let context = axis_rotation(0.1, 0);
        let initial  = axis_rotation(0.2, 1);
        let target   = axis_rotation(PI / 2.0, 0);
        let correction = sigma(&compose(&target, &inverse(&initial)));
        assert!(correction > 0.1, "test requires non-trivial correction");

        let mut g = Genome::new(config);
        g.learn_response(&context, &initial);
        let idx = g.entries.iter().position(|e| e.layer == Layer::Response).unwrap();
        let value_before = g.entries[idx].value;
        // salience_x = -1.0 → factor = 0 → no movement.
        g.credit_response(&[(idx, 1.0)], &target, correction, -1.0, 0.0);
        let value_after = g.entries[idx].value;
        let moved = sigma(&compose(&value_after, &inverse(&value_before)));
        assert!(
            moved < 1e-10,
            "salience_x = -1.0 (full anti-salient) must produce zero update: moved {:.2e}",
            moved
        );
    }

    #[test]
    fn promotion_requires_nonzero_mean_salience() {
        // Entries corrected with salience_x = 0 must not promote (|0| = 0 → mean=0).
        // Entries corrected with nonzero salience_x must promote (|x|>0 → mean>0).
        let config = GenomeConfig {
            reinforce_threshold: 0.01,
            novelty_threshold: 0.5,
            merge_threshold: 0.5,
            co_resonance_merge_threshold: CO_RESONANCE_FLOOR,
        };
        let t_val = ZREAD_T_MIN + 0.1;

        // Two distinct contexts so learn_response creates two separate entries.
        let ctx_a = axis_rotation(0.05, 0);
        let ctx_b = axis_rotation(0.12, 0); // far enough from ctx_a to create a new entry
        let resp   = axis_rotation(PI / 2.0, 1);

        // --- Both entries corrected at zero salience (should NOT promote) ---
        let mut g_zero = Genome::new(config.clone());
        g_zero.learn_response(&ctx_a, &resp);
        g_zero.learn_response(&ctx_b, &resp);
        let resp_entries: Vec<usize> = g_zero.entries.iter().enumerate()
            .filter(|(_, e)| e.layer == Layer::Response)
            .map(|(i, _)| i)
            .collect();
        assert_eq!(resp_entries.len(), 2, "setup: need two distinct Response entries");
        let (idx_za, idx_zb) = (resp_entries[0], resp_entries[1]);
        for _ in 0..2 {
            g_zero.record_co_resonance(&[(idx_za, t_val), (idx_zb, t_val)]);
            g_zero.entries[idx_za].zread_t_sum += t_val;
            g_zero.entries[idx_za].zread_read_count += 1;
            g_zero.entries[idx_zb].zread_t_sum += t_val;
            g_zero.entries[idx_zb].zread_read_count += 1;
            // salience_x = 0 → |0| = 0 → salience_sum stays 0 → mean_salience = 0.
            g_zero.credit_response(&[(idx_za, t_val), (idx_zb, t_val)], &resp, PI / 4.0, 0.0, 0.0);
        }
        let report_z = consolidate(&mut g_zero);
        assert!(
            report_z.promotion_candidates.is_empty(),
            "zero-salience entries must not produce promotion candidates"
        );

        // --- Both entries corrected with nonzero salience (SHOULD promote) ---
        let mut g_sal = Genome::new(config);
        g_sal.learn_response(&ctx_a, &resp);
        g_sal.learn_response(&ctx_b, &resp);
        let resp_sal: Vec<usize> = g_sal.entries.iter().enumerate()
            .filter(|(_, e)| e.layer == Layer::Response)
            .map(|(i, _)| i)
            .collect();
        assert_eq!(resp_sal.len(), 2, "setup: need two distinct Response entries");
        let (idx_sa, idx_sb) = (resp_sal[0], resp_sal[1]);
        for _ in 0..2 {
            g_sal.record_co_resonance(&[(idx_sa, t_val), (idx_sb, t_val)]);
            g_sal.entries[idx_sa].zread_t_sum += t_val;
            g_sal.entries[idx_sa].zread_read_count += 1;
            g_sal.entries[idx_sb].zread_t_sum += t_val;
            g_sal.entries[idx_sb].zread_read_count += 1;
            // |salience_x| = 0.5 > PROMOTION_MIN_SALIENCE → salience accumulates.
            // coherence_tone = 0.5 > 0.0 → mean_coherence > 0 → passes promotion criterion 6.
            g_sal.credit_response(&[(idx_sa, t_val), (idx_sb, t_val)], &resp, PI / 4.0, 0.5, 0.5);
        }
        let report_s = consolidate(&mut g_sal);
        assert!(
            !report_s.promotion_candidates.is_empty(),
            "nonzero-salience entries must produce promotion candidates"
        );
    }

    /// Criterion 6 promotion-coherence gate: three cases.
    ///
    /// Case A — insufficient coherence history (coherence_count < PROMOTION_MIN_ACTIVATIONS):
    ///   gate does not apply regardless of mean_coherence value.
    ///   Entry must not be blocked.
    ///
    /// Case B — sufficient history, net-negative coherence (mean < 0.0):
    ///   gate applies and blocks promotion.
    ///
    /// Case C — sufficient history, net-nonnegative coherence (mean >= 0.0):
    ///   gate applies and passes.
    #[test]
    fn promotion_coherence_gate_insufficient_history_does_not_block() {
        use std::f64::consts::PI;
        let t = BKT_THRESHOLD + 0.05;
        let config = GenomeConfig {
            reinforce_threshold: 0.001,
            novelty_threshold: 0.30,
            merge_threshold: 0.01,
            co_resonance_merge_threshold: CO_RESONANCE_FLOOR,
        };
        let ctx_a = domain_embed(b"ctx:a", 0.0);
        let ctx_b = domain_embed(b"ctx:b", 0.1);
        let resp   = domain_embed(b"resp", 0.0);

        // Build a genome with two Response entries that pass all criteria
        // except coherence history is 1 (below PROMOTION_MIN_ACTIVATIONS = 2).
        // Mean coherence is strongly negative — but the gate must NOT apply yet.
        let mut g = Genome::new(config);
        g.learn_response(&ctx_a, &resp);
        g.learn_response(&ctx_b, &resp);
        let resp_idx: Vec<usize> = g.entries.iter().enumerate()
            .filter(|(_, e)| e.layer == Layer::Response)
            .map(|(i, _)| i)
            .collect();
        assert_eq!(resp_idx.len(), 2);
        let (ia, ib) = (resp_idx[0], resp_idx[1]);

        // One correction with strongly negative coherence_tone = -0.8.
        // coherence_count will be 1 after this — below PROMOTION_MIN_ACTIVATIONS.
        g.record_co_resonance(&[(ia, t), (ib, t)]);
        g.entries[ia].zread_t_sum += t;
        g.entries[ia].zread_read_count += 1;
        g.entries[ib].zread_t_sum += t;
        g.entries[ib].zread_read_count += 1;
        g.credit_response(&[(ia, t), (ib, t)], &resp, PI / 4.0, 0.5, -0.8);

        assert_eq!(g.entries[ia].coherence_count, 1, "setup: coherence_count must be 1");
        assert!(
            g.entries[ia].mean_coherence() < PROMOTION_MIN_COHERENCE,
            "setup: mean_coherence must be below threshold"
        );

        let report = consolidate(&mut g);
        assert!(
            !report.promotion_candidates.is_empty(),
            "insufficient coherence history must not block promotion: coherence_count=1 < PROMOTION_MIN_ACTIVATIONS={PROMOTION_MIN_ACTIVATIONS}"
        );
    }

    #[test]
    fn promotion_coherence_gate_negative_history_blocks() {
        use std::f64::consts::PI;
        let t = BKT_THRESHOLD + 0.05;
        let config = GenomeConfig {
            reinforce_threshold: 0.001,
            novelty_threshold: 0.30,
            merge_threshold: 0.01,
            co_resonance_merge_threshold: CO_RESONANCE_FLOOR,
        };
        let ctx_a = domain_embed(b"ctx:a", 0.0);
        let ctx_b = domain_embed(b"ctx:b", 0.1);
        let resp   = domain_embed(b"resp", 0.0);

        let mut g = Genome::new(config);
        g.learn_response(&ctx_a, &resp);
        g.learn_response(&ctx_b, &resp);
        let resp_idx: Vec<usize> = g.entries.iter().enumerate()
            .filter(|(_, e)| e.layer == Layer::Response)
            .map(|(i, _)| i)
            .collect();
        let (ia, ib) = (resp_idx[0], resp_idx[1]);

        // PROMOTION_MIN_ACTIVATIONS = 2 corrections, both with strongly negative coherence.
        for _ in 0..PROMOTION_MIN_ACTIVATIONS {
            g.record_co_resonance(&[(ia, t), (ib, t)]);
            g.entries[ia].zread_t_sum += t;
            g.entries[ia].zread_read_count += 1;
            g.entries[ib].zread_t_sum += t;
            g.entries[ib].zread_read_count += 1;
            g.credit_response(&[(ia, t), (ib, t)], &resp, PI / 4.0, 0.5, -0.8);
        }

        assert!(g.entries[ia].coherence_count >= PROMOTION_MIN_ACTIVATIONS);
        assert!(g.entries[ia].mean_coherence() < PROMOTION_MIN_COHERENCE);

        let report = consolidate(&mut g);
        assert!(
            report.promotion_candidates.is_empty(),
            "sufficient negative coherence history must block promotion"
        );
    }

    #[test]
    fn promotion_coherence_gate_nonnegative_history_passes() {
        use std::f64::consts::PI;
        let t = BKT_THRESHOLD + 0.05;
        let config = GenomeConfig {
            reinforce_threshold: 0.001,
            novelty_threshold: 0.30,
            merge_threshold: 0.01,
            co_resonance_merge_threshold: CO_RESONANCE_FLOOR,
        };
        let ctx_a = domain_embed(b"ctx:a", 0.0);
        let ctx_b = domain_embed(b"ctx:b", 0.1);
        let resp   = domain_embed(b"resp", 0.0);

        let mut g = Genome::new(config);
        g.learn_response(&ctx_a, &resp);
        g.learn_response(&ctx_b, &resp);
        let resp_idx: Vec<usize> = g.entries.iter().enumerate()
            .filter(|(_, e)| e.layer == Layer::Response)
            .map(|(i, _)| i)
            .collect();
        let (ia, ib) = (resp_idx[0], resp_idx[1]);

        // PROMOTION_MIN_ACTIVATIONS corrections with positive coherence_tone.
        for _ in 0..PROMOTION_MIN_ACTIVATIONS {
            g.record_co_resonance(&[(ia, t), (ib, t)]);
            g.entries[ia].zread_t_sum += t;
            g.entries[ia].zread_read_count += 1;
            g.entries[ib].zread_t_sum += t;
            g.entries[ib].zread_read_count += 1;
            g.credit_response(&[(ia, t), (ib, t)], &resp, PI / 4.0, 0.5, 0.4);
        }

        assert!(g.entries[ia].coherence_count >= PROMOTION_MIN_ACTIVATIONS);
        assert!(g.entries[ia].mean_coherence() >= PROMOTION_MIN_COHERENCE);

        let report = consolidate(&mut g);
        assert!(
            !report.promotion_candidates.is_empty(),
            "nonnegative coherence history must pass promotion gate"
        );
    }
}
