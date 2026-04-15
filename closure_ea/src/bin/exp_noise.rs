//! Experiment: solenoidal noise vs deterministic Cell A.
//!
//! Runs identical curricula on two brains — one with noise disabled (the
//! current default) and one with von Mises-Fisher noise on Cell A at the
//! BKT critical scale. Compares:
//!
//!   - Genome growth (how many entries survive BKT pruning)
//!   - Closure count (how often Cell A completes a cycle)
//!   - Self-free-energy trajectory (convergence of the self-model)
//!   - Prediction error over passes (how fast the brain learns the input)
//!
//! Three experiments:
//!   1. Single orbit (easy) — repeated ε, period 6
//!   2. Two competing orbits (medium) — interleaved period-3 and period-5
//!   3. Random carriers (hard) — no structure, pure noise-vs-signal test

use closure_ea::{GenomeConfig, ThreeCell};
use closure_ea::genome::BKT_THRESHOLD;

use std::f64::consts::PI;

fn axis_rotation(angle: f64, axis: usize) -> [f64; 4] {
    let mut q = [0.0; 4];
    q[0] = (angle / 2.0).cos();
    q[1 + axis] = (angle / 2.0).sin();
    q
}

fn make_brain(threshold: f64) -> ThreeCell {
    ThreeCell::new(
        threshold,
        threshold,
        128,
        GenomeConfig {
            reinforce_threshold: 0.01,
            novelty_threshold: 0.10,
            merge_threshold: 0.01,
            co_resonance_merge_threshold: 0.0,
        },
    )
}

/// Build an orbit: ε self-composed n times gives [ε, ε², ..., εⁿ ≈ I].
fn orbit(angle: f64, axis: usize, period: usize) -> Vec<[f64; 4]> {
    let eps = axis_rotation(angle, axis);
    (0..period).map(|_| eps).collect()
}

/// Run a curriculum for N passes and collect per-pass stats.
struct PassStats {
    pass: usize,
    closures: usize,
    genome_size: usize,
    mean_prediction_error: f64,
    self_free_energy: f64,
}

fn run_passes(brain: &mut ThreeCell, corpus: &[[f64; 4]], n_passes: usize) -> Vec<PassStats> {
    let mut stats = Vec::with_capacity(n_passes);
    for pass in 0..n_passes {
        let closures_before = brain.total_closures();
        let mut total_pe = 0.0;
        let mut last_sfe = 0.0;
        for c in corpus {
            let step = brain.ingest(c);
            total_pe += step.prediction_error;
            last_sfe = step.self_free_energy;
        }
        stats.push(PassStats {
            pass,
            closures: brain.total_closures() - closures_before,
            genome_size: brain.genome_size(),
            mean_prediction_error: total_pe / corpus.len() as f64,
            self_free_energy: last_sfe,
        });
    }
    stats
}

fn print_header(name: &str) {
    println!("\n{}", "=".repeat(60));
    println!("  {name}");
    println!("{}", "=".repeat(60));
    println!(
        "{:<6} {:>8} {:>8} {:>12} {:>12}",
        "pass", "closures", "genome", "pred_err", "self_fe"
    );
    println!("{}", "-".repeat(50));
}

fn print_stats(label: &str, stats: &[PassStats]) {
    println!("\n  [{label}]");
    for s in stats {
        println!(
            "{:<6} {:>8} {:>8} {:>12.6} {:>12.6}",
            s.pass, s.closures, s.genome_size, s.mean_prediction_error, s.self_free_energy,
        );
    }
}

fn print_comparison(quiet: &[PassStats], noisy: &[PassStats]) {
    println!("\n  [comparison: noisy - quiet]");
    println!(
        "{:<6} {:>8} {:>8} {:>12} {:>12}",
        "pass", "d_clos", "d_genome", "d_pred_err", "d_self_fe"
    );
    println!("{}", "-".repeat(50));
    for (q, n) in quiet.iter().zip(noisy.iter()) {
        println!(
            "{:<6} {:>+8} {:>+8} {:>+12.6} {:>+12.6}",
            q.pass,
            n.closures as i64 - q.closures as i64,
            n.genome_size as i64 - q.genome_size as i64,
            n.mean_prediction_error - q.mean_prediction_error,
            n.self_free_energy - q.self_free_energy,
        );
    }
}

fn main() {
    let base_kappa = 1.0 / BKT_THRESHOLD;
    let n_passes = 20;

    // ── Experiment 1: single orbit (period 6 on X axis) ──────────────
    {
        print_header("Experiment 1: Single orbit (period 6)");
        let corpus = orbit(2.0 * PI / 6.0, 0, 6);

        let mut brain_q = make_brain(0.05);
        let stats_q = run_passes(&mut brain_q, &corpus, n_passes);

        let mut brain_n = make_brain(0.05);
        brain_n.enable_noise(42, base_kappa);
        let stats_n = run_passes(&mut brain_n, &corpus, n_passes);

        print_stats("quiet (no noise)", &stats_q);
        print_stats("noisy (vMF)", &stats_n);
        print_comparison(&stats_q, &stats_n);
    }

    // ── Experiment 2: two competing orbits (period 3 + period 5) ─────
    {
        print_header("Experiment 2: Competing orbits (period 3 + 5)");
        let orbit_a = orbit(2.0 * PI / 3.0, 0, 3); // X axis
        let orbit_b = orbit(2.0 * PI / 5.0, 1, 5); // Y axis
        // Interleave: AAABBBBBAAABBBBB...
        let corpus: Vec<[f64; 4]> = orbit_a.iter()
            .chain(orbit_b.iter())
            .copied()
            .collect();

        let mut brain_q = make_brain(0.05);
        let stats_q = run_passes(&mut brain_q, &corpus, n_passes);

        let mut brain_n = make_brain(0.05);
        brain_n.enable_noise(42, base_kappa);
        let stats_n = run_passes(&mut brain_n, &corpus, n_passes);

        print_stats("quiet (no noise)", &stats_q);
        print_stats("noisy (vMF)", &stats_n);
        print_comparison(&stats_q, &stats_n);
    }

    // ── Experiment 3: basin-hopping test ─────────────────────────────
    //
    // Two orbits on the same axis but different periods. The brain that
    // gets stuck in one basin sees only period-4. The brain that hops
    // should discover both period-4 and period-7.
    {
        print_header("Experiment 3: Basin hopping (period 4 vs 7, same axis)");
        let eps4 = axis_rotation(2.0 * PI / 4.0, 0);
        let eps7 = axis_rotation(2.0 * PI / 7.0, 0);
        // Alternate: 4 copies of ε₄, then 7 copies of ε₇
        let mut corpus = Vec::new();
        for _ in 0..4 { corpus.push(eps4); }
        for _ in 0..7 { corpus.push(eps7); }

        let mut brain_q = make_brain(0.05);
        let stats_q = run_passes(&mut brain_q, &corpus, n_passes);

        let mut brain_n = make_brain(0.05);
        brain_n.enable_noise(42, base_kappa);
        let stats_n = run_passes(&mut brain_n, &corpus, n_passes);

        print_stats("quiet (no noise)", &stats_q);
        print_stats("noisy (vMF)", &stats_n);
        print_comparison(&stats_q, &stats_n);

        // Report genome contents for both
        println!("\n  [quiet genome entries: {}]", brain_q.genome_size());
        println!("  [noisy genome entries: {}]", brain_n.genome_size());
    }

    // ── Experiment 4: noise sweep — vary κ, measure convergence ──────
    {
        print_header("Experiment 4: Noise sweep (fixed corpus, varying kappa)");
        let corpus = orbit(2.0 * PI / 5.0, 1, 5);
        let kappas = [0.5, 1.0, base_kappa, 4.0, 8.0, 16.0, 64.0];

        println!(
            "{:<8} {:>8} {:>8} {:>12} {:>12}",
            "kappa", "closures", "genome", "pred_err", "self_fe"
        );
        println!("{}", "-".repeat(52));

        for &k in &kappas {
            let mut brain = make_brain(0.05);
            brain.enable_noise(42, k);
            let stats = run_passes(&mut brain, &corpus, n_passes);
            let last = stats.last().unwrap();
            println!(
                "{:<8.2} {:>8} {:>8} {:>12.6} {:>12.6}",
                k, last.closures, last.genome_size,
                last.mean_prediction_error, last.self_free_energy,
            );
        }

        // Also run quiet for reference
        let mut brain = make_brain(0.05);
        let stats = run_passes(&mut brain, &corpus, n_passes);
        let last = stats.last().unwrap();
        println!(
            "{:<8} {:>8} {:>8} {:>12} {:>12}",
            "quiet", last.closures, last.genome_size,
            format!("{:.6}", last.mean_prediction_error),
            format!("{:.6}", last.self_free_energy),
        );
    }
}
