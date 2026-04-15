//! Experiment: solenoidal noise — detailed JSON trace for visualization.
//!
//! Outputs a JSON file with per-step data for both quiet and noisy brains
//! across multiple experiments. The HTML visualization reads this.

use closure_ea::{GenomeConfig, ThreeCell};
use closure_ea::genome::BKT_THRESHOLD;
use closure_ea::hopf;

use std::f64::consts::PI;
use std::io::Write;

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

/// Per-step snapshot for visualization.
#[derive(serde::Serialize)]
struct StepTrace {
    /// Global step index (across all passes).
    step: usize,
    /// Which pass this step belongs to.
    pass: usize,
    /// Step within the pass.
    step_in_pass: usize,
    /// The input carrier (before noise).
    input: [f64; 4],
    /// What Cell A actually received (after noise, if enabled).
    cell_a_carrier: [f64; 4],
    /// Cell A sigma after this step.
    cell_a_sigma: f64,
    /// Cell C sigma.
    cell_c_sigma: f64,
    /// Prediction error: sigma(cell_c, carrier).
    prediction_error: f64,
    /// Self-free-energy: sigma(cell_c, zread(cell_c)).
    self_free_energy: f64,
    /// Signed coherence change.
    valence: f64,
    /// Genome size after this step.
    genome_size: usize,
    /// Whether a closure fired this step.
    closure_fired: bool,
    /// Closure sigma (if fired).
    closure_sigma: Option<f64>,
    /// Closure support (if fired).
    closure_support: Option<usize>,
    /// Arousal tone.
    arousal_tone: f64,
    /// Coherence tone.
    coherence_tone: f64,
    /// Effective kappa (None if noise disabled).
    noise_kappa: Option<f64>,
    /// Hopf base of Cell A (S2 projection for visualization).
    cell_a_hopf_base: [f64; 3],
    /// Hopf phase of Cell A (S1 angle).
    cell_a_hopf_phase: f64,
}

/// Per-pass summary.
#[derive(serde::Serialize)]
struct PassSummary {
    pass: usize,
    closures: usize,
    genome_size: usize,
    mean_prediction_error: f64,
    final_self_free_energy: f64,
    mean_kappa: Option<f64>,
}

/// Genome entry snapshot.
#[derive(serde::Serialize)]
struct GenomeEntry {
    value: [f64; 4],
    hopf_base: [f64; 3],
    hopf_phase: f64,
    sigma: f64,
    layer: String,
    activation_count: usize,
}

/// One complete run (quiet or noisy).
#[derive(serde::Serialize)]
struct RunTrace {
    label: String,
    noise_enabled: bool,
    base_kappa: Option<f64>,
    steps: Vec<StepTrace>,
    passes: Vec<PassSummary>,
    final_genome: Vec<GenomeEntry>,
}

/// One experiment with two runs.
#[derive(serde::Serialize)]
struct Experiment {
    name: String,
    description: String,
    corpus_description: String,
    n_passes: usize,
    corpus_len: usize,
    quiet: RunTrace,
    noisy: RunTrace,
}

fn run_traced(
    brain: &mut ThreeCell,
    corpus: &[[f64; 4]],
    n_passes: usize,
    label: &str,
) -> RunTrace {
    let noise_enabled = brain.noise_enabled();
    let mut steps = Vec::new();
    let mut passes = Vec::new();
    let mut global_step = 0;

    for pass in 0..n_passes {
        let closures_before = brain.total_closures();
        let mut total_pe = 0.0;
        let mut last_sfe = 0.0;
        let mut kappa_sum = 0.0;
        let mut kappa_count = 0usize;

        for (i, c) in corpus.iter().enumerate() {
            let step = brain.ingest(c);

            let cell_a_q = brain.cell_a_state();
            let (base, phase) = hopf::decompose(&cell_a_q);

            // Reconstruct what Cell A received. If noise is on, it differs
            // from *c. We can infer it from the history — the last entry pushed.
            let cell_a_carrier = brain.cell_a_last_carrier().unwrap_or(*c);

            if let Some(k) = step.noise_kappa {
                kappa_sum += k;
                kappa_count += 1;
            }

            total_pe += step.prediction_error;
            last_sfe = step.self_free_energy;

            steps.push(StepTrace {
                step: global_step,
                pass,
                step_in_pass: i,
                input: *c,
                cell_a_carrier,
                cell_a_sigma: step.cell_a_sigma,
                cell_c_sigma: step.cell_c_sigma,
                prediction_error: step.prediction_error,
                self_free_energy: step.self_free_energy,
                valence: step.valence,
                genome_size: brain.genome_size(),
                closure_fired: step.closure.is_some(),
                closure_sigma: step.closure.as_ref().map(|e| e.sigma),
                closure_support: step.closure.as_ref().map(|e| e.support),
                arousal_tone: step.arousal_tone,
                coherence_tone: step.coherence_tone,
                noise_kappa: step.noise_kappa,
                cell_a_hopf_base: base,
                cell_a_hopf_phase: phase,
            });
            global_step += 1;
        }

        let closures = brain.total_closures() - closures_before;
        passes.push(PassSummary {
            pass,
            closures,
            genome_size: brain.genome_size(),
            mean_prediction_error: total_pe / corpus.len() as f64,
            final_self_free_energy: last_sfe,
            mean_kappa: if kappa_count > 0 {
                Some(kappa_sum / kappa_count as f64)
            } else {
                None
            },
        });
    }

    // Snapshot the final genome.
    let final_genome = brain.genome_entries().iter().map(|e| {
        let (base, phase) = hopf::decompose(&e.value);
        GenomeEntry {
            value: e.value,
            hopf_base: base,
            hopf_phase: phase,
            sigma: closure_ea::sigma(&e.value),
            layer: format!("{:?}", e.layer),
            activation_count: e.activation_count,
        }
    }).collect();

    RunTrace {
        label: label.to_string(),
        noise_enabled,
        base_kappa: if noise_enabled { Some(1.0 / BKT_THRESHOLD) } else { None },
        steps,
        passes,
        final_genome,
    }
}

fn main() {
    let base_kappa = 1.0 / BKT_THRESHOLD;
    let n_passes = 50;

    let mut experiments = Vec::new();

    // ── Experiment 1: single orbit ───────────────────────────────────
    {
        let eps = axis_rotation(2.0 * PI / 6.0, 0);
        let corpus: Vec<[f64; 4]> = (0..6).map(|_| eps).collect();

        let mut brain_q = make_brain(0.05);
        let quiet = run_traced(&mut brain_q, &corpus, n_passes, "quiet");

        let mut brain_n = make_brain(0.05);
        brain_n.enable_noise(42, base_kappa);
        let noisy = run_traced(&mut brain_n, &corpus, n_passes, "noisy");

        experiments.push(Experiment {
            name: "Single orbit (period 6)".into(),
            description: "Repeated generator on X axis. Easy — one basin.".into(),
            corpus_description: "eps = rotation(pi/3, X-axis), repeated 6x per pass".into(),
            n_passes,
            corpus_len: corpus.len(),
            quiet,
            noisy,
        });
    }

    // ── Experiment 2: competing orbits ───────────────────────────────
    {
        let orbit_a: Vec<[f64; 4]> = (0..3).map(|_| axis_rotation(2.0 * PI / 3.0, 0)).collect();
        let orbit_b: Vec<[f64; 4]> = (0..5).map(|_| axis_rotation(2.0 * PI / 5.0, 1)).collect();
        let corpus: Vec<[f64; 4]> = orbit_a.iter().chain(orbit_b.iter()).copied().collect();

        let mut brain_q = make_brain(0.05);
        let quiet = run_traced(&mut brain_q, &corpus, n_passes, "quiet");

        let mut brain_n = make_brain(0.05);
        brain_n.enable_noise(42, base_kappa);
        let noisy = run_traced(&mut brain_n, &corpus, n_passes, "noisy");

        experiments.push(Experiment {
            name: "Competing orbits (period 3 + 5)".into(),
            description: "Two orbits on different axes. Medium — two basins.".into(),
            corpus_description: "3x rot(2pi/3, X) then 5x rot(2pi/5, Y) per pass".into(),
            n_passes,
            corpus_len: corpus.len(),
            quiet,
            noisy,
        });
    }

    // ── Experiment 3: basin hopping ──────────────────────────────────
    {
        let eps4 = axis_rotation(2.0 * PI / 4.0, 0);
        let eps7 = axis_rotation(2.0 * PI / 7.0, 0);
        let mut corpus = Vec::new();
        for _ in 0..4 { corpus.push(eps4); }
        for _ in 0..7 { corpus.push(eps7); }

        let mut brain_q = make_brain(0.05);
        let quiet = run_traced(&mut brain_q, &corpus, n_passes, "quiet");

        let mut brain_n = make_brain(0.05);
        brain_n.enable_noise(42, base_kappa);
        let noisy = run_traced(&mut brain_n, &corpus, n_passes, "noisy");

        experiments.push(Experiment {
            name: "Basin hopping (period 4 + 7, same axis)".into(),
            description: "Two orbits on the same axis. Hard — overlapping basins.".into(),
            corpus_description: "4x rot(pi/2, X) then 7x rot(2pi/7, X) per pass".into(),
            n_passes,
            corpus_len: corpus.len(),
            quiet,
            noisy,
        });
    }

    // Write JSON.
    let json = serde_json::to_string(&experiments).expect("JSON serialization failed");
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../docs/noise-trace.json");
    let mut f = std::fs::File::create(path).expect("cannot create output file");
    f.write_all(json.as_bytes()).expect("write failed");
    eprintln!("Wrote {} bytes to {path}", json.len());
}
