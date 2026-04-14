//! Runtime-backed execution helpers for orbit arithmetic experiments.
//!
//! These helpers do not bypass the substrate with raw example-side
//! quaternion stepping. They seed orbit DNA into a [`ThreeCell`] and
//! route slot reads and slot-to-slot transitions through the runtime's
//! `evaluate` / `evaluate_product` path.

use crate::{inverse, sigma, GenomeConfig, ThreeCell, IDENTITY};

#[derive(Clone, Copy, Debug)]
pub struct OrbitSeed {
    pub period: usize,
    pub axis: [f64; 3],
}

#[derive(Clone, Debug)]
struct OrbitSpec {
    period: usize,
    generator: [f64; 4],
    inverse: [f64; 4],
    slots: Vec<[f64; 4]>,
    nonzero_start: usize,
    nonzero_end: usize,
}

pub struct OrbitRuntime {
    brain: ThreeCell,
    orbits: Vec<OrbitSpec>,
}

#[derive(Clone, Debug)]
pub enum MinskyInstr {
    Inc { reg: usize, next: usize },
    DecJz { reg: usize, if_zero: usize, if_pos: usize },
    Halt,
}

#[derive(Clone, Copy, Debug)]
pub struct MinskyState {
    pub pc: [f64; 4],
    pub regs: [[f64; 4]; 2],
}

pub struct MinskyMachine {
    runtime: OrbitRuntime,
    reg_orbits: [usize; 2],
    pc_orbit: usize,
    halt_slot: usize,
}

pub type Fraction = (Vec<usize>, Vec<usize>);

#[derive(Clone, Debug)]
pub struct FractranState {
    pub pc: [f64; 4],
    pub orbits: Vec<[f64; 4]>,
}

pub struct FractranMachine {
    runtime: OrbitRuntime,
    prime_orbits: Vec<usize>,
    pc_orbit: usize,
}

fn normalize_axis(axis: [f64; 3]) -> [f64; 3] {
    let norm = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
    assert!(norm > 1e-12, "orbit axis must be nonzero");
    [axis[0] / norm, axis[1] / norm, axis[2] / norm]
}

pub fn orbit_generator(period: usize, axis: [f64; 3]) -> [f64; 4] {
    assert!(period >= 2, "orbit period must be at least 2");
    let theta = 2.0 * std::f64::consts::PI / period as f64;
    let axis = normalize_axis(axis);
    [
        theta.cos(),
        theta.sin() * axis[0],
        theta.sin() * axis[1],
        theta.sin() * axis[2],
    ]
}

impl OrbitRuntime {
    pub fn new(seeds: &[OrbitSeed]) -> Self {
        let mut brain = ThreeCell::new(
            0.05,
            0.05,
            4,
            GenomeConfig {
                reinforce_threshold: 0.001,
                novelty_threshold: 0.05,
                merge_threshold: 0.001,
                co_resonance_merge_threshold: 0.0,
            },
        );

        let mut orbits = Vec::with_capacity(seeds.len());
        for seed in seeds {
            let generator = orbit_generator(seed.period, seed.axis);
            let nonzero_start = brain.hierarchy().genomes[0].len();
            brain.seed_orbit_dna(&generator, seed.period);
            let nonzero_end = brain.hierarchy().genomes[0].len();

            let mut slots = Vec::with_capacity(seed.period);
            slots.push(IDENTITY);
            for idx in nonzero_start..nonzero_end {
                slots.push(brain.hierarchy().genomes[0].entries[idx].address.geometry());
            }
            assert_eq!(
                slots.len(),
                seed.period,
                "seeded orbit must expose one slot per period position"
            );

            orbits.push(OrbitSpec {
                period: seed.period,
                generator,
                inverse: inverse(&generator),
                slots,
                nonzero_start,
                nonzero_end,
            });
        }

        Self { brain, orbits }
    }

    pub fn brain(&self) -> &ThreeCell {
        &self.brain
    }

    pub fn carrier(&self, orbit: usize, slot: usize) -> [f64; 4] {
        self.orbits[orbit].slots[slot % self.orbits[orbit].period]
    }

    pub fn slot_of(&self, orbit: usize, carrier: &[f64; 4]) -> Option<usize> {
        if sigma(carrier) < 1e-9 {
            return Some(0);
        }
        let hit = self.brain.evaluate(&[*carrier])?;
        let spec = &self.orbits[orbit];
        if hit.index >= spec.nonzero_start && hit.index < spec.nonzero_end {
            Some(hit.index - spec.nonzero_start + 1)
        } else {
            None
        }
    }

    pub fn step_forward(&self, orbit: usize, carrier: &[f64; 4]) -> Option<[f64; 4]> {
        self.brain
            .evaluate_product(carrier, &self.orbits[orbit].generator)
            .map(|hit| hit.carrier)
    }

    pub fn step_backward(&self, orbit: usize, carrier: &[f64; 4]) -> Option<[f64; 4]> {
        self.brain
            .evaluate_product(carrier, &self.orbits[orbit].inverse)
            .map(|hit| hit.carrier)
    }

    pub fn is_zero(&self, orbit: usize, carrier: &[f64; 4]) -> bool {
        matches!(self.slot_of(orbit, carrier), Some(0))
    }
}

impl MinskyMachine {
    pub fn new(counter_period: usize, pc_period: usize) -> Self {
        let runtime = OrbitRuntime::new(&[
            OrbitSeed {
                period: counter_period,
                axis: [1.0, 0.0, 0.0],
            },
            OrbitSeed {
                period: counter_period,
                axis: [0.0, 1.0, 0.0],
            },
            OrbitSeed {
                period: pc_period,
                axis: [0.0, 0.0, 1.0],
            },
        ]);

        Self {
            runtime,
            reg_orbits: [0, 1],
            pc_orbit: 2,
            halt_slot: pc_period - 1,
        }
    }

    pub fn runtime(&self) -> &OrbitRuntime {
        &self.runtime
    }

    pub fn init_state(&self, initial_pc_slot: usize, regs: [usize; 2]) -> MinskyState {
        MinskyState {
            pc: self.runtime.carrier(self.pc_orbit, initial_pc_slot),
            regs: [
                self.runtime.carrier(self.reg_orbits[0], regs[0]),
                self.runtime.carrier(self.reg_orbits[1], regs[1]),
            ],
        }
    }

    pub fn decoded_pc(&self, state: &MinskyState) -> Option<usize> {
        self.runtime.slot_of(self.pc_orbit, &state.pc)
    }

    pub fn decoded_regs(&self, state: &MinskyState) -> Option<[usize; 2]> {
        Some([
            self.runtime.slot_of(self.reg_orbits[0], &state.regs[0])?,
            self.runtime.slot_of(self.reg_orbits[1], &state.regs[1])?,
        ])
    }

    pub fn step(&self, state: &mut MinskyState, program: &[MinskyInstr]) -> bool {
        let pc_slot = self
            .decoded_pc(state)
            .expect("pc carrier must resolve on the program orbit");

        if pc_slot == self.halt_slot {
            return false;
        }

        match program
            .get(pc_slot.saturating_sub(1))
            .expect("pc slot must point at a valid instruction")
        {
            MinskyInstr::Halt => {
                state.pc = self.runtime.carrier(self.pc_orbit, self.halt_slot);
                false
            }
            MinskyInstr::Inc { reg, next } => {
                state.regs[*reg] = self
                    .runtime
                    .step_forward(self.reg_orbits[*reg], &state.regs[*reg])
                    .expect("INC must land on the next orbit slot");
                state.pc = self.runtime.carrier(self.pc_orbit, *next);
                true
            }
            MinskyInstr::DecJz {
                reg,
                if_zero,
                if_pos,
            } => {
                if self.runtime.is_zero(self.reg_orbits[*reg], &state.regs[*reg]) {
                    state.pc = self.runtime.carrier(self.pc_orbit, *if_zero);
                } else {
                    state.regs[*reg] = self
                        .runtime
                        .step_backward(self.reg_orbits[*reg], &state.regs[*reg])
                        .expect("DEC must land on the previous orbit slot");
                    state.pc = self.runtime.carrier(self.pc_orbit, *if_pos);
                }
                true
            }
        }
    }
}

impl FractranMachine {
    pub fn new_2_3_5(prime_period: usize, pc_period: usize) -> Self {
        let runtime = OrbitRuntime::new(&[
            OrbitSeed {
                period: prime_period,
                axis: [1.0, 0.0, 0.0],
            },
            OrbitSeed {
                period: prime_period,
                axis: [0.0, 1.0, 0.0],
            },
            OrbitSeed {
                period: prime_period,
                axis: [0.0, 0.0, 1.0],
            },
            OrbitSeed {
                period: pc_period,
                axis: [1.0, 1.0, 1.0],
            },
        ]);

        Self {
            runtime,
            prime_orbits: vec![0, 1, 2],
            pc_orbit: 3,
        }
    }

    pub fn runtime(&self) -> &OrbitRuntime {
        &self.runtime
    }

    pub fn init_state(&self, exponents: &[usize]) -> FractranState {
        let orbits = exponents
            .iter()
            .enumerate()
            .map(|(i, &exp)| self.runtime.carrier(self.prime_orbits[i], exp))
            .collect();
        FractranState {
            pc: self.runtime.carrier(self.pc_orbit, 1),
            orbits,
        }
    }

    pub fn decoded_pc(&self, state: &FractranState) -> Option<usize> {
        self.runtime.slot_of(self.pc_orbit, &state.pc)
    }

    pub fn exponents(&self, state: &FractranState) -> Option<Vec<usize>> {
        state
            .orbits
            .iter()
            .enumerate()
            .map(|(i, carrier)| self.runtime.slot_of(self.prime_orbits[i], carrier))
            .collect()
    }

    pub fn step(&self, state: &mut FractranState, program: &[Fraction]) -> bool {
        let pc_slot = self
            .decoded_pc(state)
            .expect("pc carrier must resolve on the program orbit");

        if pc_slot == 0 || pc_slot > program.len() {
            return false;
        }

        let (numer, denom) = program
            .get(pc_slot.saturating_sub(1))
            .expect("pc slot must point at a valid FRACTRAN fraction");

        let applicable = denom.iter().all(|&p| {
            !self
                .runtime
                .is_zero(self.prime_orbits[p], &state.orbits[p])
        });

        if applicable {
            for &p in denom {
                state.orbits[p] = self
                    .runtime
                    .step_backward(self.prime_orbits[p], &state.orbits[p])
                    .expect("denominator division must land on the previous orbit slot");
            }
            for &p in numer {
                state.orbits[p] = self
                    .runtime
                    .step_forward(self.prime_orbits[p], &state.orbits[p])
                    .expect("numerator multiplication must land on the next orbit slot");
            }
            state.pc = self.runtime.carrier(self.pc_orbit, 1);
        } else {
            state.pc = self
                .runtime
                .step_forward(self.pc_orbit, &state.pc)
                .expect("program counter must advance on its orbit");
        }

        self.decoded_pc(state)
            .expect("pc must remain on the program orbit")
            <= program.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn orbit_runtime_steps_via_threecell_lookup() {
        let runtime = OrbitRuntime::new(&[OrbitSeed {
            period: 17,
            axis: [1.0, 0.0, 0.0],
        }]);

        let slot_4 = runtime.carrier(0, 4);
        let slot_5 = runtime
            .step_forward(0, &slot_4)
            .expect("runtime must advance along the seeded orbit");
        let slot_3 = runtime
            .step_backward(0, &slot_4)
            .expect("runtime must step backward along the seeded orbit");

        assert_eq!(runtime.slot_of(0, &slot_5), Some(5));
        assert_eq!(runtime.slot_of(0, &slot_3), Some(3));
    }

    #[test]
    fn minsky_machine_uses_geometric_pc_and_registers() {
        let machine = MinskyMachine::new(31, 7);
        let program = vec![
            MinskyInstr::DecJz {
                reg: 1,
                if_zero: 3,
                if_pos: 2,
            },
            MinskyInstr::Inc { reg: 0, next: 1 },
            MinskyInstr::Halt,
        ];
        let mut state = machine.init_state(1, [4, 5]);

        for _ in 0..20 {
            if !machine.step(&mut state, &program) {
                break;
            }
        }

        assert_eq!(machine.decoded_pc(&state), Some(6));
        assert_eq!(machine.decoded_regs(&state), Some([9, 0]));
    }

    #[test]
    fn fractran_machine_executes_prime_transfer_on_runtime() {
        let machine = FractranMachine::new_2_3_5(37, 5);
        let program = vec![(vec![1], vec![0])];
        let mut state = machine.init_state(&[5, 0, 0]);

        for _ in 0..10 {
            if !machine.step(&mut state, &program) {
                break;
            }
        }

        assert_eq!(machine.exponents(&state), Some(vec![0, 5, 0]));
        assert_eq!(machine.decoded_pc(&state), Some(2));
    }
}
