//! EMBED — the brain's only input boundary.
//!
//! Bytes go in, a unit quaternion comes out. Two modes:
//!
//! * **Geometric** (`bytes_to_sphere4`): each byte composes as a
//!   rotation on S³. Similar byte sequences land at nearby quaternions.
//!   Used when the brain wants the embedding to preserve locality.
//! * **Cryptographic** (`bytes_to_sphere_hashed`): SHA-256 → Box-Muller
//!   → S³. Same bytes always map to the same point. Similarity is
//!   destroyed. Used for content-addressed identity.
//!
//! There is no GLYPH primitive. Symbol learning happens through
//! ordinary chunked promotion of carriers that EMBED produced from
//! symbol bytes.
//!
//! ── Kabbalistic correspondence ──────────────────────────────────────
//!
//! **Tzimtzum (contraction/withdrawal)** — EMBED is the moment of
//! Tzimtzum applied to raw information. The infinite variety of byte
//! sequences (Ein Sof's unstructured light) is withdrawn into a bounded
//! unit quaternion on S³ (the Vacated Space). Magnitude is withdrawn;
//! direction is retained. This is the first NOT: "not this magnitude,
//! not that magnitude — only this direction."
//!
//! **Geometric path = Kav (the channel)** — the bandwidth-limited
//! channel through which information enters with locality preserved.
//! Like the Kav, it throttles: similar sequences land nearby (high
//! throughput for similar input), dissimilar sequences land far apart
//! (low throughput for novel input). The composition of byte rotations
//! is the sequential Shefa entering through the Kav.
//!
//! **Cryptographic path = content-addressed Nitzotz** — each unique
//! byte sequence maps to a unique, fixed point on S³ regardless of
//! context. These are irreducible: no two distinct inputs share a
//! carrier. Used for domain axioms (DNA seeds) where locality must not
//! blur distinct concepts into neighbors.
//!
//! **`Vocabulary::register()` = explicit Nitzotz placement** — for
//! domains where the hash-derived carriers would scatter related concepts
//! beyond the π/3 Partzuf boundary, `register()` allows structured
//! placement: all four RNA nucleotides within σ = π/4 of each other,
//! Watson-Crick pairs symmetric about their axis. This is intentional
//! Tzimtzum geometry: choosing where in the Vacated Space each spark lands.

use sha2::{Digest, Sha256};
use std::f64::consts::{FRAC_1_SQRT_2, TAU};
use std::sync::OnceLock;

use crate::hopf::{carrier_from_hopf, decompose};
use crate::sphere::{compose, IDENTITY};

// ── S¹ parity gate ───────────────────────────────────────────────────
//
// The Hopf fibration S³ → S² × S¹ has a Z/2 fiber symmetry: composing
// with PARITY_CARRIER maps the non-trivial fiber (S¹ phase ∈ [π, 2π))
// to the trivial fiber (phase ∈ [0, π)). Applying this before the S³
// lift — between embed output and ingest input — implements the S¹
// double-cover without touching the physical core.

/// Equatorial carrier: [cos(π/4), sin(π/4), 0, 0] = [1/√2, 1/√2, 0, 0].
///
/// This carrier lives at σ = π/4, the Hopf equator and the crossing
/// point of the diabolo. Pre-composing with it maps a carrier in the
/// non-trivial S¹ fiber to the trivial fiber. It is the constant that
/// implements the parity crossing used by the double-cover: it moves a
/// carrier across the chosen S¹ seam. The SO(3) double-cover identification
/// itself is still `q ~ -q`; do not confuse this equatorial shift with the
/// antipodal identification.
pub const PARITY_CARRIER: [f64; 4] = [FRAC_1_SQRT_2, FRAC_1_SQRT_2, 0.0, 0.0];

/// Opaque-carrier S¹ parity gate — for already-lifted carriers only.
///
/// Composes [`PARITY_CARRIER`] (the equatorial carrier, σ = π/4) into
/// an already-lifted S³ carrier when `nontrivial` is true. This shifts
/// the carrier to the other S¹ fiber by post-lift quaternion multiplication.
///
/// **When to use this vs [`parity_phase_gate`]:**
/// * Use `parity_gate` for opaque carriers produced by `bytes_to_sphere4`,
///   `f64_to_sphere4`, or `i64_to_sphere4` — cases where the S¹ phase is
///   not available as a separate number.
/// * Use `parity_phase_gate` (+ `domain_embed_with_parity` /
///   `MusicEncoder::embed_with_parity`) when the domain encoder owns the
///   S¹ phase explicitly. That is the pre-lift path and is preferred for
///   structured domain data because it keeps S² and S¹ orthogonal.
///
/// ThreeCell never calls either gate internally. The gate lives at the
/// embed boundary; the caller applies it before passing to `ThreeCell::ingest`.
pub fn parity_gate(carrier: &[f64; 4], nontrivial: bool) -> [f64; 4] {
    if nontrivial {
        compose(&PARITY_CARRIER, carrier)
    } else {
        *carrier
    }
}

// ── Geometric embedding ─────────────────────────────────────────────
//
// 256 pre-computed quaternions, one per byte value, generated once
// from SHA-256 for determinism. After init, SHA-256 is never called
// again on the geometric path.

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

/// Embed a byte sequence as a quaternion on S³.
///
/// `hashed=false` → geometric embedding (locality-preserving).
/// `hashed=true`  → cryptographic embedding (content-addressed).
pub fn bytes_to_sphere4(data: &[u8], hashed: bool) -> [f64; 4] {
    if hashed {
        bytes_to_sphere_hashed(data)
    } else {
        bytes_to_sphere_geometric(data)
    }
}

/// Parity-aware byte embedding. Applies [`parity_gate`] after the S³
/// lift when `nontrivial` is true, placing the carrier in the trivial
/// S¹ fiber before it enters the physical core.
pub fn bytes_to_sphere4_with_parity(data: &[u8], hashed: bool, nontrivial: bool) -> [f64; 4] {
    parity_gate(&bytes_to_sphere4(data, hashed), nontrivial)
}

/// Deterministic semantic base for a byte string.
///
/// This is the write-side domain rule for symbolic data: bytes name a type,
/// so their hash is projected to S² and only the S² base is kept. Position in
/// a sequence is supplied separately as S¹ phase by `carrier_from_hopf`.
pub fn semantic_base_from_bytes(data: &[u8]) -> [f64; 3] {
    let carrier = bytes_to_sphere_hashed(data);
    decompose(&carrier).0
}

/// S¹ parity gate at the phase level — applied BEFORE the Hopf lift.
///
/// The S¹ double cover: the non-trivial fiber is the trivial fiber offset by π
/// in the S¹ coordinate. When `nontrivial` is true, adds π to the phase before
/// `carrier_from_hopf`. The S² base is unchanged — only the fiber coordinate shifts.
///
/// Use this gate when the S¹ phase is already known as a number (domain encoders,
/// music encoder). Use [`parity_gate`] for byte-level carriers that are already
/// lifted — the post-lift carrier multiplication is equivalent but the semantics differ.
#[inline]
pub fn parity_phase_gate(phase: f64, nontrivial: bool) -> f64 {
    if nontrivial {
        phase + std::f64::consts::PI
    } else {
        phase
    }
}

/// Canonical domain encoder: enforce the Hopf write contract for byte-sourced data.
///
/// **Hopf write contract:**
/// * `type_bytes` → S² base. The hash of `type_bytes` determines the semantic
///   type: which rotation axis this carrier sits on. Two objects of the same
///   type at different positions share the same S² base.
/// * `position_phase` → S¹ phase. The cyclic position in a sequence. Two
///   objects of different types at the same position share the same S¹ phase.
/// * W (presence/existence depth) is determined by the geometry of
///   `carrier_from_hopf(base, phase)` and reflects how far the carrier sits
///   from identity. It is NOT set independently — the geometry enforces it.
///
/// This function is the named version of `semantic_base_from_bytes` +
/// `carrier_from_hopf`. Callers that need the full type/position anatomy
/// (e.g., token encoders, position-indexed symbol tables) must use this
/// instead of raw `bytes_to_sphere4`, which does not separate type from
/// position.
///
/// Use `bytes_to_sphere4` only when the input is already a single opaque
/// signal where type and position cannot or should not be separated.
pub fn domain_embed(type_bytes: &[u8], position_phase: f64) -> [f64; 4] {
    let base = semantic_base_from_bytes(type_bytes);
    carrier_from_hopf(base, position_phase)
}

/// Parity-aware domain encoder. Applies S¹ parity as a **pre-lift phase shift**:
/// adds π to `position_phase` before `carrier_from_hopf` when `nontrivial` is true.
///
/// This is strictly pre-lift: the S² base is fixed by `type_bytes`; only the S¹
/// fiber coordinate shifts. The result is the canonical carrier in the non-trivial
/// S¹ fiber, not a post-lift carrier composition.
///
/// Contrast with [`parity_gate`], which composes PARITY_CARRIER after an already-
/// lifted carrier. Both reach "other fiber," but `domain_embed_with_parity` does
/// it at the correct algebraic level for structured domain data.
pub fn domain_embed_with_parity(
    type_bytes: &[u8],
    position_phase: f64,
    nontrivial: bool,
) -> [f64; 4] {
    let base = semantic_base_from_bytes(type_bytes);
    let phase = parity_phase_gate(position_phase, nontrivial);
    carrier_from_hopf(base, phase)
}

fn bytes_to_sphere_geometric(data: &[u8]) -> [f64; 4] {
    if data.is_empty() {
        return IDENTITY;
    }
    let table = byte_quaternions();
    let mut running = IDENTITY;
    for &byte in data {
        running = compose(&running, &table[byte as usize]);
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
        return IDENTITY;
    }
    let inv = 1.0 / norm;
    [vals[0] * inv, vals[1] * inv, vals[2] * inv, vals[3] * inv]
}

/// Embed an f64 as a quaternion using its IEEE-754 structure.
/// Deterministic; preserves local structure better than treating the
/// number as 8 raw bytes.
pub fn f64_to_sphere4(value: f64) -> [f64; 4] {
    if !value.is_finite() {
        return bytes_to_sphere4(&value.to_le_bytes(), false);
    }

    let bits = value.to_bits();
    let sign = if (bits >> 63) == 0 { 1.0 } else { -1.0 };
    let exp = ((bits >> 52) & 0x7ff) as f64 / 2047.0;
    let mant = bits & ((1u64 << 52) - 1);
    let mant_hi = ((mant >> 26) & ((1u64 << 26) - 1)) as f64 / ((1u64 << 26) - 1) as f64;
    let mant_lo = (mant & ((1u64 << 26) - 1)) as f64 / ((1u64 << 26) - 1) as f64;

    let mut q = [
        sign,
        exp * 2.0 - 1.0,
        mant_hi * 2.0 - 1.0,
        mant_lo * 2.0 - 1.0,
    ];

    let norm = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    if norm < 1e-15 {
        IDENTITY
    } else {
        let inv = 1.0 / norm;
        q[0] *= inv;
        q[1] *= inv;
        q[2] *= inv;
        q[3] *= inv;
        q
    }
}

/// Parity-aware f64 embedding. Applies [`parity_gate`] after the S³
/// lift when `nontrivial` is true.
pub fn f64_to_sphere4_with_parity(value: f64, nontrivial: bool) -> [f64; 4] {
    parity_gate(&f64_to_sphere4(value), nontrivial)
}

// ── Text vocabulary ──────────────────────────────────────────────────────────
//
// A vocabulary maps token strings to carriers on S³. The mapping is
// deterministic: the same token always produces the same carrier via
// bytes_to_sphere_hashed (content-addressed). No external tokenizer
// is required. The vocabulary table is a cache — it does not assign
// integer IDs. Carriers are the IDs.
//
// Usage:
//   let mut vocab = Vocabulary::new();
//   let carrier = vocab.embed("hello");   // carrier for "hello"
//   let tokens = vocab.tokenize("hello world");  // splits on whitespace
//   let sentence = vocab.embed_sequence("the cat sat");

/// Bit-exact key for a quaternion in a HashMap.
///
/// `[f64; 4]` doesn't implement Hash/Eq because NaN ≠ NaN. All
/// carriers on S³ are finite (the embedding functions guarantee this),
/// so transmuting each component's bits to u64 gives a safe, exact key.
/// Two carriers that are the same point on S³ will always have the same
/// key because the embedding is deterministic.
fn carrier_key(q: &[f64; 4]) -> [u64; 4] {
    [
        q[0].to_bits(),
        q[1].to_bits(),
        q[2].to_bits(),
        q[3].to_bits(),
    ]
}

/// A content-addressed vocabulary: token string ↔ carrier on S³.
///
/// The forward map (token → carrier) and the reverse map (carrier → token)
/// are kept in sync by `embed`. The embedding is deterministic and
/// locality-preserving: the same token always maps to the same carrier,
/// and tokens sharing a common prefix land in the same S³ neighborhood.
///
/// `decode` recovers a token from an exact carrier (e.g. from a genome
/// entry's stored value). `decode_nearest` recovers the closest known
/// token from any generated/read carrier, scanning by σ distance.
pub struct Vocabulary {
    /// Forward: token string → carrier.
    map: std::collections::HashMap<String, [f64; 4]>,
    /// Reverse: carrier bits → token string. Populated in lockstep with
    /// `map` inside `embed`. Enables O(1) exact decoding and serves as
    /// the scan corpus for nearest-neighbor decoding.
    reverse: std::collections::HashMap<[u64; 4], String>,
}

impl Vocabulary {
    pub fn new() -> Self {
        Self {
            map: std::collections::HashMap::new(),
            reverse: std::collections::HashMap::new(),
        }
    }

    /// Embed a single token string. Idempotent: the same string always
    /// returns the same type carrier. Token bytes define semantic type, so
    /// they are projected to an S² base. The default carrier uses phase 0.
    /// Sequence methods supply position as S¹ phase with `embed_at_phase`.
    ///
    /// Also updates the reverse map so `decode` and `decode_nearest`
    /// can recover this token from its carrier.
    pub fn embed(&mut self, token: &str) -> [f64; 4] {
        if let Some(&carrier) = self.map.get(token) {
            return carrier;
        }
        let base = semantic_base_from_bytes(token.as_bytes());
        let carrier = carrier_from_hopf(base, 0.0);
        self.map.insert(token.to_string(), carrier);
        self.reverse
            .insert(carrier_key(&carrier), token.to_string());
        carrier
    }

    /// Embed a token at a specific S¹ sequence phase.
    ///
    /// Same token at different phases shares S² base and differs in S¹.
    /// Different tokens at the same phase share S¹ and differ in S².
    pub fn embed_at_phase(&mut self, token: &str, phase: f64) -> [f64; 4] {
        let type_carrier = self.embed(token);
        let (base, _) = decompose(&type_carrier);
        carrier_from_hopf(base, phase)
    }

    /// Embed a token without caching (read-only, for inference).
    /// Returns None if the token has not been seen during training.
    pub fn lookup(&self, token: &str) -> Option<[f64; 4]> {
        self.map.get(token).copied()
    }

    /// Decode a carrier back to its token string.
    ///
    /// Uses the reverse map for O(1) exact bit-match lookup. This works
    /// for carriers that came directly out of the genome or vocabulary
    /// (they are the same bits that `embed` stored). Returns None if the
    /// carrier was never embedded — e.g. a generated carrier that drifted
    /// away from any known token. Use `decode_nearest` for those.
    pub fn decode(&self, carrier: &[f64; 4]) -> Option<&str> {
        self.reverse.get(&carrier_key(carrier)).map(|s| s.as_str())
    }

    /// Decode a generated carrier to the nearest known token by S² type.
    ///
    /// Scans the full vocabulary and returns the token whose semantic base is
    /// closest to the generated carrier's base. Position is ignored on decode
    /// because position lives in S¹ and token identity lives in S².
    ///
    /// Returns None only if the vocabulary is empty.
    pub fn decode_nearest(&self, carrier: &[f64; 4]) -> Option<&str> {
        use crate::hopf::base_distance;
        let (query_base, _) = decompose(carrier);
        let mut best_token: Option<&str> = None;
        let mut best_gap = f64::MAX;
        for (token, &stored) in &self.map {
            let (stored_base, _) = decompose(&stored);
            let gap = base_distance(&query_base, &stored_base);
            if gap < best_gap {
                best_gap = gap;
                best_token = Some(token.as_str());
            }
        }
        best_token
    }

    /// Tokenize a string by whitespace and embed each token.
    /// Returns `(token_strings, carriers)` for inspection.
    pub fn tokenize(&mut self, text: &str) -> Vec<(String, [f64; 4])> {
        let tokens: Vec<String> = text
            .split_whitespace()
            .map(|tok| tok.to_lowercase())
            .collect();
        let len = tokens.len().max(1) as f64;
        tokens
            .into_iter()
            .enumerate()
            .map(|(idx, lower)| {
                let phase = TAU * (idx as f64) / len;
                let carrier = self.embed_at_phase(&lower, phase);
                (lower, carrier)
            })
            .collect()
    }

    /// Embed a whitespace-tokenized string as a sequence of carriers.
    /// Suitable for feeding directly to `ingest_sequence()`.
    pub fn embed_sequence(&mut self, text: &str) -> Vec<[f64; 4]> {
        self.tokenize(text)
            .into_iter()
            .map(|(_, carrier)| carrier)
            .collect()
    }

    /// Decode a sequence of generated carriers back to token strings.
    ///
    /// Each carrier is decoded via `decode_nearest`, which is appropriate
    /// whenever a read/generated carrier may not exactly match a stored token.
    pub fn decode_sequence(&self, carriers: &[[f64; 4]]) -> Vec<Option<&str>> {
        carriers.iter().map(|c| self.decode_nearest(c)).collect()
    }

    // ── Persistence ───────────────────────────────────────────────────────────

    /// Save the vocabulary to a JSON file.
    ///
    /// Only the forward map (token → carrier) is written. The reverse map
    /// is derived from it and is rebuilt by `load_from_file`. Call this
    /// alongside `Genome::save_to_file` to checkpoint a full brain state.
    pub fn save_to_file(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let json = serde_json::to_string_pretty(&self.map)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Load a vocabulary from a JSON file previously written by `save_to_file`.
    ///
    /// Rebuilds both the forward and reverse maps from the stored token→carrier
    /// pairs. All methods (`embed`, `decode`, `decode_nearest`) work
    /// immediately after loading.
    pub fn load_from_file(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let json = std::fs::read_to_string(path)?;
        let map: std::collections::HashMap<String, [f64; 4]> = serde_json::from_str(&json)?;
        let mut vocab = Self::new();
        for (token, &carrier) in &map {
            vocab.map.insert(token.clone(), carrier);
            vocab.reverse.insert(carrier_key(&carrier), token.clone());
        }
        Ok(vocab)
    }

    /// Register a token with an explicit carrier rather than deriving it
    /// from bytes. Overwrites any previous mapping for this token.
    /// Use when callers need structured S³ placement (e.g. RNA nucleotides
    /// must all lie within π/3 of each other for ZREAD to work across them).
    pub fn register(&mut self, token: &str, carrier: [f64; 4]) {
        self.map.insert(token.to_string(), carrier);
        self.reverse
            .insert(carrier_key(&carrier), token.to_string());
    }

    /// Number of tokens seen so far.
    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
}

impl Default for Vocabulary {
    fn default() -> Self {
        Self::new()
    }
}

/// Embed an i64 as a quaternion. Sign + magnitude split for geometric
/// proximity between integers of similar absolute value.
pub fn i64_to_sphere4(value: i64) -> [f64; 4] {
    let sign = if value >= 0 { 1.0_f64 } else { -1.0_f64 };
    let abs_bits = value.unsigned_abs();
    let hi = ((abs_bits >> 32) as u32) as f64 / u32::MAX as f64;
    let lo = (abs_bits as u32) as f64 / u32::MAX as f64;
    let mag = (abs_bits as f64) / (i64::MAX as f64);
    let q = [sign, mag * 2.0 - 1.0, hi * 2.0 - 1.0, lo * 2.0 - 1.0];
    let norm = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    if norm < 1e-15 {
        return IDENTITY;
    }
    let inv = 1.0 / norm;
    [q[0] * inv, q[1] * inv, q[2] * inv, q[3] * inv]
}

/// Parity-aware i64 embedding. Applies [`parity_gate`] after the S³
/// lift when `nontrivial` is true.
pub fn i64_to_sphere4_with_parity(value: i64, nontrivial: bool) -> [f64; 4] {
    parity_gate(&i64_to_sphere4(value), nontrivial)
}

// ── Music encoder ────────────────────────────────────────────────────────────
//
// Music maps to S³ through the Hopf fibration exactly as every other domain:
//
//   Harmonic role (tonic, dominant, subdominant, …) → S² base.
//     Role is semantic type: what function does this event play in the key?
//     All events with the same role share the same S² axis. Two different
//     chords that are both the tonic of their respective keys occupy the
//     same S² base (same TYPE) at different metric positions (different S¹).
//
//   Bar/beat position (rational position in the measure) → S¹ phase.
//     Position is cyclic structure: where in the bar/phrase does this event
//     occur? Phase encodes the metric grid — beat 1 of bar 1, offbeat of
//     bar 2, etc. — as a fraction of the full cycle [0, 2π).
//
//   Presence / W component: determined by the Hopf lift geometry. W close
//     to 1 = event near identity (silent or identity role); W near 1/√2 =
//     equatorial / strongly active (at the Hopf equator).
//
// `MusicEncoder` owns this write contract. It routes harmonic role strings
// to S² and numeric beat positions to S¹. Callers feed the resulting carrier
// to `ThreeCell::ingest` — same as any other domain.

/// A music-domain carrier encoder: harmonic role → S², bar/beat → S¹.
///
/// Build one encoder per song or corpus. The S² base for each role is
/// derived deterministically from the role label via `semantic_base_from_bytes`,
/// so the same role name always maps to the same axis regardless of context.
/// Beat position is normalized to `[0, 2π)` from the measure fraction.
pub struct MusicEncoder {
    /// Vocabulary for roles: caches role-string → S² base.
    roles: std::collections::HashMap<String, [f64; 3]>,
}

impl MusicEncoder {
    pub fn new() -> Self {
        Self {
            roles: std::collections::HashMap::new(),
        }
    }

    /// Embed one musical event: harmonic role + position in bar.
    ///
    /// * `role` — harmonic function label, e.g. `"tonic"`, `"dominant"`,
    ///   `"subdominant"`, `"ii"`, `"V7"`. The label is hashed to an S² base;
    ///   two events with the same role land on the same axis.
    /// * `beat` — beat index within the bar (0-indexed).
    /// * `beats_per_bar` — time signature numerator (e.g. 4 for 4/4, 3 for 3/4).
    ///
    /// Returns a unit quaternion on S³ satisfying the Hopf write contract:
    /// `(role)` → S² base, `(beat / beats_per_bar)` → S¹ phase.
    pub fn embed(&mut self, role: &str, beat: usize, beats_per_bar: usize) -> [f64; 4] {
        let base = *self
            .roles
            .entry(role.to_string())
            .or_insert_with(|| semantic_base_from_bytes(role.as_bytes()));
        let phase = if beats_per_bar == 0 {
            0.0
        } else {
            std::f64::consts::TAU * (beat as f64) / (beats_per_bar as f64)
        };
        carrier_from_hopf(base, phase)
    }

    /// Embed with sub-beat resolution. `sub_beat` ∈ [0, sub_divisions) provides
    /// finer metric placement within a beat.
    ///
    /// Full phase = 2π * (beat + sub_beat / sub_divisions) / beats_per_bar.
    pub fn embed_sub_beat(
        &mut self,
        role: &str,
        beat: usize,
        sub_beat: usize,
        sub_divisions: usize,
        beats_per_bar: usize,
    ) -> [f64; 4] {
        let base = *self
            .roles
            .entry(role.to_string())
            .or_insert_with(|| semantic_base_from_bytes(role.as_bytes()));
        let sub = if sub_divisions == 0 {
            0.0
        } else {
            sub_beat as f64 / sub_divisions as f64
        };
        let phase = if beats_per_bar == 0 {
            0.0
        } else {
            std::f64::consts::TAU * (beat as f64 + sub) / (beats_per_bar as f64)
        };
        carrier_from_hopf(base, phase)
    }

    /// Parity-aware embed. Applies S¹ parity as a **pre-lift phase shift**
    /// (adds π to the beat phase before `carrier_from_hopf`) when `nontrivial`
    /// is true. The S² base (harmonic role) is unchanged.
    pub fn embed_with_parity(
        &mut self,
        role: &str,
        beat: usize,
        beats_per_bar: usize,
        nontrivial: bool,
    ) -> [f64; 4] {
        let base = *self
            .roles
            .entry(role.to_string())
            .or_insert_with(|| semantic_base_from_bytes(role.as_bytes()));
        let raw_phase = if beats_per_bar == 0 {
            0.0
        } else {
            std::f64::consts::TAU * (beat as f64) / (beats_per_bar as f64)
        };
        carrier_from_hopf(base, parity_phase_gate(raw_phase, nontrivial))
    }

    /// Retrieve the S² base direction associated with a role (computed on demand).
    ///
    /// Two events with the same role have the same base regardless of their
    /// beat position. This is the "semantic type" in the Hopf anatomy.
    pub fn role_base(&mut self, role: &str) -> [f64; 3] {
        *self
            .roles
            .entry(role.to_string())
            .or_insert_with(|| semantic_base_from_bytes(role.as_bytes()))
    }

    pub fn role_count(&self) -> usize {
        self.roles.len()
    }
}

impl Default for MusicEncoder {
    fn default() -> Self {
        Self::new()
    }
}
