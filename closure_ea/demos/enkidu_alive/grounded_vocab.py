"""
Grounded vocabulary — Enkidu labels his internal states with words.

Each word maps to an EXACT geometric state. Not statistical.
Not measured from text. The position IS the experience.

    "hungry"  = the quaternion of high σ_hunger
    "food"    = the direction toward food on S³
    "home"    = identity [1,0,0,0]
    "safe"    = both drives low, at home
    "danger"  = σ near π (death approaching)

These positions don't collapse to identity because they ARE
geometric states, not averages of context votes. The scaffold
that all text-taught words attach to.

Usage:
    from brahman.enkidu_alive.grounded_vocab import build_grounded_vocab
    vocab, embeddings = build_grounded_vocab()
    # Save as base model for text teaching:
    # teach.py --base grounded.json --corpus text.txt
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import json
from pathlib import Path

from geometry import (
    IDENTITY, normalize, qmul, invert, sigma, distance,
    hopf_decompose, _clifford, _THETA
)
from homeostasis import World


def _q(w, x, y, z):
    return normalize(np.array([w, x, y, z]))


def build_grounded_vocab(n_runs=50, ticks_per_run=300):
    """Run Enkidu, label his states, extract grounded word positions.

    Returns (vocab_list, embeddings_dict) where embeddings are
    m=2 free quaternions (8 floats) per word, ready for the
    Gilgamesh JSON format.
    """

    # ── CORE VOCABULARY: words grounded in geometric states ──
    #
    # Each word has a REASON for its position. The position is
    # the geometric state the word names. No statistics needed.

    words = {}

    # IDENTITY — the self, the center, silence, home
    words["home"] = IDENTITY.copy()
    words["self"] = IDENTITY.copy()
    words["silence"] = IDENTITY.copy()
    words["rest"] = IDENTITY.copy()
    words["nothing"] = IDENTITY.copy()
    words["zero"] = IDENTITY.copy()
    words["identity"] = IDENTITY.copy()

    # DIRECTIONS — the four grid moves on the Clifford torus
    # These are EXACT: one step in each direction
    words["up"] = _clifford(_THETA, 0)
    words["down"] = _clifford(-_THETA, 0)
    words["left"] = _clifford(0, -_THETA)
    words["right"] = _clifford(0, _THETA)

    # Direction inverses compose to identity:
    # up · down → identity, left · right → identity
    # This is GUARANTEED by the algebra, not learned.

    # DRIVES — the two axes of Enkidu's experience
    # Hunger: rises along the W fiber (existence axis)
    # A hungry quaternion departs identity along the x-axis
    hunger_low = _clifford(_THETA * 2, 0)       # slightly hungry
    hunger_mid = _clifford(_THETA * 6, 0)       # moderately hungry
    hunger_high = _clifford(_THETA * 12, 0)     # very hungry
    hunger_lethal = _clifford(np.pi/2, 0)       # death

    words["hungry"] = hunger_mid
    words["starving"] = hunger_high
    words["full"] = invert(hunger_mid)           # the inverse of hungry
    words["eat"] = invert(hunger_mid)            # eating closes hunger
    words["food"] = invert(hunger_high)          # food closes starvation
    words["death"] = hunger_lethal               # σ = π/2, lethal

    # Cold: rises along the y-axis (orthogonal to hunger on the torus)
    cold_low = _clifford(0, _THETA * 2)
    cold_mid = _clifford(0, _THETA * 6)
    cold_high = _clifford(0, _THETA * 12)

    words["cold"] = cold_mid
    words["freezing"] = cold_high
    words["warm"] = invert(cold_mid)             # the inverse of cold
    words["shelter"] = invert(cold_high)          # shelter closes freezing
    words["fire"] = invert(cold_mid)             # fire warms

    # DISTANCE concepts — how far from home
    words["near"] = _clifford(_THETA * 2, _THETA * 2)    # small σ from home
    words["far"] = _clifford(_THETA * 8, _THETA * 8)     # large σ from home
    words["close"] = invert(words["far"])                  # inverse of far

    # ACTIONS — what Enkidu does
    words["walk"] = _clifford(_THETA * 3, _THETA)  # movement (increases cold)
    words["stay"] = IDENTITY.copy()                  # no movement
    words["go"] = _clifford(_THETA * 4, _THETA * 2) # purposeful movement
    words["come"] = invert(words["go"])               # return

    # STATES — compound experiences
    words["safe"] = normalize(qmul(invert(hunger_low), invert(cold_low)))  # low on both drives
    words["danger"] = normalize(qmul(hunger_high, cold_high))              # high on both drives
    words["lost"] = normalize(qmul(hunger_mid, cold_high))                 # hungry AND cold

    # EVALUATIVE — Enkidu's judgment of states
    words["good"] = words["safe"].copy()    # good = safe (both drives low)
    words["bad"] = words["danger"].copy()   # bad = dangerous
    words["yes"] = words["food"].copy()     # yes = what closes hunger (affirmative)
    words["no"] = words["danger"].copy()    # no = what opens both drives (negative)

    # RELATIONAL — how things relate
    words["and"] = _clifford(_THETA * 0.5, _THETA * 0.5)  # small departure, connective
    words["or"] = _clifford(_THETA, -_THETA)               # branch point
    words["not"] = _clifford(-_THETA * 4, -_THETA * 4)     # negation (reversal direction)
    words["is"] = _clifford(_THETA * 0.3, 0)               # tiny departure (copula)
    words["the"] = _clifford(_THETA * 0.1, _THETA * 0.1)   # nearly identity (determiner)
    words["a"] = _clifford(_THETA * 0.1, -_THETA * 0.1)    # nearly identity
    words["to"] = _clifford(0, _THETA * 0.5)               # directional marker
    words["from"] = _clifford(0, -_THETA * 0.5)            # inverse of "to"

    # EMOTIONAL — higher-level states (compound of drives)
    words["fear"] = normalize(qmul(hunger_high, cold_mid))  # hungry + cold
    words["relief"] = invert(words["fear"])                   # inverse of fear
    words["joy"] = words["safe"].copy()                       # low drives = joy
    words["pain"] = words["danger"].copy()                    # high drives = pain
    words["love"] = normalize(qmul(invert(cold_high), invert(hunger_low)))  # warm + fed
    words["hate"] = invert(words["love"])                     # inverse of love
    words["hope"] = normalize(qmul(words["food"], cold_low)) # food visible, slightly cold
    words["despair"] = invert(words["hope"])                  # no food, very cold

    # ── EXPERIENTIAL ENRICHMENT ──
    # Run Enkidu for many ticks, collect geometric states at key moments.
    # These provide ADDITIONAL grounding from actual navigation experience.

    food_directions = []
    home_directions = []
    danger_moments = []
    relief_moments = []

    import io, contextlib

    for _ in range(n_runs):
        world = World(grid_size=10, hunger_rate=0.06, food_spawn_rate=0.08)
        world.spawn_food()
        world.spawn_food()

        prev_hunger = 0.0
        for _ in range(ticks_per_run):
            if not world.alive:
                break
            # Suppress World's print output
            with contextlib.redirect_stdout(io.StringIO()):
                snap = world.tick()
            if snap is None:
                break

            # Collect geometric states at key moments
            pos = world.enkidu.position
            q_pos = _clifford(pos[0] * _THETA, pos[1] * _THETA)

            # When moving toward food: the direction IS the meaning of "food"
            if snap["drive"] == "hunger" and snap["target"]:
                tx, ty = snap["target"]
                q_food = _clifford(tx * _THETA, ty * _THETA)
                food_directions.append(normalize(qmul(invert(q_pos), q_food)))

            # When moving toward home: the direction IS the meaning of "home"
            if snap["drive"] == "shelter":
                q_home = IDENTITY.copy()
                home_directions.append(normalize(qmul(invert(q_pos), q_home)))

            # Danger moments: high combined σ
            if snap["hunger"] > 2.0 and snap["coldness"] > 1.5:
                danger_moments.append(q_pos.copy())

            # Relief moments: just ate
            if snap.get("ate"):
                relief_moments.append(q_pos.copy())

            prev_hunger = snap["hunger"]

    # Average the experiential vectors to refine grounded positions
    if food_directions:
        avg_food_dir = normalize(np.mean(food_directions, axis=0))
        # Blend with the designed position (70% designed, 30% experiential)
        words["food"] = normalize(0.7 * words["food"] + 0.3 * avg_food_dir)
        words["eat"] = normalize(0.7 * words["eat"] + 0.3 * avg_food_dir)

    if home_directions:
        avg_home_dir = normalize(np.mean(home_directions, axis=0))
        words["come"] = normalize(0.7 * words["come"] + 0.3 * avg_home_dir)

    if danger_moments:
        avg_danger = normalize(np.mean(danger_moments, axis=0))
        words["danger"] = normalize(0.7 * words["danger"] + 0.3 * avg_danger)
        words["fear"] = normalize(0.7 * words["fear"] + 0.3 * avg_danger)

    if relief_moments:
        avg_relief = normalize(np.mean(relief_moments, axis=0))
        words["relief"] = normalize(0.7 * words["relief"] + 0.3 * avg_relief)
        words["joy"] = normalize(0.7 * words["joy"] + 0.3 * avg_relief)

    # ── CONVERT TO m=2+2 FORMAT ──
    # The grounded words are on a single S³. For m=2+2, we need
    # two free quaternions per word. Use the same quaternion for
    # factor 0, and a rotated version for factor 1 (so the two
    # factors carry different information).
    #
    # Factor 0: the direct grounded position
    # Factor 1: the position rotated by π/6 around a fixed axis
    #           (gives a second view of the same state)

    rotation = normalize(np.array([np.cos(np.pi/12), np.sin(np.pi/12), 0, 0]))

    vocab_list = sorted(words.keys())
    embeddings = []
    for w in vocab_list:
        q0 = words[w]
        q1 = normalize(qmul(rotation, qmul(words[w], invert(rotation))))
        free = np.concatenate([q0, q1])
        embeddings.append(free.tolist())

    return vocab_list, embeddings


def save_grounded_model(output_path="brahman/gilgamesh_grounded.json",
                        n_runs=50, ticks=300):
    """Build and save the grounded vocabulary as a Gilgamesh model."""
    vocab, embeddings = build_grounded_vocab(n_runs=n_runs, ticks_per_run=ticks)

    data = {
        "vocab": vocab,
        "embeddings": embeddings,
        "m": 2,
        "note": "Grounded vocabulary from Enkidu Alive — EXACT geometric positions from experience"
    }

    Path(output_path).write_text(json.dumps(data, indent=2))
    print(f"Saved {len(vocab)} grounded words to {output_path}")

    # Verify key relationships
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from enkidu_cell import expand, mf_mul, mf_sigma
    full = {}
    for i, w in enumerate(vocab):
        full[w] = expand(np.array(embeddings[i]))

    print("\nGrounded relationships (should be near-identity):")
    pairs = [
        ("hungry", "food"), ("cold", "warm"), ("up", "down"),
        ("left", "right"), ("go", "come"), ("love", "hate"),
        ("fear", "relief"), ("hope", "despair"), ("good", "bad"),
        ("far", "close"), ("danger", "safe"),
    ]
    for a, b in pairs:
        if a in full and b in full:
            composed = mf_mul(full[a], full[b])
            s = mf_sigma(composed)
            mark = "✓" if s < 0.3 else "✗"
            print(f"  {mark} {a:>10} · {b:<10} σ = {s:.4f}")

    print(f"\nσ spread:")
    sigmas = [mf_sigma(full[w]) for w in vocab]
    print(f"  min={min(sigmas):.4f}  max={max(sigmas):.4f}  mean={np.mean(sigmas):.4f}")
    print(f"  identity words (home, self, rest): σ ≈ {mf_sigma(full['home']):.4f}")
    print(f"  drive words (hungry, cold): σ ≈ {mf_sigma(full['hungry']):.4f}, {mf_sigma(full['cold']):.4f}")
    print(f"  extreme words (death, danger): σ ≈ {mf_sigma(full['death']):.4f}, {mf_sigma(full['danger']):.4f}")

    return data


if __name__ == "__main__":
    output = "brahman/gilgamesh_grounded.json"
    if len(sys.argv) > 1:
        output = sys.argv[1]
    save_grounded_model(output)
