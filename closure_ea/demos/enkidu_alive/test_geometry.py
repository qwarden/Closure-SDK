"""Test the geometry primitives.

1. Verify inverse pairs: UP then DOWN returns home
2. Drop Enkidu somewhere, can he walk home?
3. Navigate to a food target
4. Verify Hopf decomposition reads correctly
5. Random walk round trip
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from geometry import (
    sigma, distance, hopf_decompose, position_to_quaternion,
    IDENTITY, EnkiduState,
)


def test_inverse_pairs():
    """UP then DOWN should return to identity."""
    print("=== Inverse pairs ===")

    e = EnkiduState()
    e.move("up")
    e.move("down")
    print(f"  UP then DOWN: pos={e.position}  σ={e.sigma_home():.6f}")
    assert e.position == (0, 0)
    assert e.sigma_home() < 0.001

    e = EnkiduState()
    e.move("left")
    e.move("right")
    print(f"  LEFT then RIGHT: pos={e.position}  σ={e.sigma_home():.6f}")
    assert e.position == (0, 0)
    assert e.sigma_home() < 0.001

    print("  PASS\n")


def test_commutativity():
    """UP then LEFT should equal LEFT then UP on the Clifford torus."""
    print("=== Commutativity on Clifford torus ===")

    e1 = EnkiduState()
    e1.move("up")
    e1.move("left")

    e2 = EnkiduState()
    e2.move("left")
    e2.move("up")

    d = distance(e1.quaternion, e2.quaternion)
    print(f"  UP·LEFT vs LEFT·UP: distance = {d:.6f}")
    print(f"  pos1 = {e1.position}, pos2 = {e2.position}")
    assert e1.position == e2.position
    assert d < 0.001
    print("  PASS\n")


def test_walk_home():
    """Walk UP UP LEFT, then navigate home."""
    print("=== Walk home ===")

    e = EnkiduState()
    for step in ["up", "up", "left"]:
        e.move(step)
        print(f"  {step:>5}  →  pos={e.position}  σ_home={e.sigma_home():.4f}")

    print(f"\n  Away from home. pos={e.position}  σ={e.sigma_home():.4f}")

    # Walk home
    print("\n  Walking home:")
    steps = 0
    while e.sigma_home() > 0.001 and steps < 20:
        move_name, dist_after = e.best_step_toward(0, 0)
        e.move(move_name)
        steps += 1
        print(f"  {move_name:>5}  →  pos={e.position}  σ_home={e.sigma_home():.4f}")

    print(f"\n  Home in {steps} steps. pos={e.position} σ={e.sigma_home():.6f}")
    assert e.position == (0, 0)
    assert e.sigma_home() < 0.001
    print("  PASS\n")


def test_navigate_to_food():
    """Start at home, navigate to food at (3, -2)."""
    print("=== Navigate to food ===")

    e = EnkiduState()
    food_x, food_y = 3, -2
    food_q = position_to_quaternion(food_x, food_y)

    print(f"  Food at ({food_x}, {food_y})")
    print(f"  Distance to food: σ = {e.sigma_from(food_q):.4f}")
    print()

    steps = 0
    while e.sigma_from(food_q) > 0.001 and steps < 20:
        move_name, dist_after = e.best_step_toward(food_x, food_y)
        e.move(move_name)
        steps += 1
        print(f"  {move_name:>5}  →  pos={e.position}  σ_food={e.sigma_from(food_q):.4f}")

    print(f"\n  At food in {steps} steps. pos={e.position}")
    assert e.position == (food_x, food_y)
    print("  PASS\n")


def test_hopf_channels():
    """Hopf decomposition gives readable channels."""
    print("=== Hopf decomposition ===")

    # At identity: σ = 0
    h = hopf_decompose(IDENTITY)
    print(f"  At identity:  σ={h['sigma']:.4f}  R={h['R']:.3f}  G={h['G']:.3f}  B={h['B']:.3f}  W={h['W']:.4f}")
    assert h["sigma"] < 0.01

    # After moving UP: displacement in x-component
    e = EnkiduState()
    e.move("up")
    h = e.hopf()
    print(f"  After UP:     σ={h['sigma']:.4f}  R={h['R']:.3f}  G={h['G']:.3f}  B={h['B']:.3f}  W={h['W']:.4f}")
    assert h["sigma"] > 0.1

    # After moving LEFT: displacement in y-component
    e = EnkiduState()
    e.move("left")
    h = e.hopf()
    print(f"  After LEFT:   σ={h['sigma']:.4f}  R={h['R']:.3f}  G={h['G']:.3f}  B={h['B']:.3f}  W={h['W']:.4f}")
    assert h["sigma"] > 0.1

    # Far from home
    e = EnkiduState()
    for _ in range(5):
        e.move("up")
    h = e.hopf()
    print(f"  After 5×UP:   σ={h['sigma']:.4f}  R={h['R']:.3f}  G={h['G']:.3f}  B={h['B']:.3f}  W={h['W']:.4f}")
    assert h["sigma"] > 0.5

    print("  PASS\n")


def test_random_walk_round_trip():
    """Walk a random path, then walk home."""
    print("=== Random walk round trip ===")
    import random
    random.seed(42)

    e = EnkiduState()
    path = [random.choice(["up", "down", "left", "right"]) for _ in range(8)]
    print(f"  Forward path: {' '.join(path)}")

    for step in path:
        e.move(step)

    print(f"  After walk: pos={e.position}  σ={e.sigma_home():.4f}")

    # Walk home
    home_path = []
    steps = 0
    while e.sigma_home() > 0.001 and steps < 30:
        move_name, _ = e.best_step_toward(0, 0)
        e.move(move_name)
        home_path.append(move_name)
        steps += 1

    print(f"  Home path:    {' '.join(home_path)}")
    print(f"  Final pos:    {e.position}")
    print(f"  Final σ:      {e.sigma_home():.6f}")
    assert e.position == (0, 0)
    assert e.sigma_home() < 0.001
    print("  PASS\n")


if __name__ == "__main__":
    test_inverse_pairs()
    test_commutativity()
    test_walk_home()
    test_navigate_to_food()
    test_hopf_channels()
    test_random_walk_round_trip()
    print("All geometry tests passed.")
