"""Step 1: Enkidu visits waypoints then walks straight home.

Give him a list of places to go. He walks to each one in order.
Then from wherever he ends up, he walks straight home — not
retracing his path, but closing the residual. The algebra
cancels the invertibles.

This IS Gilgamesh. The composition accumulates, C⁻¹ points home.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from geometry import EnkiduState


def visit_waypoints_then_home(waypoints, verbose=True):
    """Enkidu visits a list of (x, y) waypoints, then returns home.

    Returns the full trace: list of (x, y, sigma_home, phase) at each step.
    """
    e = EnkiduState()
    trace = [(e.x, e.y, e.sigma_home(), "start")]

    # Visit each waypoint
    for i, (wx, wy) in enumerate(waypoints):
        if verbose:
            print(f"\n  Waypoint {i+1}: ({wx}, {wy})")

        while e.position != (wx, wy):
            move, _ = e.best_step_toward(wx, wy)
            e.move(move)
            trace.append((e.x, e.y, e.sigma_home(), f"→wp{i+1}"))
            if verbose:
                print(f"    {move:>5}  →  pos={e.position}  σ_home={e.sigma_home():.4f}")

        if verbose:
            print(f"    Arrived at waypoint {i+1}.")

    # Now walk straight home
    if verbose:
        print(f"\n  All waypoints visited. Position: {e.position}, σ_home={e.sigma_home():.4f}")
        print(f"  Walking straight home:")

    steps_home = 0
    while e.position != (0, 0):
        move, _ = e.best_step_toward(0, 0)
        e.move(move)
        steps_home += 1
        trace.append((e.x, e.y, e.sigma_home(), "→home"))
        if verbose:
            print(f"    {move:>5}  →  pos={e.position}  σ_home={e.sigma_home():.4f}")

    if verbose:
        print(f"\n  Home in {steps_home} steps. σ = {e.sigma_home():.6f}")

    return trace, steps_home


def test_single_waypoint():
    print("=" * 50)
    print("TEST 1: Single waypoint (3, 2)")
    print("=" * 50)
    trace, steps = visit_waypoints_then_home([(3, 2)])
    assert trace[-1][:2] == (0, 0)
    print("  PASS\n")


def test_two_waypoints():
    print("=" * 50)
    print("TEST 2: Two waypoints (3, 2) then (-1, 4)")
    print("=" * 50)
    trace, steps = visit_waypoints_then_home([(3, 2), (-1, 4)])
    assert trace[-1][:2] == (0, 0)
    print("  PASS\n")


def test_loop_that_cancels():
    """Visit (3, 0) then (-3, 0) — net displacement is zero.
    Enkidu should already be near home after the second waypoint."""
    print("=" * 50)
    print("TEST 3: Cancelling loop (3,0) → (-3,0) — net ≈ (0,0)")
    print("=" * 50)
    trace, steps = visit_waypoints_then_home([(3, 0), (-3, 0)])
    # After visiting (-3,0) from (3,0), he's at (-3,0), not (0,0).
    # But the WALK from (-3,0) is 3 steps. The algebra doesn't cancel
    # the waypoint path — it cancels the final position to home.
    # The straight line home from wherever you are IS the closure.
    assert trace[-1][:2] == (0, 0)
    print("  PASS\n")


def test_many_waypoints():
    """5 waypoints in a wandering pattern."""
    print("=" * 50)
    print("TEST 4: Five waypoints — wandering path")
    print("=" * 50)
    waypoints = [(2, 1), (-1, 3), (0, -2), (4, 0), (-2, -1)]
    trace, steps = visit_waypoints_then_home(waypoints)
    assert trace[-1][:2] == (0, 0)
    print("  PASS\n")


def test_random_waypoints():
    """Random waypoints, verify Enkidu always gets home."""
    import random
    random.seed(42)

    print("=" * 50)
    print("TEST 5: 20 random trials, 1-5 waypoints each")
    print("=" * 50)

    for trial in range(20):
        n_wp = random.randint(1, 5)
        waypoints = [(random.randint(-5, 5), random.randint(-5, 5)) for _ in range(n_wp)]
        trace, steps = visit_waypoints_then_home(waypoints, verbose=False)
        final_pos = trace[-1][:2]
        final_sigma = trace[-1][2]
        wp_str = " → ".join(f"({x},{y})" for x, y in waypoints)
        status = "OK" if final_pos == (0, 0) else "FAIL"
        print(f"  Trial {trial+1:>2}: {wp_str:<40} home in {steps} steps [{status}]")
        assert final_pos == (0, 0), f"Trial {trial+1} failed: ended at {final_pos}"

    print("\n  All 20 trials: PASS\n")


def test_sigma_trace():
    """Show the σ trace — it should rise as Enkidu walks away,
    then fall monotonically as he walks home."""
    print("=" * 50)
    print("TEST 6: σ trace (should fall monotonically going home)")
    print("=" * 50)

    trace, _ = visit_waypoints_then_home([(4, 3)], verbose=False)

    print("\n  Full σ trace:")
    for x, y, sig, phase in trace:
        bar = "█" * int(sig * 20)
        print(f"    ({x:>2},{y:>2})  σ={sig:.4f}  {bar}  [{phase}]")

    # Check: the home-bound segment has monotonically decreasing σ
    home_sigmas = [sig for _, _, sig, phase in trace if phase == "→home"]
    for i in range(1, len(home_sigmas)):
        assert home_sigmas[i] <= home_sigmas[i-1] + 0.001, \
            f"σ increased going home: {home_sigmas[i-1]:.4f} → {home_sigmas[i]:.4f}"

    print("\n  σ monotonically decreases going home.")
    print("  PASS\n")


if __name__ == "__main__":
    test_single_waypoint()
    test_two_waypoints()
    test_loop_that_cancels()
    test_many_waypoints()
    test_random_waypoints()
    test_sigma_trace()
    print("All waypoint tests passed.")
