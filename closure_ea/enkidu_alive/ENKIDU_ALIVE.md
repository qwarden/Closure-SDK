# Enkidu Alive

Minimal instantiation of a proto-mind using Zeroth Law geometry.
No neural network. No training. Pure algebra as neural signal.

## What it is

A creature (Enkidu) lives on a 2D grid that projects onto S³ via
Clifford torus embedding. Home = identity quaternion = (1,0,0,0).

Enkidu has two drives:
- **Hunger** — scalar that rises over time. Resets to 0 when eating.
- **Coldness** — σ (geodesic distance from identity). Rises as he walks away from home.

The drive with higher σ wins. That's the entire decision system.

## The core insight

There is only ONE function: `best_step_toward(target)`.

- Going to food = `best_step_toward(food_position)`
- Going home = `best_step_toward(0, 0)`

Homeostasis is identity (σ = 0 on all axes). Disruption (hunger rising,
distance increasing) triggers the closure-seeking walk. The walk IS the
closure. The seeing IS the walking read backwards.

## Architecture

```
geometry.py       — Quaternion math: pos2quat(), sigma(), best_step_toward()
homeostasis.py    — World state, drives, tick loop (Python backend)
enkidu_alive.html — Self-contained visual demo (JS, same logic)
```

## How to use the demo

Open `enkidu_alive.html` in a browser.

**Manual mode:** Click the grid to place food. Keep Enkidu alive.

**Auto modes:**
- Thriving — food spawns frequently, close to home. Watch foraging emerge.
- Scarce — less food, further away. Hesitation at the hunger/cold boundary.
- Famine — rare food at the edge of survival radius. He dies a lot.

**Controls:** Pause, speed toggle, reset.

## What to watch for

1. **The hesitation** — when hunger ≈ coldness, Enkidu oscillates. That's
   competing closure loops. Not a bug. That's proto-attention.
2. **The straight-line home** — he never retraces his outbound path.
   The algebraic residual C⁻¹ decomposes into the minimal steps home.
3. **The rhythm** — in thriving mode, a foraging cycle emerges:
   hungry → walk to food → eat → cold → walk home → rest → hungry again.
   This rhythm was not programmed. It falls out of max(σ_hunger, σ_cold).

## Death

Hunger at π = starvation. Enkidu dies. The mortality pressure is what
makes food-seeking non-optional. Without death, he can rest at home
forever. With death, the homeostasis loop MUST cycle.
