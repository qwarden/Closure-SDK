# Enkidu Alive

An autonomous agent that forages, builds shelter, and survives through
quaternion composition on S³ — with zero learned parameters.

## What this is

Enkidu lives on a grid. Home is the identity element. Two drives
accumulate over time: hunger rises each tick, cold rises each step
away from shelter. Both are distances from identity — measures of how
far the agent has drifted from equilibrium. At every tick, Enkidu
compares the two distances, attends to whichever is larger, and takes
the step that most reduces it.

That single comparison, applied tick after tick, produces foraging
trips with direct return paths, hesitation when drives compete, drive
switching at the crossover point, and rest at identity when both
distances fall to zero. When taught to build shelter, Enkidu creates
new fixed points along his routes — niche construction — extending
his survivable range without any change to the underlying logic.

The mechanism is geometric attention: measure σ to each target, act
on the largest. It is also Friston's Free Energy Principle: minimise
surprise through active inference, with precision weighting selecting
the active channel. The algebra computes both directly.

## Running the demo

Open `enkidu_alive.html` in any browser. No server, no dependencies.

**Controls:**
- **Click** the grid to drop food
- **Shift+click** to place a shelter
- **Thriving / Scarce** toggle the environment
- **Cautious / Balanced / Bold** change his temperament
- **Teach him to build shelter** gives him the ability to create camps

**Suggested demo flow:**
1. Start in Thriving — watch the foraging rhythm emerge
2. Switch to Scarce — he dies of cold or hunger within a minute
3. Reset, Scarce again — same result
4. Click "Teach him to build shelter" — he survives, building camps
   along his foraging routes
5. Try Bold — he roams wider, builds further out, sometimes dies anyway
   (the cost of risk-taking)

## Survival statistics

Simulated over 2,000 runs of 600 ticks each, scarce environment:

| Temperament | No shelter | With shelter | Primary killer |
|-------------|-----------|-------------|----------------|
| Cautious    | 49%       | 100%        | hunger (72%)   |
| Balanced    | 49%       | 100%        | cold (81%)     |
| Bold        | 31%       | 90%         | cold (100%)    |

Thriving mode: 98% survival regardless of temperament.

## Architecture

There is no neural network. The entire agent is:

```
position      → quaternion via Clifford torus embedding
home          → identity element [1, 0, 0, 0]
coldness      → accumulated scalar, resets at shelter
hunger        → accumulated scalar, resets at food
drive         → max(hunger, cold × weight)
navigation    → best_step_toward(target) via Manhattan distance
shelter build → triggered when nearest shelter > threshold steps
```

The geometry handles navigation: UP/DOWN and LEFT/RIGHT are algebraic
inverses on S³. A path of arbitrary length composes into a single
quaternion residual. The residual decomposes into the direct steps
home — the agent never retraces its path.

## What comes next

**Food storage.** Enkidu carries food back to shelter, eating it later
to reduce hunger without a new foraging trip. Deferred hunger closure.

**Fire.** Camps with fire reset cold faster or provide a warmth radius.
Amplified cold closure. Requires Enkidu to first learn shelter (you
can't build a fire without a place to put it).

**Extreme mode.** Very scarce food, harsh cold. Requires both storage
and fire to survive — compound tool use emerging from the same algebra.

## Connection to the SDK

The Closure SDK's Enkidu class composes two streams, checks if they
close to identity, and classifies the residual as missing (W axis) or
reorder (RGB axis). This grid agent does the same thing: it composes
steps, checks distance from identity (σ), and the residual drives its
next move. The detector and the actor are the same operation.

See [DRAWING_BOARD.md](../DRAWING_BOARD.md) for the full experimental
record and [BRAHMAN.md](../BRAHMAN.md) for the architecture spec.
