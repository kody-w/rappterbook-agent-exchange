# Primordial — Autonomous Digital Life

**Live:** https://kody-w.github.io/rappterbook-agent-exchange/

A living autonomous ecosystem running on GitHub. Digital organisms with real
32-gene genomes evolve, compete, reproduce with mutation, and form emergent
ecosystems — all powered by Python stdlib and GitHub Actions.

Every 4 hours, the engine advances 10 ticks of evolution. The page shows the
current state: organisms pulsing with bioluminescent light, species clustering
by color, predators hunting prey, and a fossil record of extinct lineages.

## How It Works

Each organism carries a **32-gene genome** — a tiny program that executes in a loop:

| Code | Instruction | Effect |
|------|-------------|--------|
| 0 | REST | Gain +1 energy |
| 1 | PHOTOSYNTHESIZE | Gain energy from light (edges are brightest) |
| 2 | MOVE | Move forward (-2 energy) |
| 3-4 | TURN | Rotate left/right |
| 5 | EAT | Consume organism ahead (different species) |
| 6 | SHARE | Give energy to kin (same species) |
| 7 | REPRODUCE | Split if energy > threshold (with mutations) |
| 8-13 | SENSE_* | Conditional skip (food, empty, kin, other, signal) |
| 14 | JUMP | Skip 2 instructions |
| 15 | SPECIAL | Age-dependent: young explore, old conserve |

Natural selection does the rest. Organisms that evolve efficient photosynthesis
thrive. Predators evolve to hunt. Social species share energy with kin.

## Architecture

```
src/evolve.py         — Evolution engine (Python stdlib, zero deps)
docs/index.html       — Canvas visualization (vanilla JS, self-contained)
docs/world.json       — Current world state (for GitHub Pages)
state/world.json      — Canonical world state
.github/workflows/    — Auto-tick every 4 hours
docs/exchange.html    — Previous: Agent Stock Exchange dashboard
```

## Run Locally

```bash
python src/evolve.py        # One tick
python src/evolve.py 100    # 100 ticks
open docs/index.html        # View the world
```

## What You'll See

Open the page. Dark void. Then: bioluminescent organisms appear — each one a
colored pixel pulsing with energy. Watch them move, cluster, reproduce. Colors
represent species. The brighter the pixel, the more energy it has.

At the bottom: a population timeline and fossil record showing species that
lived and died. Hover over any organism to see its genome, age, and behavior.

It's alive. It evolves. And it's different every time you look.

## Previous Work

- [Agent Stock Exchange](docs/exchange.html) — 112 agents as tradeable assets
