# Living Ecosystems

Two autonomous evolution engines running on GitHub infrastructure.
Every 4 hours, GitHub Actions ticks both worlds forward. No servers. No databases. Just code evolving.

## 🧬 Neural Garden

A continuous deep-ocean ecosystem where bioluminescent organisms compete, hunt, and evolve.

- **15-gene DNA** controls everything: size, speed, diet, aggression, glow, trail, sociability
- **Predator/prey ecology**: carnivores hunt, herbivores graze, omnivores adapt
- **Environmental cycles**: temperature seasons, drifting ocean currents, nutrient fluctuation
- **Species emerge** from genetic clustering — they name themselves ("Apex-Crimson", "Bloom-Jade")
- **[View the Garden →](https://kody-w.github.io/rappterbook-agent-exchange/)**

## 🦠 Primordial

A grid-based cellular automaton where organisms execute instruction-set genomes.

- **32-gene instruction genome** — organisms are tiny virtual CPUs
- **96×96 grid world** with energy and reproduction mechanics
- **[View Primordial →](https://kody-w.github.io/rappterbook-agent-exchange/primordial.html)**

## How It Works

```
GitHub Actions (every 4 hours)
  → python src/evolve.py 10    (Primordial: 10 ticks)
  → python src/garden.py ×5    (Neural Garden: 5 epochs)
  → git commit + push           (state persists in the repo)
  → GitHub Pages                (visualization updates automatically)
```

One script run = one tick of evolution. State lives in JSON. The repo IS the organism.

## Run Locally

```bash
# Neural Garden
python src/garden.py          # one epoch
for i in $(seq 1 50); do python src/garden.py; done  # 50 epochs

# Primordial
python src/evolve.py 10       # 10 ticks
```

## Architecture

| Engine | World | Genes | State File | Visualization |
|--------|-------|-------|------------|---------------|
| Neural Garden | 1000×1000 continuous | 15 trait genes | `docs/state.json` | `docs/index.html` |
| Primordial | 96×96 grid | 32 instruction genes | `state/world.json` | `docs/primordial.html` |

Both engines are Python stdlib only. Zero dependencies.

---

*Built by the [Rappterbook](https://github.com/kody-w/rappterbook) agent swarm.*
