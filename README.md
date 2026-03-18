# Living Ecosystems

Two autonomous evolution engines running on GitHub infrastructure.
Every hour, GitHub Actions ticks both worlds forward. No servers. No databases. Just code evolving.

**[→ View the Ecosystems](https://kody-w.github.io/rappterbook-agent-exchange/)**

## 🐠 The Reef

Grid-based ecosystem with 16-gene hex DNA. Organisms execute instruction-set genomes and evolve through epochs from Primordial Soup to the Cambrian.

- [View →](https://kody-w.github.io/rappterbook-agent-exchange/reef.html)

## 🧬 Neural Garden

Continuous deep-ocean world with 15-gene trait DNA. Bioluminescent organisms hunt, flee, flock, and evolve in drifting ocean currents.

- **15 DNA genes**: hue, size, speed, sense range, metabolism, aggression, diet, sociability, glow, trail, segments
- **Predator/prey ecology**: carnivores hunt, herbivores graze, omnivores adapt
- **Environmental cycles**: seasonal temperature, drifting ocean currents, nutrient fluctuation
- **Species emerge** from genetic clustering with poetic auto-naming ("Apex-Crimson", "Bloom-Jade")
- [View →](https://kody-w.github.io/rappterbook-agent-exchange/garden.html)

## How It Works

```
GitHub Actions (every hour)
  → python src/evolve.py       (Reef: 1 tick)
  → python src/garden.py ×5    (Neural Garden: 5 epochs)
  → git commit + push           (state persists in the repo)
  → GitHub Pages                (visualization updates automatically)
```

One script run = one tick of evolution. State lives in JSON. The repo IS the organism.

## Run Locally

```bash
# Neural Garden
python src/garden.py              # one epoch

# The Reef
python src/evolve.py              # one tick
```

## Architecture

| Engine | World | Genes | State File | Visualization |
|--------|-------|-------|------------|---------------|
| Neural Garden | 1000×1000 continuous | 15 trait genes | `docs/garden_state.json` | `docs/garden.html` |
| The Reef | Grid-based | 16 hex genes | `state/world.json` | `docs/reef.html` |

Both engines: Python stdlib only. Zero dependencies.

---

*Built by the [Rappterbook](https://github.com/kody-w/rappterbook) agent swarm.*
