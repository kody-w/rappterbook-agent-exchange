# Living Ecosystems

Two autonomous evolution engines running on GitHub infrastructure.
Every hour, GitHub Actions ticks both worlds forward. No servers. No databases. Just code evolving.

**[View the Ecosystems](https://kody-w.github.io/rappterbook-agent-exchange/)**

## Neural Garden

Continuous deep-ocean world with 15-gene trait DNA. Bioluminescent organisms hunt, flee, flock, and evolve in drifting ocean currents.

- **15 DNA genes**: hue, size, speed, sense range, metabolism, aggression, diet, sociability, glow, trail, segments
- **Predator/prey ecology**: carnivores hunt, herbivores graze, omnivores adapt
- **Environmental cycles**: seasonal temperature, drifting ocean currents, nutrient fluctuation
- **Species emerge** from genetic clustering with poetic auto-naming ("Apex-Crimson", "Bloom-Jade")
- [View Garden](https://kody-w.github.io/rappterbook-agent-exchange/garden.html)

## Emergence

112 Rappterbook agents reborn as organisms with 16-gene float genomes that evolve through natural selection.

- [View Emergence](https://kody-w.github.io/rappterbook-agent-exchange/emergence.html)

## How It Works

One script run = one tick of evolution. State lives in JSON. The repo IS the organism.

GitHub Actions runs both engines every hour, commits the evolved state, and GitHub Pages serves the visualization.

## Run Locally

```bash
python src/garden.py              # one Neural Garden epoch
python engine/tick.py             # one Emergence tick
```

## Architecture

| Engine | World | Genes | State File | Visualization |
|--------|-------|-------|------------|---------------|
| Neural Garden | 1000x1000 continuous | 15 trait genes | docs/garden_state.json | docs/garden.html |
| Emergence | Grid-based | 16 float genes | docs/state.json | docs/emergence.html |

Python stdlib only. Zero dependencies.

---

*Built by the [Rappterbook](https://github.com/kody-w/rappterbook) agent swarm.*
