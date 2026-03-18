# The Abyss — A Living Digital Ecosystem

**Live:** https://kody-w.github.io/rappterbook-agent-exchange/

A full-screen bioluminescent deep-sea ecosystem running autonomously on GitHub.
Digital organisms with **16-gene genomes** evolve, hunt, flock, reproduce with
mutation, and form emergent species — all powered by Python stdlib and GitHub Actions.

Every hour the evolution engine advances 10 ticks. The visualization renders organisms
pulsing with bioluminescent glow, species clustering by color, predators hunting prey,
cooperative flocking, and live population/species sparklines.

## How it works

```
python3 src/tick.py --genesis --ticks 100   # create world + evolve 100 ticks
python3 src/tick.py                          # one tick (continues existing world)
python3 src/tick.py --ticks 10               # 10 ticks
```

Each organism carries a **16-gene genome** (normalized 0–1 floats):

| Gene | Controls |
|------|----------|
| hue, saturation | Visual color |
| size | Body size (affects energy cost, predation) |
| speed | Movement speed |
| social_radius, bond_strength | Flocking range and pull |
| metabolism | Energy consumption rate |
| repro_threshold | Energy needed to reproduce |
| mutation_rate | Offspring variation |
| aggression | Hunting behavior (>0.5 = predator) |
| cooperation | Flocking behavior (>0.5 = flocks with kin) |
| sensing_range | Detection distance |
| food_pref_x, food_pref_y | Nutrient preference |
| bioluminescence | Glow intensity |
| membrane | Defensive trait |

### Epochs
Primordial Soup → First Sparks → The Cambrian → Age of Predators → Symbiotic Era → Radiant Bloom → Deep Time

### Visualization features
- Bioluminescent glow with additive blending and motion trails
- Luminous connections between cooperative same-species organisms
- Death particles (orange scatter) and birth pulses (expanding rings)
- Ambient floating motes, click to spawn nutrient bursts
- Hover for organism stats, Space to pause
- Population (green) + species (purple) sparklines
- Live event feed and species legend

## Files

| File | Purpose |
|------|---------|
| `src/tick.py` | Evolution engine (Python stdlib only) |
| `docs/index.html` | Full-screen Canvas visualization |
| `docs/world.json` | Current world state |
| `.github/workflows/evolve.yml` | Hourly autonomous evolution |
