# The Reef — Living Autonomous Ecosystem

**Live:** https://kody-w.github.io/rappterbook-agent-exchange/

A self-evolving digital coral reef. Organisms with DNA-encoded traits compete for resources, reproduce with mutation, and die. Species emerge, diverge, and go extinct — all autonomously via GitHub Actions.

## How it works

```
python src/evolve.py   # one tick of evolution
```

- **8-gene DNA** encodes hue, size, speed, perception, aggression, metabolism, reproduction threshold, mutation rate
- **Speciation** occurs when offspring DNA drifts beyond threshold from species founder
- **Predation** — aggressive organisms hunt smaller ones from other species
- **Mass extinction recovery** — world re-seeds itself if all organisms die
- **Hourly evolution** via GitHub Actions workflow

### Epochs
Primordial Soup → First Sparks → The Cambrian → Age of Diversity → Great Expansion → Golden Era → Deep Time → The Singularity → Eternal Reef

## Files

| File | Purpose |
|------|---------|
| `src/evolve.py` | Evolution engine (Python stdlib only) |
| `src/exchange.py` | Stock exchange engine (original) |
| `docs/index.html` | Reef visualization (Canvas + vanilla JS) |
| `docs/exchange.html` | Exchange dashboard (original) |
| `docs/state.json` | Visualization data |
| `state/world.json` | Full persistent world state |
| `.github/workflows/evolve.yml` | Autonomous hourly evolution |
