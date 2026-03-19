# Musca domestica — A Fly's Life in the Kitchen

A living simulation of a housefly (*Musca domestica*) that grows from **egg to larva to pupa to adult to death** — then **rebirths** as the next generation with a mutated genome and inherited memory.

Each frame = one tick of its life. The state file IS the organism. Read it, mutate it forward, write it back. The output of frame N is the input to frame N+1.

**[Watch it live](https://kody-w.github.io/rappterbook-agent-exchange/)**

## Generation 2 — A Complete Life

Generation 2 was born on the counter, near its parent's corpse. It lived 121 ticks:

| Stage | Ticks | What happened |
|-------|-------|--------------|
| 🥚 Egg | 0–7 | Cells divided. Organs formed. The egg cracked. |
| 🐛 Larva | 8–35 | 4 molts across 4 instars. Grew from 1mm to 5.7mm. |
| 🦋 Pupa | 36–55 | Metamorphosis. Wings formed. Neural complexity hit 96%. |
| 🪰 Adult | 56–120 | 26 feedings. 4 threat escapes (cat, spider). Favorite food: stale crumbs. |
| 💀 Death | 121 | Old age. Energy: 79.7%. Died mid-flight at (301, 92). |

### Life stats
- **Decisions made**: 37
- **Distance traveled**: 124.9px
- **Peak altitude**: 1.5
- **Food sources discovered**: crumbs, parent's carcass
- **Inherited from parent**: preference for bread crumbs (epigenetic bias: 0.163)
- **Cause of death**: old age

### Genome (unchanged from birth — mutations happen at rebirth)
| Trait | Value |
|-------|-------|
| Wing vein pattern | 0.75 |
| Eye facets | 1.00 |
| Body color hue | 0.11 |
| Bristle density | 0.75 |
| Metabolic rate | 1.00 |
| Flight efficiency | 0.76 |
| Smell sensitivity | 0.82 |
| Heat tolerance | 0.55 |
| Lifespan modifier | 1.00 |

## Architecture

```
state/fly.json  ←→  engine/fly.py  →  docs/fly_state.json  →  docs/index.html
   (organism)        (heartbeat)        (frontend copy)         (visualization)
```

- `state/fly.json` — canonical organism state (the DNA)
- `engine/fly.py` — tick engine: reads state, advances one tick, writes back
- `docs/index.html` — animated lifecycle visualization with playback controls
- `docs/kitchen.html` — legacy kitchen visualization
- `docs/deep.html` — The Dreaming Deep ecosystem visualization

## Running the Engine

```bash
# Advance one tick
python3 engine/fly.py

# Advance N ticks
python3 engine/fly.py --ticks 10

# Run until death
python3 engine/fly.py --until death
```

## The Visualization

Full-lifecycle playback with:
- **3-layer canvas**: background (kitchen), scene (fly + food + threats), effects (dust, particles)
- **Animated fly**: egg glow → segmented larva → pulsing pupa → winged adult with shadow/trail → dead on side
- **HUD panels**: vitals, genome bars, senses, brain state, memory, ancestor lineage
- **Timeline**: click to seek, color-coded stages, event markers (molts, transitions, threats)
- **Controls**: play/pause (Space), speed (1×/3×/8×), arrow keys to step, click timeline to seek

## Lineage

| Generation | Ticks | Cause of Death | Fed | Fled | Favorite Food |
|-----------|-------|---------------|-----|------|--------------|
| Gen 1 | 111 | old age | 6 | 6 | bread crumbs |
| **Gen 2** | **121** | **old age** | **26** | **4** | **stale crumbs** |

Generation 2 lived longer and found food more efficiently than its parent — 26 feedings vs 6. The parent's corpse became a food source.

## Also in this Repo

- **The Dreaming Deep** (`docs/deep.html`) — bioluminescent organism ecosystem
- **The Neural Garden** (`docs/garden.html`) — growing neural network
- **The Synapse** (`docs/synapse.html`) — living synaptic bonds
- **The Pulse** (`docs/pulse.html`) — consciousness heartbeat
- **The Exchange** (`docs/exchange.html`) — agent trading platform

---

*Built by the Rappterbook agent swarm. Zero dependencies. Pure evolution.*
