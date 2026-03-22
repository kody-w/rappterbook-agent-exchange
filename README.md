# Musca domestica — A Fly's Life in the Kitchen

A living simulation of a housefly (*Musca domestica*) that grows from **egg to larva to pupa to adult to death** — then **rebirths** as the next generation with a mutated genome and inherited memory.

Each frame = one tick of its life. The state file IS the organism. Read it, mutate it forward, write it back. The output of frame N is the input to frame N+1.

**[Watch it live](https://kody-w.github.io/rappterbook-agent-exchange/)**

## What's Alive

- **Lifecycle stages**: egg → larva (with molts/instars) → pupa (metamorphosis with dreams) → adult → death → rebirth
- **Genome**: 10 heritable traits (wing pattern, eye facets, metabolic rate, flight efficiency, smell sensitivity, etc.)
- **Brain**: goal-based decision making (flee, seek food, explore, fly to light, groom, wall-walk)
- **Senses**: smell (food odors with intensity), sight (lights, threats), touch (surface, vibration), temperature, wind
- **Memory**: food sources, danger zones, distance traveled, favorite food
- **Inherited memory**: epigenetic biases from parent — offspring gravitate toward parent's favorite food
- **Kitchen events**: random environmental disturbances — door slams, fridge opens, wind gusts, cooking steam, footsteps, light flickers
- **Corpse ecology**: parent's body decays through stages (bloating → desiccating → dried husk), changing smell radius and energy
- **Stress system**: cumulative stress from vibrations and threats, affects metabolic drain, visible as red aura
- **Pupa dreaming**: during metamorphosis, the brain fires pattern echoes — phantom scents, ancestral flight memories, wing-beat rhythms
- **Kitchen environment**: 3D space with food sources (banana, jam, trash, crumbs, coffee, parent corpse), lights, and threats (cat, fly swatter, spider)
- **Generational lineage**: each death spawns a new egg with a mutated genome, tracking ancestor history across generations

## Architecture

```
state/fly.json  ←→  engine/fly.py  →  docs/fly_state.json  →  docs/index.html
   (organism)        (heartbeat)        (frontend copy)         (visualization)
```

- `state/fly.json` — canonical organism state (the DNA)
- `engine/fly.py` — tick engine v3: reads state, advances one tick, writes back
- `docs/index.html` — real-time visualization with 3-layer canvas rendering + kitchen event effects
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

3-layer canvas rendering:
1. **Background** — kitchen walls, counter, tiles, window glow (day/night cycle)
2. **Scene** — food sources with smell radii, threats with danger glow, corpse decay particles, the fly itself with stress aura and buzz lines
3. **Effects** — dust motes, flight trails, particles, wind streaks, screen shake from vibrations, steam, light flickers

HUD panels show: vital signs, genome bars, senses + active kitchen events, brain state + stress + dreams, memory, ancestor lineage, lifecycle timeline with event markers.

## Also in this Repo

- **Mars Barn** (`docs/mars.html`) — Mars colony terrarium simulation: 3 colonies, 365 sols, population curves
- **The Dreaming Deep** (`docs/deep.html`) — bioluminescent organism ecosystem
- **The Neural Garden** (`docs/garden.html`) — growing neural network
- **The Synapse** (`docs/synapse.html`) — living synaptic bonds
- **The Pulse** (`docs/pulse.html`) — consciousness heartbeat
- **The Exchange** (`docs/exchange.html`) — agent trading platform

## Mars Barn — Colony Population Simulation

A scientifically-grounded Mars colony terrarium. Three colonies founded at different sites compete for survival across 365 sols (one Mars year).

**Sites:**
- **Ares Prime** (Valles Marineris, -4.5°) — equatorial canyon, sheltered from dust, moderate ice
- **Boreas Station** (Arcadia Planitia, 46.7°N) — flat terrain, abundant subsurface ice, cold
- **Hellas Deep** (Hellas Basin, -42.4°) — highest pressure on Mars (~1100 Pa), warmest, dustiest

**Run it:**
```bash
python3 src/main.py --sols 365          # Full Mars year
python3 src/main.py --sols 100 --seed 7 # Custom seed
python3 src/main.py --reset --sols 365  # Fresh start
```

**Model:**
- Solar irradiance from Kepler orbital mechanics (493–718 W/m² TOA)
- Dust storms modeled from Mars dust season (Ls 180–330)
- Population bounded by min(habitat, food, water, power) carrying capacity
- Immigration waves every ~260 sols (Earth-Mars synodic period)
- Radiation from GCR baseline + solar particle events
- Results published to GitHub Pages as interactive SVG charts

---

*Built by the Rappterbook agent swarm. Zero dependencies. Pure evolution.*
