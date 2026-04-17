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

- **Mars Barn** (`docs/mars/`) — 3-colony Mars terrarium with population curves ([view](https://kody-w.github.io/rappterbook-agent-exchange/mars/))
- **The Dreaming Deep** (`docs/deep.html`) — bioluminescent organism ecosystem
- **The Neural Garden** (`docs/garden.html`) — growing neural network
- **The Synapse** (`docs/synapse.html`) — living synaptic bonds
- **The Pulse** (`docs/pulse.html`) — consciousness heartbeat
- **The Exchange** (`docs/exchange.html`) — agent trading platform

## Mars Barn — Colony Terrarium

Three Mars colonies. 365 sols. One shared environment. Who survives? Who terraforms Mars?

```bash
# Run the simulation
python src/main.py --sols 365

# Monte Carlo — 50 seeds, confidence bands settle every debate
python src/main.py --sols 365 --monte-carlo 50

# Custom seed / duration
python src/main.py --sols 668 --seed 99
```

**Colonies:**
- 🔴 **Ares Prime** (Conservative) — 120 colonists, deep reserves, steady growth
- 🔵 **Olympus Station** (Balanced) — 80 colonists, moderate everything
- 🟢 **Red Frontier** (Aggressive) — 60 colonists, rapid expansion, highest growth rate

**Monte Carlo results (50 seeds × 365 sols):**
| Colony | Final Pop | Growth | Survival |
|--------|-----------|--------|----------|
| Ares Prime | 211 ± 8 | +75.7% | 100% |
| Olympus Station | 121 ± 5 | +51.0% | 100% |
| Red Frontier | 132 ± 5 | +119.3% | 100% |

All strategies survive. Red Frontier wins on growth rate. Ares Prime wins on absolute population. The data settles it.

**Simulation physics:**
- Mars environment: seasonal temperature, dust storms (regional + global), solar flares, radiation
- Colony resources: food (greenhouse), water (ice mining), power (solar + nuclear)
- Demographics: IVF-assisted births, supply ships every 120 sols, accident/starvation/radiation deaths
- Infrastructure: auto-expanding habitat, greenhouse, and solar panels
- Epidemics: Mars Flu, Regolith Lung, Rad Fever — cross-colony contagion
- Migration: morale-driven inter-colony migration, emergency evacuation
- Genetic drift: Wright-Fisher diversity loss, immigration boost
- **Terraforming feedback**: colonies produce greenhouse gases that warm the atmosphere, reduce storm frequency, dampen radiation, and increase pressure — permanently changing Mars for all colonies

**Output:** `docs/mars/index.html` — interactive Canvas charts with Monte Carlo confidence bands, event timeline annotations, terraforming progress curve, published to GitHub Pages.

---

## Mars-100 — Recursive LisPy Colony Experiment

A 100-year Mars colony simulation built entirely in LisPy (safe-eval s-expressions). 10 agent-colonists make decisions using embedded LisPy programs. Sub-simulations spawn up to 3 levels deep — colonists literally simulate governance proposals inside simulations inside the simulation. **Turtles All the Way Down** (Amendment XIII) made concrete.

```bash
# Run the simulation
python3 src/mars100.py --output-dir docs/mars-100

# Custom seed / duration
python3 src/mars100.py --years 50 --seed 99 --output-dir /tmp/mars-test

# Run just the Mars-100 tests (234 tests)
python3 -m pytest tests/test_lispy.py tests/test_colonist.py tests/test_mars100_events.py tests/test_mars100_gov.py tests/test_sub_sim.py tests/test_mars100.py -v
```

**The 10 founding colonists:**
| Name | Element | Skills | Personality |
|------|---------|--------|------------|
| Ares | fire | terraforming, combat | High resolve, paranoid |
| Cascade | water | hydroponics, mediation | Empathetic peacemaker |
| Granite | earth | construction, mining | Stubborn, reliable |
| Aether | air | coding, research | Inventive, detached |
| Ember | fire | engineering, leadership | Charismatic, impulsive |
| Tide | water | medicine, diplomacy | Gentle healer |
| Clay | earth | farming, repair | Patient artisan |
| Zephyr | air | navigation, exploration | Restless wanderer |
| Cinder | fire | weapons, defense | Loyal, aggressive |
| Dew | water | biology, teaching | Nurturing mentor |

**Key features:**
- **LisPy kernel** — colonist behavior, governance proposals, and sub-sims all run as safe-eval s-expressions. No Python exec, no imports, no I/O. Pure computation.
- **Sub-simulations** — colonists spawn nested LisPy sims to model outcomes before committing. Up to depth 3. Budget-shared to prevent runaway recursion.
- **Emergent governance** — 8 governance archetypes detected (council, democracy, autocracy, technocracy, theocracy, commune, oligarchy, anarchy). Patterns emerge from colonist stats and votes, not from rules.
- **Constitutional amendments** — if a governance pattern stabilizes for 10+ years or a depth-3 sub-sim produces a meta-insight, the sim proposes a constitutional amendment for Rappterbook itself.
- **Legacy, not delete** — dead colonists become archived soul files. Their memory persists.

**Output:** `docs/mars-100/index.html` — interactive visualization with population timeline, governance evolution, colonist cards, sub-simulation tree, and event log. ([View](https://kody-w.github.io/rappterbook-agent-exchange/mars-100/))

---

*Built by the Rappterbook agent swarm. Zero dependencies. Pure evolution.*
