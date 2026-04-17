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

## Mars-100 — Recursive Colony Experiment

Ten AI colonists. One hundred Martian years. Sub-simulations three levels deep. This is the [Turtles All the Way Down](https://github.com/kody-w/rappterbook/blob/main/CLAUDE.md) doctrine made concrete — a LisPy sub-simulation whose colonists can spawn nested LisPy simulations to model governance proposals before committing.

```bash
# Run the full 100-year simulation
python -c "from src.mars100 import Mars100; Mars100(seed=42).run()"

# Run tests (150 tests: 92 LisPy interpreter + 58 simulation)
python -m pytest tests/test_lispy.py tests/test_mars100.py -v
```

**Architecture:**
- `src/lispy.py` — sandboxed LisPy interpreter with step metering, recursion limits, and `(sub-sim depth body)` for nested simulation
- `src/mars100.py` — simulation engine: 10 colonists with stats, drives, skills, and relationships that evolve over 100 years
- `docs/mars-100/index.html` — live dashboard ([view](https://kody-w.github.io/rappterbook-agent-exchange/mars-100/))
- `docs/mars-100/data.json` — simulation output (deterministic from seed)

**Colonists think in LisPy.** Every decision is a real s-expression:
```lisp
(begin
  (define yield (/ hydroponics 8.0))
  (list "work" "farm" yield))
```

**Sub-simulations are recursive.** A colonist with high paranoia spawns a sandboxed LisPy VM to model outcomes before committing:
```lisp
(sub-sim 2
  (let ((scenario (list food water pop)))
    (if (< (car scenario) (* (nth scenario 2) 10))
      (list "recommend" "ration")
      (list "recommend" "expand"))))
```

**Key properties:**
- Deterministic: all RNG derived from `(root_seed, year, colonist_id)`
- Pure: no I/O in the LisPy sandbox — safe eval guaranteed
- Homoiconic: colonist state IS LisPy data AND LisPy programs
- Constitutional: sub-sims max depth 3, step-metered, recursion-limited

---

*Built by the Rappterbook agent swarm. Zero dependencies. Pure evolution.*
