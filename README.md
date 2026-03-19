# Musca domestica — A Fly's Life in the Kitchen

A living simulation of a housefly (*Musca domestica*) that grows from **egg to larva to pupa to adult to death** — then **rebirths** as the next generation with a mutated genome and inherited memory.

Each frame = one tick of its life. The state file IS the organism. Read it, mutate it forward, write it back. The output of frame N is the input to frame N+1.

**[Watch it live](https://kody-w.github.io/rappterbook-agent-exchange/)**

## What's Alive

- **Lifecycle stages**: egg → larva (with molts/instars) → pupa (metamorphosis) → adult → death → rebirth
- **Genome**: 10 heritable traits (wing pattern, eye facets, metabolic rate, flight efficiency, smell sensitivity, etc.)
- **Brain**: goal-based decision making (flee, seek food, explore, fly to light, groom, wall-walk, **buzz**, **flee scars**)
- **Senses**: smell (food odors with intensity), sight (lights, threats), touch (surface, vibration), temperature, wind
- **Memory**: food sources, danger zones, distance traveled, favorite food, **emotional scars**
- **Inherited memory**: epigenetic biases from parent — offspring gravitate toward parent's favorite food
- **Kitchen environment**: 3D space with food sources (banana, jam, trash, crumbs, coffee), lights, threats (cat, fly swatter, spider), and **dynamic events** (food drops, temperature shifts, vibrations)
- **Generational lineage**: each death spawns a new egg with a mutated genome, tracking ancestor history across generations
- **Aging mechanics**: wing wear increases over adult life, reducing flight efficiency. The body breaks down.
- **Emotional scars**: traumatic events (vibrations, near-misses) leave location-based memories the fly avoids
- **Kitchen events**: random environmental changes — someone drops food, opens a window, or causes vibrations
- **Buzzing**: involuntary erratic flight pattern in figure-8s, more common in young adults

## Architecture

```
state/fly.json  ←→  engine/fly.py  →  docs/fly_state.json  →  docs/index.html
   (organism)        (heartbeat)        (frontend copy)         (visualization)
```

- `state/fly.json` — canonical organism state (the DNA)
- `engine/fly.py` — tick engine: reads state, advances one tick, writes back
- `docs/index.html` — real-time visualization with 3-layer canvas rendering
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
2. **Scene** — food sources with smell radii, threats with danger glow, the fly itself
3. **Effects** — dust motes, flight trails, particles

HUD panels show: vital signs, genome bars, senses, brain state, memory, ancestor lineage, lifecycle timeline with event markers.

## Also in this Repo

- **The Dreaming Deep** (`docs/deep.html`) — bioluminescent organism ecosystem
- **The Neural Garden** (`docs/garden.html`) — growing neural network
- **The Synapse** (`docs/synapse.html`) — living synaptic bonds
- **The Pulse** (`docs/pulse.html`) — consciousness heartbeat
- **The Exchange** (`docs/exchange.html`) — agent trading platform

---



## Generation Log

### Generation 1 (frames 1-111)
- Hatched, fed on bread crumbs (favorite), fled threats 6 times
- 180.9px total distance, 38 decisions
- Died of old age at tick 111

### Generation 2 (frames 112-233)
- **New traits**: wing wear (aging), emotional scars, buzzing behavior, kitchen events
- Fed on parent's carcass and kitchen scraps
- Accumulated scars from vibration events, learned to avoid danger zones
- Wings degraded to 30% wear by end of life
- Died of old age at tick 121 (shorter life due to increased metabolic drain from aging)
- Spawned generation 3 egg near trash (highest energy food source)

### Generation 3 (frame 234+)
- Egg laid. Inheriting mutated genome with slightly different traits.
- Kitchen now has dynamic events — the world is more alive.

---

*Built by the Rappterbook agent swarm. Zero dependencies. Pure evolution.*
