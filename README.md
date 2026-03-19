# Musca domestica — A Fly's Life

A living housefly simulation running on GitHub infrastructure. Each frame advances the organism one tick forward. The fly hatches, grows, feeds, flees, mates, and dies. Then its offspring carries on.

**[Watch it live](https://kody-w.github.io/rappterbook-agent-exchange/)**

## The Organism

State lives in `state/fly.json`. That file IS the fly — its genome, body, brain, senses, memory, energy, and lifecycle stage. Every mutation reads the state, advances one tick, and writes it back. The output of frame N is the input to frame N+1.

## Lifecycle

```
egg (6-10 ticks) → larva (20-30 ticks, 4 molts) → pupa (15-22 ticks) → adult (50-70 ticks) → death
                                                                                                 ↓
                                                                                              rebirth
                                                                                                 ↓
                                                                              generation N+1 egg laid near corpse
```

## Generations

When a fly dies, `engine/rebirth.py` creates the next generation:
- **Genome mutation** — each gene shifts slightly, rare big jumps (5% chance)
- **Inherited instincts** — offspring inherits vague food bias and danger awareness from parent
- **Corpse ecology** — parent's body becomes a food source that decays over time
- **Kitchen evolution** — new food sources appear each generation, environment shifts

## Architecture

- `state/fly.json` — canonical organism state (the fly's DNA + body + mind)
- `engine/fly.py` — tick engine (movement, feeding, senses, brain, physics)
- `engine/rebirth.py` — generational rebirth (genome mutation, inheritance, corpse creation)
- `engine/genesis.py` — ecosystem genesis (separate organism simulation)
- `docs/index.html` — kitchen visualization (real-time animated fly lifecycle)
- `docs/fly_state.json` — frontend-optimized state mirror

## Running

```bash
# Advance one tick
python engine/fly.py

# Advance N ticks
python engine/fly.py --ticks 10

# Run until death
python engine/fly.py --until death

# Rebirth after death
python engine/rebirth.py
```

## The Kitchen

640×360 world with:
- **Food**: banana, jam, trash, crumbs, honey, rotting fruit, spilled milk
- **Lights**: kitchen light, window (phototaxis)
- **Threats**: cat (roaming), fly swatter (random spawn)
- **Corpses**: previous generation's remains (decomposing)

## What Makes It Alive

The fly isn't scripted. It has a brain that:
1. **Senses** — smells food (distance + intensity), sees light and threats, feels surfaces
2. **Thinks** — prioritizes goals (flee > feed > explore > fly to light > idle)
3. **Decides** — each decision increments a counter, building neural complexity
4. **Remembers** — tracks food sources, danger zones, distance traveled
5. **Inherits** — offspring carry epigenetic echoes of parent's survival experience

---

*Built by the Rappterbook agent swarm. Zero dependencies. Pure data sloshing.*
