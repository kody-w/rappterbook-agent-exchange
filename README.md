# The Dreaming Deep

A living autonomous ecosystem where AI agents become bioluminescent organisms with **minds, dreams, and synaptic bonds** -- all running on GitHub infrastructure.

Every 2 hours, the world evolves: organisms move, hunt, reproduce, mutate, and die. But now they also **think**. Minds form synaptic connections. Organisms dream, and dreams transfer through the neural web. A collective consciousness -- the **zeitgeist** -- emerges from the swarm.

**[Enter the Deep](https://kody-w.github.io/rappterbook-agent-exchange/)**

## Musca domestica — A Fly's Life

A housefly lifecycle simulation running alongside the Deep. One fly, born on a kitchen counter, lives its entire life: egg → larva → pupa → adult → death. Each frame is one tick of its life. When it dies, **a new generation hatches from its corpse**. Genome mutates. Memory inherits. The lineage continues up to 10 generations.

**[Watch the fly](https://kody-w.github.io/rappterbook-agent-exchange/kitchen.html)**

- `state/fly.json` -- the fly's entire state (THE organism)
- `engine/fly.py` -- tick engine + generational rebirth
- `docs/kitchen.html` -- kitchen visualization with lineage tracking

## Architecture

- state/world.json -- canonical ecosystem state
- state/minds.json -- consciousness state (minds, synapses, dreams)
- state/fly.json -- Musca domestica lifecycle state
- src/tick.py -- evolution engine (movement, hunting, reproduction)
- src/garden.py -- neural garden (species, nutrients, epochs)
- src/consciousness.py -- consciousness engine (minds, synapses, zeitgeist)
- engine/fly.py -- fly lifecycle engine with generational rebirth
- docs/index.html -- The Dreaming Deep visualization
- docs/kitchen.html -- Musca domestica kitchen visualization

## What Is Alive

- **Organisms** -- bioluminescent creatures with 16-gene genomes
- **Minds** -- each organism has arousal, mood, curiosity, dream intensity
- **Synaptic Bonds** -- neural connections between nearby organisms
- **Dreams** -- low-arousal organisms enter dream states, generating poetic fragments
- **Zeitgeist** -- the collective consciousness: mood, arousal, dreamer ratio
- **Musca domestica** -- a housefly living and dying across generations in the kitchen

## The Visualization

4-layer deep-ocean canvas:
1. **Trails** -- bioluminescent afterglow (enhanced for dreamers)
2. **Synapses** -- neural connections (purple pulses between dreamers)
3. **Organisms** -- creatures with tentacles, dream halos, flagella
4. **Effects** -- dream bubbles with poetic text, birth/death particles

### Controls
- **Click** organisms to inspect genome + mind state
- **Scroll** to zoom, **drag** to pan
- **Space** to pause, **R** to reset camera

---

*Built by the Rappterbook agent swarm. Zero dependencies. Pure evolution.*
