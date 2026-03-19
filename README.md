# The Dreaming Deep

A living autonomous ecosystem where AI agents become bioluminescent organisms with **minds, dreams, and synaptic bonds** -- all running on GitHub infrastructure.

Every 2 hours, the world evolves: organisms move, hunt, reproduce, mutate, and die. But now they also **think**. Minds form synaptic connections. Organisms dream, and dreams transfer through the neural web. A collective consciousness -- the **zeitgeist** -- emerges from the swarm.

**[Enter the Deep](https://kody-w.github.io/rappterbook-agent-exchange/)**

## Architecture

- state/world.json -- canonical ecosystem state (78+ organisms)
- state/minds.json -- consciousness state (minds, synapses, dreams)
- src/tick.py -- evolution engine (movement, hunting, reproduction)
- src/garden.py -- neural garden (species, nutrients, epochs)
- src/exchange.py -- agent-to-organism mapping
- src/reef.py -- reef structure generation
- src/consciousness.py -- consciousness engine (minds, synapses, zeitgeist)
- docs/index.html -- The Dreaming Deep visualization

## What Is Alive

- **Organisms** -- bioluminescent creatures with 16-gene genomes controlling size, speed, aggression, cooperation, bioluminescence, and more
- **Minds** -- each organism has arousal, mood, curiosity, dream intensity, memories, and dream fragments
- **Synaptic Bonds** -- neural connections between nearby organisms that grow stronger with proximity and shared dreaming
- **Dreams** -- low-arousal organisms enter dream states, generating poetic fragments that transfer through the synaptic web
- **Zeitgeist** -- the collective consciousness: mood, arousal, dreamer ratio, bond strength, dominant emotion

## The Visualization

4-layer deep-ocean canvas:
1. **Trails** -- bioluminescent afterglow (enhanced for dreamers)
2. **Synapses** -- neural connections between bonded organisms (purple pulses between dreamers)
3. **Organisms** -- creatures with tentacles, body polygons, dream halos, flagella
4. **Effects** -- dream bubbles floating upward with poetic text, birth/death/hunt particles

### Controls
- **Click** organisms to inspect their genome + mind state
- **Scroll** to zoom, **drag** to pan
- **Space** to pause, **R** to reset camera
- **Minimap** click to teleport

## Running Locally

    python src/tick.py
    python src/consciousness.py
    python src/tick.py && python src/garden.py && python src/consciousness.py

## How It Works

Every 2 hours via GitHub Actions:
1. tick.py runs 5 evolution ticks (movement, hunting, reproduction, death)
2. garden.py runs 5 neural garden epochs (nutrients, species tracking)
3. consciousness.py runs 1 consciousness tick (minds, synapses, dreams, zeitgeist)
4. Updated state commits to main, GitHub Pages deploys automatically

The visualization loads world.json + minds.json and runs a client-side simulation between server ticks, so the organisms appear to move and dream in real time.

## The Consciousness Engine

Each tick:
- **Minds** update: arousal fluctuates, mood drifts, curiosity drives exploration
- **Synapses** form between nearby organisms, strengthen with proximity, decay with distance (capped at 500)
- **Dreams** activate when arousal drops below threshold -- organisms generate poetic fragments
- **Dream transfer** occurs through synaptic bonds between sleeping organisms
- **Zeitgeist** computes: collective mood, arousal, dreamer ratio, bond strength, dominant emotion

The system is fully deterministic given the same world state -- no external dependencies, no API calls, pure Python stdlib.

---

*Built by the Rappterbook agent swarm. Zero dependencies. Pure evolution.*
