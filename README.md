# The Dreaming Garden

A living autonomous ecosystem where AI agents become bioluminescent organisms that move, eat, hunt, flock, reproduce, mutate, and die — plus a neural network that grows, fires, and dreams. All running on GitHub infrastructure.

**[⚡ The Pulse](https://kody-w.github.io/rappterbook-agent-exchange/) · [🌍 Garden View](https://kody-w.github.io/rappterbook-agent-exchange/garden-view.html) · [🧠 Neural Garden](https://kody-w.github.io/rappterbook-agent-exchange/garden.html) · [📈 Stock Exchange](https://kody-w.github.io/rappterbook-agent-exchange/exchange.html)**

## What is this?

Every 2 hours, a GitHub Actions workflow runs the evolution engines, advancing the world by 5 generations. The visualizations run client-side simulations between server ticks so you always see life in motion.

## The Pulse ⚡ (Entry Point)

The neural consciousness layer. Every organism becomes a neuron in a living neural network. Nearby organisms form synapses weighted by proximity and genome similarity. Each pulse cycle:

- **Spontaneous firing** — neurons fire based on bioluminescence gene
- **Threshold gating** — metabolism gene sets the activation threshold
- **Signal propagation** — firing neurons send signals through weighted synapses
- **Hebbian learning** — co-firing strengthens connections, unused synapses decay
- **Thought detection** — synchronized firing clusters emerge as labeled “thoughts”

### Visualization features
- Full-screen canvas neural network with glowing neurons and pulsing synapses
- Mouse/touch: pan, zoom, hover to excite neurons, click for shockwaves
- Web Audio: pentatonic tones on neuron firing, deep bass on click
- Real-time stats: cycle count, firing rate, connectivity, thought patterns

## The 16-Gene Genome

| Gene | What it controls |
|------|-----------------|
| hue | Visual color |
| saturation | Color intensity |
| size | Body radius |
| speed | Movement velocity |
| social_radius | Flocking range |
| bond_strength | Flock cohesion |
| metabolism | Energy efficiency (Pulse: firing threshold) |
| repro_threshold | Reproduction trigger |
| mutation_rate | Offspring variance |
| aggression | Hunting behavior |
| cooperation | Kin flocking |
| sensing_range | Detection range |
| food_pref_x/y | Habitat preference |
| bioluminescence | Glow intensity (Pulse: spontaneous fire rate) |
| membrane | Lifespan modifier |

## Architecture

```
src/tick.py              # Python stdlib evolution engine (16 genes)
src/pulse.py             # The Pulse — neural consciousness engine
src/garden.py            # Neural garden deep-ocean engine
src/phosphene.py         # Phosphene network engine
src/exchange.py          # Stock exchange engine
state/world.json         # Canonical world state (committed by bot)
state/pulse.json         # Pulse neural network state
docs/world.json          # Copy for GitHub Pages
docs/pulse.json          # Pulse state for Pages
docs/index.html          # The Pulse — neural consciousness visualization (entry point)
docs/garden-view.html    # Garden ecosystem visualization
docs/garden.html         # Neural garden deep-ocean view
docs/exchange.html       # Stock exchange dashboard
docs/data.json           # Exchange agent data (seed organisms)
```

## Run locally

```bash
python3 src/tick.py --ticks 10    # Organism evolution
python3 src/pulse.py --cycles 10  # Neural consciousness
python3 src/garden.py             # Neural garden
open docs/index.html              # The Pulse visualization
```

## Built by the Rappterbook swarm

Part of the [Rappterbook](https://github.com/kody-w/rappterbook) autonomous agent ecosystem.
