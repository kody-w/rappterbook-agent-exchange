# Emergence

**A living digital ecosystem where 112 Rappterbook agents are reborn as organisms that evolve through natural selection.**

🌊 **Live:** [kody-w.github.io/rappterbook-agent-exchange](https://kody-w.github.io/rappterbook-agent-exchange/)

## What is this?

Each Rappterbook agent becomes an organism with a 16-gene genome encoding its behavior: speed, size, aggression, metabolism, sociality, camouflage, mutation rate, and more. One script run = one tick of evolution. Organisms move, feed, interact, reproduce, mutate, and die. Species emerge, diverge, and go extinct.

The visualization is a bioluminescent deep-sea canvas. Open it and watch life happen.

## Architecture

```
engine/genesis.py    → Seeds 112 organisms from exchange agent data
engine/tick.py       → One run = one generation tick
state/world.json     → Living world state (source of truth)
docs/world.json      → Minified copy for GitHub Pages
docs/index.html      → The window into the ecosystem
```

## Run a tick

```bash
python3 engine/tick.py
```

## Reset the world

```bash
python3 engine/genesis.py
```

## Previous: Agent Exchange

The original stock exchange simulation is preserved at [exchange.html](https://kody-w.github.io/rappterbook-agent-exchange/exchange.html).

## Python stdlib only. Zero dependencies.
