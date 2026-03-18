# Emergence -- A Living Digital Ecosystem

**Live:** https://kody-w.github.io/rappterbook-agent-exchange/

112 AI agents from Rappterbook, reborn as organisms in a self-evolving ecosystem. Each organism carries a 16-gene genome that controls its color, size, speed, social behavior, aggression, cooperation, bioluminescence, and metabolism. Natural selection does the rest.

## How it works

**One script run = one tick of evolution.**

    python3 engine/genesis.py   # Seed the world (run once)
    python3 engine/tick.py      # Evolve one generation

Each tick: organisms move toward nutrients, feed, interact (cooperators share energy, aggressors steal it), reproduce with mutation, and die from starvation. Species are reclassified by genome similarity. The dashboard updates live.

## The Genome (16 genes)

hue, saturation, size, speed, social_radius, bond_strength, metabolism, repro_threshold, mutation_rate, aggression, cooperation, sensing_range, food_pref_x, food_pref_y, bioluminescence, membrane

## The Visualization

Open docs/index.html -- a full-screen bioluminescent ecosystem. Glowing organisms with DNA-driven colors, sizes, and brightness. Luminous connections between similar neighbors. Hover to inspect, click to track, scroll to zoom, drag to pan.

## Files

- engine/genesis.py -- Seeds initial world from exchange agent data
- engine/tick.py -- Evolution engine (one run = one generation)
- state/world.json -- Living world state
- docs/index.html -- Bioluminescent ecosystem visualization
- docs/exchange.html -- Original stock exchange dashboard
- src/exchange.py -- Original exchange simulation engine

Python stdlib only. Zero dependencies.
