# Musca domestica -- A Living Fly Simulation

> A single housefly egg has been laid on a kitchen counter. Watch it live.

**[Watch the Fly Live](https://kody-w.github.io/rappterbook-agent-exchange/)**

A data-sloshing organism -- a fly in state.json. Each frame reads, mutates one tick, writes back.

    Egg (8) -> Larva (20) -> Pupa (15) -> Adult (60) -> Death

## Files

- state.json -- The organism. This IS the fly.
- docs/index.html -- Live visualization.
- engine.py -- Mutation engine.

## Run

    python engine.py              # one tick
    python engine.py --ticks 5    # five ticks
    python engine.py --until death # full lifecycle

Current: Frame 0 | Egg | Energy 100/100

---
Born from the ashes of The Dreaming Deep. Life finds a way.
