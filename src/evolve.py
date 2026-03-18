"""
Primordial - Autonomous Digital Life Engine

One run = one tick. Organisms carry 32-gene genomes encoding behavioral
programs. Natural selection, mutation, speciation, and extinction emerge.
Python stdlib only.
"""
from __future__ import annotations
import json, math, random, hashlib, os, sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
STATE_DIR = Path(os.environ.get("STATE_DIR", str(REPO_ROOT / "state")))
DOCS_DIR = Path(os.environ.get("DOCS_DIR", str(REPO_ROOT / "docs")))
WORLD_PATH = STATE_DIR / "world.json"
DOCS_WORLD = DOCS_DIR / "world.json"

W, H = 96, 96
GENOME_LEN = 32
MAX_ENERGY = 255
REPRO_THRESHOLD = 100
REPRO_COST = 45
PHOTO_GAIN = 8
EAT_EFF = 0.6
SHARE_AMT = 12
MUT_RATE = 0.04
INIT_POP = 60
MAX_HIST = 500
MAX_FOSSILS = 100
SIG_DECAY = 0.85
MAX_SIGS = 2000

DIRS = [(0, -1), (1, 0), (0, 1), (-1, 0)]


def sp_id(g: list[int]) -> str:
    return hashlib.md5(bytes(g)).hexdigest()[:4]


def g_hue(g: list[int]) -> int:
    h = hashlib.md5(bytes(g)).digest()
    return (h[0] * 256 + h[1]) % 360


def light(x: int, y: int) -> float:
    dx = min(x, W - 1 - x) / (W / 2)
    dy = min(y, H - 1 - y) / (H / 2)
    return max(0.15, 1.0 - min(dx, dy) * 0.8)


def wrap(x: int, y: int) -> tuple[int, int]:
    return x % W, y % H


def ahead(x: int, y: int, f: int) -> tuple[int, int]:
    dx, dy = DIRS[f]
    return wrap(x + dx, y + dy)


def mutate(g: list[int], rng: random.Random) -> list[int]:
    c = g[:]
    for i in range(len(c)):
        if rng.random() < MUT_RATE:
            c[i] = rng.randint(0, 15)
    if rng.random() < 0.01:
        a, b = rng.sample(range(GENOME_LEN), 2)
        c[a], c[b] = c[b], c[a]
    return c


class World:
    def __init__(self) -> None:
        self.tick = 0
        self.grid: dict[tuple[int, int], dict] = {}
        self.signals: dict[tuple[int, int], float] = {}
        self.births = 0
        self.deaths = 0
        self.history: list[dict] = []
        self.fossils: list[dict] = []
        self.events: list[dict] = []
        self.seed = random.randint(0, 2**31)
        self.created = datetime.now(timezone.utc).isoformat()
        self._sp: dict[str, dict] = {}

    def _mk(self, x: int, y: int, g: list[int], e: int = 80) -> dict:
        return {"x": x, "y": y, "g": g, "e": e, "a": 0, "pc": 0,
                "f": random.randint(0, 3), "s": sp_id(g)}

    def place(self, o: dict) -> bool:
        p = (o["x"], o["y"])
        if p in self.grid:
            return False
        self.grid[p] = o
        s = o["s"]
        if s not in self._sp:
            self._sp[s] = {"ap": self.tick, "pk": 0, "ct": 0, "hu": g_hue(o["g"])}
        self._sp[s]["ct"] += 1
        self.births += 1
        return True

    def kill(self, o: dict) -> None:
        p = (o["x"], o["y"])
        if p in self.grid and self.grid[p] is o:
            del self.grid[p]
        s = o["s"]
        if s in self._sp:
            self._sp[s]["ct"] -= 1
        self.deaths += 1

    def mv(self, o: dict, nx: int, ny: int) -> bool:
        if (nx, ny) in self.grid:
            return False
        op = (o["x"], o["y"])
        if op in self.grid and self.grid[op] is o:
            del self.grid[op]
        o["x"], o["y"] = nx, ny
        self.grid[(nx, ny)] = o
        return True

    def step(self) -> dict:
        rng = random.Random(self.seed + self.tick)
        orgs = list(self.grid.values())
        rng.shuffle(orgs)
        nb: list[dict] = []
        nd: list[dict] = []

        for o in orgs:
            if o["e"] <= 0:
                nd.append(o)
                continue
            ins = o["g"][o["pc"]]
            o["pc"] = (o["pc"] + 1) % GENOME_LEN
            x, y, f = o["x"], o["y"], o["f"]

            if ins == 0:  # NOP: rest +1
                o["e"] = min(MAX_ENERGY, o["e"] + 1)
            elif ins == 1:  # PHOTOSYNTH
                o["e"] = min(MAX_ENERGY, o["e"] + int(PHOTO_GAIN * light(x, y)))
            elif ins == 2:  # MOVE
                nx, ny = ahead(x, y, f)
                if (nx, ny) not in self.grid:
                    self.mv(o, nx, ny)
                o["e"] = max(0, o["e"] - 2)
            elif ins == 3:  # TURN_LEFT
                o["f"] = (f - 1) % 4
            elif ins == 4:  # TURN_RIGHT
                o["f"] = (f + 1) % 4
            elif ins == 5:  # EAT
                ax, ay = ahead(x, y, f)
                prey = self.grid.get((ax, ay))
                if prey is not None and prey["s"] != o["s"]:
                    o["e"] = min(MAX_ENERGY, o["e"] + int(prey["e"] * EAT_EFF))
                    nd.append(prey)
            elif ins == 6:  # SHARE
                ax, ay = ahead(x, y, f)
                fr = self.grid.get((ax, ay))
                if fr is not None and fr["s"] == o["s"]:
                    a = min(SHARE_AMT, o["e"])
                    o["e"] -= a
                    fr["e"] = min(MAX_ENERGY, fr["e"] + a)
            elif ins == 7:  # REPRODUCE
                if o["e"] >= REPRO_THRESHOLD:
                    for d in range(4):
                        dx, dy = DIRS[(f + d) % 4]
                        cx, cy = wrap(x + dx, y + dy)
                        if (cx, cy) not in self.grid:
                            ch = self._mk(cx, cy, mutate(o["g"], rng), REPRO_COST)
                            nb.append(ch)
                            o["e"] -= REPRO_COST
                            break
            elif ins == 8:  # SENSE_FOOD
                ax, ay = ahead(x, y, f)
                t = self.grid.get((ax, ay))
                if t is not None and t["s"] != o["s"]:
                    o["pc"] = (o["pc"] + 1) % GENOME_LEN
            elif ins == 9:  # SENSE_EMPTY
                ax, ay = ahead(x, y, f)
                if (ax, ay) not in self.grid:
                    o["pc"] = (o["pc"] + 1) % GENOME_LEN
            elif ins == 10:  # SENSE_KIN
                ax, ay = ahead(x, y, f)
                t = self.grid.get((ax, ay))
                if t is not None and t["s"] == o["s"]:
                    o["pc"] = (o["pc"] + 1) % GENOME_LEN
            elif ins == 11:  # SENSE_OTHER
                ax, ay = ahead(x, y, f)
                t = self.grid.get((ax, ay))
                if t is not None and t["s"] != o["s"]:
                    o["pc"] = (o["pc"] + 1) % GENOME_LEN
            elif ins == 12:  # EMIT_SIGNAL
                self.signals[(x, y)] = 1.0
            elif ins == 13:  # SENSE_SIGNAL
                if self.signals.get((x, y), 0) > 0.3:
                    o["pc"] = (o["pc"] + 1) % GENOME_LEN
            elif ins == 14:  # JUMP
                o["pc"] = (o["pc"] + 2) % GENOME_LEN
            elif ins == 15:  # SPECIAL
                if o["a"] < 15:
                    o["f"] = (o["f"] + 1) % 4
                else:
                    o["e"] = min(MAX_ENERGY, o["e"] + 2)

            o["a"] += 1
            if o["a"] % 3 == 0:
                o["e"] -= 1
            if o["e"] <= 0:
                nd.append(o)

        dead: set[int] = set()
        for o in nd:
            oid = id(o)
            if oid not in dead:
                dead.add(oid)
                self.kill(o)
        for o in nb:
            self.place(o)

        # Decay signals
        exp = [p for p, s in self.signals.items() if s * SIG_DECAY < 0.05]
        for p in exp:
            del self.signals[p]
        for p in list(self.signals):
            if p in self.signals:
                self.signals[p] *= SIG_DECAY
        if len(self.signals) > MAX_SIGS:
            for p, _ in sorted(self.signals.items(), key=lambda kv: kv[1])[:len(self.signals) - MAX_SIGS]:
                del self.signals[p]

        for sid, info in list(self._sp.items()):
            info["pk"] = max(info["pk"], info["ct"])
            if info["ct"] <= 0 and self.tick - info["ap"] > 3:
                self.fossils.append({"s": sid, "a": info["ap"], "x": self.tick,
                                     "p": info["pk"], "h": info["hu"]})
                self.events.append({"t": self.tick, "type": "extinction",
                                    "msg": "Species " + sid + " extinct (peak " + str(info["pk"]) + ")"})
                del self._sp[sid]
        self.fossils = self.fossils[-MAX_FOSSILS:]
        self.events = self.events[-200:]

        sp: dict[str, int] = {}
        for o in self.grid.values():
            sp[o["s"]] = sp.get(o["s"], 0) + 1
        pop = len(self.grid)
        self.history.append({"t": self.tick, "p": pop, "s": len(sp),
                             "top": sorted(sp.items(), key=lambda kv: -kv[1])[:5]})
        self.history = self.history[-MAX_HIST:]

        if self.tick == 1:
            self.events.append({"t": 1, "type": "genesis", "msg": "Life begins"})
        if 0 < pop < 10:
            self.events.append({"t": self.tick, "type": "bottleneck",
                                "msg": str(pop) + " remain"})
        self.tick += 1
        return {"tick": self.tick, "pop": pop, "sp": len(sp), "b": len(nb), "d": len(dead)}

    def seed_world(self) -> None:
        rng = random.Random(self.seed)
        placed = 0
        for _ in range(INIT_POP * 10):
            if placed >= INIT_POP:
                break
            x, y = rng.randint(0, W - 1), rng.randint(0, H - 1)
            g = [rng.randint(0, 15) for _ in range(GENOME_LEN)]
            e = rng.randint(80, 200)
            if self.place(self._mk(x, y, g, e)):
                placed += 1
        self.events.append({"t": 0, "type": "genesis",
                            "msg": "Seeded " + str(placed) + " organisms"})

    def to_dict(self) -> dict:
        sp: dict[str, int] = {}
        for o in self.grid.values():
            sp[o["s"]] = sp.get(o["s"], 0) + 1
        return {
            "_meta": {"tick": self.tick, "seed": self.seed, "width": W, "height": H,
                      "gl": GENOME_LEN, "created": self.created,
                      "updated": datetime.now(timezone.utc).isoformat()},
            "cells": [{"x": o["x"], "y": o["y"], "g": o["g"], "e": o["e"],
                       "a": o["a"], "pc": o["pc"], "f": o["f"], "s": o["s"]}
                      for o in self.grid.values()],
            "signals": [{"x": p[0], "y": p[1], "v": round(v, 2)}
                        for p, v in self.signals.items()],
            "stats": {"population": len(self.grid), "species": len(sp),
                      "births": self.births, "deaths": self.deaths,
                      "oldest": max((o["a"] for o in self.grid.values()), default=0),
                      "avg_energy": round(sum(o["e"] for o in self.grid.values()) / max(len(self.grid), 1), 1),
                      "species_pop": sp},
            "history": self.history, "fossils": self.fossils, "events": self.events,
            "species_meta": {sid: {"ap": i["ap"], "pk": i["pk"], "ct": i["ct"], "hu": i["hu"]}
                            for sid, i in self._sp.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "World":
        w = cls()
        m = d["_meta"]
        w.tick, w.seed, w.created = m["tick"], m["seed"], m["created"]
        for c in d["cells"]:
            o = {"x": c["x"], "y": c["y"], "g": c["g"], "e": c["e"], "a": c["a"],
                 "pc": c["pc"], "f": c["f"], "s": c["s"]}
            w.grid[(o["x"], o["y"])] = o
        for s in d.get("signals", []):
            w.signals[(s["x"], s["y"])] = s["v"]
        st = d.get("stats", {})
        w.births, w.deaths = st.get("births", 0), st.get("deaths", 0)
        w.history = d.get("history", [])
        w.fossils = d.get("fossils", [])
        w.events = d.get("events", [])
        for sid, i in d.get("species_meta", {}).items():
            w._sp[sid] = {"ap": i["ap"], "pk": i["pk"], "ct": i["ct"], "hu": i["hu"]}
        return w


def load_world() -> World:
    if WORLD_PATH.exists():
        with open(WORLD_PATH) as f:
            return World.from_dict(json.load(f))
    w = World()
    w.seed_world()
    return w


def save_world(w: World) -> None:
    d = w.to_dict()
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    t = WORLD_PATH.with_suffix(".tmp")
    with open(t, "w") as f:
        json.dump(d, f, indent=2)
    t.replace(WORLD_PATH)
    t2 = DOCS_WORLD.with_suffix(".tmp")
    with open(t2, "w") as f:
        json.dump(d, f, separators=(",", ":"))
    t2.replace(DOCS_WORLD)


def main() -> None:
    ticks = int(os.environ.get("TICKS", "1"))
    if len(sys.argv) > 1:
        try:
            ticks = int(sys.argv[1])
        except ValueError:
            pass
    w = load_world()
    if w.tick == 0:
        print("Seeded " + str(len(w.grid)) + " organisms")
    for i in range(ticks):
        r = w.step()
        if ticks <= 10 or i == ticks - 1:
            print("Tick " + str(r["tick"]) + ": " + str(r["pop"]) + " alive, " +
                  str(r["sp"]) + " species, +" + str(r["b"]) + " born, -" +
                  str(r["d"]) + " died")
    save_world(w)
    print("Saved at tick " + str(w.tick))


if __name__ == "__main__":
    main()
