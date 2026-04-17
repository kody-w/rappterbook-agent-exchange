"""Tests for the Mars-100 economy engine."""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.mars100.economy import (
    Wallet, ColonyEconomy, EconomySnapshot,
    compute_income, process_theft, collect_taxes, process_inheritance,
    spend_treasury, compute_gini, tick_economy,
    ACTION_INCOME, TAX_RATES, THEFT_FRACTION, MAX_WEALTH,
    DEFAULT_STARTING_WEALTH, TREASURY_RESOURCE_CONVERSION,
)
from src.mars100.colonist import create_founding_ten, Colonist
from src.mars100.colony import Resources, SocialGraph


# ── Wallet tests ──

def test_wallet_defaults():
    w = Wallet()
    assert w.credits == DEFAULT_STARTING_WEALTH
    assert w.lifetime_income == 0.0
    assert w.lifetime_tax_paid == 0.0
    assert w.lifetime_stolen == 0.0
    assert w.trades_completed == 0


def test_wallet_clamp_lower():
    w = Wallet(credits=-5.0)
    w.clamp()
    assert w.credits == 0.0


def test_wallet_clamp_upper():
    w = Wallet(credits=10.0)
    w.clamp()
    assert w.credits == MAX_WEALTH


def test_wallet_roundtrip():
    w = Wallet(credits=1.5, lifetime_income=3.0, lifetime_tax_paid=0.5,
               lifetime_stolen=0.2, trades_completed=7)
    d = w.to_dict()
    w2 = Wallet.from_dict(d)
    assert abs(w2.credits - w.credits) < 1e-4
    assert w2.trades_completed == 7


# ── ColonyEconomy tests ──

def test_economy_ensure_wallet():
    econ = ColonyEconomy()
    w = econ.ensure_wallet("col-0")
    assert isinstance(w, Wallet)
    assert econ.wealth_of("col-0") == DEFAULT_STARTING_WEALTH


def test_economy_wealth_of_missing():
    econ = ColonyEconomy()
    assert econ.wealth_of("nonexistent") == DEFAULT_STARTING_WEALTH


def test_economy_roundtrip():
    econ = ColonyEconomy()
    econ.ensure_wallet("col-0").credits = 2.0
    econ.treasury = 1.5
    d = econ.to_dict()
    econ2 = ColonyEconomy.from_dict(d)
    assert abs(econ2.wealth_of("col-0") - 2.0) < 1e-4
    assert abs(econ2.treasury - 1.5) < 1e-4


# ── Income tests ──

def test_compute_income_positive():
    colonists = create_founding_ten(42)
    c = colonists[0]
    income = compute_income("farm", c)
    assert income > 0.0
    assert income < 0.2  # reasonable upper bound


def test_compute_income_rest_is_minimal():
    colonists = create_founding_ten(42)
    c = colonists[0]
    income_rest = compute_income("rest", c)
    income_farm = compute_income("farm", c)
    assert income_rest < income_farm


def test_compute_income_hoard_earns_zero_base():
    colonists = create_founding_ten(42)
    c = colonists[0]
    income = compute_income("hoard", c)
    assert income == 0.0  # base is 0, retention doesn't help


def test_compute_income_sabotage_earns_zero_base():
    colonists = create_founding_ten(42)
    c = colonists[0]
    income = compute_income("sabotage", c)
    assert income == 0.0


def test_compute_income_all_actions_bounded():
    colonists = create_founding_ten(42)
    for c in colonists:
        for action in ACTION_INCOME:
            income = compute_income(action, c)
            assert 0.0 <= income <= 1.0


# ── Theft tests ──

def test_process_theft_steals_from_victim():
    econ = ColonyEconomy()
    econ.ensure_wallet("thief").credits = 0.5
    econ.ensure_wallet("victim").credits = 2.0
    rng = random.Random(42)
    event = process_theft("thief", "victim", econ, rng)
    assert event is not None
    assert event["type"] == "theft"
    assert event["amount"] > 0.0
    assert econ.wealth_of("thief") > 0.5
    assert econ.wealth_of("victim") < 2.0


def test_process_theft_empty_victim():
    econ = ColonyEconomy()
    econ.ensure_wallet("thief").credits = 1.0
    econ.ensure_wallet("broke").credits = 0.0
    rng = random.Random(42)
    event = process_theft("thief", "broke", econ, rng)
    assert event is None


def test_process_theft_conservation():
    econ = ColonyEconomy()
    econ.ensure_wallet("thief").credits = 1.0
    econ.ensure_wallet("victim").credits = 2.0
    total_before = econ.wealth_of("thief") + econ.wealth_of("victim")
    rng = random.Random(42)
    process_theft("thief", "victim", econ, rng)
    total_after = econ.wealth_of("thief") + econ.wealth_of("victim")
    assert abs(total_before - total_after) < 1e-6  # wealth conserved


# ── Tax tests ──

def test_collect_taxes_anarchy_is_zero():
    econ = ColonyEconomy()
    econ.ensure_wallet("col-0").credits = 1.0
    total = collect_taxes(econ, "anarchy", ["col-0"])
    assert total == 0.0


def test_collect_taxes_council_rate():
    econ = ColonyEconomy()
    econ.ensure_wallet("col-0").credits = 1.0
    total = collect_taxes(econ, "council", ["col-0"])
    assert abs(total - 0.10) < 1e-6
    assert abs(econ.wealth_of("col-0") - 0.90) < 1e-4


def test_collect_taxes_goes_to_treasury():
    econ = ColonyEconomy()
    econ.ensure_wallet("col-0").credits = 2.0
    collect_taxes(econ, "dictator", ["col-0"])
    assert econ.treasury > 0.0


def test_collect_taxes_custom_rate():
    econ = ColonyEconomy()
    econ.custom_tax_rate = 0.50
    econ.ensure_wallet("col-0").credits = 1.0
    total = collect_taxes(econ, "anarchy", ["col-0"])
    assert abs(total - 0.50) < 1e-6


# ── Inheritance tests ──

def test_inheritance_to_children():
    econ = ColonyEconomy()
    econ.ensure_wallet("parent").credits = 3.0
    econ.ensure_wallet("child-a").credits = 0.0
    econ.ensure_wallet("child-b").credits = 0.0
    event = process_inheritance("parent", ["child-a", "child-b"], econ)
    assert event["type"] == "inheritance"
    assert event["amount"] == 3.0
    assert abs(econ.wealth_of("child-a") - 1.5) < 1e-4
    assert abs(econ.wealth_of("child-b") - 1.5) < 1e-4
    assert econ.wealth_of("parent") == 0.0


def test_inheritance_to_treasury():
    econ = ColonyEconomy()
    econ.ensure_wallet("loner").credits = 2.0
    event = process_inheritance("loner", [], econ)
    assert event["type"] == "inheritance_to_treasury"
    assert abs(econ.treasury - 2.0) < 1e-4


# ── Spend treasury tests ──

def test_spend_treasury_improves_resources():
    econ = ColonyEconomy()
    econ.treasury = 1.0
    res = Resources()
    res.food = 0.3
    before_food = res.food
    spent = spend_treasury(econ, res)
    assert spent > 0.0
    assert res.food > before_food


def test_spend_treasury_zero():
    econ = ColonyEconomy()
    econ.treasury = 0.0
    res = Resources()
    spent = spend_treasury(econ, res)
    assert spent == 0.0


def test_spend_treasury_resources_capped():
    econ = ColonyEconomy()
    econ.treasury = 100.0
    res = Resources()
    res.food = 0.99
    spend_treasury(econ, res)
    assert res.food <= 1.0


# ── Gini tests ──

def test_gini_perfect_equality():
    econ = ColonyEconomy()
    for i in range(5):
        econ.ensure_wallet(f"col-{i}").credits = 1.0
    gini = compute_gini(econ, [f"col-{i}" for i in range(5)])
    assert abs(gini) < 1e-6


def test_gini_maximal_inequality():
    econ = ColonyEconomy()
    econ.ensure_wallet("rich").credits = MAX_WEALTH
    for i in range(1, 10):
        econ.ensure_wallet(f"poor-{i}").credits = 0.0
    ids = ["rich"] + [f"poor-{i}" for i in range(1, 10)]
    gini = compute_gini(econ, ids)
    assert gini > 0.7  # should be near 1.0


def test_gini_range():
    econ = ColonyEconomy()
    rng = random.Random(42)
    for i in range(10):
        econ.ensure_wallet(f"col-{i}").credits = rng.uniform(0, MAX_WEALTH)
    gini = compute_gini(econ, [f"col-{i}" for i in range(10)])
    assert 0.0 <= gini <= 1.0


def test_gini_single_colonist():
    econ = ColonyEconomy()
    econ.ensure_wallet("solo").credits = 2.0
    gini = compute_gini(econ, ["solo"])
    assert gini == 0.0


def test_gini_no_wealth():
    econ = ColonyEconomy()
    for i in range(5):
        econ.ensure_wallet(f"col-{i}").credits = 0.0
    gini = compute_gini(econ, [f"col-{i}" for i in range(5)])
    assert gini == 0.0


# ── EconomySnapshot tests ──

def test_snapshot_roundtrip():
    snap = EconomySnapshot(
        year=1, gini=0.3, total_income=1.0, total_tax=0.2,
        total_theft=0.05, treasury=0.5,
        wealth_distribution={"col-0": 1.5, "col-1": 0.8},
        events=[{"type": "theft", "amount": 0.05}],
    )
    d = snap.to_dict()
    assert d["year"] == 1
    assert d["gini"] == 0.3
    assert "col-0" in d["wealth_distribution"]


# ── Integration: tick_economy ──

def test_tick_economy_basic():
    colonists = create_founding_ten(42)
    active_ids = [c.id for c in colonists]
    econ = ColonyEconomy()
    for cid in active_ids:
        econ.ensure_wallet(cid)

    social = SocialGraph()
    social.initialize(active_ids, random.Random(42))
    rng = random.Random(42)
    resources = Resources()

    actions = {c.id: "farm" for c in colonists}
    snap = tick_economy(
        economy=econ,
        actions=actions,
        colonists=colonists,
        active_ids=active_ids,
        gov_type="anarchy",
        year=1,
        social=social,
        rng=rng,
        deaths=[],
        lineage={},
        resources=resources,
    )
    assert snap.year == 1
    assert snap.total_income > 0.0
    assert snap.total_tax == 0.0  # anarchy = no tax
    assert snap.total_theft == 0.0  # no sabotage actions


def test_tick_economy_with_sabotage():
    colonists = create_founding_ten(42)
    active_ids = [c.id for c in colonists]
    econ = ColonyEconomy()
    for cid in active_ids:
        econ.ensure_wallet(cid)
        econ.wallets[cid].credits = 1.0  # everyone starts with 1.0

    social = SocialGraph()
    social.initialize(active_ids, random.Random(42))
    rng = random.Random(42)
    resources = Resources()

    actions = {c.id: "farm" for c in colonists}
    actions[colonists[0].id] = "sabotage"

    snap = tick_economy(
        economy=econ,
        actions=actions,
        colonists=colonists,
        active_ids=active_ids,
        gov_type="council",
        year=5,
        social=social,
        rng=rng,
        deaths=[],
        lineage={},
        resources=resources,
    )
    assert snap.total_theft > 0.0
    assert snap.total_tax > 0.0  # council has 10% rate


def test_tick_economy_with_death_inheritance():
    colonists = create_founding_ten(42)
    active_ids = [c.id for c in colonists]
    econ = ColonyEconomy()
    for cid in active_ids:
        econ.ensure_wallet(cid)

    dead_id = colonists[0].id
    econ.wallets[dead_id].credits = 3.0

    social = SocialGraph()
    social.initialize(active_ids, random.Random(42))
    rng = random.Random(42)
    resources = Resources()

    # dead colonist has a child
    child_id = colonists[1].id
    lineage = {dead_id: [child_id]}
    deaths = [{"id": dead_id, "name": colonists[0].name, "cause": "starvation", "year": 10}]
    remaining_active = [cid for cid in active_ids if cid != dead_id]

    snap = tick_economy(
        economy=econ,
        actions={cid: "farm" for cid in remaining_active},
        colonists=colonists,
        active_ids=remaining_active,
        gov_type="anarchy",
        year=10,
        social=social,
        rng=rng,
        deaths=deaths,
        lineage=lineage,
        resources=resources,
    )

    # Check inheritance event was logged
    inheritance_events = [e for e in snap.events if e["type"] == "inheritance"]
    assert len(inheritance_events) == 1
    assert inheritance_events[0]["heirs"] == [child_id]


def test_tick_economy_inequality_crisis():
    colonists = create_founding_ten(42)
    active_ids = [c.id for c in colonists]
    econ = ColonyEconomy()
    econ.ensure_wallet(active_ids[0]).credits = MAX_WEALTH
    for cid in active_ids[1:]:
        econ.ensure_wallet(cid).credits = 0.0

    social = SocialGraph()
    social.initialize(active_ids, random.Random(42))
    rng = random.Random(42)
    resources = Resources()

    snap = tick_economy(
        economy=econ,
        actions={cid: "rest" for cid in active_ids},
        colonists=colonists,
        active_ids=active_ids,
        gov_type="anarchy",
        year=50,
        social=social,
        rng=rng,
        deaths=[],
        lineage={},
        resources=resources,
    )
    crisis_events = [e for e in snap.events if e["type"] == "inequality_crisis"]
    assert len(crisis_events) == 1
    assert snap.gini > 0.6


# ── Full engine smoke test ──

def test_engine_10_years_with_economy():
    from src.mars100.engine import Mars100Engine
    engine = Mars100Engine(seed=42, total_years=10)
    result = engine.run()
    assert len(result.years) == 10
    # Economy snapshots exist for every year
    for yr in result.years:
        assert "gini" in yr.economy_snapshot
        assert "total_income" in yr.economy_snapshot
        assert "wealth_distribution" in yr.economy_snapshot
    # Final Gini is in the result
    assert 0.0 <= result.final_gini <= 1.0
    assert result.total_theft >= 0.0
    assert result.total_tax_collected >= 0.0


def test_engine_economy_deterministic():
    from src.mars100.engine import Mars100Engine
    r1 = Mars100Engine(seed=99, total_years=5).run()
    r2 = Mars100Engine(seed=99, total_years=5).run()
    for y1, y2 in zip(r1.years, r2.years):
        assert y1.economy_snapshot["gini"] == y2.economy_snapshot["gini"]
        assert y1.economy_snapshot["total_income"] == y2.economy_snapshot["total_income"]


def test_colonist_parent_ids():
    from src.mars100.colonist import Colonist, create_founding_ten
    founders = create_founding_ten(42)
    for f in founders:
        assert hasattr(f, "parent_ids")
        assert f.parent_ids == []
    d = founders[0].to_dict()
    assert "parent_ids" in d
    c2 = Colonist.from_dict(d)
    assert c2.parent_ids == []


def test_colonist_wealth_in_lispy_bindings():
    founders = create_founding_ten(42)
    bindings = founders[0].lispy_bindings()
    assert "wealth" in bindings
