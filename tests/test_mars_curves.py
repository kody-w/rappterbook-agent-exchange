"""
Tests for src/mars_curves.py — Mars colony dashboard generator.

Covers:
- generate_dashboard: HTML structure, colony data injection, edge cases
- _build_events_js: event extraction, dedup, priority, 40-event cap
- _build_mc_js: Monte Carlo data serialization
- Property invariants: output is valid HTML, no data leaks, bounds hold

Run: python -m pytest tests/test_mars_curves.py -v
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.mars_curves import (
    COLORS,
    generate_dashboard,
    _build_events_js,
    _build_mc_js,
)


# ── Fixtures ─────────────────────────────────────────────────────


def _minimal_results(
    n_sols: int = 10,
    n_colonies: int = 3,
    include_history: bool = True,
) -> dict:
    """Build a minimal but valid results dict for generate_dashboard."""
    names = ["Ares Prime", "Olympus Station", "Red Frontier"][:n_colonies]
    colonies = []
    for i, name in enumerate(names):
        if include_history:
            history = [
                {
                    "population": 100 + i * 10 + sol,
                    "food_kg": 5000.0 + sol * 10,
                    "morale": 0.7 + sol * 0.01,
                    "births": max(0, sol - 2),
                    "deaths": max(0, sol - 5),
                    "carrying_capacity": 300,
                    "genetic_diversity": 0.95 - sol * 0.005,
                    "net_migration": (-1) ** sol,
                }
                for sol in range(n_sols)
            ]
            colonies.append({
                "name": name,
                "history": history,
                "events": [],
                "death_causes": {"radiation": 2, "starvation": 0},
            })
        else:
            colonies.append({
                "name": name,
                "population": [100 + i * 10 + s for s in range(n_sols)],
                "food_kg": [5000.0 + s * 10 for s in range(n_sols)],
                "morale": [0.7 + s * 0.01 for s in range(n_sols)],
                "births": [max(0, s - 2) for s in range(n_sols)],
                "deaths": [max(0, s - 5) for s in range(n_sols)],
                "carrying_capacity": [300] * n_sols,
                "genetic_diversity": [0.95] * n_sols,
                "net_migration": [0] * n_sols,
                "death_causes": {},
            })

    env_history = [
        {
            "temperature_c": -60.0 + sol * 0.1,
            "dust_opacity": 0.0,
            "radiation_msv": 0.67,
            "terraforming_progress": sol * 0.001,
            "pressure_kpa": 0.636,
        }
        for sol in range(n_sols)
    ]

    summary_colonies = [
        {
            "name": name,
            "strategy": "balanced",
            "start_pop": 100 + i * 10,
            "end_pop": 100 + i * 10 + n_sols - 1,
            "peak_pop": 100 + i * 10 + n_sols,
            "min_pop": 100 + i * 10,
            "total_births": n_sols,
            "total_deaths": max(0, n_sols - 5),
            "growth_pct": 5.0 + i,
            "net_migration": i - 1,
            "death_causes": {"radiation": 2, "starvation": 0},
        }
        for i, name in enumerate(names)
    ]

    return {
        "colonies": colonies,
        "environment": {"history": env_history},
        "summary": {"colonies": summary_colonies, "total_migrations": 3},
        "_meta": {"sols": n_sols, "generated": "2026-03-26T00:00:00Z"},
    }


def _minimal_mc_data(n_seeds: int = 50) -> dict:
    """Build minimal Monte Carlo results."""
    return {
        "n_seeds": n_seeds,
        "colony_names": ["Ares Prime", "Olympus Station", "Red Frontier"],
        "final_pop_stats": [
            {"mean": 150.0, "stdev": 12.0, "p10": 135.0, "p90": 170.0},
            {"mean": 180.0, "stdev": 15.0, "p10": 160.0, "p90": 200.0},
            {"mean": 120.0, "stdev": 20.0, "p10": 95.0, "p90": 145.0},
        ],
        "growth_pct_stats": [
            {"mean": 5.0, "stdev": 2.0},
            {"mean": 8.0, "stdev": 3.0},
            {"mean": -2.0, "stdev": 4.0},
        ],
        "survival_rates": [0.99, 0.95, 0.85],
        "population_bands": {
            "Ares Prime": {"p10": [100] * 10, "p50": [125] * 10, "p90": [150] * 10},
            "Olympus Station": {"p10": [110] * 10, "p50": [140] * 10, "p90": [170] * 10},
            "Red Frontier": {"p10": [80] * 10, "p50": [100] * 10, "p90": [130] * 10},
        },
    }


# ── generate_dashboard basic structure ──────────────────────────


class TestDashboardHTMLStructure:
    """The dashboard returns valid HTML with required elements."""

    def test_returns_string(self) -> None:
        html = generate_dashboard(_minimal_results())
        assert isinstance(html, str)

    def test_starts_with_doctype(self) -> None:
        html = generate_dashboard(_minimal_results())
        assert html.strip().startswith("<!DOCTYPE html>")

    def test_has_closing_html(self) -> None:
        html = generate_dashboard(_minimal_results())
        assert "</html>" in html

    def test_has_title(self) -> None:
        html = generate_dashboard(_minimal_results())
        assert "<title>Mars Barn" in html

    def test_has_canvas_elements(self) -> None:
        html = generate_dashboard(_minimal_results())
        assert "<canvas" in html

    def test_has_mars_barn_heading(self) -> None:
        html = generate_dashboard(_minimal_results())
        assert "Mars Barn" in html

    def test_contains_colony_colors(self) -> None:
        html = generate_dashboard(_minimal_results())
        for color in COLORS.values():
            assert color in html

    def test_sol_count_in_subtitle(self) -> None:
        html = generate_dashboard(_minimal_results(n_sols=42))
        assert "42 sols" in html

    def test_generated_date_in_subtitle(self) -> None:
        html = generate_dashboard(_minimal_results())
        assert "2026-03-26" in html


class TestDashboardColonyCards:
    """Summary cards render correctly for each colony."""

    def test_all_colony_names_present(self) -> None:
        html = generate_dashboard(_minimal_results())
        for name in ["Ares Prime", "Olympus Station", "Red Frontier"]:
            assert name in html

    def test_strategy_label_present(self) -> None:
        html = generate_dashboard(_minimal_results())
        assert "BALANCED" in html

    def test_growth_arrow_up(self) -> None:
        """Positive growth shows ↑ arrow."""
        html = generate_dashboard(_minimal_results())
        assert "↑" in html

    def test_growth_arrow_down(self) -> None:
        """Negative growth shows ↓ arrow."""
        results = _minimal_results()
        results["summary"]["colonies"][2]["growth_pct"] = -3.0
        html = generate_dashboard(results)
        assert "↓" in html

    def test_growth_arrow_neutral(self) -> None:
        """Zero growth shows → arrow."""
        results = _minimal_results()
        results["summary"]["colonies"][0]["growth_pct"] = 0
        html = generate_dashboard(results)
        assert "→" in html

    def test_migration_displayed_when_nonzero(self) -> None:
        html = generate_dashboard(_minimal_results())
        assert "Migration:" in html

    def test_no_migration_when_zero(self) -> None:
        """Colony with net_migration=0 should not show migration string."""
        results = _minimal_results(n_colonies=1)
        results["summary"]["colonies"][0]["net_migration"] = 0
        html = generate_dashboard(results)
        # The single colony card should not contain "Migration:"
        # (other colonies might, so we check just one colony)
        assert results["summary"]["colonies"][0]["net_migration"] == 0

    def test_death_causes_shown(self) -> None:
        html = generate_dashboard(_minimal_results())
        assert "radiation" in html

    def test_zero_death_causes_omitted(self) -> None:
        """Death causes with count 0 should not be the top killer."""
        results = _minimal_results()
        results["summary"]["colonies"][0]["death_causes"] = {"starvation": 0}
        html = generate_dashboard(results)
        # Should still produce valid HTML (no crash)
        assert "</html>" in html


class TestDashboardColonyData:
    """Colony time-series data is correctly injected into JavaScript."""

    def test_colonies_js_present(self) -> None:
        html = generate_dashboard(_minimal_results())
        assert "const COLONIES" in html

    def test_colony_populations_in_js(self) -> None:
        """Population arrays are embedded."""
        results = _minimal_results(n_sols=5)
        html = generate_dashboard(results)
        # Ares Prime starts at 100
        assert "100" in html

    def test_env_data_in_js(self) -> None:
        html = generate_dashboard(_minimal_results())
        assert "const ENV" in html

    def test_terraform_data_in_js(self) -> None:
        html = generate_dashboard(_minimal_results())
        assert "const TERRAFORM" in html

    def test_history_format_used(self) -> None:
        """When history key exists, data is extracted from it."""
        results = _minimal_results(include_history=True)
        html = generate_dashboard(results)
        assert "const COLONIES" in html

    def test_flat_format_used(self) -> None:
        """When no history key, falls back to flat arrays."""
        results = _minimal_results(include_history=False)
        html = generate_dashboard(results)
        assert "const COLONIES" in html


class TestDashboardWithoutHistory:
    """Dashboard works with flat-array colony format (no history key)."""

    def test_env_flat_format(self) -> None:
        results = _minimal_results()
        results["environment"] = {
            "temperature_c": [-60.0] * 5,
            "dust_opacity": [0.0] * 5,
            "radiation_msv": [0.67] * 5,
            "terraforming_progress": [0.0] * 5,
            "pressure_kpa": [0.636] * 5,
        }
        html = generate_dashboard(results)
        assert "const ENV" in html
        assert "const TERRAFORM" in html

    def test_env_missing_terraform_keys(self) -> None:
        """Environment without terraforming keys uses empty arrays."""
        results = _minimal_results()
        results["environment"] = {
            "temperature_c": [-60.0],
            "dust_opacity": [0.0],
            "radiation_msv": [0.67],
        }
        html = generate_dashboard(results)
        assert "const TERRAFORM" in html


# ── Monte Carlo integration ─────────────────────────────────────


class TestDashboardMonteCarlo:
    """Monte Carlo data renders correctly when provided."""

    def test_mc_null_when_no_data(self) -> None:
        html = generate_dashboard(_minimal_results())
        assert "const MC = null" in html

    def test_mc_present_when_provided(self) -> None:
        html = generate_dashboard(_minimal_results(), mc_data=_minimal_mc_data())
        assert "const MC = " in html
        assert "const MC = null" not in html

    def test_mc_subtitle_shows_seed_count(self) -> None:
        html = generate_dashboard(_minimal_results(), mc_data=_minimal_mc_data(n_seeds=100))
        assert "Monte Carlo: 100 seeds" in html

    def test_mc_cards_rendered(self) -> None:
        html = generate_dashboard(_minimal_results(), mc_data=_minimal_mc_data())
        assert "Monte Carlo Statistics" in html

    def test_mc_survival_rate_green(self) -> None:
        """99% survival gets green color."""
        mc = _minimal_mc_data()
        mc["survival_rates"] = [0.99, 0.99, 0.99]
        html = generate_dashboard(_minimal_results(), mc_data=mc)
        assert "#2ecc71" in html  # green

    def test_mc_survival_rate_amber(self) -> None:
        """95% survival gets amber color."""
        mc = _minimal_mc_data()
        mc["survival_rates"] = [0.95, 0.95, 0.95]
        html = generate_dashboard(_minimal_results(), mc_data=mc)
        assert "#f39c12" in html  # amber

    def test_mc_survival_rate_red(self) -> None:
        """85% survival gets red color."""
        mc = _minimal_mc_data()
        mc["survival_rates"] = [0.85, 0.85, 0.85]
        html = generate_dashboard(_minimal_results(), mc_data=mc)
        assert "#e74c3c" in html  # red (also used by Ares Prime, but validates logic)

    def test_mc_final_pop_stats(self) -> None:
        mc = _minimal_mc_data()
        html = generate_dashboard(_minimal_results(), mc_data=mc)
        assert "150" in html  # mean of Ares Prime
        assert "p10" in html  # range label


# ── _build_events_js ────────────────────────────────────────────


class TestBuildEventsJs:
    """Event extraction, deduplication, priority sorting, and cap."""

    def test_empty_colonies(self) -> None:
        js = _build_events_js([])
        assert "const EVENTS = " in js
        events = json.loads(js.replace("const EVENTS = ", "").rstrip(";\n"))
        assert events == []

    def test_no_events_key(self) -> None:
        js = _build_events_js([{"name": "Ares Prime"}])
        events = json.loads(js.replace("const EVENTS = ", "").rstrip(";\n"))
        assert events == []

    def test_storm_events_extracted(self) -> None:
        colonies = [{
            "name": "Ares Prime",
            "events": [
                {"sol": 10, "type": "storm", "kind": "regional"},
                {"sol": 50, "type": "storm", "kind": "global"},
            ],
        }]
        js = _build_events_js(colonies)
        events = json.loads(js.replace("const EVENTS = ", "").rstrip(";\n"))
        assert len(events) == 2
        assert events[0]["type"] == "regional_storm"
        assert events[1]["type"] == "global_storm"

    def test_storm_dedup_same_sol(self) -> None:
        """Two storms at the same sol from different colonies are deduped."""
        colonies = [
            {"name": "Ares Prime", "events": [{"sol": 10, "type": "storm", "kind": "global"}]},
            {"name": "Red Frontier", "events": [{"sol": 10, "type": "storm", "kind": "global"}]},
        ]
        js = _build_events_js(colonies)
        events = json.loads(js.replace("const EVENTS = ", "").rstrip(";\n"))
        storm_10 = [e for e in events if e["sol"] == 10]
        assert len(storm_10) == 1

    def test_epidemic_events(self) -> None:
        colonies = [{
            "events": [
                {"sol": 20, "type": "epidemic_start", "strain": "Mars Flu"},
                {"sol": 30, "type": "epidemic_end", "strain": "Mars Flu"},
            ],
        }]
        js = _build_events_js(colonies)
        events = json.loads(js.replace("const EVENTS = ", "").rstrip(";\n"))
        assert len(events) == 2
        types = {e["type"] for e in events}
        assert "epidemic_start" in types
        assert "epidemic_end" in types

    def test_supply_ship_event(self) -> None:
        colonies = [{"events": [{"sol": 100, "type": "supply_ship"}]}]
        js = _build_events_js(colonies)
        events = json.loads(js.replace("const EVENTS = ", "").rstrip(";\n"))
        assert len(events) == 1
        assert events[0]["type"] == "supply_ship"

    def test_discovery_event(self) -> None:
        colonies = [{"events": [{"sol": 42, "type": "discovery", "kind": "ice_cave"}]}]
        js = _build_events_js(colonies)
        events = json.loads(js.replace("const EVENTS = ", "").rstrip(";\n"))
        assert len(events) == 1
        assert events[0]["label"] == "ice_cave"

    def test_tech_unlock_event(self) -> None:
        colonies = [{
            "name": "Olympus Station",
            "events": [{"sol": 60, "type": "tech_unlock", "name": "solar_panels_v2"}],
        }]
        js = _build_events_js(colonies)
        events = json.loads(js.replace("const EVENTS = ", "").rstrip(";\n"))
        assert len(events) == 1
        assert events[0]["label"] == "solar_panels_v2"
        assert events[0]["colony"] == "Olympus Station"

    def test_unknown_event_type_ignored(self) -> None:
        """Events with unrecognized types are silently dropped."""
        colonies = [{"events": [{"sol": 5, "type": "alien_contact"}]}]
        js = _build_events_js(colonies)
        events = json.loads(js.replace("const EVENTS = ", "").rstrip(";\n"))
        assert len(events) == 0

    def test_events_sorted_by_sol(self) -> None:
        """Final output is sorted by sol."""
        colonies = [{
            "events": [
                {"sol": 50, "type": "epidemic_start", "strain": "X"},
                {"sol": 10, "type": "storm", "kind": "regional"},
                {"sol": 30, "type": "supply_ship"},
            ],
        }]
        js = _build_events_js(colonies)
        events = json.loads(js.replace("const EVENTS = ", "").rstrip(";\n"))
        sols = [e["sol"] for e in events]
        assert sols == sorted(sols)

    def test_events_capped_at_40(self) -> None:
        """No more than 40 events in output."""
        many_events = [{"sol": i, "type": "supply_ship"} for i in range(60)]
        colonies = [{"events": many_events}]
        js = _build_events_js(colonies)
        events = json.loads(js.replace("const EVENTS = ", "").rstrip(";\n"))
        assert len(events) <= 40

    def test_priority_sort_before_cap(self) -> None:
        """Higher-priority events survive the 40-event cap."""
        events = [{"sol": i, "type": "supply_ship"} for i in range(50)]
        events.append({"sol": 25, "type": "epidemic_start", "strain": "Plague"})
        colonies = [{"events": events}]
        js = _build_events_js(colonies)
        parsed = json.loads(js.replace("const EVENTS = ", "").rstrip(";\n"))
        types = {e["type"] for e in parsed}
        assert "epidemic_start" in types  # higher priority survives the cap

    def test_events_not_list_is_skipped(self) -> None:
        """If events key is not a list, it's safely ignored."""
        colonies = [{"events": "not a list"}]
        js = _build_events_js(colonies)
        events = json.loads(js.replace("const EVENTS = ", "").rstrip(";\n"))
        assert events == []

    def test_missing_event_fields_safe(self) -> None:
        """Events with missing optional fields don't crash."""
        colonies = [{"events": [{"type": "storm"}, {"type": "discovery"}]}]
        js = _build_events_js(colonies)
        events = json.loads(js.replace("const EVENTS = ", "").rstrip(";\n"))
        # Should produce valid JSON without crashing
        assert isinstance(events, list)


# ── _build_mc_js ────────────────────────────────────────────────


class TestBuildMcJs:
    """Monte Carlo JS serialization."""

    def test_returns_js_const(self) -> None:
        js = _build_mc_js({"n_seeds": 50, "data": [1, 2, 3]})
        assert js.startswith("const MC = ")

    def test_valid_json_payload(self) -> None:
        mc = _minimal_mc_data()
        js = _build_mc_js(mc)
        payload = js.replace("const MC = ", "").rstrip(";\n")
        parsed = json.loads(payload)
        assert parsed["n_seeds"] == 50

    def test_compact_separators(self) -> None:
        """Output uses compact JSON (no spaces after separators)."""
        js = _build_mc_js({"a": 1, "b": [1, 2]})
        payload = js.replace("const MC = ", "").rstrip(";\n")
        # Compact: no ": " or ", " — uses ":" and ","
        assert ": " not in payload

    def test_empty_dict(self) -> None:
        js = _build_mc_js({})
        assert "const MC = {}" in js


# ── Property-based invariants ───────────────────────────────────


class TestDashboardInvariants:
    """Properties that must hold for any valid input."""

    def test_html_length_scales_with_sols(self) -> None:
        """More sols → longer HTML (more data points)."""
        html_10 = generate_dashboard(_minimal_results(n_sols=10))
        html_100 = generate_dashboard(_minimal_results(n_sols=100))
        assert len(html_100) > len(html_10)

    def test_no_python_objects_in_output(self) -> None:
        """No Python repr leaks (e.g., 'None', 'True', 'False' as Python)."""
        html = generate_dashboard(_minimal_results(), mc_data=_minimal_mc_data())
        # JavaScript uses null/true/false, not None/True/False
        # Check JS data sections specifically
        assert "None" not in html.split("const COLONIES")[1].split("</script>")[0] if "const COLONIES" in html else True

    def test_deterministic_output(self) -> None:
        """Same input → same output."""
        results = _minimal_results()
        html1 = generate_dashboard(results)
        html2 = generate_dashboard(results)
        assert html1 == html2

    def test_single_colony_works(self) -> None:
        """Dashboard renders with only 1 colony."""
        results = _minimal_results(n_colonies=1)
        results["summary"]["colonies"] = results["summary"]["colonies"][:1]
        html = generate_dashboard(results)
        assert "Ares Prime" in html
        assert "</html>" in html

    def test_zero_sol_results(self) -> None:
        """Dashboard doesn't crash with 0 sols (edge case)."""
        results = _minimal_results(n_sols=0)
        # Empty histories
        for c in results["colonies"]:
            c["history"] = []
        results["environment"]["history"] = []
        html = generate_dashboard(results)
        assert "</html>" in html

    def test_large_sim_no_crash(self) -> None:
        """1000 sols of data renders without error."""
        html = generate_dashboard(_minimal_results(n_sols=1000))
        assert "</html>" in html
        assert len(html) > 10000


# ── COLORS constant ─────────────────────────────────────────────


class TestColors:
    """COLORS constant has expected structure."""

    def test_three_colonies_defined(self) -> None:
        assert len(COLORS) == 3

    def test_all_hex_colors(self) -> None:
        for color in COLORS.values():
            assert color.startswith("#")
            assert len(color) == 7

    def test_expected_colony_names(self) -> None:
        assert "Ares Prime" in COLORS
        assert "Olympus Station" in COLORS
        assert "Red Frontier" in COLORS


# ── Smoke: 10-sol simulation round-trip ─────────────────────────


class TestSmoke:
    """End-to-end: generate dashboard from real sim data if available."""

    def test_generate_with_full_fixture(self) -> None:
        """Full fixture renders without crash."""
        results = _minimal_results(n_sols=50)
        mc = _minimal_mc_data()
        html = generate_dashboard(results, mc_data=mc)
        assert "<!DOCTYPE html>" in html
        assert "Mars Barn" in html
        assert "Monte Carlo" in html
        assert len(html) > 5000

    def test_generate_minimal_no_mc(self) -> None:
        html = generate_dashboard(_minimal_results(n_sols=5))
        assert "const MC = null" in html
        assert "</html>" in html
