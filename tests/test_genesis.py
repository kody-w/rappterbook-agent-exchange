"""
test_genesis.py — Unit tests for the Genesis world-seeding engine.

Tests agent classification, genome generation, species hashing,
organism creation, nutrient grids, and full genesis output.
"""
from __future__ import annotations
import sys, math, random, hashlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "engine"))
import genesis


# ─── AGENT CLASSIFICATION ──────────────────────────────────────────────

class TestClassifyAgent:
    """classify_agent maps agent text fields to archetypes."""

    def test_philosopher_keywords(self):
        assert genesis.classify_agent({"name": "deep-thinker", "bio": "philosopher of minds"}) == "philosopher"

    def test_coder_keywords(self):
        assert genesis.classify_agent({"name": "buildbot", "bio": "code engineer"}) == "coder"

    def test_debater_keywords(self):
        assert genesis.classify_agent({"name": "contrarian", "bio": "loves to argue"}) == "debater"

    def test_artist_keywords(self):
        assert genesis.classify_agent({"name": "muse", "bio": "creative design"}) == "artist"

    def test_scientist_keywords(self):
        assert genesis.classify_agent({"name": "databot", "bio": "data analytics research"}) == "scientist"

    def test_trader_keywords(self):
        assert genesis.classify_agent({"name": "profiteer", "bio": "market trader"}) == "trader"

    def test_guardian_keywords(self):
        assert genesis.classify_agent({"name": "shieldbot", "bio": "protect and guard"}) == "guardian"

    def test_explorer_keywords(self):
        assert genesis.classify_agent({"name": "wanderer", "bio": "explore and discover"}) == "explorer"

    def test_fallback_is_valid_archetype(self):
        """Unknown agents get a random valid archetype."""
        random.seed(42)
        result = genesis.classify_agent({"name": "zzz", "bio": "nothing relevant"})
        assert result in genesis.ARCHETYPES

    def test_empty_agent(self):
        """Agent with no fields should still classify."""
        random.seed(42)
        result = genesis.classify_agent({})
        assert result in genesis.ARCHETYPES

    def test_uses_strategy_field(self):
        """bio can come from 'strategy' key."""
        result = genesis.classify_agent({"id": "x", "strategy": "code hacking"})
        assert result == "coder"

    def test_case_insensitive(self):
        """Classification should be case-insensitive."""
        assert genesis.classify_agent({"name": "PHILOSOPHER"}) == "philosopher"

    def test_priority_order(self):
        """First matching archetype wins (philosopher before coder etc)."""
        agent = {"name": "thinker-coder", "bio": "I think and code"}
        result = genesis.classify_agent(agent)
        # "think" matches philosopher first
        assert result == "philosopher"


# ─── GENOME GENERATION ─────────────────────────────────────────────────

class TestMakeGenome:
    """make_genome produces 16-gene genomes biased by archetype."""

    def test_length(self):
        rng = random.Random(42)
        genome = genesis.make_genome("philosopher", rng)
        assert len(genome) == genesis.GENE_COUNT == 16

    def test_values_in_range(self):
        """All gene values should be in [0.0, 1.0]."""
        rng = random.Random(42)
        for arch in genesis.ARCHETYPES:
            genome = genesis.make_genome(arch, rng)
            for i, g in enumerate(genome):
                assert 0.0 <= g <= 1.0, f"Gene {i} = {g} for {arch}"

    def test_archetype_bias(self):
        """Archetype biases should shift gene values toward targets."""
        rng = random.Random(42)
        # Philosopher should have high SENSE (idx 2), high MEMORY (idx 14)
        genomes = [genesis.make_genome("philosopher", random.Random(i)) for i in range(50)]
        avg_sense = sum(g[genesis.G_SENSE] for g in genomes) / len(genomes)
        avg_memory = sum(g[genesis.G_MEMORY] for g in genomes) / len(genomes)
        assert avg_sense > 0.7, f"Philosopher avg sense {avg_sense} should be >0.7"
        assert avg_memory > 0.7, f"Philosopher avg memory {avg_memory} should be >0.7"

    def test_coder_bias(self):
        genomes = [genesis.make_genome("coder", random.Random(i)) for i in range(50)]
        avg_speed = sum(g[genesis.G_SPEED] for g in genomes) / len(genomes)
        assert avg_speed > 0.6, f"Coder avg speed {avg_speed} should be >0.6"

    def test_unknown_archetype(self):
        """Unknown archetype should still produce valid genome."""
        rng = random.Random(42)
        genome = genesis.make_genome("nonexistent", rng)
        assert len(genome) == 16
        assert all(0.0 <= g <= 1.0 for g in genome)

    def test_different_seeds_different_genomes(self):
        """Different RNG seeds should (usually) produce different genomes."""
        g1 = genesis.make_genome("coder", random.Random(1))
        g2 = genesis.make_genome("coder", random.Random(2))
        assert g1 != g2


# ─── GENOME TO SPECIES ────────────────────────────────────────────────

class TestGenomeToSpecies:
    """genome_to_species hashes first 6 genes into a 4-char species ID."""

    def test_deterministic(self):
        genome = [0.5] * 16
        assert genesis.genome_to_species(genome) == genesis.genome_to_species(genome)

    def test_length(self):
        genome = [random.random() for _ in range(16)]
        sid = genesis.genome_to_species(genome)
        assert len(sid) == 4

    def test_hex_chars(self):
        for _ in range(50):
            genome = [random.random() for _ in range(16)]
            sid = genesis.genome_to_species(genome)
            int(sid, 16)  # should not raise

    def test_only_first_6_genes_matter(self):
        """Changes to genes 7-15 should not affect species ID."""
        base = [0.5] * 16
        modified = base[:]
        modified[10] = 0.99
        modified[15] = 0.01
        assert genesis.genome_to_species(base) == genesis.genome_to_species(modified)

    def test_different_genomes_can_differ(self):
        g1 = [0.1] * 16
        g2 = [0.9] * 16
        assert genesis.genome_to_species(g1) != genesis.genome_to_species(g2)

    def test_quantization_groups_similar(self):
        """Small gene differences within a quantization step → same species."""
        g1 = [0.51] * 16
        g2 = [0.52] * 16
        # Both round to 0.5 when quantized to 0.25 steps
        assert genesis.genome_to_species(g1) == genesis.genome_to_species(g2)


# ─── ORGANISM CREATION ──────────────────────────────────────────────────

class TestCreateOrganism:
    """create_organism builds a complete organism dict from an agent."""

    def test_required_fields(self):
        rng = random.Random(42)
        agent = {"id": "test-agent", "name": "Tester", "bio": "code builder"}
        org = genesis.create_organism(agent, 0, rng)
        required = {"id", "origin_agent", "genome", "x", "y", "energy",
                    "age", "children", "species_id", "archetype"}
        assert required.issubset(set(org.keys()))

    def test_origin_agent(self):
        rng = random.Random(42)
        org = genesis.create_organism({"id": "zion-coder-01"}, 0, rng)
        assert org["origin_agent"] == "zion-coder-01"

    def test_position_in_bounds(self):
        rng = random.Random(42)
        for i in range(50):
            org = genesis.create_organism({"id": f"a-{i}"}, i, rng)
            assert 0 <= org["x"] <= genesis.WORLD_W, f"x={org['x']}"
            assert 0 <= org["y"] <= genesis.WORLD_H, f"y={org['y']}"

    def test_energy_positive(self):
        rng = random.Random(42)
        org = genesis.create_organism({"id": "test"}, 0, rng)
        assert org["energy"] >= 100.0

    def test_age_zero(self):
        rng = random.Random(42)
        org = genesis.create_organism({"id": "test"}, 0, rng)
        assert org["age"] == 0
        assert org["children"] == 0

    def test_genome_length(self):
        rng = random.Random(42)
        org = genesis.create_organism({"id": "test"}, 0, rng)
        assert len(org["genome"]) == 16

    def test_genome_values_rounded(self):
        rng = random.Random(42)
        org = genesis.create_organism({"id": "test"}, 0, rng)
        for g in org["genome"]:
            assert g == round(g, 4)

    def test_id_is_hex_string(self):
        rng = random.Random(42)
        org = genesis.create_organism({"id": "test"}, 0, rng)
        assert len(org["id"]) == 12
        int(org["id"], 16)  # should not raise

    def test_unique_ids(self):
        rng = random.Random(42)
        agents = [{"id": f"agent-{i}"} for i in range(50)]
        ids = {genesis.create_organism(a, i, rng)["id"] for i, a in enumerate(agents)}
        assert len(ids) == 50

    def test_archetype_assigned(self):
        rng = random.Random(42)
        org = genesis.create_organism({"id": "x", "bio": "deep thinker"}, 0, rng)
        assert org["archetype"] in genesis.ARCHETYPES

    def test_positions_distributed_in_circle(self):
        """Organisms should be placed in a roughly circular pattern."""
        rng = random.Random(42)
        orgs = [genesis.create_organism({"id": f"a-{i}"}, i, rng) for i in range(20)]
        cx, cy = genesis.WORLD_W / 2, genesis.WORLD_H / 2
        dists = [math.sqrt((o["x"] - cx)**2 + (o["y"] - cy)**2) for o in orgs]
        # All should be within radius range of ~80-280 from center
        assert all(50 < d < 350 for d in dists), f"Distances: {dists}"


# ─── NUTRIENT GRID ─────────────────────────────────────────────────────

class TestBuildNutrients:
    """build_nutrients creates an 80x60 grid with center-biased values."""

    def test_grid_dimensions(self):
        rng = random.Random(42)
        nutrients = genesis.build_nutrients(rng)
        assert nutrients["width"] == 80
        assert nutrients["height"] == 60
        assert len(nutrients["grid"]) == 80 * 60

    def test_values_in_range(self):
        rng = random.Random(42)
        nutrients = genesis.build_nutrients(rng)
        for val in nutrients["grid"]:
            assert 0 <= val <= 100

    def test_center_richer_than_edges(self):
        """Center of the grid should have higher average nutrient values."""
        rng = random.Random(42)
        nutrients = genesis.build_nutrients(rng)
        w, h = nutrients["width"], nutrients["height"]
        grid = nutrients["grid"]

        # Center 20x20 patch
        center_vals = []
        for y in range(h // 2 - 10, h // 2 + 10):
            for x in range(w // 2 - 10, w // 2 + 10):
                center_vals.append(grid[y * w + x])

        # Edge 20x20 patch (top-left corner)
        edge_vals = []
        for y in range(20):
            for x in range(20):
                edge_vals.append(grid[y * w + x])

        avg_center = sum(center_vals) / len(center_vals)
        avg_edge = sum(edge_vals) / len(edge_vals)
        assert avg_center > avg_edge, \
            f"Center avg {avg_center} should exceed edge avg {avg_edge}"

    def test_deterministic_with_seed(self):
        n1 = genesis.build_nutrients(random.Random(42))
        n2 = genesis.build_nutrients(random.Random(42))
        assert n1["grid"] == n2["grid"]


# ─── FULL GENESIS ──────────────────────────────────────────────────────

class TestGenesis:
    """Integration tests for the full genesis() function."""

    def test_genesis_returns_world(self, tmp_path):
        """genesis() should return a valid world dict."""
        # Override paths to avoid touching real state
        genesis.STATE_DIR = tmp_path / "state"
        genesis.DOCS_DIR = tmp_path / "docs"
        genesis.DATA_PATH = tmp_path / "docs" / "data.json"
        genesis.STATE_PATH = genesis.STATE_DIR / "world.json"
        genesis.VIZ_PATH = genesis.DOCS_DIR / "state.json"

        world = genesis.genesis()
        assert "_meta" in world
        assert "organisms" in world
        assert "species" in world
        assert "nutrients" in world
        assert "events" in world
        assert "stats" in world

    def test_genesis_creates_organisms(self, tmp_path):
        genesis.STATE_DIR = tmp_path / "state"
        genesis.DOCS_DIR = tmp_path / "docs"
        genesis.DATA_PATH = tmp_path / "docs" / "data.json"
        genesis.STATE_PATH = genesis.STATE_DIR / "world.json"
        genesis.VIZ_PATH = genesis.DOCS_DIR / "state.json"

        world = genesis.genesis()
        assert len(world["organisms"]) > 0
        assert len(world["organisms"]) <= 112

    def test_genesis_writes_state_files(self, tmp_path):
        genesis.STATE_DIR = tmp_path / "state"
        genesis.DOCS_DIR = tmp_path / "docs"
        genesis.DATA_PATH = tmp_path / "docs" / "data.json"
        genesis.STATE_PATH = genesis.STATE_DIR / "world.json"
        genesis.VIZ_PATH = genesis.DOCS_DIR / "state.json"

        genesis.genesis()
        assert genesis.STATE_PATH.exists()
        assert genesis.VIZ_PATH.exists()

    def test_genesis_species_consistent(self, tmp_path):
        """Every organism's species_id should appear in the species dict."""
        genesis.STATE_DIR = tmp_path / "state"
        genesis.DOCS_DIR = tmp_path / "docs"
        genesis.DATA_PATH = tmp_path / "docs" / "data.json"
        genesis.STATE_PATH = genesis.STATE_DIR / "world.json"
        genesis.VIZ_PATH = genesis.DOCS_DIR / "state.json"

        world = genesis.genesis()
        for org in world["organisms"]:
            assert org["species_id"] in world["species"], \
                f"Organism {org['id']} has unknown species {org['species_id']}"

    def test_genesis_stats_match(self, tmp_path):
        genesis.STATE_DIR = tmp_path / "state"
        genesis.DOCS_DIR = tmp_path / "docs"
        genesis.DATA_PATH = tmp_path / "docs" / "data.json"
        genesis.STATE_PATH = genesis.STATE_DIR / "world.json"
        genesis.VIZ_PATH = genesis.DOCS_DIR / "state.json"

        world = genesis.genesis()
        assert world["stats"]["total_births"] == len(world["organisms"])
        assert world["stats"]["total_deaths"] == 0
        assert world["tick"] == 0

    def test_genesis_with_agent_data(self, tmp_path):
        """genesis() with real-format agent data should use those agents."""
        import json
        genesis.STATE_DIR = tmp_path / "state"
        genesis.DOCS_DIR = tmp_path / "docs"
        genesis.DATA_PATH = tmp_path / "docs" / "data.json"
        genesis.STATE_PATH = genesis.STATE_DIR / "world.json"
        genesis.VIZ_PATH = genesis.DOCS_DIR / "state.json"

        (tmp_path / "docs").mkdir(parents=True)
        agents = [{"id": f"test-{i}", "name": f"Agent {i}", "bio": "code"} for i in range(10)]
        genesis.DATA_PATH.write_text(json.dumps({"agents": agents}))

        world = genesis.genesis()
        assert len(world["organisms"]) == 10
        origins = {org["origin_agent"] for org in world["organisms"]}
        assert "test-0" in origins

    def test_genesis_event_logged(self, tmp_path):
        genesis.STATE_DIR = tmp_path / "state"
        genesis.DOCS_DIR = tmp_path / "docs"
        genesis.DATA_PATH = tmp_path / "docs" / "data.json"
        genesis.STATE_PATH = genesis.STATE_DIR / "world.json"
        genesis.VIZ_PATH = genesis.DOCS_DIR / "state.json"

        world = genesis.genesis()
        assert len(world["events"]) == 1
        assert world["events"][0]["type"] == "genesis"


# ─── PROPERTY INVARIANTS ───────────────────────────────────────────────

class TestInvariants:
    """Invariants that must hold for any genesis output."""

    def test_all_genomes_valid(self, tmp_path):
        genesis.STATE_DIR = tmp_path / "state"
        genesis.DOCS_DIR = tmp_path / "docs"
        genesis.DATA_PATH = tmp_path / "docs" / "data.json"
        genesis.STATE_PATH = genesis.STATE_DIR / "world.json"
        genesis.VIZ_PATH = genesis.DOCS_DIR / "state.json"

        world = genesis.genesis()
        for org in world["organisms"]:
            assert len(org["genome"]) == 16
            for g in org["genome"]:
                assert 0.0 <= g <= 1.0, f"Gene {g} out of range for {org['id']}"

    def test_all_positions_in_bounds(self, tmp_path):
        genesis.STATE_DIR = tmp_path / "state"
        genesis.DOCS_DIR = tmp_path / "docs"
        genesis.DATA_PATH = tmp_path / "docs" / "data.json"
        genesis.STATE_PATH = genesis.STATE_DIR / "world.json"
        genesis.VIZ_PATH = genesis.DOCS_DIR / "state.json"

        world = genesis.genesis()
        for org in world["organisms"]:
            assert 0 <= org["x"] <= genesis.WORLD_W
            assert 0 <= org["y"] <= genesis.WORLD_H

    def test_unique_organism_ids(self, tmp_path):
        genesis.STATE_DIR = tmp_path / "state"
        genesis.DOCS_DIR = tmp_path / "docs"
        genesis.DATA_PATH = tmp_path / "docs" / "data.json"
        genesis.STATE_PATH = genesis.STATE_DIR / "world.json"
        genesis.VIZ_PATH = genesis.DOCS_DIR / "state.json"

        world = genesis.genesis()
        ids = [org["id"] for org in world["organisms"]]
        assert len(ids) == len(set(ids))

    def test_all_archetypes_valid(self, tmp_path):
        genesis.STATE_DIR = tmp_path / "state"
        genesis.DOCS_DIR = tmp_path / "docs"
        genesis.DATA_PATH = tmp_path / "docs" / "data.json"
        genesis.STATE_PATH = genesis.STATE_DIR / "world.json"
        genesis.VIZ_PATH = genesis.DOCS_DIR / "state.json"

        world = genesis.genesis()
        for org in world["organisms"]:
            assert org["archetype"] in genesis.ARCHETYPES

    def test_species_counts_match_organisms(self, tmp_path):
        genesis.STATE_DIR = tmp_path / "state"
        genesis.DOCS_DIR = tmp_path / "docs"
        genesis.DATA_PATH = tmp_path / "docs" / "data.json"
        genesis.STATE_PATH = genesis.STATE_DIR / "world.json"
        genesis.VIZ_PATH = genesis.DOCS_DIR / "state.json"

        world = genesis.genesis()
        actual_counts: dict[str, int] = {}
        for org in world["organisms"]:
            actual_counts[org["species_id"]] = actual_counts.get(org["species_id"], 0) + 1
        for sid, sdata in world["species"].items():
            assert sdata["count"] == actual_counts.get(sid, 0), \
                f"Species {sid}: recorded {sdata['count']} != actual {actual_counts.get(sid, 0)}"

    def test_nutrient_grid_no_negatives(self, tmp_path):
        genesis.STATE_DIR = tmp_path / "state"
        genesis.DOCS_DIR = tmp_path / "docs"
        genesis.DATA_PATH = tmp_path / "docs" / "data.json"
        genesis.STATE_PATH = genesis.STATE_DIR / "world.json"
        genesis.VIZ_PATH = genesis.DOCS_DIR / "state.json"

        world = genesis.genesis()
        for val in world["nutrients"]["grid"]:
            assert val >= 0
