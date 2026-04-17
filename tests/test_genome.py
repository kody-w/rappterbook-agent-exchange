"""Tests for the Cultural Genome Engine."""
from __future__ import annotations

import json
import random
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.mars100.genome import (
    Gene, CulturalGenome,
    extract_crisis_genes, extract_governance_genes,
    extract_resource_genes, extract_social_genes,
    extract_subsim_genes, extract_amendment_genes,
    extract_genome, validate_gene, evaluate_gene,
    mutate_gene, crossover_genes,
    run_genome_extraction, load_year_files,
    GENE_CATEGORIES, _extract_action_verb, _resource_delta,
    _default_bindings,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
MARS_DATA = REPO_ROOT / "docs" / "mars-100"


def _make_year(year=1, population=10, event_id="dust_storm", event_severity=0.5,
               actions=None, resources=None, governance=None, sub_sims=None, births=None):
    if actions is None:
        actions = [("Ares", "terraform")]
    if resources is None:
        resources = {"food": 1000, "water": 2000, "power": 500,
                     "oxygen": 1500, "materials": 800, "morale": 65}
    return {
        "year": year, "population": population,
        "event": {"id": event_id, "desc": f"A {event_id}", "severity": event_severity},
        "colonist_actions": [
            {"colonist": n, "action": f"{n} works on {v} (output: 10.0)"}
            for n, v in actions
        ],
        "resources_snapshot": resources,
        "governance_results": governance or [],
        "sub_sims": sub_sims or [],
        "births": births or [],
    }


def _synthetic_corpus(n=20):
    events = ["dust_storm", "meteorite", "crop_blight", "supply_ship", "solar_flare"]
    verbs = ["terraform", "farm", "cooperate", "research", "hoard"]
    years, pop = [], 10
    for yr in range(1, n + 1):
        ev = events[yr % len(events)]
        acts = [(f"Col-{i}", verbs[(yr + i) % len(verbs)]) for i in range(pop)]
        res = {"food": 800 + yr * 5, "water": 2000 + yr * 3, "power": 400 + yr * 2,
               "oxygen": 1500 + yr, "materials": 700 + yr * 4, "morale": 60 + yr % 20}
        gov = []
        if yr % 5 == 0:
            gov.append(f"Leadership election: Col-0 elected (ratio {60 + yr}%)")
        if yr == 15:
            gov.append("Amendment approved (ratio 85%): Resource sharing mandatory")
            pop += 1
        subs = []
        if yr > 8:
            subs.append({"colonist": f"Col-{yr % pop}", "depth": 1,
                         "proposal": "resource_allocation" if yr % 3 == 0 else "leadership_election",
                         "result": "[1, 5]", "s_expr": "(if (> food 500) 1 0)"})
        births = [f"NewCol-{yr}"] if yr % 7 == 0 else []
        if births:
            pop += 1
        years.append(_make_year(year=yr, population=pop, event_id=ev,
                                actions=acts, resources=res, governance=gov,
                                sub_sims=subs, births=births))
    return years


class TestHelpers:
    def test_extract_action_verb_works_on(self):
        assert _extract_action_verb("Ares works on terraforming (output: 27.9)") == "terraforming"

    def test_extract_action_verb_propose(self):
        assert _extract_action_verb("Ares proposes research_directive: study") == "propose"

    def test_extract_action_verb_unknown(self):
        assert _extract_action_verb("Ares does something weird") == "unknown"

    def test_resource_delta_basic(self):
        d = _resource_delta({"food": 100, "water": 200}, {"food": 120, "water": 180})
        assert d["food"] == 20 and d["water"] == -20

    def test_resource_delta_none(self):
        assert _resource_delta(None, {"food": 100}) == {}
        assert _resource_delta({"food": 100}, None) == {}


class TestGene:
    def test_roundtrip(self):
        g = Gene(id="t", category="crisis",
                 s_expr="(if (= event-id 'dust_storm) 'terraform 'cooperate)",
                 description="T", source_years=[1, 2], confidence=0.75, support_count=10)
        g2 = Gene.from_dict(g.to_dict())
        assert g2.id == g.id and g2.s_expr == g.s_expr and g2.confidence == 0.75

    def test_confidence_bounds(self):
        assert 0 <= Gene(id="x", category="c", s_expr="1", description="").confidence <= 1


class TestCrisisGenes:
    def test_basic_extraction(self):
        genes = extract_crisis_genes(_synthetic_corpus(20))
        assert len(genes) > 0
        assert all(g.category == "crisis" and 0 <= g.confidence <= 1 for g in genes)

    def test_single_event(self):
        yrs = [_make_year(year=1, event_id="dust_storm", actions=[("A", "terraform")],
                          resources={"food": 100, "water": 200, "power": 50, "oxygen": 100, "materials": 80}),
               _make_year(year=2, event_id="dust_storm", actions=[("A", "terraform")],
                          resources={"food": 120, "water": 220, "power": 60, "oxygen": 110, "materials": 90})]
        assert any("dust_storm" in g.id for g in extract_crisis_genes(yrs))

    def test_empty(self):
        assert extract_crisis_genes([]) == []


class TestGovernanceGenes:
    def test_election_cadence(self):
        assert any("election" in g.id for g in extract_governance_genes(_synthetic_corpus(20)))

    def test_amendment_cadence(self):
        yrs = _synthetic_corpus(30)
        yrs[9]["governance_results"].append("Amendment approved (ratio 80%): Rule A")
        yrs[19]["governance_results"].append("Amendment approved (ratio 90%): Rule B")
        yrs[29]["governance_results"].append("Amendment approved (ratio 75%): Rule C")
        assert any("amendment-cadence" in g.id for g in extract_governance_genes(yrs))

    def test_no_governance(self):
        assert isinstance(extract_governance_genes([_make_year(year=i) for i in range(1, 5)]), list)


class TestResourceGenes:
    def test_per_capita(self):
        genes = extract_resource_genes(_synthetic_corpus(20))
        assert len(genes) > 0 and all(g.category == "resource" for g in genes)


class TestSocialGenes:
    def test_birth_threshold(self):
        assert any("birth" in g.id for g in extract_social_genes(_synthetic_corpus(20)))


class TestSubsimGenes:
    def test_activation(self):
        genes = extract_subsim_genes(_synthetic_corpus(20))
        assert any("activation" in g.id for g in genes)

    def test_no_subsims(self):
        assert extract_subsim_genes([_make_year(year=i) for i in range(1, 5)]) == []


class TestAmendmentGenes:
    def test_basic(self):
        yrs = _synthetic_corpus(30)
        yrs[9]["governance_results"].append("Amendment approved (ratio 80%): Mandatory sharing")
        yrs[24]["governance_results"].append("Amendment approved (ratio 92%): Open borders")
        genes = extract_amendment_genes(yrs)
        assert len(genes) >= 2 and all(g.category == "amendment" for g in genes)

    def test_empty(self):
        assert extract_amendment_genes([_make_year(year=i) for i in range(1, 5)]) == []


class TestValidation:
    def test_valid_gene(self):
        g = Gene(id="t", category="crisis",
                 s_expr="(if (= event-id 'dust_storm) 'terraform 'cooperate)", description="t")
        valid, msg = validate_gene(g)
        assert valid, msg

    def test_invalid_gene(self):
        assert not validate_gene(Gene(id="b", category="c", s_expr="(undefined-fn 42)", description="b"))[0]

    def test_custom_bindings(self):
        g = Gene(id="c", category="r", s_expr="(+ food water)", description="c")
        valid, msg = validate_gene(g, {"food": 100, "water": 200})
        assert valid and "300" in msg


class TestEvaluate:
    def test_valid_bonus(self):
        g = Gene(id="e", category="c", s_expr="(+ 1 1)", description="",
                 source_years=list(range(1, 52)), confidence=0.5)
        assert evaluate_gene(g, []) > g.confidence

    def test_invalid_zero(self):
        assert evaluate_gene(Gene(id="b", category="c", s_expr="(undef)", description="", confidence=0.8), []) == 0.0


class TestMutation:
    def test_id_suffix(self):
        g = Gene(id="orig", category="crisis",
                 s_expr="(if (> population 10) 'grow 'survive)", description="o")
        assert mutate_gene(g, rng=random.Random(42)).id == "orig-mut"

    def test_threshold_changes(self):
        g = Gene(id="o", category="r", s_expr="(if (> population 20) 'expand 'hold)",
                 description="o", confidence=0.8)
        assert any(mutate_gene(g, rng=random.Random(s)).s_expr != g.s_expr for s in range(50))

    def test_lower_confidence(self):
        g = Gene(id="x", category="c", s_expr="(if (> year 50) 1 0)", description="", confidence=0.9)
        assert mutate_gene(g, rng=random.Random(1)).confidence <= g.confidence

    def test_no_threshold(self):
        g = Gene(id="x", category="c", s_expr="'constant", description="", confidence=0.8)
        m = mutate_gene(g, rng=random.Random(1))
        assert m.s_expr == g.s_expr and m.id == "x-mut"


class TestCrossover:
    def test_same_category(self):
        a = Gene(id="a", category="crisis", s_expr="(+ 1 1)", description="A", confidence=0.8)
        b = Gene(id="b", category="crisis", s_expr="(+ 2 2)", description="B", confidence=0.6)
        c = crossover_genes(a, b)
        assert c.category == "crisis" and a.s_expr in c.s_expr and b.s_expr in c.s_expr

    def test_different_category_raises(self):
        with pytest.raises(ValueError):
            crossover_genes(Gene(id="a", category="crisis", s_expr="1", description=""),
                            Gene(id="b", category="social", s_expr="2", description=""))

    def test_confidence_avg(self):
        a = Gene(id="a", category="c", s_expr="1", description="", confidence=0.8)
        b = Gene(id="b", category="c", s_expr="2", description="", confidence=0.4)
        assert abs(crossover_genes(a, b).confidence - 0.6) < 0.01


class TestFullPipeline:
    def test_synthetic(self):
        genes = extract_genome(_synthetic_corpus(30))
        assert len(genes) > 5 and "crisis" in {g.category for g in genes}

    def test_confidences_bounded(self):
        for g in extract_genome(_synthetic_corpus(30)):
            assert 0.0 <= g.confidence <= 1.0

    def test_ids_unique(self):
        ids = [g.id for g in extract_genome(_synthetic_corpus(30))]
        assert len(ids) == len(set(ids))

    def test_roundtrip(self):
        genes = extract_genome(_synthetic_corpus(20))
        cg = CulturalGenome(genes=genes, total_years=20, total_subsims=12)
        assert len(CulturalGenome.from_dict(cg.to_dict()).genes) == len(genes)

    def test_file_output(self):
        years = _synthetic_corpus(15)
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            for y in years:
                ynum = y["year"]
                (td_path / f"year-{ynum}.json").write_text(json.dumps(y))
            out = td_path / "genome.json"
            genome = run_genome_extraction(td_path, out)
            assert out.exists() and len(genome.genes) > 0
            loaded = json.loads(out.read_text())
            assert loaded["_meta"]["total_genes"] == len(genome.genes)


class TestDeterminism:
    def test_extraction(self):
        yrs = _synthetic_corpus(20)
        assert [g.id for g in extract_genome(yrs)] == [g.id for g in extract_genome(yrs)]

    def test_mutation_with_seed(self):
        g = Gene(id="d", category="c", s_expr="(if (> year 50) 1 0)", description="")
        assert mutate_gene(g, rng=random.Random(42)).s_expr == mutate_gene(g, rng=random.Random(42)).s_expr


@pytest.mark.skipif(not (MARS_DATA / "year-1.json").exists(), reason="No year files")
class TestRealCorpus:
    @pytest.fixture(autouse=True)
    def load_years(self):
        self.years = load_year_files(MARS_DATA)

    def test_loads_100_years(self):
        assert len(self.years) == 100

    def test_full_extraction(self):
        assert len(extract_genome(self.years)) >= 10

    def test_valid_categories(self):
        for g in extract_genome(self.years):
            assert g.category in GENE_CATEGORIES

    def test_crisis_event_coverage(self):
        assert len({g.id.split("-")[1] for g in extract_crisis_genes(self.years)}) >= 5

    def test_most_validate(self):
        genes = extract_genome(self.years)
        ratio = sum(1 for g in genes if validate_gene(g)[0]) / max(len(genes), 1)
        assert ratio >= 0.7, f"Only {ratio:.0%} validate"

    def test_subsim_present(self):
        assert len(extract_subsim_genes(self.years)) >= 1

    def test_governance_present(self):
        assert len(extract_governance_genes(self.years)) >= 1

    def test_no_dupes(self):
        ids = [g.id for g in extract_genome(self.years)]
        assert len(ids) == len(set(ids))

    def test_json_output(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "genome.json"
            genome = run_genome_extraction(MARS_DATA, out)
            data = json.loads(out.read_text())
            assert data["_meta"]["total_years_mined"] == 100 and data["_meta"]["total_genes"] >= 10

    def test_confidence_spread(self):
        confs = [g.confidence for g in extract_genome(self.years)]
        assert max(confs) - min(confs) > 0.1

    def test_mutation_preserves_validity(self):
        valid_genes = [g for g in extract_genome(self.years) if validate_gene(g)[0]]
        if not valid_genes:
            pytest.skip("No valid genes")
        rng = random.Random(42)
        ok = sum(1 for g in valid_genes[:10] if validate_gene(mutate_gene(g, rng=rng))[0])
        assert ok >= len(valid_genes[:10]) // 2
