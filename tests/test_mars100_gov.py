"""test_mars100_gov.py -- Tests for governance pattern detection.

Covers: proposals, voting, pattern detection, role assignment,
amendment detection.
"""
from __future__ import annotations

import pytest
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.mars100_gov import (
    create_proposal, cast_vote, tally_votes,
    detect_pattern, assign_governance_roles,
    check_amendment_worthy, PATTERNS,
)
from src.colonist import create_colonists


@pytest.fixture
def colonists():
    return create_colonists(seed=42)


class TestProposal:
    def test_create(self):
        p = create_proposal(10, 'ares', 'Build dome', 'Expand habitat')
        assert p['year'] == 10
        assert p['proposer'] == 'ares'
        assert p['outcome'] is None

    def test_id_format(self):
        p = create_proposal(10, 'ares', 'Build dome', 'Expand')
        assert p['id'] == 'prop-y10-ares'


class TestVoting:
    def test_cast_vote(self):
        p = create_proposal(10, 'ares', 'Test', '')
        cast_vote(p, 'marina', 'yes')
        assert p['votes']['marina'] == 'yes'

    def test_invalid_vote(self):
        p = create_proposal(10, 'ares', 'Test', '')
        with pytest.raises(ValueError):
            cast_vote(p, 'marina', 'maybe')

    def test_tally_passes(self, colonists):
        p = create_proposal(10, 'ares', 'Test', '')
        for c in colonists[:6]:
            cast_vote(p, c['id'], 'yes')
        for c in colonists[6:]:
            cast_vote(p, c['id'], 'no')
        result = tally_votes(p, colonists)
        assert result == 'passed'

    def test_tally_fails(self, colonists):
        p = create_proposal(10, 'ares', 'Test', '')
        for c in colonists[:3]:
            cast_vote(p, c['id'], 'yes')
        for c in colonists[3:]:
            cast_vote(p, c['id'], 'no')
        result = tally_votes(p, colonists)
        assert result == 'failed'

    def test_tally_split(self, colonists):
        p = create_proposal(10, 'ares', 'Test', '')
        for c in colonists[:5]:
            cast_vote(p, c['id'], 'yes')
        for c in colonists[5:]:
            cast_vote(p, c['id'], 'no')
        result = tally_votes(p, colonists)
        assert result == 'split'

    def test_dead_dont_count(self, colonists):
        p = create_proposal(10, 'ares', 'Test', '')
        colonists[0]['alive'] = False
        colonists[1]['alive'] = False
        for c in colonists[2:7]:
            cast_vote(p, c['id'], 'yes')
        for c in colonists[7:]:
            cast_vote(p, c['id'], 'no')
        # Dead votes from colonists[0:2] should be ignored
        cast_vote(p, colonists[0]['id'], 'no')
        result = tally_votes(p, colonists)
        assert result == 'passed'


class TestPatternDetection:
    def test_anarchy_with_no_proposals(self, colonists):
        roles = {c['id']: None for c in colonists}
        pattern = detect_pattern([], colonists, roles)
        assert pattern == 'anarchy'

    def test_autocracy(self, colonists):
        """One proposer dominates."""
        proposals = [create_proposal(y, 'ignis', f'Rule {y}', '') for y in range(20)]
        roles = {c['id']: None for c in colonists}
        roles['ignis'] = 'leader'
        pattern = detect_pattern(proposals, colonists, roles)
        assert pattern == 'autocracy'

    def test_democracy_high_participation(self, colonists):
        proposals = []
        for y in range(20):
            p = create_proposal(y, colonists[y % 10]['id'], f'Prop {y}', '')
            for c in colonists:
                cast_vote(p, c['id'], random.choice(['yes', 'no']))
            proposals.append(p)
        roles = {c['id']: 'council_member' for c in colonists[:5]}
        for c in colonists[5:]:
            roles[c['id']] = None
        pattern = detect_pattern(proposals, colonists, roles)
        assert pattern in ('democracy', 'council', 'commune')

    def test_single_colonist_is_autocracy(self):
        colonists = create_colonists()
        for c in colonists[1:]:
            c['alive'] = False
        roles = {c['id']: None for c in colonists}
        proposals = [create_proposal(1, colonists[0]['id'], 'Survive', '')]
        pattern = detect_pattern(proposals, colonists, roles)
        assert pattern == 'autocracy'


class TestRoleAssignment:
    def test_returns_all_colonists(self, colonists):
        roles = assign_governance_roles(colonists, [], 10, random.Random(42))
        for c in colonists:
            assert c['id'] in roles

    def test_leader_exists_with_proposals(self, colonists):
        proposals = [create_proposal(y, 'ares', f'Prop {y}', '') for y in range(5)]
        for p in proposals:
            tally_votes(p, colonists)
        roles = assign_governance_roles(colonists, proposals, 10, random.Random(42))
        leaders = [cid for cid, role in roles.items() if role == 'leader']
        assert len(leaders) >= 1

    def test_no_roles_without_proposals(self, colonists):
        roles = assign_governance_roles(colonists, [], 10, random.Random(42))
        assert all(role is None for role in roles.values())


class TestAmendment:
    def test_none_with_short_history(self):
        result = check_amendment_worthy([(i, 'anarchy') for i in range(5)], [])
        assert result is None

    def test_amendment_from_stable_pattern(self):
        history = [(i, 'democracy') for i in range(30)]
        result = check_amendment_worthy(history, [])
        assert result is not None
        assert 'democracy' in result

    def test_amendment_from_depth3_insight(self):
        history = [(i, 'council') for i in range(25)]
        insights = [{'depth': 3, 'insight': 'Impermanence is the only constant.'}]
        result = check_amendment_worthy(history, insights)
        assert result is not None
        assert 'Impermanence' in result or 'council' in result.lower()

    def test_no_amendment_from_anarchy(self):
        """Anarchy streaks don't generate amendments."""
        history = [(i, 'anarchy') for i in range(30)]
        result = check_amendment_worthy(history, [])
        assert result is None

    def test_amendment_includes_pattern_description(self):
        history = [(i, 'council') for i in range(25)]
        result = check_amendment_worthy(history, [])
        assert result is not None
        assert 'council' in result.lower()
