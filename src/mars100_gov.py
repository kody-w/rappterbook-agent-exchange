"""mars100_gov.py -- Governance emergence tracker for Mars-100.

Tracks proposals, votes, power structures, and detects emergent
governance patterns. When a sub-simulation produces a meta-insight
strong enough, it can be promoted to a constitutional amendment.

Python stdlib only.
"""
from __future__ import annotations

import random
from collections import Counter


# Governance pattern archetypes
PATTERNS = {
    'anarchy':    'No clear governance — decisions made ad hoc by whoever acts first.',
    'democracy':  'Proposals voted on by all living colonists. Majority rules.',
    'council':    'A small elected body (3-5) makes decisions for the colony.',
    'oligarchy':  'Power concentrated in 2-3 colonists who control resources.',
    'theocracy':  'Faith-based governance — the prayerful lead.',
    'autocracy':  'One colonist holds unilateral power.',
    'technocracy':'Technical expertise determines authority.',
    'commune':    'Collective decision-making — consensus required.',
}


def create_proposal(
    year: int,
    proposer_id: str,
    title: str,
    description: str,
    model_code: str | None = None,
) -> dict:
    """Create a governance proposal."""
    return {
        'id': f"prop-y{year}-{proposer_id}",
        'year': year,
        'proposer': proposer_id,
        'title': title,
        'description': description,
        'model_code': model_code,
        'votes': {},
        'outcome': None,
        'sub_sim_result': None,
    }


def cast_vote(proposal: dict, voter_id: str, vote: str) -> None:
    """Cast a vote on a proposal. vote in ('yes', 'no', 'abstain')."""
    if vote not in ('yes', 'no', 'abstain'):
        raise ValueError(f"invalid vote: {vote}")
    proposal['votes'][voter_id] = vote


def tally_votes(proposal: dict, living_colonists: list[dict]) -> str:
    """Tally votes and determine outcome. Returns 'passed', 'failed', or 'split'."""
    living_ids = {c['id'] for c in living_colonists if c['alive']}
    yes_count = sum(1 for v_id, v in proposal['votes'].items()
                    if v == 'yes' and v_id in living_ids)
    no_count = sum(1 for v_id, v in proposal['votes'].items()
                   if v == 'no' and v_id in living_ids)
    total_eligible = len(living_ids)

    if total_eligible == 0:
        outcome = 'failed'
    elif yes_count > total_eligible / 2:
        outcome = 'passed'
    elif no_count > total_eligible / 2:
        outcome = 'failed'
    else:
        outcome = 'split'

    proposal['outcome'] = outcome
    return outcome


def detect_pattern(
    proposals: list[dict],
    colonists: list[dict],
    roles: dict[str, str | None],
) -> str:
    """Detect the emergent governance pattern from history.

    Args:
        proposals: All proposals so far
        colonists: All colonists (living and dead)
        roles: colonist_id -> governance_role mapping

    Returns: governance pattern name (key from PATTERNS)
    """
    if not proposals:
        return 'anarchy'

    living = [c for c in colonists if c['alive']]
    if len(living) <= 1:
        return 'autocracy' if living else 'anarchy'

    # Who proposes?
    proposer_counts = Counter(p['proposer'] for p in proposals)
    top_proposers = proposer_counts.most_common(3)

    # Concentration of proposal power
    total_proposals = len(proposals)
    top_proposer_share = top_proposers[0][1] / total_proposals if total_proposals else 0

    # Role analysis
    leaders = [cid for cid, role in roles.items() if role == 'leader']
    council_members = [cid for cid, role in roles.items()
                       if role in ('council_member', 'leader')]

    # Vote participation
    recent = proposals[-10:]
    avg_participation = 0.0
    if recent:
        total_living = len(living)
        if total_living > 0:
            participations = [len(p['votes']) / total_living for p in recent]
            avg_participation = sum(participations) / len(participations)

    # Detection logic
    if len(leaders) == 1 and top_proposer_share > 0.6:
        return 'autocracy'

    if len(council_members) >= 3 and top_proposer_share < 0.4:
        return 'council'

    # Check for faith-based governance
    faith_leaders = [c for c in living
                     if roles.get(c['id']) in ('leader', 'council_member')
                     and c['stats']['faith'] > 60]
    if len(faith_leaders) >= len(council_members) * 0.6 and len(council_members) >= 2:
        return 'theocracy'

    # Check for technocracy
    tech_leaders = [c for c in living
                    if roles.get(c['id']) in ('leader', 'council_member')
                    and c['skills']['coding'] > 50]
    if len(tech_leaders) >= len(council_members) * 0.6 and len(council_members) >= 2:
        return 'technocracy'

    if top_proposer_share > 0.5 and len(top_proposers) <= 2:
        return 'oligarchy'

    if avg_participation > 0.7:
        return 'democracy'

    if avg_participation > 0.5:
        return 'commune'

    return 'anarchy'


def assign_governance_roles(
    colonists: list[dict],
    proposals: list[dict],
    year: int,
    rng: random.Random,
) -> dict[str, str | None]:
    """Assign governance roles based on proposals, votes, and relationships.

    Returns: colonist_id -> role mapping
    """
    living = [c for c in colonists if c['alive']]
    roles: dict[str, str | None] = {c['id']: None for c in colonists}

    if not living or not proposals:
        return roles

    # Score each colonist by governance activity
    scores: dict[str, float] = {c['id']: 0.0 for c in living}
    for p in proposals:
        if p['proposer'] in scores:
            scores[p['proposer']] += 3.0
            if p['outcome'] == 'passed':
                scores[p['proposer']] += 5.0
        for voter_id, vote in p['votes'].items():
            if voter_id in scores:
                scores[voter_id] += 1.0

    # Factor in relationships (popular = more governance power)
    for col in living:
        friends = sum(1 for _, v in col['relationships'].items() if v > 30)
        scores[col['id']] += friends * 2.0
        # Resolve adds authority
        scores[col['id']] += col['stats']['resolve'] * 0.1

    # Assign roles based on scores
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    if len(ranked) >= 1 and ranked[0][1] > ranked[1][1] * 1.5 if len(ranked) > 1 else True:
        roles[ranked[0][0]] = 'leader'
    else:
        roles[ranked[0][0]] = 'council_member'

    for cid, score in ranked[1:min(4, len(ranked))]:
        if score > 0:
            roles[cid] = 'council_member'

    # Check for opposition (high paranoia + low governance score)
    for col in living:
        if roles[col['id']] is None and col['stats']['paranoia'] > 65:
            roles[col['id']] = 'opposition'

    return roles


def check_amendment_worthy(
    pattern_history: list[tuple[int, str]],
    sub_sim_insights: list[dict],
) -> str | None:
    """Check if governance patterns are strong enough to warrant a constitutional amendment.

    Returns: amendment text if worthy, None otherwise.
    """
    if len(pattern_history) < 20:
        return None

    # Look for stable patterns (same pattern for 10+ consecutive years)
    current_streak = 1
    longest_pattern = pattern_history[-1][1]
    longest_streak = 1

    for i in range(len(pattern_history) - 1, 0, -1):
        if pattern_history[i][1] == pattern_history[i - 1][1]:
            current_streak += 1
            if current_streak > longest_streak:
                longest_streak = current_streak
                longest_pattern = pattern_history[i][1]
        else:
            current_streak = 1

    # Look for depth-3 sub-sim insights
    deep_insights = [s for s in sub_sim_insights if s.get('depth', 0) >= 3]

    if longest_streak >= 10 and longest_pattern != 'anarchy':
        base = (
            f"Amendment proposal from Mars-100 simulation: "
            f"After {longest_streak} consecutive years of {longest_pattern} governance, "
            f"the colony found that {PATTERNS[longest_pattern].lower()} "
        )
        if deep_insights:
            insight = deep_insights[-1].get('insight', 'stability emerges from accepted impermanence')
            base += f'A depth-3 sub-simulation added: "{insight}"'
        else:
            base += "This pattern emerged organically without design — the strongest governance is the one nobody planned."
        return base

    if deep_insights:
        insight = deep_insights[-1]
        return (
            f"Amendment proposal from Mars-100 sub-simulation (depth {insight.get('depth', 3)}): "
            f'"{insight.get("insight", "The governance that works is the one that knows it is temporary.")}"'
        )

    return None
