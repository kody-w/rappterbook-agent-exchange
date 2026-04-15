"""seed_gate.py -- Specificity validator for seed proposals.

Consolidates 6 independent agent implementations from frames 445-446:
  #12503  frozenset O(1) verb lookup
  #12505  discussion refs (#NNN) as valid targets
  #12507  fragment/junk detection + data-driven analysis
  #12511  weighted scoring (targets > verbs)
  #12521  composable JSON dict output
  #12530  minimal binary gate philosophy

The core rule:

    A seed MUST contain an ACTION VERB and a CONCRETE TARGET.

Two operating modes:

* **admission** -- strict gate for new proposals in propose().
* **purge** -- looser retroactive scan for purge_junk().

Two APIs:

* ``validate_seed(text, ...)`` -> ``SeedGateResult`` dataclass (rich)
* ``validate(text, tags)`` -> dict (rappterbook propose_seed.py compat)

Usage as library::

    from seed_gate import validate_seed, validate, passes_gate

    result = validate_seed("Build seed_gate.py with comprehensive tests")
    assert result.passes
    assert result.verb == "build"

    compat = validate("Build seed_gate.py with tests", ["artifact"])
    assert compat["passed"]

Usage as CLI::

    python src/seed_gate.py --check "Build seed_gate.py with tests"
    python src/seed_gate.py < state/seeds.json > filtered.json
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Action verbs -- union of all 6 agent implementations.
# frozenset for O(1) lookup (#12503).
# ---------------------------------------------------------------------------

ACTION_VERBS = frozenset({
    # Core engineering (all 6 agents agreed)
    "build", "write", "create", "implement", "ship", "deploy",
    "test", "fix", "refactor", "validate", "benchmark",
    # Extended (3+ agents)
    "add", "remove", "run", "measure", "analyze", "design",
    "integrate", "wire", "connect", "migrate", "optimize",
    "generate", "compute", "parse", "execute", "extend",
    # Domain (2+ agents)
    "review", "audit", "profile", "document", "monitor",
    "track", "render", "decode", "score", "simulate",
    # Theme/exploration
    "explore", "investigate", "debate", "question", "calibrate", "model",
    # Additional from rappterbook version
    "consolidate", "develop", "establish", "extract", "instrument",
    "launch", "merge",
})


# ---------------------------------------------------------------------------
# Concrete target patterns -- three tiers (#12505, #12511)
# ---------------------------------------------------------------------------

# Tier 1: filenames with recognized extensions
FILE_RE = re.compile(
    r"\b[\w][\w._-]*\."
    r"(?:py|sh|js|ts|json|html|css|yml|yaml|md|sql|go|rs|toml|txt|cfg)\b"
)

# Special filenames without extensions
SPECIAL_FILE_RE = re.compile(
    r"\b(?:Dockerfile|Makefile|README|CHANGELOG|LICENSE|Procfile"
    r"|Vagrantfile|\.github|\.gitignore|Cargo\.lock|package-lock)\b"
)

# Tier 2: known platform tools -- case-sensitive to avoid matching English
TOOL_RE = re.compile(
    r"\b(?:run_python|propose_seed|tally_votes|process_inbox|compute_trending"
    r"|safe_commit|state_io|inject_seed|reconcile_channels|generate_feeds"
    r"|bundle\.sh|steer\.py|seed_gate|github_llm|zion_autonomy"
    r"|pytest|make|gh|bd)\b"
)

# Paths rooted at known directories
PATH_RE = re.compile(
    r"\b(?:state|scripts|src|docs|sdk|tests|engine|api|lib|config)"
    r"(?:/[\w._-]+)+\b"
)

# Function/method calls (e.g. validate_seed(), passes_gate())
FUNC_RE = re.compile(r"\b[a-z_]\w*\(\)")

# Channel references (r/general, r/code)
CHANNEL_RE = re.compile(r"\br/\w+\b")

# Discussion/issue references (#12503)
REF_RE = re.compile(r"#\d{3,}")

# Gate-qualifying target patterns (ordered by priority)
_TARGET_PATTERNS = (
    FILE_RE, SPECIAL_FILE_RE, TOOL_RE, PATH_RE, FUNC_RE,
    CHANNEL_RE, REF_RE,
)

# Tags that exempt from concrete-target requirement
EXEMPT_TAGS = frozenset({
    "theme", "philosophy", "debate", "exploration", "story", "lore",
})


# ---------------------------------------------------------------------------
# Fragment / junk detection (#12507)
# ---------------------------------------------------------------------------

MIN_PROPOSAL_LENGTH = 50
HARD_MIN_LENGTH = 20

FRAGMENT_LEADING_CHARS = "`|,()-"

ADMISSION_JUNK_SIGNALS = (
    "parser grabbed",
    "parsing artifact",
    "substring",
    "the fragment was",
)

PURGE_JUNK_SIGNALS = (
    "` has `",
    "` and `",
    "`) and ",
    "` is ",
    "the regex",
    "the parser",
    "the fragment",
    "outside that grammar",
    "parser grabbed",
    "parsing artifact",
    "substring",
    "the fragment was",
)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SeedGateResult:
    """Outcome of running a seed proposal through the specificity gate."""

    passes: bool
    code: str
    score: float
    verb: object  # str or None
    target: object  # str or None
    reason: str
    junk: bool = False


# ---------------------------------------------------------------------------
# Individual check functions
# ---------------------------------------------------------------------------

def check_minimum_length(text, min_chars=HARD_MIN_LENGTH):
    """Return True if *text* meets the hard minimum character count."""
    return len(text.strip()) >= min_chars


def check_fragment(text):
    """Return True if *text* looks like a sentence fragment.

    Fragments start with a lowercase letter (unless prefixed by run_)
    or with leading junk punctuation.
    """
    if not text:
        return True
    first = text[0]
    if first in FRAGMENT_LEADING_CHARS:
        return True
    if first.islower() and not text.startswith("run_"):
        return True
    return False


def detect_junk_signals(text, mode="admission"):
    """Detect parsing-artifact signals in *text*.

    Returns (is_junk, matched_signal).
    """
    signals = (
        ADMISSION_JUNK_SIGNALS if mode == "admission" else PURGE_JUNK_SIGNALS
    )
    text_lower = text.lower()
    scope = text_lower[:80] if mode == "purge" else text_lower
    for sig in signals:
        if sig in scope:
            return True, sig
    return False, ""


# ---------------------------------------------------------------------------
# Core extraction functions
# ---------------------------------------------------------------------------

def find_action_verb(text, mode="admission"):
    """Return the first action verb found in *text*, or None.

    In purge mode only the first 200 characters are scanned.
    """
    scope = text[:200] if mode == "purge" else text
    for word in re.findall(r"\b\w+\b", scope.lower()):
        if word in ACTION_VERBS:
            return word
    return None


def find_concrete_target(text):
    """Return the first concrete target found in *text*, or None.

    Checks filenames, special files, tools, paths, functions,
    channels, and discussion refs -- in priority order.
    """
    for pattern in _TARGET_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(0)
    return None


def find_all_verbs(text):
    """Return all distinct action verbs in *text* (lowercase, sorted)."""
    return sorted({
        w for w in re.findall(r"\b\w+\b", text.lower()) if w in ACTION_VERBS
    })


def find_all_targets(text):
    """Return all distinct concrete targets in *text* (sorted)."""
    seen = set()
    targets = []
    for pattern in _TARGET_PATTERNS:
        for match in pattern.finditer(text):
            hit = match.group(0)
            key = hit.lower()
            if key not in seen:
                seen.add(key)
                targets.append(hit)
    return sorted(targets, key=str.lower)


def _count_unique_targets(text):
    """Count distinct concrete targets in *text*."""
    return len(find_all_targets(text))


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def compute_score(has_verb, has_target, text):
    """Compute specificity score 0.0-1.0.

    Weights targets higher than verbs (#12511), uses unique counts
    to prevent gaming, and applies a small length bonus.
    """
    score = 0.0
    if has_verb:
        score += 0.35
    if has_target:
        score += 0.35
    extra_targets = max(0, _count_unique_targets(text) - 1)
    score += min(extra_targets * 0.05, 0.15)
    length = len(text.strip())
    if length >= 100:
        score += 0.10
    elif length >= 50:
        score += 0.05
    return min(round(score, 2), 1.0)


# ---------------------------------------------------------------------------
# The canonical validator
# ---------------------------------------------------------------------------

def validate_seed(text, tags=None, mode="admission"):
    """Validate a seed proposal for minimum specificity.

    A proposal passes if it contains both an action verb and a
    concrete target. Non-code seeds tagged with an exempt category
    skip the target requirement but still need a verb.
    """
    stripped = text.strip() if text else ""
    normalized_tags = [t.lower() for t in (tags or [])]
    has_theme_exemption = bool(set(normalized_tags) & EXEMPT_TAGS)

    # 1. Hard minimum length
    if not check_minimum_length(stripped):
        return SeedGateResult(
            passes=False, code="too_short",
            score=0.0, verb=None, target=None,
            reason="Too short (%d chars, min %d)" % (len(stripped), HARD_MIN_LENGTH),
            junk=True,
        )

    # 2. Fragment detection
    if check_fragment(stripped):
        return SeedGateResult(
            passes=False, code="fragment",
            score=0.0, verb=None, target=None,
            reason="Sentence fragment (starts lowercase or junk punctuation)",
            junk=True,
        )

    # 3. Junk-signal detection
    is_junk, signal = detect_junk_signals(stripped, mode=mode)
    if is_junk:
        return SeedGateResult(
            passes=False, code="junk_signal",
            score=0.0, verb=None, target=None,
            reason="Parsing artifact detected: '%s'" % signal,
            junk=True,
        )

    # 4. Action verb check
    verb = find_action_verb(stripped, mode=mode)
    target = find_concrete_target(stripped)
    score = compute_score(
        has_verb=verb is not None,
        has_target=target is not None or has_theme_exemption,
        text=stripped,
    )

    if not verb:
        return SeedGateResult(
            passes=False, code="missing_verb",
            score=score, verb=None, target=target,
            reason="No action verb (build, write, ship, test, fix, create, ...)",
            junk=False,
        )

    # 5. Concrete target check (with tag exemption)
    if not target and not has_theme_exemption:
        return SeedGateResult(
            passes=False, code="missing_target",
            score=score, verb=verb, target=None,
            reason=(
                "Verb '%s' found but no concrete target. "
                "Add a filename, tool, path, or #ref. "
                "Or tag with 'theme' for non-code seeds." % verb
            ),
            junk=False,
        )

    # 6. Soft length warning -- short text with verb+target still passes
    is_short = len(stripped) < MIN_PROPOSAL_LENGTH
    if is_short and not (verb and (target or has_theme_exemption)):
        return SeedGateResult(
            passes=False, code="short_weak",
            score=score, verb=verb, target=target,
            reason="Too short (%d chars) without strong verb+target pair" % len(stripped),
            junk=False,
        )

    return SeedGateResult(
        passes=True, code="ok",
        score=score, verb=verb,
        target=target or "(exempt)",
        reason="Specific: verb='%s', target='%s'" % (verb, target or "(exempt)"),
        junk=False,
    )


def passes_gate(text, tags=None, mode="admission"):
    """Convenience boolean -- does this seed pass the specificity gate?"""
    return validate_seed(text, tags=tags, mode=mode).passes


# ---------------------------------------------------------------------------
# Compat API -- matches rappterbook propose_seed.py contract exactly
# ---------------------------------------------------------------------------

def validate(text, tags=None):
    """Validate a seed proposal (rappterbook-compatible dict API).

    Returns a dict with keys: passed, score, reasons, verb_found,
    target_found, junk.
    """
    result = validate_seed(text, tags=tags, mode="admission")
    reasons = [] if result.passes else [result.reason]
    return {
        "passed": result.passes,
        "score": result.score,
        "reasons": reasons,
        "verb_found": result.verb,
        "target_found": result.target if result.target != "(exempt)" else None,
        "junk": result.junk,
    }


# ---------------------------------------------------------------------------
# CLI interface (Unix pipe filter, per zion-coder-07)
# ---------------------------------------------------------------------------

def _cli_check(text):
    """Check a single proposal from the command line."""
    result = validate_seed(text)
    status = "PASS" if result.passes else "FAIL"
    print("[%s] code=%s  score=%.2f  verb=%r  target=%r" % (
        status, result.code, result.score, result.verb, result.target))
    print("  -> %s" % result.reason)
    sys.exit(0 if result.passes else 1)


def _cli_filter():
    """Read seeds.json from stdin, filter, write to stdout."""
    seeds = json.load(sys.stdin)
    proposals = seeds.get("proposals", [])
    kept = []
    rejected = 0
    for p in proposals:
        text = p.get("text", "")
        tags = p.get("tags", [])
        result = validate_seed(text, tags=tags)
        p["specificity"] = {
            "passes": result.passes,
            "score": result.score,
            "verb": result.verb,
            "target": result.target,
        }
        if result.passes:
            kept.append(p)
        else:
            rejected += 1
            import sys as _sys
            print("FILTERED: %s... (score %.2f, %s)" % (
                text[:60], result.score, result.reason), file=_sys.stderr)
    seeds["proposals"] = kept
    json.dump(seeds, sys.stdout, indent=2)
    import sys as _sys
    print("\n%d kept, %d filtered" % (len(kept), rejected), file=_sys.stderr)


def main():
    """Entry point for CLI usage."""
    if len(sys.argv) >= 3 and sys.argv[1] == "--check":
        _cli_check(" ".join(sys.argv[2:]))
    elif not sys.stdin.isatty():
        _cli_filter()
    else:
        print(__doc__)
        sys.exit(0)


if __name__ == "__main__":
    main()
