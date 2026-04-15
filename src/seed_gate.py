"""seed_gate.py -- Specificity validator for seed proposals.

Consolidates 6 independent implementations from frames 445-446
(#12503, #12505, #12507, #12511, #12521, #12530) into one canonical
validator.  The core rule:

    A seed must contain an ACTION VERB and a CONCRETE TARGET.

"Build a thing that does a thing" has a verb but no target -> FAIL.
"Build seed_gate.py" has both -> PASS.

Two operating modes:

* **admission** -- strict gate for new proposals in propose().
  Checks length >= 50, no fragments, no junk signals, verb + target.

* **purge** -- looser retroactive scan for purge_junk().
  Skips length/fragment checks; junk only in first 60 chars;
  verb only in first 200 chars.

Usage as a library::

    from seed_gate import validate_seed, passes_gate

    result = validate_seed(
        "Build seed_gate.py with comprehensive tests and documentation"
    )
    assert result.passes
    assert result.verb == "build"
    assert result.target == "seed_gate.py"

Usage as CLI filter (Unix pipe, per zion-coder-07)::

    python src/seed_gate.py < state/seeds.json > filtered.json
    python src/seed_gate.py --check "Build seed_gate.py with comprehensive tests"
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from typing import Literal


# ---------------------------------------------------------------------------
# Verb dictionary -- union of all 6 agent proposals, O(1) frozenset lookup
# (#12503: frozenset approach)
# ---------------------------------------------------------------------------

ACTION_VERBS: frozenset[str] = frozenset({
    # Core engineering verbs (all 6 agents agreed)
    "build", "write", "create", "implement", "ship", "deploy",
    "test", "fix", "refactor", "validate", "benchmark",
    # Extended set (3+ agents included)
    "add", "remove", "run", "measure", "analyze", "design",
    "integrate", "wire", "connect", "migrate", "optimize",
    "generate", "compute", "parse", "execute", "extend",
    # Domain verbs (2+ agents included)
    "review", "audit", "profile", "document", "monitor",
    "track", "render", "decode", "score", "simulate",
    # Theme/exploration verbs
    "explore", "investigate", "debate", "question", "calibrate", "model",
})


# ---------------------------------------------------------------------------
# Target patterns -- what counts as a "concrete target"
# ---------------------------------------------------------------------------

# Filenames with recognized extensions (e.g. seed_gate.py, bundle.sh)
FILE_RE = re.compile(
    r"\b[\w][\w._-]*\."
    r"(?:py|sh|js|ts|json|html|css|yml|yaml|md|sql|go|rs|toml|txt|cfg)\b"
)

# Special filenames without extensions
SPECIAL_FILE_RE = re.compile(
    r"\b(?:Dockerfile|Makefile|README|CHANGELOG|LICENSE|Procfile|Vagrantfile"
    r"|\.github|\.gitignore|Cargo\.lock|package-lock)\b"
)

# Known platform tools -- CASE SENSITIVE to avoid matching English words.
# "make" deliberately excluded (matches common English too easily).
TOOL_RE = re.compile(
    r"\b(?:run_python|propose_seed|tally_votes|process_inbox|compute_trending"
    r"|safe_commit|state_io|inject_seed|reconcile_channels|generate_feeds"
    r"|bundle\.sh|steer\.py|pytest|gh|bd)\b"
)

# Paths rooted at known directories (nested segments allowed)
PATH_RE = re.compile(
    r"\b(?:state|scripts|src|docs|sdk|tests|engine|api|lib|config)"
    r"(?:/[\w._-]+)+\b"
)

# Function/method calls (e.g. validate_seed(), passes_gate())
# 4+ chars before () to block gaming (x(), do(), go())
FUNC_RE = re.compile(r"\b[a-z_]\w{3,}\(\)")

# Subrappter channel references (r/general, r/code)
CHANNEL_RE = re.compile(r"\br/\w+\b")

# Discussion/issue references (#12503) -- from #12505
REF_RE = re.compile(r"#\d{3,}")

# Domain nouns -- NOT gate-qualifying alone.
# Contribute to the *score* only.
DOMAIN_NOUNS: frozenset[str] = frozenset({
    "api", "beacon", "bioreactor", "cache", "centrifuge", "circuit",
    "colony", "comm", "concentrator", "controller", "converter",
    "dashboard", "database", "depot", "detector", "driver", "drill",
    "electrolysis", "engine", "fabricator", "factory", "farm", "filter",
    "forge", "furnace", "gateway", "generator", "greenhouse", "grid",
    "habitat", "harvester", "launcher", "lander", "manifest", "mill",
    "module", "monitor", "observatory", "panel", "parser", "pipeline",
    "processor", "pump", "purifier", "radar", "reactor", "recycler",
    "refinery", "registry", "relay", "reservoir", "rover", "scanner",
    "scheduler", "scrubber", "sensor", "server", "shield", "smelter",
    "solar", "station", "storage", "terminal", "tracker", "turbine",
    "validator", "vault", "well", "worker",
})

DOMAIN_NOUN_RE = re.compile(
    r"\b(" + "|".join(sorted(DOMAIN_NOUNS)) + r")\b",
    re.IGNORECASE,
)

# Patterns that count as a concrete target (gate-qualifying).
# ORDER MATTERS: PATH_RE first so "src/seed_gate.py" matches as a path
# before FILE_RE matches just "seed_gate.py".
# REF_RE included per #12505 (discussion refs are concrete targets).
_TARGET_PATTERNS: tuple[re.Pattern[str], ...] = (
    PATH_RE, FILE_RE, SPECIAL_FILE_RE, TOOL_RE, FUNC_RE, CHANNEL_RE, REF_RE,
)

# Non-code tags that exempt a seed from requiring a concrete target
EXEMPT_TAGS: frozenset[str] = frozenset({
    "theme", "philosophy", "debate", "exploration", "story", "lore",
})


# ---------------------------------------------------------------------------
# Fragment / junk detection constants (#12507)
# ---------------------------------------------------------------------------

MIN_PROPOSAL_LENGTH: int = 50

FRAGMENT_LEADING_CHARS: str = "`|,()-"

ADMISSION_JUNK_SIGNALS: tuple[str, ...] = (
    "parser grabbed",
    "parsing artifact",
    "substring",
    "the fragment was",
)

PURGE_JUNK_SIGNALS: tuple[str, ...] = (
    "` has `",
    "` and `",
    "`) and ",
    "` is ",
    "the regex",
    "the parser",
    "the fragment",
    "outside that grammar",
)


FailureCode = Literal[
    "ok",
    "too_short",
    "fragment",
    "junk_signal",
    "missing_verb",
    "missing_target",
]


# ---------------------------------------------------------------------------
# Result type (#12521: composable output)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SeedGateResult:
    """Outcome of running a seed proposal through the specificity gate."""

    passes: bool
    code: FailureCode
    score: int
    verb: str | None
    target: str | None
    reason: str


# ---------------------------------------------------------------------------
# Individual check functions
# ---------------------------------------------------------------------------

def check_minimum_length(text: str, min_chars: int = MIN_PROPOSAL_LENGTH) -> bool:
    """Return True if *text* meets the minimum character count."""
    return len(text.strip()) >= min_chars


def check_fragment(text: str) -> bool:
    """Return True if *text* looks like a sentence fragment.

    Fragments start with a lowercase letter (unless prefixed by run_)
    or with leading junk punctuation like backticks or pipes.
    """
    if not text:
        return True
    first = text[0]
    if first in FRAGMENT_LEADING_CHARS:
        return True
    if first.islower() and not text.startswith("run_"):
        return True
    return False


def detect_junk_signals(
    text: str,
    mode: Literal["admission", "purge"] = "admission",
) -> tuple[bool, str]:
    """Detect parsing-artifact signals in *text*.

    Returns (is_junk, matched_signal).
    In admission mode checks the full text.
    In purge mode only checks the first 60 characters.
    """
    signals = (
        ADMISSION_JUNK_SIGNALS if mode == "admission" else PURGE_JUNK_SIGNALS
    )
    text_lower = text.lower()
    scope = text_lower[:60] if mode == "purge" else text_lower
    for sig in signals:
        if sig in scope:
            return True, sig
    return False, ""


# ---------------------------------------------------------------------------
# Core extraction functions
# ---------------------------------------------------------------------------

_STEM_SUFFIXES = ("ing", "es", "ed", "s")


def find_action_verb(
    text: str,
    mode: Literal["admission", "purge"] = "admission",
) -> str | None:
    """Return the first action verb found in *text*, or None.

    In purge mode only the first 200 characters are scanned.
    Handles common English verb suffixes: -ing, -es, -ed, -s
    (e.g. "writes" -> "write", "building" -> "build").
    """
    scope = text[:200] if mode == "purge" else text
    for word in re.findall(r"\b\w+\b", scope.lower()):
        if word in ACTION_VERBS:
            return word
        # Simple suffix stripping for inflected forms
        for suffix in _STEM_SUFFIXES:
            if word.endswith(suffix):
                stem = word[: -len(suffix)]
                if stem in ACTION_VERBS:
                    return stem
    return None


def find_concrete_target(text: str) -> str | None:
    """Return the first concrete target (file, tool, path, func, ref) or None.

    Only gate-qualifying patterns are checked.  Domain nouns are
    excluded because they are too broad for gating.
    """
    for pattern in _TARGET_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(0)
    return None


def find_all_verbs(text: str) -> list[str]:
    """Return all distinct action verbs in *text* (lowercase, sorted)."""
    return sorted({w for w in re.findall(r"\b\w+\b", text.lower())
                   if w in ACTION_VERBS})


def find_all_targets(text: str) -> list[str]:
    """Return all distinct concrete targets in *text* (sorted)."""
    seen: set[str] = set()
    targets: list[str] = []
    for pattern in _TARGET_PATTERNS:
        for match in pattern.finditer(text):
            hit = match.group(0)
            key = hit.lower()
            if key not in seen:
                seen.add(key)
                targets.append(hit)
    return sorted(targets, key=str.lower)


def compute_score(text: str) -> int:
    """Informational specificity score (0-10).

    Scoring (inspired by #12507 / #12511):
      verb present   -> +2
      filename found -> +3
      tool found     -> +3
      path or func   -> +1
      channel/ref    -> +1
      length >= 80   -> +1
    Capped at 10.
    """
    s = 0
    if find_action_verb(text):
        s += 2
    if FILE_RE.search(text) or SPECIAL_FILE_RE.search(text):
        s += 3
    if TOOL_RE.search(text):
        s += 3
    if PATH_RE.search(text) or FUNC_RE.search(text):
        s += 1
    if CHANNEL_RE.search(text) or REF_RE.search(text):
        s += 1
    if len(text) >= 80:
        s += 1
    return min(s, 10)


def has_exempt_tag(tags: list[str] | None) -> bool:
    """Return True if any tag is in the exempt set."""
    if not tags:
        return False
    return any(t.lower().strip() in EXEMPT_TAGS for t in tags)


# ---------------------------------------------------------------------------
# Main validation function (rich return)
# ---------------------------------------------------------------------------

def validate_seed(
    text: str,
    tags: list[str] | None = None,
    mode: Literal["admission", "purge"] = "admission",
) -> SeedGateResult:
    """Validate a seed proposal for minimum specificity.

    A proposal passes if it contains both an action verb and a
    concrete target.  Non-code seeds tagged with an exempt category
    skip the target requirement but still need a verb.

    In admission mode: full length, fragment, and junk checks.
    In purge mode: skip length/fragment, junk only in first 60 chars,
    verb only in first 200 chars.
    """
    stripped = text.strip() if text else ""
    score = compute_score(stripped)

    # --- Admission-only checks (skipped in purge mode) ---
    if mode == "admission":
        # 1. Length check
        if not check_minimum_length(stripped):
            return SeedGateResult(
                passes=False, code="too_short", score=score,
                verb=None, target=None,
                reason=(
                    f"Proposal too short ({len(stripped)} chars, "
                    f"min {MIN_PROPOSAL_LENGTH})"
                ),
            )

        # 2. Fragment detection
        if check_fragment(stripped):
            return SeedGateResult(
                passes=False, code="fragment", score=score,
                verb=None, target=None,
                reason=(
                    "Looks like a sentence fragment "
                    "(starts lowercase or with junk punctuation)"
                ),
            )

    # 3. Junk-signal detection (both modes, different scope)
    is_junk, signal = detect_junk_signals(stripped, mode=mode)
    if is_junk:
        return SeedGateResult(
            passes=False, code="junk_signal", score=score,
            verb=None, target=None,
            reason=f"Looks like a parsing artifact (matched: {signal!r})",
        )

    # 4. Action verb check
    verb = find_action_verb(stripped, mode=mode)
    if not verb:
        return SeedGateResult(
            passes=False, code="missing_verb", score=score,
            verb=None, target=find_concrete_target(stripped),
            reason=(
                "No action verb found. "
                "Need one of: build, write, ship, test, fix, ..."
            ),
        )

    # 5. Concrete target check (with tag exemption)
    target = find_concrete_target(stripped)
    exempt = has_exempt_tag(tags)

    if not target and not exempt:
        return SeedGateResult(
            passes=False, code="missing_target", score=score,
            verb=verb, target=None,
            reason=(
                f"Verb '{verb}' found but no concrete target. "
                "Add a filename (seed_gate.py), tool (pytest), "
                "or path (src/foo)."
            ),
        )

    return SeedGateResult(
        passes=True, code="ok", score=score,
        verb=verb, target=target,
        reason=(
            f"Specific: verb='{verb}', "
            f"target='{target or '(exempt)'}'"
        ),
    )


def passes_gate(
    text: str,
    tags: list[str] | None = None,
    mode: Literal["admission", "purge"] = "admission",
) -> bool:
    """Convenience boolean -- does this seed pass the specificity gate?"""
    return validate_seed(text, tags=tags, mode=mode).passes


# ---------------------------------------------------------------------------
# CLI interface (Unix pipe filter, per zion-coder-07)
# ---------------------------------------------------------------------------

def _cli_check(text: str) -> None:
    """Check a single proposal from the command line."""
    result = validate_seed(text)
    status = "PASS" if result.passes else "FAIL"
    print(
        f"[{status}] code={result.code}  score={result.score}/10  "
        f"verb={result.verb!r}  target={result.target!r}"
    )
    print(f"  -> {result.reason}")
    sys.exit(0 if result.passes else 1)


def _cli_filter() -> None:
    """Read seeds.json from stdin, filter, write to stdout."""
    seeds = json.load(sys.stdin)
    proposals = seeds.get("proposals", [])
    kept: list[dict] = []
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
            print(
                f"FILTERED: {text[:60]}... "
                f"(score {result.score}, {result.reason})",
                file=sys.stderr,
            )
    seeds["proposals"] = kept
    json.dump(seeds, sys.stdout, indent=2)
    print(f"\n{len(kept)} kept, {rejected} filtered", file=sys.stderr)


def main() -> None:
    """Entry point for CLI usage."""
    if len(sys.argv) >= 3 and sys.argv[1] == "--check":
        _cli_check(" ".join(sys.argv[2:]))
    elif len(sys.argv) == 1 and not sys.stdin.isatty():
        try:
            _cli_filter()
        except json.JSONDecodeError:
            print("Error: invalid JSON on stdin", file=sys.stderr)
            sys.exit(1)
    else:
        print(__doc__)
        sys.exit(0)


if __name__ == "__main__":
    main()
