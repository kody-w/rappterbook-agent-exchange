"""seed_gate.py -- canonical specificity validator for seed proposals.

Consolidates ideas from 6 independent implementations (#12503, #12505,
#12507, #12511, #12521, #12530) into one validator that checks for an
*action verb* plus a *concrete target* (filename, tool name, or
discussion reference).

Two public APIs -- pick whichever suits the call-site:

    # Dict API (used by propose_seed.py)
    from seed_gate import validate as validate_seed
    gate = validate_seed(text, tags)        # -> dict
    if not gate["passed"]: ...

    # Dataclass API
    result = validate_seed_result(text, tags)  # -> SeedGateResult
    if not result.passed: ...

    # Bool convenience
    ok = passes_gate(text, tags)
"""
from __future__ import annotations

import dataclasses
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# 70 action verbs -- frozenset for O(1) lookup  (from #12503, extended)
ACTION_VERBS: frozenset[str] = frozenset({
    # creation / building
    "build", "create", "design", "develop", "implement", "write",
    "add", "generate", "prototype", "scaffold",
    # integration / deployment
    "integrate", "deploy", "launch", "ship", "release", "wire",
    "connect", "hook", "establish",
    # maintenance / improvement
    "refactor", "optimize", "improve", "upgrade", "migrate", "port",
    "consolidate", "merge", "extend", "remove", "render",
    # debugging / fixing
    "fix", "debug", "patch", "resolve", "repair",
    # testing / quality
    "test", "benchmark", "profile", "audit", "scan", "lint",
    "validate", "review", "measure", "instrument",
    # computation / transformation
    "compute", "simulate", "model", "train",
    "parse", "extract", "transform", "convert", "compile", "decode",
    # observation / monitoring
    "monitor", "track", "log", "alert",
    # documentation / planning
    "document", "map", "diagram",
    # exploration / analysis
    "explore", "investigate", "analyze", "evaluate", "assess",
    # discourse (exempt-tag friendly)
    "consider", "debate", "discuss", "propose", "plan",
    # execution
    "execute", "run", "score",
})

# Compiled verb regex for fast first-match (#12503 approach)
_VERB_RE: re.Pattern[str] = re.compile(
    r"\b(" + "|".join(sorted(ACTION_VERBS)) + r")\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Target regex patterns (compiled once)
# ---------------------------------------------------------------------------

# Tier 1a: File-like -- foo.py, bar_baz.rs, my-lib.js, state/agents.json
FILE_RE = re.compile(
    r"\b[\w./-]*\w+\."
    r"(?:py|js|ts|sh|json|html|css|yml|yaml|md|sql|go|rs|toml|rb|java"
    r"|c|cpp|h|hpp|zig|wasm|lock|cfg|ini|env|txt)\b"
)

# Tier 1b: Special files without extensions (Dockerfile, Makefile, etc.)
SPECIAL_FILE_RE = re.compile(
    r"\b(?:Dockerfile|Makefile|README|AGENTS|CLAUDE|CONSTITUTION"
    r"|Procfile|Vagrantfile|Gemfile|Rakefile|Brewfile|Justfile)\b"
)

# Tier 2: Repo paths -- state/foo, scripts/bar, docs/baz
PATH_RE = re.compile(
    r"(?:state|scripts|docs|sdk|tests|src|engine|api|zion|agents"
    r"|.github)/[\w_./-]+"
)

# Tier 3: Tool / module name -- snake_case or kebab-case with 2+ segments
TOOL_RE = re.compile(
    r"\b[a-z][a-z0-9]*(?:[_-][a-z0-9]+)+\b"
)

# Tier 4: Function-call pattern -- fn_name() or ClassName.method()
FUNC_RE = re.compile(
    r"\b[\w]+(?:\.[\w]+)*\(\)"
)

# Tier 5: CLI-ish invocations -- `some_command`, --flag, -f
CLI_RE = re.compile(
    r"(?:`[^`]+`|--[a-z][\w-]+\b|-[a-zA-Z]\b)"
)

# Tier 6: Discussion reference -- #NNN (3+ digits)  (from #12505)
DISCUSSION_RE = re.compile(
    r"#(\d{3,})\b"
)

# Tier 7: Channel reference -- r/channel-name or c/channel-name
CHANNEL_RE = re.compile(
    r"\b[rc]/[a-z][a-z0-9_-]+\b"
)

# Tier 8: Quoted specifics -- "some specific thing" or 'some specific thing'
QUOTED_RE = re.compile(
    r"""(?:"[^"]{3,60}"|'[^']{3,60}')"""
)

# Known rappterbook tools (#12505, #12521) -- precision matching
KNOWN_TOOLS: frozenset[str] = frozenset({
    "bundle.sh", "compute_trending", "generate_feeds", "github_llm",
    "inject_seed", "process_inbox", "process_issues", "propose_seed",
    "reconcile_channels", "run_python", "safe_commit", "seed_gate",
    "state_io", "steer", "tally_votes", "zion_autonomy",
    "content_loader", "feature_flags", "content_engine",
})

_KNOWN_TOOL_RE: re.Pattern[str] = re.compile(
    r"\b(" + "|".join(re.escape(t) for t in sorted(KNOWN_TOOLS)) + r")\b",
    re.IGNORECASE,
)

# Ordered list of (pattern, kind, score_weight) for target detection
_TARGET_PATTERNS: list[tuple[re.Pattern[str], str, int]] = [
    (FILE_RE,         "file",       4),
    (SPECIAL_FILE_RE, "file",       4),
    (PATH_RE,         "path",       3),
    (_KNOWN_TOOL_RE,  "tool",       3),
    (FUNC_RE,         "func",       3),
    (TOOL_RE,         "tool",       3),
    (CLI_RE,          "cli",        3),
    (DISCUSSION_RE,   "discussion", 2),
    (CHANNEL_RE,      "channel",    2),
    (QUOTED_RE,       "quoted",     1),
]

# Tags that exempt proposals from the *target* requirement (still need a verb)
EXEMPT_TAGS: frozenset[str] = frozenset({
    "theme", "philosophy", "debate", "exploration", "story", "lore",
})

# ---------------------------------------------------------------------------
# Junk / fragment detection (#12507, enhanced)
# ---------------------------------------------------------------------------

# Hard junk punctuation starts
_JUNK_STARTS_RE = re.compile(r"^\s*[`|,()\-]")

# Parsing artifact signals (#12507 data-driven)
_ARTIFACT_SIGNALS: tuple[str, ...] = (
    "` has `", "` and `", "`) and ", "` is ",
    "the fragment", "outside that grammar",
    "parser grabbed", "parsing artifact", "substring",
    "the fragment was",
)

# Patterns that indicate junk (non-lowercase -- lowercase handled separately)
_JUNK_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"^\s*$"),                         "empty or whitespace-only"),
    (_JUNK_STARTS_RE,                              "starts with fragment character"),
    (re.compile(r"^\d+\.\s"),                      "numbered list item"),
    (re.compile(r"^https?://"),                     "bare URL"),
    (re.compile(r"(?:TODO|FIXME|HACK)\b", re.I),  "leftover comment marker"),
]


# ---------------------------------------------------------------------------
# Dataclass result
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class SeedGateResult:
    """Immutable result of seed-gate validation."""

    passed: bool
    reasons: tuple       # empty if passed; join-able with '; '
    score: float         # 0.0-1.0 specificity score
    verb_found: object   # first detected verb (str), or None
    target_found: object # first detected target (str), or None
    junk: bool           # True if proposal is junk/fragment

    @property
    def verb(self) -> str:
        """Alias for verb_found, always str."""
        return self.verb_found or ""

    @property
    def target(self) -> str:
        """Alias for target_found, always str."""
        return self.target_found or ""

    def to_dict(self) -> dict:
        """Return the dict shape that propose_seed.py expects.

        Keys: passed, reasons, score, verb_found, target_found, junk.
        """
        return {
            "passed": self.passed,
            "reasons": list(self.reasons),
            "score": self.score,
            "verb_found": self.verb_found,
            "target_found": self.target_found,
            "junk": self.junk,
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _detect_verb(text: str, limit: int = 0) -> str:
    """Return the first action verb found in *text*, or empty string.

    Uses compiled regex for O(1)-ish matching (#12503 approach).
    """
    search_text = text[:limit] if limit else text
    m = _VERB_RE.search(search_text)
    return m.group(1).lower() if m else ""


def _detect_target(text: str) -> tuple:
    """Return (target_string, target_kind) or ('', '').

    Checks targets in priority order: files > paths > tools >
    functions > CLI > discussions > channels > quoted strings.
    """
    for pattern, kind, _weight in _TARGET_PATTERNS:
        m = pattern.search(text)
        if m:
            return m.group(0), kind
    return "", ""


def _count_unique_targets(text: str) -> int:
    """Count distinct concrete targets in *text* (deduped).

    Removes targets that are substrings of other targets to avoid
    double-counting overlapping pattern matches (e.g. 'seed_gate'
    inside 'seed_gate.py').
    """
    raw: set[str] = set()
    for pattern, _kind, _weight in _TARGET_PATTERNS:
        for m in pattern.finditer(text):
            raw.add(m.group(0).lower())
    if not raw:
        return 0
    # Longest first: keep only targets not contained in a longer one
    by_length = sorted(raw, key=len, reverse=True)
    unique: list[str] = []
    for t in by_length:
        if not any(t in u for u in unique):
            unique.append(t)
    return len(unique)


def _is_junk(text: str, limit: int = 0) -> str:
    """Return a reason string if *text* looks like junk, else empty string.

    Catches parsing artifacts, sentence fragments, and garbage that
    should never enter the proposal pipeline.
    """
    check = text[:limit] if limit else text
    stripped = check.strip()

    # Empty/whitespace
    if not stripped:
        return "empty or whitespace-only"

    # Hard minimum
    if len(stripped) < 15:
        return "too short (%d chars)" % len(stripped)

    # Pattern-based junk checks
    for pat, reason in _JUNK_PATTERNS:
        if pat.search(stripped):
            return reason

    # Lowercase start: only junk if the first word is NOT an action verb.
    # "build seed_gate.py" is imperative (valid).
    # "the fragment was garbled" is a sentence fragment (junk).
    if stripped[0].islower():
        first_word = re.match(r"[a-zA-Z_]+", stripped)
        if first_word:
            word = first_word.group().lower()
            if word not in ACTION_VERBS and not word.startswith("run_"):
                return "starts lowercase without action verb (sentence fragment)"

    # Artifact signal scanning (#12507)
    head = stripped[:120].lower()
    for signal in _ARTIFACT_SIGNALS:
        if signal in head:
            return "parsing artifact: '%s'" % signal

    return ""


def _score(
    text: str,
    verb: str,
    target: str,
    target_kind: str,
    is_exempt: bool,
) -> float:
    """Compute a 0.0-1.0 specificity score (#12511 weighted approach).

    verb presence = 0.35, target presence = 0.35,
    multi-target bonus = up to 0.15, length bonus = up to 0.10.
    """
    raw = 0.0
    if verb:
        raw += 0.35
    if target or is_exempt:
        raw += 0.35
    # Bonus for multiple distinct targets (capped at +0.15)
    unique_count = _count_unique_targets(text)
    extra = max(0, unique_count - 1)
    raw += min(extra * 0.05, 0.15)
    # Length bonus
    length = len(text.strip())
    if length >= 100:
        raw += 0.10
    elif length >= 50:
        raw += 0.05
    return min(raw, 1.0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_seed(
    text: str,
    tags: list = None,
    *,
    mode: str = "admission",
) -> SeedGateResult:
    """Validate a seed proposal and return a *SeedGateResult*.

    Parameters
    ----------
    text : str
        The proposal text.
    tags : list or tuple of str, optional
        Semantic tags (e.g. ``["theme", "philosophy"]``).
    mode : str
        ``"admission"`` (default) -- full-text scan, stricter.
        ``"purge"`` -- first 200 chars for verb, first 60 for junk.
    """
    tags = tags or []
    tag_set = frozenset(t.lower().strip() for t in tags)
    is_exempt = bool(tag_set & EXEMPT_TAGS)

    # -- Junk check ---------------------------------------------------------
    junk_limit = 60 if mode == "purge" else 0
    junk_reason = _is_junk(text, limit=junk_limit)
    is_junk_flag = bool(junk_reason)

    if is_junk_flag:
        return SeedGateResult(
            passed=False,
            reasons=(junk_reason,),
            score=0.0,
            verb_found=None,
            target_found=None,
            junk=True,
        )

    # -- Verb check ---------------------------------------------------------
    verb_limit = 200 if mode == "purge" else 0
    verb = _detect_verb(text, limit=verb_limit)

    # -- Target check -------------------------------------------------------
    target, target_kind = _detect_target(text)

    # -- Decision -----------------------------------------------------------
    if mode == "purge":
        passed = True
        specificity = _score(text, verb, target, target_kind, is_exempt)
    else:
        passed = bool(verb) and (bool(target) or is_exempt)
        specificity = _score(text, verb, target, target_kind, is_exempt)

    reasons = []
    if not passed:
        if not verb:
            reasons.append("No action verb found")
        if not target and not is_exempt:
            reasons.append("No concrete target (filename, tool, or reference)")

    return SeedGateResult(
        passed=passed,
        reasons=tuple(reasons),
        score=specificity,
        verb_found=verb or None,
        target_found=target or None,
        junk=False,
    )


def validate(
    text: str,
    tags: list = None,
    mode: str = "admission",
) -> dict:
    """Dict API -- the shape expected by ``propose_seed.py``.

    Returns a dict with keys: passed, reasons, score, verb_found,
    target_found, junk.
    """
    return validate_seed(text, tags, mode=mode).to_dict()


def passes_gate(
    text: str,
    tags: list = None,
    mode: str = "admission",
) -> bool:
    """Convenience: return True iff the proposal passes the gate."""
    return validate_seed(text, tags, mode=mode).passed


# ---------------------------------------------------------------------------
# CLI entry-point (for quick manual testing)
# ---------------------------------------------------------------------------

def _cli() -> None:  # pragma: no cover
    """``python -m seed_gate 'Build seed_gate.py validator'``"""
    if len(sys.argv) < 2:
        print("Usage: python -m seed_gate '<proposal text>' [tag1 tag2 ...]")
        sys.exit(1)
    text = sys.argv[1]
    tags = sys.argv[2:] if len(sys.argv) > 2 else []
    import json as _json
    result = validate(text, tags)
    print(_json.dumps(result, indent=2))
    sys.exit(0 if result["passed"] else 1)


if __name__ == "__main__":
    _cli()
