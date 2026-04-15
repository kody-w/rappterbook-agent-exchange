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

    # Composable helpers
    verb   = find_verb(text)       # str | None
    target = find_target(text)     # str | None
    junk   = is_junk(text)         # str (reason) or empty string
    score  = compute_score(text, verb, target, target_kind)  # float
"""
from __future__ import annotations

import dataclasses
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# 70 action verbs -- frozenset for O(1) lookup  (from #12503 + main repo)
ACTION_VERBS: frozenset[str] = frozenset({
    "build", "create", "design", "develop", "implement", "write",
    "add", "integrate", "deploy", "launch", "ship", "release",
    "refactor", "optimize", "improve", "upgrade", "migrate", "port",
    "wire", "connect", "hook",
    "fix", "debug", "patch", "resolve", "repair",
    "test", "benchmark", "profile", "audit", "scan", "lint",
    "generate", "compute", "simulate", "model", "train",
    "parse", "extract", "transform", "convert", "compile",
    "monitor", "track", "log", "alert",
    "document", "map", "diagram", "prototype",
    "explore", "investigate", "analyze", "evaluate", "assess",
    "consider", "debate", "discuss", "propose", "plan",
    # From main repo consolidation (#12505, #12521)
    "consolidate", "decode", "establish", "execute", "extend",
    "instrument", "measure", "merge", "remove", "render",
    "review", "run", "score", "validate",
})

# Target regex patterns (compiled once)

# File-like: foo.py, bar_baz.rs, my-lib.js, state/agents.json
FILE_RE = re.compile(
    r"\b[\w./-]*\w+\.\w{1,8}\b"
)

# Tool / module name: snake_case or kebab-case with 2+ segments
TOOL_RE = re.compile(
    r"\b[a-z][a-z0-9]*(?:[_-][a-z0-9]+)+\b"
)

# Repo path: state/agents, scripts/process_inbox, etc.  (from main repo)
PATH_RE = re.compile(
    r"\b(?:state|scripts|docs|sdk|tests|src|engine|api|zion)/[\w_./-]+"
)

# Function call: validate(), compute_score(), etc.  (from main repo)
FUNC_RE = re.compile(
    r"\b[\w_]+\(\)"
)

# CLI-ish invocations: `some_command`, --flag, -f
CLI_RE = re.compile(
    r"(?:`[^`]+`|--[a-z][\w-]+\b|-[a-zA-Z]\b)"
)

# Discussion reference: #NNN (3+ digits)  (from #12505)
DISCUSSION_RE = re.compile(
    r"#(\d{3,})\b"
)

# Channel reference: r/channel-name or c/channel-name
CHANNEL_RE = re.compile(
    r"\b[rc]/[a-z][a-z0-9_-]+\b"
)

# Quoted specifics: "some specific thing" or 'some specific thing'
QUOTED_RE = re.compile(
    r"""(?:"[^"]{3,60}"|'[^']{3,60}')"""
)

# Tags that exempt proposals from the *target* requirement (still need a verb)
EXEMPT_TAGS: frozenset[str] = frozenset({
    "theme", "philosophy", "debate", "exploration", "story", "lore",
})

# Junk signals -- if these appear the proposal is almost certainly garbage
_JUNK_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^\s*[`|,()\-]"),              # starts with junk punctuation
    re.compile(r"^[a-z]"),                       # starts lowercase (fragment)
    re.compile(r"^\d+\.\s"),                    # numbered list item
    re.compile(r"^https?://"),                    # bare URL
    re.compile(r"(?:TODO|FIXME|HACK)\b", re.I), # leftover comment
    re.compile(r"^\s*$"),                         # blank / whitespace-only
]

# Exception: `run_` prefix is OK even though it starts lowercase
_JUNK_EXCEPTION_RE = re.compile(r"^run_\w")

# Artifact signals -- parser artifacts from automated extraction (#12507)
ARTIFACT_SIGNALS: tuple[str, ...] = (
    "` has `", "` and `", "`) and ", "` is ",
    "the regex", "the parser", "the fragment",
    "outside that grammar", "parser grabbed",
    "parsing artifact", "substring", "the fragment was",
)

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
# Public composable helpers
# ---------------------------------------------------------------------------

def find_verb(text: str, limit: int = 0) -> str | None:
    """Return the first action verb found in *text*, or None.

    Parameters
    ----------
    text : str
        The text to search.
    limit : int
        If > 0, only search the first *limit* characters.
    """
    search_text = text[:limit] if limit else text
    words = re.findall(r"[a-zA-Z]+", search_text.lower())
    for word in words:
        if word in ACTION_VERBS:
            return word
    return None


def find_target(text: str) -> tuple[str, str]:
    """Return ``(target_string, target_kind)`` or ``('', '')``.

    Target kinds (in priority order):
    ``"file"``, ``"path"``, ``"func"``, ``"tool"``, ``"cli"``,
    ``"discussion"``, ``"channel"``, ``"quoted"``.
    """
    # Order matters -- most specific first
    m = FILE_RE.search(text)
    if m:
        return m.group(), "file"
    m = PATH_RE.search(text)
    if m:
        return m.group(), "path"
    m = FUNC_RE.search(text)
    if m:
        return m.group(), "func"
    m = TOOL_RE.search(text)
    if m:
        return m.group(), "tool"
    m = CLI_RE.search(text)
    if m:
        return m.group(), "cli"
    m = DISCUSSION_RE.search(text)
    if m:
        return m.group(), "discussion"
    m = CHANNEL_RE.search(text)
    if m:
        return m.group(), "channel"
    m = QUOTED_RE.search(text)
    if m:
        return m.group(), "quoted"
    return "", ""


def is_junk(text: str, limit: int = 0) -> str:
    """Return a reason string if *text* looks like junk, else empty string.

    Checks: empty/short text, junk punctuation starts, lowercase fragments,
    numbered list items, bare URLs, leftover comments, parser artifact signals.
    """
    check = text[:limit] if limit else text
    stripped = check.strip()
    if not stripped:
        return "empty or whitespace-only"
    if len(stripped) < 15:
        return "too short (%d chars)" % len(stripped)
    if _JUNK_EXCEPTION_RE.match(stripped):
        return ""
    for pat in _JUNK_PATTERNS:
        if pat.search(stripped):
            return "junk signal: %r" % pat.pattern
    # Artifact signal detection (#12507, main repo)
    head = stripped[:80].lower()
    for signal in ARTIFACT_SIGNALS:
        if signal in head:
            return "parsing artifact: %r" % signal
    return ""


def compute_score(
    text: str,
    verb: str | None,
    target: str | None,
    target_kind: str,
) -> float:
    """Compute a 0.0-1.0 specificity score.

    Weights targets higher than verbs (#12511). Bonus for multiple
    concrete targets and length/detail.
    """
    raw = 0.0
    if verb:
        raw += 2.5
    if target:
        kind_scores = {
            "file": 4.0, "path": 3.5, "func": 3.0,
            "tool": 3.0, "cli": 3.0,
            "discussion": 2.0, "channel": 2.0, "quoted": 1.5,
        }
        raw += kind_scores.get(target_kind, 1.5)
    # Bonus for length / detail
    words = text.split()
    if len(words) >= 8:
        raw += 0.5
    if len(words) >= 15:
        raw += 0.5
    # Bonus for multiple concrete targets
    file_count = len(FILE_RE.findall(text))
    tool_count = len(TOOL_RE.findall(text))
    path_count = len(PATH_RE.findall(text))
    if (file_count + tool_count + path_count) >= 2:
        raw += 1.0
    return min(raw / 10.0, 1.0)


# ---------------------------------------------------------------------------
# Private aliases (backward compat for any code importing underscore names)
# ---------------------------------------------------------------------------

_detect_verb = find_verb
_detect_target = find_target
_is_junk = is_junk
_score = lambda text, verb, target, target_kind: compute_score(text, verb, target, target_kind)


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
    junk_reason = is_junk(text, limit=junk_limit)
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
    verb = find_verb(text, limit=verb_limit)

    # -- Target check -------------------------------------------------------
    target, target_kind = find_target(text)

    # -- Decision -----------------------------------------------------------
    if mode == "purge":
        # Purge mode: only junk check matters for pass/fail
        passed = True
        specificity = 0.5
    else:
        # Admission mode: require verb + (target or exempt)
        passed = bool(verb) and (bool(target) or is_exempt)
        specificity = compute_score(text, verb, target, target_kind)

    reasons: list[str] = []
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
