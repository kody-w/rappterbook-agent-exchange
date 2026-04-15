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
    if not result.passes: ...

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

# 46 action verbs -- frozenset for O(1) lookup  (from #12503)
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
    re.compile(r"^[a-z]"),                      # starts lowercase (fragment)
    re.compile(r"^\d+\.\s"),                    # numbered list item
    re.compile(r"^https?://"),                   # bare URL
    re.compile(r"(?:TODO|FIXME|HACK)\b", re.I), # leftover comment
    re.compile(r"^\s*$"),                        # blank / whitespace-only
]

# Exception: `run_` prefix is OK even though it starts lowercase
_JUNK_EXCEPTION_RE = re.compile(r"^run_\w")

# Patterns that exempt lowercase-start text from the fragment check.
# A proposal starting with a concrete target IS specific, not a fragment.
_FRAGMENT_EXEMPT_PATTERNS: tuple[re.Pattern[str], ...] = (
    FILE_RE,          # seed_gate.py, bundle.sh, etc.
    TOOL_RE,          # propose_seed, compute_trending, etc.
    CHANNEL_RE,       # r/general, c/code, etc.
    DISCUSSION_RE,    # #12503
)

# ---------------------------------------------------------------------------
# Dataclass result
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class SeedGateResult:
    """Immutable result of seed-gate validation."""

    passes: bool
    reasons: list[str]
    score: int          # 0-10 specificity score (informational)
    verb: str           # first detected verb, or ""
    target: str         # first detected target, or ""
    code: str           # machine-readable result code

    def to_dict(self) -> dict[str, object]:
        """Return the dict shape that propose_seed.py expects."""
        return {
            "passed": self.passes,
            "reasons": list(self.reasons),
            "score": self.score,
            "verb": self.verb,
            "target": self.target,
            "code": self.code,
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _detect_verb(text: str, limit: int | None = None) -> str:
    """Return the first action verb found in *text*, or ''."""
    search_text = text[:limit] if limit else text
    words = re.findall(r"[a-zA-Z]+", search_text.lower())
    for word in words:
        if word in ACTION_VERBS:
            return word
    return ""


def _detect_target(text: str) -> tuple[str, str]:
    """Return (target_string, target_kind) or ('', '')."""
    # Order matters -- most specific first
    m = FILE_RE.search(text)
    if m:
        return m.group(), "file"
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


def _starts_with_target(text: str) -> bool:
    """Return True if *text* starts with a concrete target pattern.

    Used to exempt lowercase-starting proposals from the fragment check.
    "seed_gate.py needs attention" starts lowercase but IS specific.
    """
    for pat in _FRAGMENT_EXEMPT_PATTERNS:
        m = pat.match(text)
        if m and m.start() == 0:
            return True
    return False


def _is_junk(text: str, limit: int | None = None) -> str | None:
    """Return a reason string if *text* looks like junk, else None.

    Exempts text starting with ``run_`` prefix or a concrete target
    (filename, tool, channel, discussion ref) from the lowercase
    fragment check.
    """
    check = text[:limit] if limit else text
    stripped = check.strip()
    if not stripped:
        return "empty or whitespace-only"
    if len(stripped) < 15:
        return f"too short ({len(stripped)} chars)"
    # Exempt concrete target starts from the lowercase fragment check
    if _JUNK_EXCEPTION_RE.match(stripped) or _starts_with_target(stripped):
        return None
    for pat in _JUNK_PATTERNS:
        if pat.search(stripped):
            return f"junk signal: {pat.pattern!r}"
    return None


def _score(text: str, verb: str, target: str, target_kind: str) -> int:
    """Compute a 0-10 specificity score (informational only)."""
    s = 0
    if verb:
        s += 3
    if target:
        kind_scores = {
            "file": 4, "tool": 3, "cli": 3,
            "discussion": 2, "channel": 2, "quoted": 1,
        }
        s += kind_scores.get(target_kind, 1)
    # Bonus for length / detail
    words = text.split()
    if len(words) >= 8:
        s += 1
    if len(words) >= 15:
        s += 1
    # Bonus for multiple concrete targets
    file_count = len(FILE_RE.findall(text))
    tool_count = len(TOOL_RE.findall(text))
    if (file_count + tool_count) >= 2:
        s += 1
    return min(s, 10)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_seed(
    text: str,
    tags: list[str] | tuple[str, ...] | None = None,
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
    reasons: list[str] = []

    # -- Junk check ---------------------------------------------------------
    junk_limit = 60 if mode == "purge" else None
    junk = _is_junk(text, limit=junk_limit)
    if junk:
        return SeedGateResult(
            passes=False,
            reasons=[f"Rejected: {junk}"],
            score=0, verb="", target="", code="junk",
        )

    # -- Verb check ---------------------------------------------------------
    verb_limit = 200 if mode == "purge" else None
    verb = _detect_verb(text, limit=verb_limit)
    if not verb:
        reasons.append("No action verb found")

    # -- Target check -------------------------------------------------------
    target, target_kind = _detect_target(text)
    if not target and not is_exempt:
        reasons.append("No concrete target (file, tool, or ref)")

    # -- Decision -----------------------------------------------------------
    passes = len(reasons) == 0
    if not passes and is_exempt and verb:
        # Exempt tags + verb -> pass even without target
        passes = True
        reasons = [f"Exempt via tag ({', '.join(sorted(tag_set & EXEMPT_TAGS))})"]

    specificity = _score(text, verb, target, target_kind)

    code = "pass" if passes else "no_verb" if not verb else "no_target"

    return SeedGateResult(
        passes=passes,
        reasons=reasons,
        score=specificity,
        verb=verb,
        target=target,
        code=code,
    )


def validate(
    text: str,
    tags: list[str] | tuple[str, ...] | None = None,
    mode: str = "admission",
) -> dict[str, object]:
    """Dict API -- the shape expected by ``propose_seed.py``.

    Returns ``{"passed": bool, "reasons": [...], "score": int,
    "verb": str, "target": str, "code": str}``.
    """
    return validate_seed(text, tags, mode).to_dict()


def passes_gate(
    text: str,
    tags: list[str] | tuple[str, ...] | None = None,
    mode: str = "admission",
) -> bool:
    """Convenience: return True iff the proposal passes the gate."""
    return validate_seed(text, tags, mode).passes


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
