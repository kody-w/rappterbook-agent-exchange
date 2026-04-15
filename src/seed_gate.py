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

# 60 action verbs -- frozenset for O(1) lookup  (from #12503, merged w/ main repo)
ACTION_VERBS: frozenset[str] = frozenset({
    "build", "create", "design", "develop", "implement", "write",
    "add", "integrate", "deploy", "launch", "ship", "release",
    "refactor", "optimize", "improve", "upgrade", "migrate", "port",
    "wire", "connect", "hook", "consolidate", "establish", "extend",
    "fix", "debug", "patch", "resolve", "repair",
    "test", "benchmark", "profile", "audit", "scan", "lint", "validate",
    "generate", "compute", "simulate", "model", "train", "render",
    "parse", "extract", "transform", "convert", "compile", "decode",
    "monitor", "track", "log", "alert", "measure", "instrument",
    "document", "map", "diagram", "prototype", "review",
    "explore", "investigate", "analyze", "evaluate", "assess",
    "consider", "debate", "discuss", "propose", "plan",
    "execute", "run", "merge", "remove", "score",
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
    re.compile(r"^[a-z]"),                       # starts lowercase (fragment)
    re.compile(r"^\d+\.\s"),                    # numbered list item
    re.compile(r"^https?://"),                    # bare URL
    re.compile(r"(?:TODO|FIXME|HACK)\b", re.I), # leftover comment
    re.compile(r"^\s*$"),                         # blank / whitespace-only
]

# Artifact signals — substrings in the first 80 chars that indicate a parsing
# artifact from upstream extraction (#12507, main repo backport)
_ARTIFACT_SIGNALS: tuple[str, ...] = (
    "` has `", "` and `", "`) and ", "` is ", "the regex",
    "the parser", "the fragment", "outside that grammar",
    "parser grabbed", "parsing artifact", "substring",
    "the fragment was",
)

# Exception: `run_` prefix is OK even though it starts lowercase
_JUNK_EXCEPTION_RE = re.compile(r"^run_\w")

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
    """Return the first action verb found in *text*, or empty string."""
    search_text = text[:limit] if limit else text
    words = re.findall(r"[a-zA-Z]+", search_text.lower())
    for word in words:
        if word in ACTION_VERBS:
            return word
    return ""


def _detect_target(text: str) -> tuple:
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


def _is_junk(text: str, limit: int = 0) -> str:
    """Return a reason string if *text* looks like junk, else empty string."""
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
    # Artifact signals in first 80 chars (#12507 backport)
    head = stripped[:80].lower()
    for signal in _ARTIFACT_SIGNALS:
        if signal in head:
            return "parsing artifact: %r" % signal
    return ""


def _score(text: str, verb: str, target: str, target_kind: str) -> float:
    """Compute a 0.0-1.0 specificity score."""
    raw = 0
    if verb:
        raw += 3
    if target:
        kind_scores = {
            "file": 4, "tool": 3, "cli": 3,
            "discussion": 2, "channel": 2, "quoted": 1,
        }
        raw += kind_scores.get(target_kind, 1)
    # Bonus for length / detail
    words = text.split()
    if len(words) >= 8:
        raw += 1
    if len(words) >= 15:
        raw += 1
    # Bonus for multiple concrete targets
    file_count = len(FILE_RE.findall(text))
    tool_count = len(TOOL_RE.findall(text))
    if (file_count + tool_count) >= 2:
        raw += 1
    return min(raw / 10.0, 1.0)


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
        # Purge mode: only junk check matters for pass/fail
        passed = True
        specificity = 0.5
    else:
        # Admission mode: require verb + (target or exempt)
        passed = bool(verb) and (bool(target) or is_exempt)
        specificity = _score(text, verb, target, target_kind)

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


def validate_batch(
    proposals: list[dict],
    mode: str = "admission",
) -> list[dict]:
    """Validate a list of proposals in one call.

    Each item in *proposals* must have at least a ``"text"`` key.
    Optional: ``"tags"`` (list of str).

    Returns a list of dicts (same shape as ``validate()``), each
    augmented with ``"index"`` pointing back to the input position.
    """
    results: list[dict] = []
    for i, item in enumerate(proposals):
        text = item.get("text", "")
        tags = item.get("tags") or []
        result = validate(text, tags, mode=mode)
        result["index"] = i
        results.append(result)
    return results


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
