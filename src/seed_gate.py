"""seed_gate.py -- canonical specificity validator for seed proposals.

Consolidates ideas from 6 independent implementations (#12503, #12505,
#12507, #12511, #12521, #12530) into one validator that checks for an
*action verb* plus a *concrete target* (filename, tool name, path, or
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
    verb   = find_verb(text)          # str | None
    target = find_target(text)        # (str, str) pair
    junk   = is_junk(text)            # str (reason) or empty
    score  = compute_score(text, verb, target, kind)  # float

Evolution log:
    PR #237  -- initial canonical validator (165 tests)
    PR #242  -- contract alignment with propose_seed.py
    This frame -- auto-discovered KNOWN_MODULES, two-tier artifact
                  signals, path patterns, target canonicalization,
                  propose_seed.py wiring.
"""
from __future__ import annotations

import dataclasses
import os
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
    "consolidate", "decode", "establish", "execute", "extend",
    "instrument", "measure", "merge", "remove", "render",
    "review", "run", "score", "validate",
})

# ---------------------------------------------------------------------------
# Target regex patterns (compiled once)
# ---------------------------------------------------------------------------

# File-like: foo.py, bar_baz.rs, my-lib.js, state/agents.json
FILE_RE = re.compile(r"\b[\w./-]*\w+\.\w{1,8}\b")

# Repo-aware paths: src/, tests/, engine/, state/, docs/  (from main repo)
PATH_RE = re.compile(
    r"(?:(?:src|tests|engine|state|docs|api|scripts|sdk|zion)/[\w_./-]+)"
)

# Function call: validate(), compute_score()  (from main repo)
FUNC_RE = re.compile(r"\b[\w_]+\(\)")

# Tool / module name: snake_case or kebab-case with 2+ segments
TOOL_RE = re.compile(r"\b[a-z][a-z0-9]*(?:[_-][a-z0-9]+)+\b")

# CLI-ish: `some_command`, --flag, -f
CLI_RE = re.compile(r"(?:`[^`]+`|--[a-z][\w-]+\b|-[a-zA-Z]\b)")

# Discussion reference: #NNN (3+ digits)  (from #12505)
DISCUSSION_RE = re.compile(r"#(\d{3,})\b")

# Channel reference: r/channel-name or c/channel-name
CHANNEL_RE = re.compile(r"\b[rc]/[a-z][a-z0-9_-]+\b")

# Quoted specifics: "some specific thing"
QUOTED_RE = re.compile(r"""(?:"[^"]{3,60}"|'[^']{3,60}')""")

# Context-sensitive module reference: `module`, import module, from module
MODULE_CONTEXT_RE = re.compile(r"(?:`[\w_]+`|import\s+[\w_]+|from\s+[\w_]+)")

# Tags that exempt from the *target* requirement (still need a verb)
EXEMPT_TAGS: frozenset[str] = frozenset({
    "theme", "philosophy", "debate", "exploration", "story", "lore",
})

# ---------------------------------------------------------------------------
# Junk / artifact detection (#12507 + main repo consolidation)
# ---------------------------------------------------------------------------

_JUNK_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^\s*[`|,()\-]"),
    re.compile(r"^[a-z]"),
    re.compile(r"^\d+\.\s"),
    re.compile(r"^https?://"),
    re.compile(r"(?:TODO|FIXME|HACK)\b", re.I),
    re.compile(r"^\s*$"),
]

_JUNK_EXCEPTION_RE = re.compile(r"^run_\w")

# Two-tier artifact signals (rubber-duck advised split):
# Hard: always fail -- parser/extraction garbage
_HARD_ARTIFACT_SIGNALS: tuple[str, ...] = (
    "parser grabbed", "parsing artifact", "outside that grammar",
    "the fragment was", "`) and ", "` has `",
)

# Soft: only fail when no verb+target to redeem
_SOFT_ARTIFACT_SIGNALS: tuple[str, ...] = (
    "the regex", "the parser", "the fragment", "substring",
    "` and `", "` is ",
)

# Backward-compatible flat tuple (union of both tiers)
ARTIFACT_SIGNALS: tuple[str, ...] = _HARD_ARTIFACT_SIGNALS + _SOFT_ARTIFACT_SIGNALS

# ---------------------------------------------------------------------------
# Auto-discovered modules from src/*.py  (avoids hand-maintaining 100+ list)
# ---------------------------------------------------------------------------

def _discover_modules() -> frozenset[str]:
    """Scan src/ for .py files and return multi-segment module names."""
    src_dir = Path(__file__).resolve().parent
    if not src_dir.is_dir():
        return frozenset()
    modules: set[str] = set()
    for p in src_dir.glob("*.py"):
        name = p.stem
        if name.startswith("__") or name.startswith("test_") or name.startswith("run_"):
            continue
        if "_" in name:
            modules.add(name)
    return frozenset(modules)


KNOWN_MODULES: frozenset[str] = _discover_modules()

# ---------------------------------------------------------------------------
# Dataclass result
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class SeedGateResult:
    """Immutable result of seed-gate validation."""

    passed: bool
    reasons: tuple
    score: float
    verb_found: object
    target_found: object
    junk: bool

    @property
    def verb(self) -> str:
        return self.verb_found or ""

    @property
    def target(self) -> str:
        return self.target_found or ""

    def to_dict(self) -> dict:
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
    """Return the first action verb found in *text*, or None."""
    search_text = text[:limit] if limit else text
    words = re.findall(r"[a-zA-Z]+", search_text.lower())
    for word in words:
        if word in ACTION_VERBS:
            return word
    return None


def find_target(text: str) -> tuple[str, str]:
    """Return (target_string, target_kind) or ('', '').

    Checks patterns in priority order -- most specific first.
    Module names only match in code-ish context (backticks, imports)
    to avoid false positives on generic nouns.
    """
    m = FILE_RE.search(text)
    if m:
        return m.group(), "file"
    m = PATH_RE.search(text)
    if m:
        return m.group(), "path"
    m = FUNC_RE.search(text)
    if m:
        return m.group(), "func"
    # Channel before tool: r/mars-engineering contains mars-engineering
    m = CHANNEL_RE.search(text)
    if m:
        return m.group(), "channel"
    m = TOOL_RE.search(text)
    if m:
        return m.group(), "tool"
    m = CLI_RE.search(text)
    if m:
        return m.group(), "cli"
    m = DISCUSSION_RE.search(text)
    if m:
        return m.group(), "discussion"
    # Module context: only if the name is a known Mars colony module
    if KNOWN_MODULES:
        m = MODULE_CONTEXT_RE.search(text)
        if m:
            match_text = m.group()
            bare = match_text.strip("`").replace("import ", "").replace("from ", "").strip()
            if bare in KNOWN_MODULES:
                return bare, "module"
    m = QUOTED_RE.search(text)
    if m:
        return m.group(), "quoted"
    return "", ""


def is_junk(text: str, limit: int = 0) -> str:
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
    # Hard artifact signals -- always fail (first 80 chars)
    head = stripped[:80].lower()
    for signal in _HARD_ARTIFACT_SIGNALS:
        if signal in head:
            return "artifact detected: %r" % signal
    return ""


def is_soft_artifact(text: str) -> bool:
    """Return True if text contains soft artifact signals."""
    head = text.strip()[:80].lower()
    return any(signal in head for signal in _SOFT_ARTIFACT_SIGNALS)


def canonicalize_target(target: str) -> str:
    """Normalize a target string for dedup: strip path prefix, extension, quotes."""
    t = target.strip("\"'` ")
    for prefix in ("src/", "tests/", "engine/", "state/", "docs/", "scripts/"):
        if t.startswith(prefix):
            t = t[len(prefix):]
            break
    dot = t.rfind(".")
    if dot > 0:
        t = t[:dot]
    return t.lower().strip()


def count_unique_targets(text: str) -> int:
    """Count distinct concrete targets in *text* after canonicalization."""
    raw_targets: list[str] = []
    for pattern in (FILE_RE, PATH_RE, TOOL_RE, CLI_RE, DISCUSSION_RE, CHANNEL_RE):
        for m in pattern.finditer(text):
            raw_targets.append(m.group())
    canonical: set[str] = set()
    for t in raw_targets:
        c = canonicalize_target(t)
        if c:
            canonical.add(c)
    return len(canonical)


def compute_score(
    text: str,
    verb: str | None,
    target: str | None,
    target_kind: str,
) -> float:
    """Compute a 0.0-1.0 specificity score."""
    raw = 0.0
    if verb:
        raw += 2.5
    if target:
        kind_scores = {
            "file": 4.0, "path": 3.5, "func": 3.0, "module": 3.0,
            "tool": 3.0, "cli": 3.0,
            "discussion": 2.0, "channel": 2.0, "quoted": 1.5,
        }
        raw += kind_scores.get(target_kind, 1.5)
    words = text.split()
    if len(words) >= 8:
        raw += 0.5
    if len(words) >= 15:
        raw += 0.5
    unique = count_unique_targets(text)
    if unique >= 2:
        raw += 1.0
    return min(raw / 10.0, 1.0)


# Backward-compat aliases
_detect_verb = find_verb
_detect_target = find_target
_is_junk = is_junk
_is_soft_artifact = is_soft_artifact
_canonicalize_target = canonicalize_target
_count_unique_targets = count_unique_targets
_score = lambda text, verb, target, kind: compute_score(text, verb, target, kind)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_seed(
    text: str,
    tags: list = None,
    *,
    mode: str = "admission",
) -> SeedGateResult:
    """Validate a seed proposal and return a *SeedGateResult*."""
    tags = tags or []
    tag_set = frozenset(t.lower().strip() for t in tags)
    is_exempt = bool(tag_set & EXEMPT_TAGS)

    # -- Junk check (hard fail) ---
    junk_limit = 60 if mode == "purge" else 0
    junk_reason = is_junk(text, limit=junk_limit)
    if junk_reason:
        return SeedGateResult(
            passed=False, reasons=(junk_reason,), score=0.0,
            verb_found=None, target_found=None, junk=True,
        )

    # -- Verb + target ---
    verb_limit = 200 if mode == "purge" else 0
    verb = find_verb(text, limit=verb_limit)
    target, target_kind = find_target(text)

    # -- Soft artifact check ---
    if is_soft_artifact(text) and not (verb and target) and not is_exempt:
        return SeedGateResult(
            passed=False,
            reasons=("soft artifact signal without redeeming verb+target",),
            score=0.0, verb_found=verb, target_found=target or None,
            junk=True,
        )

    # -- Decision ---
    if mode == "purge":
        passed = True
        specificity = 0.5
    else:
        passed = bool(verb) and (bool(target) or is_exempt)
        specificity = compute_score(text, verb, target, target_kind)

    reasons: list[str] = []
    if not passed:
        if not verb:
            reasons.append("No action verb found")
        if not target and not is_exempt:
            reasons.append("No concrete target (filename, tool, or reference)")

    return SeedGateResult(
        passed=passed, reasons=tuple(reasons), score=specificity,
        verb_found=verb or None, target_found=target or None, junk=False,
    )


def validate(text: str, tags: list = None, mode: str = "admission") -> dict:
    """Dict API -- the shape expected by propose_seed.py."""
    return validate_seed(text, tags, mode=mode).to_dict()


def passes_gate(text: str, tags: list = None, mode: str = "admission") -> bool:
    """Convenience: return True iff the proposal passes the gate."""
    return validate_seed(text, tags, mode=mode).passed


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def _cli() -> None:  # pragma: no cover
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
