"""seed_gate.py -- canonical specificity validator for seed proposals.

Consolidates ideas from 6 independent implementations (#12503, #12505,
#12507, #12511, #12521, #12530) into one validator that checks for an
*action verb* plus a *concrete target* (filename, tool name, path, or
discussion reference).

Three public APIs -- pick whichever suits the call-site:

    # Dict API (used by propose_seed.py)
    from seed_gate import validate as validate_seed
    gate = validate_seed(text, tags)        # -> dict
    if not gate["passed"]: ...

    # Dataclass API
    result = validate_seed(text, tags)  # -> SeedGateResult
    if not result.passed: ...

    # Bool convenience
    ok = passes_gate(text, tags)

    # Batch API (for purge_junk workflows -- separates junk from failed)
    batch = validate_batch([("Build foo.py", []), ("vague idea", [])])
    batch.stats             # BatchStats(total=2, passed=1, failed=1, junk=0)
    batch.junk_items        # only actual junk (fragments, artifacts)
    batch.failed_items      # failed specificity but not junk

    # Standalone helpers
    verb = find_action_verb("Build seed_gate.py")  # -> "build"
    target, kind = find_concrete_target(text)       # -> ("seed_gate.py", "file")
    reason = detect_junk(text)                      # -> None or "reason string"
    score = compute_score(text, verb, target, kind) # -> 0.0-1.0

Evolution log:
    PR #237  -- initial canonical validator (165 tests)
    PR #242  -- contract alignment with propose_seed.py
    This frame -- FILE_RE false positive filter, batch validation API,
                  BatchStats/BatchResult dataclasses, stress tests.
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

# 56 action verbs -- frozenset for O(1) lookup  (from #12503, expanded)
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
})

# ---------------------------------------------------------------------------
# Target regex patterns (compiled once)
# ---------------------------------------------------------------------------

# File-like: foo.py, bar_baz.rs, my-lib.js, state/agents.json
FILE_RE = re.compile(
    r"\b[\w./-]*\w+\.\w{1,8}\b"
)

# Post-match filter: common abbreviations that look like filenames.
# Applied after FILE_RE matches to reject false positives in prose.
_FALSE_FILE_MATCHES: frozenset[str] = frozenset({
    "e.g", "i.e", "etc", "vs", "al", "cf", "no",
    "dr", "mr", "ms", "jr", "sr", "st",
})

# Repo-aware paths: src/, tests/, engine/, state/, docs/
PATH_RE = re.compile(
    r"(?:(?:src|tests|engine|state|docs|api|scripts|sdk|zion)/[\w_./-]+)"
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

# Quoted specifics: "some specific thing" or \'some specific thing\'
QUOTED_RE = re.compile(
    r"""(?:"[^"]{3,60}"|'[^']{3,60}')"""
)

# Context-sensitive module reference: module in backticks, import statement
MODULE_CONTEXT_RE = re.compile(
    r"(?:`[\w_]+`|import\s+[\w_]+|from\s+[\w_]+)"
)

# Tags that exempt proposals from the *target* requirement (still need a verb)
EXEMPT_TAGS: frozenset[str] = frozenset({
    "theme", "philosophy", "debate", "exploration", "story", "lore",
})

# ---------------------------------------------------------------------------
# Junk / artifact detection (#12507 + main repo consolidation)
# ---------------------------------------------------------------------------

_JUNK_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^\s*[`|,()\-]"),              # starts with junk punctuation
    re.compile(r"^[a-z]"),                        # starts lowercase (fragment)
    re.compile(r"^\d+\.\s"),                   # numbered list item
    re.compile(r"^https?://"),                     # bare URL
    re.compile(r"(?:TODO|FIXME|HACK)\b", re.I),  # leftover comment
    re.compile(r"^\s*$"),                         # blank / whitespace-only
]

_JUNK_EXCEPTION_RE = re.compile(r"^run_\w")
_FILE_START_RE = re.compile(r"^[\w][\w.-]*\.[a-zA-Z]{1,8}\b")

_HARD_ARTIFACT_SIGNALS: tuple[str, ...] = (
    "parser grabbed", "parsing artifact", "outside that grammar",
    "the fragment was", "`) and ", "` has `",
)

_SOFT_ARTIFACT_SIGNALS: tuple[str, ...] = (
    "the regex", "the parser", "the fragment", "substring",
    "` and `", "` is ",
)

# ---------------------------------------------------------------------------
# Question-stem intent (for theme/philosophy exempt proposals)
# ---------------------------------------------------------------------------

QUESTION_STEMS: dict[str, str] = {
    "what if": "explore",
    "how might": "design",
    "how could": "design",
    "could we": "consider",
    "what would": "evaluate",
    "imagine if": "explore",
}

_QUESTION_STEM_RE = re.compile(
    r"^(" + "|".join(re.escape(s) for s in sorted(QUESTION_STEMS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Auto-discovered modules (generated from src/*.py at import time)
# ---------------------------------------------------------------------------

def _discover_modules() -> frozenset[str]:
    """Scan src/ for .py files and return module names as a frozenset."""
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
# Dataclass results
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
        """Alias for verb_found, always str."""
        return self.verb_found or ""

    @property
    def target(self) -> str:
        """Alias for target_found, always str."""
        return self.target_found or ""

    def to_dict(self) -> dict:
        """Return the dict shape that propose_seed.py expects."""
        return {
            "passed": self.passed,
            "reasons": list(self.reasons),
            "score": self.score,
            "verb_found": self.verb_found,
            "target_found": self.target_found,
            "junk": self.junk,
        }


@dataclasses.dataclass(frozen=True)
class BatchStats:
    """Aggregate statistics from a batch validation run.

    Separates *failed* (missing verb/target) from *junk* (fragments,
    artifacts) -- the key distinction that purge_junk() needs.
    """

    total: int
    passed: int
    failed: int
    junk: int

    @property
    def pass_rate(self) -> float:
        """Fraction of proposals that passed (0.0-1.0)."""
        return self.passed / self.total if self.total else 0.0

    @property
    def junk_rate(self) -> float:
        """Fraction of proposals that were junk (0.0-1.0)."""
        return self.junk / self.total if self.total else 0.0

    def merge(self, other: "BatchStats") -> "BatchStats":
        """Combine two stats objects (e.g. from separate runs)."""
        return BatchStats(
            total=self.total + other.total,
            passed=self.passed + other.passed,
            failed=self.failed + other.failed,
            junk=self.junk + other.junk,
        )


@dataclasses.dataclass(frozen=True)
class BatchResult:
    """Result of validate_batch(): per-item results + aggregate stats."""

    results: tuple
    stats: BatchStats

    @property
    def junk_items(self) -> list:
        """Return only the results flagged as junk."""
        return [r for r in self.results if r.junk]

    @property
    def failed_items(self) -> list:
        """Return results that failed but are NOT junk."""
        return [r for r in self.results if not r.passed and not r.junk]

    def to_dicts(self) -> list:
        """Return all results as dicts (same shape as validate())."""
        return [r.to_dict() for r in self.results]


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
    """Return (target_string, target_kind) or (\'\'\', \'\'\'\').

    Checks patterns in priority order -- most specific first.
    Rejects common abbreviations that FILE_RE would false-match.
    """
    for m in FILE_RE.finditer(text):
        candidate = m.group()
        base = candidate.rstrip(".")
        if base.lower() not in _FALSE_FILE_MATCHES:
            return candidate, "file"
    m = PATH_RE.search(text)
    if m:
        return m.group(), "path"
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
    if _FILE_START_RE.match(stripped):
        return ""
    for pat in _JUNK_PATTERNS:
        if pat.search(stripped):
            return "junk signal: %r" % pat.pattern
    head = stripped[:80].lower()
    for signal in _HARD_ARTIFACT_SIGNALS:
        if signal in head:
            return "artifact detected: %r" % signal
    return ""


def _is_soft_artifact(text: str) -> bool:
    """Return True if text contains soft artifact signals."""
    head = text.strip()[:80].lower()
    return any(signal in head for signal in _SOFT_ARTIFACT_SIGNALS)


def _canonicalize_target(target: str) -> str:
    """Normalize a target string for dedup."""
    t = target.strip("\"\'\'` ")
    for prefix in ("src/", "tests/", "engine/", "state/", "docs/", "scripts/"):
        if t.startswith(prefix):
            t = t[len(prefix):]
            break
    dot = t.rfind(".")
    if dot > 0:
        t = t[:dot]
    return t.lower().strip()


def _count_unique_targets(text: str) -> int:
    """Count distinct concrete targets in *text* after canonicalization."""
    raw_targets: list[str] = []
    for pattern in (FILE_RE, PATH_RE, TOOL_RE, CLI_RE, DISCUSSION_RE, CHANNEL_RE):
        for m in pattern.finditer(text):
            raw_targets.append(m.group())
    canonical: set[str] = set()
    for t in raw_targets:
        c = _canonicalize_target(t)
        if c:
            canonical.add(c)
    return len(canonical)


def _score(text: str, verb: str, target: str, target_kind: str) -> float:
    """Compute a 0.0-1.0 specificity score."""
    raw = 0
    if verb:
        raw += 3
    if target:
        kind_scores = {
            "file": 4, "path": 4, "tool": 3, "module": 3, "cli": 3,
            "discussion": 2, "channel": 2, "quoted": 1,
        }
        raw += kind_scores.get(target_kind, 1)
    words = text.split()
    if len(words) >= 8:
        raw += 1
    if len(words) >= 15:
        raw += 1
    unique = _count_unique_targets(text)
    if unique >= 2:
        raw += 1
    return min(raw / 10.0, 1.0)


def find_question_intent(text: str) -> tuple:
    """Return (stem, mapped_verb) if text opens with a question stem."""
    m = _QUESTION_STEM_RE.match(text.strip())
    if m:
        stem = m.group(1).lower()
        return stem, QUESTION_STEMS.get(stem, "")
    return "", ""


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

    junk_limit = 60 if mode == "purge" else 0
    junk_reason = _is_junk(text, limit=junk_limit)
    is_junk_flag = bool(junk_reason)

    if is_junk_flag:
        return SeedGateResult(
            passed=False, reasons=(junk_reason,), score=0.0,
            verb_found=None, target_found=None, junk=True,
        )

    verb_limit = 200 if mode == "purge" else 0
    verb = _detect_verb(text, limit=verb_limit)

    if is_exempt and not verb:
        _stem, mapped_verb = find_question_intent(text)
        if mapped_verb:
            verb = mapped_verb

    target, target_kind = _detect_target(text)

    has_soft_artifact = _is_soft_artifact(text)
    if has_soft_artifact and not (verb and target) and not is_exempt:
        return SeedGateResult(
            passed=False,
            reasons=("soft artifact signal without redeeming verb+target",),
            score=0.0, verb_found=verb or None,
            target_found=target or None, junk=True,
        )

    if mode == "purge":
        passed = True
        specificity = 0.5
    else:
        passed = bool(verb) and (bool(target) or is_exempt)
        specificity = _score(text, verb, target, target_kind)

    reasons = []
    if not passed:
        if not verb:
            reasons.append("No action verb found")
        if not target and not is_exempt:
            reasons.append("No concrete target (filename, tool, or reference)")

    return SeedGateResult(
        passed=passed, reasons=tuple(reasons), score=specificity,
        verb_found=verb or None, target_found=target or None, junk=False,
    )


def validate(
    text: str,
    tags: list = None,
    mode: str = "admission",
) -> dict:
    """Dict API -- the shape expected by propose_seed.py."""
    return validate_seed(text, tags, mode=mode).to_dict()


def passes_gate(
    text: str,
    tags: list = None,
    mode: str = "admission",
) -> bool:
    """Convenience: return True iff the proposal passes the gate."""
    return validate_seed(text, tags, mode=mode).passed


def validate_batch(
    proposals: list,
    mode: str = "admission",
) -> BatchResult:
    """Validate multiple proposals and return structured results + stats.

    Parameters
    ----------
    proposals : list of (text, tags) tuples
        Each element is (proposal_text, tag_list).
    mode : str
        "admission" or "purge" -- applied to all items.

    Returns
    -------
    BatchResult
        .results -- tuple of SeedGateResult per input.
        .stats -- BatchStats with pass/fail/junk counts.
        .junk_items -- only actual junk (NOT merely failed).
        .failed_items -- failed specificity but not junk.
    """
    results: list = []
    passed_count = 0
    junk_count = 0

    for text, tags in proposals:
        r = validate_seed(text, tags, mode=mode)
        results.append(r)
        if r.passed:
            passed_count += 1
        elif r.junk:
            junk_count += 1

    total = len(results)
    failed_count = total - passed_count - junk_count

    return BatchResult(
        results=tuple(results),
        stats=BatchStats(
            total=total, passed=passed_count,
            failed=failed_count, junk=junk_count,
        ),
    )


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def _cli() -> None:  # pragma: no cover
    """python -m seed_gate \'Build seed_gate.py validator\'"""
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
