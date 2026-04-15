"""seed_gate.py -- canonical specificity validator for seed proposals.

Consolidates ideas from 6 independent implementations (#12503, #12505,
#12507, #12511, #12521, #12530) and PRs #245, #246, #247 into one
validator that checks for an *action verb* plus a *concrete target*
(filename, tool name, path, or discussion reference).

Two public APIs -- pick whichever suits the call-site:

    # Dict API (used by propose_seed.py)
    from seed_gate import validate
    gate = validate(text, tags)
    if not gate["passed"]: ...

    # Dataclass API
    result = validate_seed(text, tags)  # -> SeedGateResult
    if not result.passed: ...

    # Bool convenience
    ok = passes_gate(text, tags)

    # Batch API (for purge_junk in propose_seed.py)
    br = validate_batch(proposals)      # -> BatchResult
    for item in br.junk_items: ...

    # Composable helpers
    verb   = find_verb(text)          # str | None
    target = find_target(text)        # (str, str) pair
    junk   = is_junk(text)            # str (reason) or empty
    score  = compute_score(text, verb, target, kind)  # float

    # Diagnostics
    bd  = score_breakdown(text, verb, target, kind)  # dict
    info = explain(text, tags)                        # dict
    v, pos = find_verb_with_position(text)            # (str|None, int)

Evolution log:
    PR #237  -- initial canonical validator (165 tests)
    PR #242  -- contract alignment with propose_seed.py
    PR #245  -- auto-discovered modules, two-tier artifacts, propose_seed wiring
    PR #248  -- consolidated #245/#246/#247: false-file filter, special
                files, known tools, question stems, batch API, smart
                lowercase, substring dedup.
    This frame -- 11 DevOps verbs (81 total), version/abbrev false-file
                  filters, env var + class targets, find_verb_with_position(),
                  imperative bonus, score_breakdown(), explain() diagnostics.
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

# 81 action verbs -- frozenset for O(1) lookup  (from #12503 + main repo + DevOps)
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
    # DevOps / lifecycle verbs (this frame)
    "configure", "scaffold", "bootstrap", "provision", "install",
    "deprecate", "archive", "rollback", "revert", "containerize", "expose",
})

# ---------------------------------------------------------------------------
# Target regex patterns (compiled once)
# ---------------------------------------------------------------------------

# File-like: foo.py, bar_baz.rs, my-lib.js, state/agents.json
FILE_RE = re.compile(r"\b[\w./-]*\w+\.\w{1,8}\b")

# False positives that FILE_RE catches (abbreviations with periods)
_FALSE_FILE_MATCHES: frozenset[str] = frozenset({
    "e.g", "i.e", "a.m", "p.m", "vs.",
    "ph.d", "u.s",
})

# Version patterns that FILE_RE catches: v1.2, v0.1, v2.0
_VERSION_RE = re.compile(r"^v\d+\.\d")

# Numbered reference patterns: no.5, fig.1, vol.2, ch.3
_NUMBERED_REF_RE = re.compile(r"^(?:no|fig|vol|ch|pt|sec)\.\d")

# Special files without extensions (PR #246)
SPECIAL_FILE_RE = re.compile(
    r"\b(?:Dockerfile|Makefile|Vagrantfile|Procfile|Gemfile|Rakefile"
    r"|README|AGENTS|CLAUDE|CONSTITUTION|CONTRIBUTING|LICENSE"
    r"|CHANGELOG|ROADMAP|MANIFEST|FEATURE_FREEZE)\b"
)

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

# Environment variable reference: $STATE_DIR, ${GITHUB_TOKEN}
# Requires 3+ uppercase chars to avoid $X or matching dollar amounts
ENV_VAR_RE = re.compile(r"\$\{?[A-Z][A-Z0-9_]{2,}\}?")

# Context-constrained PascalCase class: `SeedGateResult`, class MarsColony
# Only matches in code context (backticks or class keyword) to avoid
# false positives on normal capitalized words.
CLASS_CONTEXT_RE = re.compile(
    r"(?:`([A-Z][a-z]\w*(?:[A-Z][a-z]\w*)+)`"
    r"|class\s+([A-Z][a-z]\w*(?:[A-Z][a-z]\w*)+))"
)

# Known rappterbook tools -- precision-matched before generic TOOL_RE (PR #246)
KNOWN_TOOLS: frozenset[str] = frozenset({
    "state_io", "process_inbox", "process_issues", "propose_seed",
    "seed_gate", "compute_trending", "generate_feeds", "safe_commit",
    "content_loader", "content_engine", "feature_flags", "github_llm",
    "zion_autonomy", "heartbeat_audit", "pii_scan", "bundle",
    "compute_analytics", "reconcile_channels", "git_scrape_analytics",
})

_KNOWN_TOOL_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(t) for t in sorted(KNOWN_TOOLS)) + r")\b"
)

# Tags that exempt from the *target* requirement (still need a verb)
EXEMPT_TAGS: frozenset[str] = frozenset({
    "theme", "philosophy", "debate", "exploration", "story", "lore",
})

# Question stems -- map to implicit verbs for exempt-tag proposals (PR #247)
QUESTION_STEMS: dict[str, str] = {
    "what if": "explore",
    "how might": "design",
    "how could": "design",
    "how would": "design",
    "how should": "evaluate",
    "should we": "evaluate",
    "could we": "explore",
    "what would": "explore",
    "why not": "propose",
    "why do": "investigate",
    "why does": "investigate",
}

_QUESTION_STEM_RE = re.compile(
    r"^(?:" + "|".join(re.escape(s) for s in sorted(QUESTION_STEMS, key=len, reverse=True)) + r")\b",
    re.I,
)

# ---------------------------------------------------------------------------
# Junk / artifact detection (#12507 + main repo consolidation)
# ---------------------------------------------------------------------------

# Core junk signals -- note: ^[a-z] is handled separately to allow
# verb-starting lowercase text (the old pattern rejected "build seed_gate.py")
_JUNK_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^\s*[`|,()\-]"),
    re.compile(r"^\d+\.\s"),
    re.compile(r"^https?://"),
    re.compile(r"(?:TODO|FIXME|HACK)\b", re.I),
    re.compile(r"^\s*$"),
]

# Lowercase start -- only junk if first word is NOT an action verb
# and text does NOT start with a known file
_LOWERCASE_START_RE = re.compile(r"^[a-z]")
_FILE_START_RE = re.compile(r"^[\w./-]*\w+\.\w{1,8}\b")

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


@dataclasses.dataclass(frozen=True)
class BatchStats:
    """Aggregate stats for a batch validation run (PR #247)."""
    total: int
    passed: int
    failed: int
    junk: int

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0

    @property
    def junk_rate(self) -> float:
        return self.junk / self.total if self.total else 0.0

    def merge(self, other: "BatchStats") -> "BatchStats":
        return BatchStats(
            total=self.total + other.total,
            passed=self.passed + other.passed,
            failed=self.failed + other.failed,
            junk=self.junk + other.junk,
        )


@dataclasses.dataclass(frozen=True)
class BatchResult:
    """Result of validate_batch() -- separates junk from merely-failed."""
    stats: BatchStats
    passed_items: tuple
    failed_items: tuple
    junk_items: tuple


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_false_file_match(match_text: str) -> bool:
    """Return True if a FILE_RE match is actually a false positive.

    Catches abbreviations (e.g, i.e, Ph.D, U.S), version numbers (v1.2),
    and numbered references (no.5, fig.1, vol.2, ch.3).
    """
    lower = match_text.lower().rstrip(".")
    if lower in _FALSE_FILE_MATCHES:
        return True
    if _VERSION_RE.match(lower):
        return True
    if _NUMBERED_REF_RE.match(lower):
        return True
    return False


def _starts_with_verb(text: str) -> bool:
    """Return True if text starts with an action verb."""
    first_word = text.split()[0].lower() if text.split() else ""
    return first_word in ACTION_VERBS


def _starts_with_file(text: str) -> bool:
    """Return True if text starts with a file-like pattern."""
    m = _FILE_START_RE.match(text.strip())
    if m:
        return not _is_false_file_match(m.group())
    return False


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


def find_verb_with_position(text: str, limit: int = 0) -> tuple:
    """Return (verb, word_position) or (None, -1).

    Word position is 0-based index among whitespace-delimited words.
    Verbs in position 0-2 are considered imperative (command form).
    """
    search_text = text[:limit] if limit else text
    words = search_text.split()
    for i, word in enumerate(words):
        clean = re.sub(r"[^a-zA-Z]", "", word).lower()
        if clean in ACTION_VERBS:
            return clean, i
    return None, -1


def find_target(text: str) -> tuple:
    """Return (target_string, target_kind) or ('', '').

    Checks patterns in priority order -- most specific first.
    Rejects common abbreviations that FILE_RE would false-match.
    Module names only match in code-ish context (backticks, imports)
    to avoid false positives on generic nouns.
    """
    # File-like (with false-positive filtering)
    for m in FILE_RE.finditer(text):
        if not _is_false_file_match(m.group()):
            return m.group(), "file"
    # Special files (Dockerfile, Makefile, README, etc.)
    m = SPECIAL_FILE_RE.search(text)
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
    # Known tools first (precision), then generic TOOL_RE
    m = _KNOWN_TOOL_RE.search(text)
    if m:
        return m.group(), "tool"
    m = TOOL_RE.search(text)
    if m:
        return m.group(), "tool"
    # PascalCase class in code context -- before CLI so `SeedGateResult`
    # matches as class, not as a generic backtick CLI snippet
    m = CLASS_CONTEXT_RE.search(text)
    if m:
        cls_name = m.group(1) or m.group(2)
        return cls_name, "class"
    m = CLI_RE.search(text)
    if m:
        return m.group(), "cli"
    # Env vars after CLI: $STATE_DIR, ${GITHUB_TOKEN}
    m = ENV_VAR_RE.search(text)
    if m:
        return m.group(), "env_var"
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


def find_all_targets(text: str) -> list:
    """Return all concrete targets in *text* as a list of dicts.

    Each dict has keys: target (str), kind (str).
    Deduplicates by canonical form.
    """
    targets = []
    seen_canonical = set()

    def _add(match_text, kind):
        c = canonicalize_target(match_text)
        if c and c not in seen_canonical:
            seen_canonical.add(c)
            targets.append({"target": match_text, "kind": kind})

    for m in FILE_RE.finditer(text):
        if not _is_false_file_match(m.group()):
            _add(m.group(), "file")
    for m in SPECIAL_FILE_RE.finditer(text):
        _add(m.group(), "file")
    for m in PATH_RE.finditer(text):
        _add(m.group(), "path")
    for m in FUNC_RE.finditer(text):
        _add(m.group(), "func")
    for m in CHANNEL_RE.finditer(text):
        _add(m.group(), "channel")
    for m in _KNOWN_TOOL_RE.finditer(text):
        _add(m.group(), "tool")
    for m in TOOL_RE.finditer(text):
        _add(m.group(), "tool")
    for m in CLI_RE.finditer(text):
        _add(m.group(), "cli")
    for m in ENV_VAR_RE.finditer(text):
        _add(m.group(), "env_var")
    for m in DISCUSSION_RE.finditer(text):
        _add(m.group(), "discussion")
    if KNOWN_MODULES:
        for m in MODULE_CONTEXT_RE.finditer(text):
            bare = m.group().strip("`").replace("import ", "").replace("from ", "").strip()
            if bare in KNOWN_MODULES:
                _add(bare, "module")
    for m in CLASS_CONTEXT_RE.finditer(text):
        cls_name = m.group(1) or m.group(2)
        if cls_name:
            _add(cls_name, "class")
    for m in QUOTED_RE.finditer(text):
        _add(m.group(), "quoted")

    return targets


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
    # Smart lowercase handling: verb-starting or file-starting text is OK
    if _LOWERCASE_START_RE.match(stripped):
        if not _starts_with_verb(stripped) and not _starts_with_file(stripped):
            return "starts lowercase (not a verb or file)"
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
    t = target.strip("\"' `${}").rstrip(".")
    for prefix in ("src/", "tests/", "engine/", "state/", "docs/", "scripts/"):
        if t.startswith(prefix):
            t = t[len(prefix):]
            break
    dot = t.rfind(".")
    if dot > 0:
        t = t[:dot]
    return t.lower().strip()


def count_unique_targets(text: str) -> int:
    """Count distinct concrete targets in *text* after canonicalization.

    Uses substring-aware dedup: if 'seed_gate' and 'seed_gate.py'
    both appear, they count as one (PR #246).
    """
    raw_targets = []
    for pattern in (FILE_RE, PATH_RE, TOOL_RE, CLI_RE, DISCUSSION_RE, CHANNEL_RE, ENV_VAR_RE):
        for m in pattern.finditer(text):
            raw_targets.append(m.group())
    # Also count class targets from context regex
    for m in CLASS_CONTEXT_RE.finditer(text):
        cls_name = m.group(1) or m.group(2)
        if cls_name:
            raw_targets.append(cls_name)
    canonical = []
    for t in raw_targets:
        c = canonicalize_target(t)
        if c:
            canonical.append(c)
    # Substring dedup: remove shorter forms that are substrings of longer ones
    canonical_sorted = sorted(set(canonical), key=len, reverse=True)
    unique = []
    for c in canonical_sorted:
        if not any(c in u for u in unique):
            unique.append(c)
    return len(unique)


def compute_score(
    text: str,
    verb: str | None,
    target: str | None,
    target_kind: str,
) -> float:
    """Compute a 0.0-1.0 specificity score.

    Includes an imperative bonus for verbs in the first 3 words.
    """
    bd = score_breakdown(text, verb, target, target_kind)
    return bd["final_score"]


def score_breakdown(
    text: str,
    verb: str | None,
    target: str | None,
    target_kind: str,
) -> dict:
    """Return a decomposed scoring showing each component's contribution.

    Keys: verb_pts, target_pts, target_kind, length_pts, multi_target_pts,
    unique_targets, imperative_pts, raw_total, final_score.
    """
    verb_pts = 2.5 if verb else 0.0

    kind_scores = {
        "file": 4.0, "path": 3.5, "func": 3.0, "module": 3.0,
        "tool": 3.0, "cli": 3.0, "class": 2.5, "env_var": 2.0,
        "discussion": 2.0, "channel": 2.0, "quoted": 1.5,
    }
    target_pts = kind_scores.get(target_kind, 1.5) if target else 0.0

    words = text.split()
    length_pts = 0.0
    if len(words) >= 15:
        length_pts = 1.5
    elif len(words) >= 8:
        length_pts = 0.5

    unique = count_unique_targets(text)
    multi_target_pts = 1.0 if unique >= 2 else 0.0

    # Imperative bonus: verb in the first 3 words
    imperative_pts = 0.0
    if verb:
        _, pos = find_verb_with_position(text)
        if 0 <= pos <= 2:
            imperative_pts = 0.5

    raw_total = verb_pts + target_pts + length_pts + multi_target_pts + imperative_pts
    final_score = min(raw_total / 10.0, 1.0)

    return {
        "verb_pts": verb_pts,
        "target_pts": target_pts,
        "target_kind": target_kind or "",
        "length_pts": length_pts,
        "multi_target_pts": multi_target_pts,
        "imperative_pts": imperative_pts,
        "unique_targets": unique,
        "raw_total": raw_total,
        "final_score": final_score,
    }


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

    # -- Question stem inference (exempt tags only) ---
    if not verb and is_exempt:
        m = _QUESTION_STEM_RE.match(text.strip())
        if m:
            verb = QUESTION_STEMS.get(m.group().lower())

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


def validate(text: str, tags: list = None, mode: str = "admission") -> dict:
    """Dict API -- the shape expected by propose_seed.py."""
    return validate_seed(text, tags, mode=mode).to_dict()


def passes_gate(text: str, tags: list = None, mode: str = "admission") -> bool:
    """Convenience: return True iff the proposal passes the gate."""
    return validate_seed(text, tags, mode=mode).passed


def explain(text: str, tags: list = None) -> dict:
    """Return a structured diagnostic dict for *text*.

    Includes everything from validate() plus: verb_position,
    verb_imperative, primary_target, primary_target_kind, all_targets,
    unique_target_count, exempt_tags, stem_verb, word_count, soft_artifact,
    and the full score_breakdown.

    Always consistent with validate() -- calls it internally.
    """
    tags = tags or []
    result = validate_seed(text, tags)
    tag_set = frozenset(t.lower().strip() for t in tags)

    # Junk short-circuit
    junk_reason = is_junk(text) or None
    if result.junk:
        return {
            "passed": False,
            "verb": None,
            "verb_position": -1,
            "verb_imperative": False,
            "primary_target": None,
            "primary_target_kind": "",
            "all_targets": [],
            "score": 0.0,
            "junk_reason": junk_reason,
            "unique_target_count": 0,
            "exempt_tags": sorted(tag_set & EXEMPT_TAGS),
            "stem_verb": None,
            "word_count": len(text.split()),
            "soft_artifact": is_soft_artifact(text),
        }

    verb, verb_pos = find_verb_with_position(text)
    target, target_kind = find_target(text)
    all_targets = find_all_targets(text)

    # Question stem inference
    stem_verb = None
    if not verb and (tag_set & EXEMPT_TAGS):
        m = _QUESTION_STEM_RE.match(text.strip())
        if m:
            stem_verb = QUESTION_STEMS.get(m.group().lower())

    # Use the stem verb if no direct verb found
    effective_verb = verb or stem_verb

    return {
        "passed": result.passed,
        "verb": effective_verb,
        "verb_position": verb_pos,
        "verb_imperative": 0 <= verb_pos <= 2,
        "primary_target": target or None,
        "primary_target_kind": target_kind,
        "all_targets": all_targets,
        "score": result.score,
        "junk_reason": junk_reason,
        "unique_target_count": count_unique_targets(text),
        "exempt_tags": sorted(tag_set & EXEMPT_TAGS),
        "stem_verb": stem_verb,
        "word_count": len(text.split()),
        "soft_artifact": is_soft_artifact(text),
    }


def validate_batch(
    proposals: list,
    tags: list = None,
    mode: str = "admission",
) -> BatchResult:
    """Validate a batch of proposals; separate junk from merely-failed.

    Returns a BatchResult with stats + categorized items so callers
    (like propose_seed.purge_junk) can treat junk and vague-but-salvageable
    proposals differently.
    """
    passed_items = []
    failed_items = []
    junk_items = []

    for text in proposals:
        result = validate(text, tags, mode=mode)
        if result["junk"]:
            junk_items.append((text, result))
        elif result["passed"]:
            passed_items.append((text, result))
        else:
            failed_items.append((text, result))

    stats = BatchStats(
        total=len(proposals),
        passed=len(passed_items),
        failed=len(failed_items),
        junk=len(junk_items),
    )
    return BatchResult(
        stats=stats,
        passed_items=tuple(passed_items),
        failed_items=tuple(failed_items),
        junk_items=tuple(junk_items),
    )


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
