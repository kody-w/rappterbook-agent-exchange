"""seed_gate.py -- canonical specificity validator for seed proposals.

Consolidates ideas from 6 independent implementations (#12503, #12505,
#12507, #12511, #12521, #12530) and PRs #245, #246, #247 into one
validator that checks for an *action verb* plus a *concrete target*
(filename, tool name, path, or discussion reference).

Two public APIs -- pick whichever suits the call-site:

    # Dict API (used by propose_seed.py)
    from seed_gate import validate
    gate = validate(text, tags)         # -> dict
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
    verb   = find_verb(text)          # str | None (handles inflected forms)
    target = find_target(text)        # (str, str) pair
    junk   = is_junk(text)            # str (reason) or empty
    score  = compute_score(text, verb, target, kind)  # float
    hints  = suggest(text, tags)      # list[str] -- actionable hints

    result.confidence                   # 'high' | 'medium' | 'low' | 'none'

Evolution log:
    PR #237  -- initial canonical validator (165 tests)
    PR #242  -- contract alignment with propose_seed.py
    PR #245  -- auto-discovered modules, two-tier artifacts, propose_seed wiring
    PR #248  -- consolidated #245/#246/#247: false-file filter, special
                  files, known tools, question stems, batch API, smart
                  lowercase, substring dedup.
    This frame -- phrasal verbs, tag-implied verbs (#12530), advisory
                  labeling (#12507), rich match lists (#12521), case-
                  insensitive module matching, CONST_RE in find_target.
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

# 95 action verbs -- frozenset for O(1) lookup
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
    "configure", "scaffold", "bootstrap", "provision",
    "automate", "archive", "inject", "normalize",
    "update", "delete", "enable", "disable",
    "install", "deprecate", "rewrite", "standardize",
    "containerize", "expose", "wrap", "stub", "mock",
    "isolate", "define", "declare", "register",
    "secure", "clean", "schedule", "cache", "publish",
    "annotate", "version", "backup", "package",
})

# Phrasal verbs -- two-word engineering verbs (#12521)
PHRASAL_VERBS: dict[str, str] = {
    "set up": "set up", "roll back": "roll back", "wire up": "wire up",
    "clean up": "clean up", "spin up": "spin up", "tear down": "tear down",
    "break down": "break down", "plug in": "plug in", "hook up": "hook up",
    "scale up": "scale up", "scale down": "scale down", "lock down": "lock down",
    "back up": "back up", "phase out": "phase out", "ramp up": "ramp up",
    "opt in": "opt in", "opt out": "opt out", "switch out": "switch out",
    "swap out": "swap out", "flesh out": "flesh out", "stub out": "stub out",
    "carve out": "carve out", "pull in": "pull in",
}

_PHRASAL_FIRST: dict[str, dict[str, str]] = {}
for _phrase, _canonical in PHRASAL_VERBS.items():
    _first, _second = _phrase.split()
    _PHRASAL_FIRST.setdefault(_first, {})[_second] = _canonical


# ---------------------------------------------------------------------------
# Verb normalization -- explicit map + safe suffix rules (PR #252 idea)
# ---------------------------------------------------------------------------

# Explicit irregular/tricky forms that suffix rules can't handle
_VERB_NORM_MAP: dict[str, str] = {
    # -ing forms
    "building": "build", "creating": "create", "designing": "design",
    "developing": "develop", "implementing": "implement", "writing": "write",
    "adding": "add", "deploying": "deploy", "launching": "launch",
    "shipping": "ship", "releasing": "release", "refactoring": "refactor",
    "optimizing": "optimize", "improving": "improve", "upgrading": "upgrade",
    "migrating": "migrate", "porting": "port", "wiring": "wire",
    "connecting": "connect", "hooking": "hook", "fixing": "fix",
    "debugging": "debug", "patching": "patch", "resolving": "resolve",
    "repairing": "repair", "testing": "test", "profiling": "profile",
    "auditing": "audit", "scanning": "scan", "linting": "lint",
    "generating": "generate", "computing": "compute", "simulating": "simulate",
    "modeling": "model", "training": "train", "parsing": "parse",
    "extracting": "extract", "converting": "convert", "compiling": "compile",
    "monitoring": "monitor", "tracking": "track", "logging": "log",
    "alerting": "alert", "mapping": "map", "exploring": "explore",
    "investigating": "investigate", "analyzing": "analyze",
    "evaluating": "evaluate", "assessing": "assess", "consolidating": "consolidate",
    "merging": "merge", "removing": "remove", "rendering": "render",
    "reviewing": "review", "running": "run", "scoring": "score",
    "validating": "validate", "configuring": "configure", "automating": "automate",
    "archiving": "archive", "injecting": "inject", "normalizing": "normalize",
    "updating": "update", "deleting": "delete", "enabling": "enable",
    "disabling": "disable", "installing": "install", "rewriting": "rewrite",
    "standardizing": "standardize", "containerizing": "containerize",
    "exposing": "expose", "wrapping": "wrap", "stubbing": "stub",
    "mocking": "mock", "isolating": "isolate", "defining": "define",
    "declaring": "declare", "registering": "register", "securing": "secure",
    "cleaning": "clean", "scheduling": "schedule", "caching": "cache",
    "publishing": "publish", "annotating": "annotate", "packaging": "package",
    "benchmarking": "benchmark", "documenting": "document",
    # Past tense -ed
    "built": "build", "created": "create", "designed": "design",
    "developed": "develop", "implemented": "implement", "deployed": "deploy",
    "launched": "launch", "shipped": "ship", "released": "release",
    "refactored": "refactor", "optimized": "optimize", "improved": "improve",
    "upgraded": "upgrade", "migrated": "migrate", "wired": "wire",
    "connected": "connect", "hooked": "hook", "fixed": "fix",
    "debugged": "debug", "patched": "patch", "resolved": "resolve",
    "repaired": "repair", "tested": "test", "profiled": "profile",
    "audited": "audit", "scanned": "scan", "linted": "lint",
    "generated": "generate", "computed": "compute", "simulated": "simulate",
    "modeled": "model", "trained": "train", "parsed": "parse",
    "extracted": "extract", "converted": "convert", "compiled": "compile",
    "monitored": "monitor", "tracked": "track", "logged": "log",
    "mapped": "map", "merged": "merge", "removed": "remove",
    "rendered": "render", "reviewed": "review", "scored": "score",
    "validated": "validate", "configured": "configure", "automated": "automate",
    "archived": "archive", "injected": "inject", "normalized": "normalize",
    "updated": "update", "deleted": "delete", "enabled": "enable",
    "disabled": "disable", "installed": "install", "exposed": "expose",
    "wrapped": "wrap", "stubbed": "stub", "mocked": "mock",
    "isolated": "isolate", "defined": "define", "declared": "declare",
    "registered": "register", "secured": "secure", "cleaned": "clean",
    "scheduled": "schedule", "cached": "cache", "published": "publish",
    "benchmarked": "benchmark", "documented": "document", "simplified": "simplify",
    # Third-person -s/-es
    "builds": "build", "creates": "create", "deploys": "deploy",
    "tests": "test", "fixes": "fix", "runs": "run", "adds": "add",
    "ships": "ship", "wires": "wire", "hooks": "hook", "logs": "log",
    "scans": "scan", "tracks": "track", "merges": "merge",
    "updates": "update", "deletes": "delete", "exposes": "expose",
    "wraps": "wrap", "stubs": "stub", "mocks": "mock",
    "cleans": "clean", "scores": "score", "defines": "define",
    "optimizes": "optimize", "patches": "patch", "launches": "launch",
    "analyzes": "analyze", "configures": "configure", "generates": "generate",
    "compiles": "compile", "validates": "validate", "resolves": "resolve",
    "explores": "explore", "removes": "remove", "enables": "enable",
    "disables": "disable", "migrates": "migrate", "converts": "convert",
    "simulates": "simulate", "automates": "automate", "publishes": "publish",
    "consolidates": "consolidate", "evaluates": "evaluate",
    "simplifies": "simplify", "documents": "document",
    # Irregular past
    "wrote": "write", "written": "write", "ran": "run",
    "sent": "deploy",
}


def _normalize_verb(word: str) -> str | None:
    """Normalize an inflected verb to its base form in ACTION_VERBS.

    Uses an explicit map for safety -- no generic suffix stripping that
    could turn nouns like 'configuration' into verbs.  Returns None if
    the word is not a recognized verb form.
    """
    low = word.lower()
    if low in ACTION_VERBS:
        return low
    return _VERB_NORM_MAP.get(low)

# SCREAMING_SNAKE_CASE constants -- e.g. ACTION_VERBS, MAX_RETRIES
CONST_RE = re.compile(r"\b[A-Z][A-Z0-9_]{2,}\b")

# ---------------------------------------------------------------------------
# Commit-prefix stripping (conventional commits: "fix: ...", "feat(scope): ...")
# ---------------------------------------------------------------------------

_COMMIT_TYPES: frozenset[str] = frozenset({
    "feat", "fix", "docs", "test", "refactor", "perf",
    "build", "ci", "chore", "revert", "style",
})

_COMMIT_PREFIX_RE = re.compile(
    r"^(" + "|".join(sorted(_COMMIT_TYPES)) + r")(?:\([^)]*\))?!?:\s*",
    re.IGNORECASE,
)


def _strip_commit_prefix(text: str) -> tuple[str, str]:
    """Strip a conventional-commit prefix, returning (cleaned, commit_type).

    Returns (text, "") if no prefix matched.
    """
    m = _COMMIT_PREFIX_RE.match(text)
    if m:
        return text[m.end():], m.group(1).lower()
    return text, ""

# ---------------------------------------------------------------------------
# Target regex patterns (compiled once)
# ---------------------------------------------------------------------------

# File-like: foo.py, bar_baz.rs, my-lib.js, state/agents.json
FILE_RE = re.compile(r"\b[\w./-]*\w+\.\w{1,8}\b")

# Version strings that FILE_RE would false-match (v2.0.0, 1.2.3)
_VERSION_RE = re.compile(r"^v?\d+\.\d+(?:\.\d+)?(?:[-+.]\w+)*$", re.IGNORECASE)

# Numbered references: fig.1, no.5, vol.2, ch.3, pt.1
_NUMBERED_REF_RE = re.compile(
    r"^(?:fig|no|vol|ch|pt|sec|eq|ref|ex)\.\d+$", re.IGNORECASE
)

# False positives that FILE_RE catches (abbreviations with periods)
_FALSE_FILE_MATCHES: frozenset[str] = frozenset({
    "e.g", "i.e", "a.m", "p.m", "vs.",
})

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

# Known rappterbook tools -- precision-matched before generic TOOL_RE (PR #246)
KNOWN_TOOLS: frozenset[str] = frozenset({
    "state_io", "process_inbox", "process_issues", "propose_seed",
    "seed_gate", "compute_trending", "generate_feeds", "safe_commit",
    "content_loader", "content_engine", "feature_flags", "github_llm",
    "zion_autonomy", "heartbeat_audit", "pii_scan", "bundle",
    "compute_analytics", "reconcile_channels", "git_scrape_analytics",
    "inject_seed", "tally_votes", "steer", "harvest_artifact",
    "build_seed_tracker", "build_harness_dashboard", "vlink",
})

_KNOWN_TOOL_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(t) for t in sorted(KNOWN_TOOLS)) + r")\b"
)

# Tags that exempt from the *target* requirement (still need a verb)
EXEMPT_TAGS: frozenset[str] = frozenset({
    "theme", "philosophy", "debate", "exploration", "story", "lore",
})

TAG_IMPLIED_VERBS: dict[str, str] = {
    "code": "build", "build": "build", "test": "test", "debug": "debug",
    "docs": "document", "refactor": "refactor", "security": "secure",
    "deploy": "deploy", "monitor": "monitor",
}

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
_KNOWN_MODULES_LOWER: frozenset[str] = frozenset(m.lower() for m in KNOWN_MODULES)

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
    advisory: str = ""
    all_verbs: tuple = ()
    all_targets: tuple = ()

    @property
    def confidence(self) -> str:
        """Gate-aware confidence: 'none' when not passed, else score-based."""
        if not self.passed:
            return "none"
        if self.score >= 0.7:
            return "high"
        if self.score >= 0.4:
            return "medium"
        return "low"

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
            "advisory": self.advisory,
            "all_verbs": list(self.all_verbs),
            "all_targets": [list(t) for t in self.all_targets],
            "confidence": self.confidence,
            "suggestions": [],
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

    def merge(self, other: BatchStats) -> BatchStats:
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
    passed_items: tuple[tuple[str, dict], ...]
    failed_items: tuple[tuple[str, dict], ...]
    junk_items: tuple[tuple[str, dict], ...]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_false_file_match(match_text: str) -> bool:
    """Return True if a FILE_RE match is actually a false positive."""
    low = match_text.lower().rstrip(".")
    if low in _FALSE_FILE_MATCHES:
        return True
    # Version strings (v2.0.0, 1.2.3) are not files
    if _VERSION_RE.match(match_text):
        return True
    # Numbered references (fig.1, ch.3) are not files
    if _NUMBERED_REF_RE.match(match_text):
        return True
    return False


def _starts_with_verb(text: str) -> bool:
    """Return True if text starts with an action verb (single or phrasal).

    Supports inflected forms via _normalize_verb().
    """
    words = text.split()
    if not words:
        return False
    first = words[0].lower()
    if first in _PHRASAL_FIRST and len(words) > 1:
        second = words[1].lower()
        if second in _PHRASAL_FIRST[first]:
            return True
    return _normalize_verb(first) is not None


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
    """Return the first action verb found in *text*, or None.

    Handles inflected forms (Building -> build, Created -> create).
    """
    search_text = text[:limit] if limit else text
    words = re.findall(r"[a-zA-Z]+", search_text.lower())
    for i, word in enumerate(words):
        if word in _PHRASAL_FIRST and i + 1 < len(words):
            next_word = words[i + 1]
            if next_word in _PHRASAL_FIRST[word]:
                return _PHRASAL_FIRST[word][next_word]
        normalized = _normalize_verb(word)
        if normalized:
            return normalized
    return None


def find_all_verbs(text: str) -> list[str]:
    """Return all action verbs in *text* (deduped, order-preserving)."""
    words = re.findall(r"[a-zA-Z]+", text.lower())
    seen: set[str] = set()
    result: list[str] = []
    skip_next = False
    for i, word in enumerate(words):
        if skip_next:
            skip_next = False
            continue
        if word in _PHRASAL_FIRST and i + 1 < len(words):
            next_word = words[i + 1]
            if next_word in _PHRASAL_FIRST[word]:
                v = _PHRASAL_FIRST[word][next_word]
                if v not in seen:
                    seen.add(v)
                    result.append(v)
                skip_next = True
                continue
        normalized = _normalize_verb(word)
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


def find_target(text: str) -> tuple[str, str]:
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
    # SCREAMING_SNAKE_CASE constants
    m = CONST_RE.search(text)
    if m:
        return m.group(), "const"
    # Known tools first (precision), then generic TOOL_RE
    m = _KNOWN_TOOL_RE.search(text)
    if m:
        return m.group(), "tool"
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
            if bare.lower() in _KNOWN_MODULES_LOWER:
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
    # Strip commit prefix before junk checks so "fix: ..." is not flagged lowercase
    cleaned, _ctype = _strip_commit_prefix(stripped)
    if _ctype:
        stripped = cleaned.strip() or stripped
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


def _find_all_targets(text: str) -> tuple[tuple[str, str], ...]:
    """Return all targets in *text* as ((target, kind), ...) (#12521)."""
    found: list[tuple[str, str]] = []
    seen_canonical: set[str] = set()
    def _add(t: str, k: str) -> None:
        c = canonicalize_target(t)
        if c and c not in seen_canonical:
            seen_canonical.add(c)
            found.append((t, k))
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
    for m in CONST_RE.finditer(text):
        _add(m.group(), "const")
    for m in _KNOWN_TOOL_RE.finditer(text):
        _add(m.group(), "tool")
    for m in TOOL_RE.finditer(text):
        _add(m.group(), "tool")
    for m in CLI_RE.finditer(text):
        _add(m.group(), "cli")
    for m in DISCUSSION_RE.finditer(text):
        _add(m.group(), "discussion")
    for m in QUOTED_RE.finditer(text):
        _add(m.group(), "quoted")
    return tuple(found)


def is_soft_artifact(text: str) -> bool:
    """Return True if text contains soft artifact signals."""
    head = text.strip()[:80].lower()
    return any(signal in head for signal in _SOFT_ARTIFACT_SIGNALS)


def canonicalize_target(target: str) -> str:
    """Normalize a target string for dedup: strip path prefix, extension, quotes."""
    t = target.strip("\"' `")
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
    raw_targets: list[str] = []
    for pattern in (FILE_RE, PATH_RE, CONST_RE, TOOL_RE, CLI_RE, DISCUSSION_RE, CHANNEL_RE):
        for m in pattern.finditer(text):
            t = m.group()
            if pattern is FILE_RE and _is_false_file_match(t):
                continue
            raw_targets.append(t)
    canonical: list[str] = []
    for t in raw_targets:
        c = canonicalize_target(t)
        if c:
            canonical.append(c)
    # Substring dedup: remove shorter forms that are substrings of longer ones
    canonical_sorted = sorted(set(canonical), key=len, reverse=True)
    unique: list[str] = []
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
    """Compute a 0.0-1.0 specificity score."""
    raw = 0.0
    if verb:
        raw += 2.5
    if target:
        kind_scores = {
            "file": 4.0, "path": 3.5, "func": 3.0, "module": 3.0,
            "tool": 3.0, "cli": 3.0, "const": 2.5,
            "discussion": 2.0, "channel": 2.0, "quoted": 1.5,
        }
        raw += kind_scores.get(target_kind, 1.5)
    words = text.split()
    if len(words) >= 8:
        raw += 0.5
    if len(words) >= 15:
        raw += 1.0
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
_normalize = _normalize_verb


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

    # -- Strip conventional-commit prefix once at the top ---
    cleaned_text, commit_type = _strip_commit_prefix(text.strip())
    working_text = cleaned_text if commit_type else text

    # -- Junk check (hard fail) ---
    junk_limit = 60 if mode == "purge" else 0
    junk_reason = is_junk(text, limit=junk_limit)
    if junk_reason:
        return SeedGateResult(
            passed=False, reasons=(junk_reason,), score=0.0,
            verb_found=None, target_found=None, junk=True,
        )

    # -- Verb + target (use cleaned text so commit prefix doesn't interfere) ---
    verb_limit = 200 if mode == "purge" else 0
    verb = find_verb(working_text, limit=verb_limit)
    target, target_kind = find_target(working_text)

    # -- Commit-type as fallback verb ---
    if not verb and commit_type and commit_type in ACTION_VERBS:
        verb = commit_type

    # -- Tag-implied verb inference (#12530) ---
    if not verb:
        for tag in tag_set:
            if tag in TAG_IMPLIED_VERBS:
                verb = TAG_IMPLIED_VERBS[tag]
                break

    # -- Question stem inference (exempt tags only) ---
    if not verb and is_exempt:
        m = _QUESTION_STEM_RE.match(working_text.strip())
        if m:
            verb = QUESTION_STEMS.get(m.group().lower())

    # -- Rich match info (#12521) ---
    all_verbs = tuple(find_all_verbs(text))
    all_targets = _find_all_targets(text)

    # -- Soft artifact check ---
    if is_soft_artifact(text) and not (verb and target) and not is_exempt:
        return SeedGateResult(
            passed=False,
            reasons=("soft artifact signal without redeeming verb+target",),
            score=0.0, verb_found=verb, target_found=target or None,
            junk=True, all_verbs=all_verbs, all_targets=all_targets,
        )

    # -- Decision ---
    if mode == "purge":
        passed = True
        specificity = 0.5
    else:
        passed = bool(verb) and (bool(target) or is_exempt)
        specificity = compute_score(text, verb, target, target_kind)

    reasons: list[str] = []
    advisory = ""
    if not passed:
        if not verb:
            reasons.append("No action verb found")
        if not target and not is_exempt:
            reasons.append("No concrete target (filename, tool, or reference)")
        if verb and not target and not is_exempt:
            advisory = "needs-specificity"
    elif verb and not target:
        advisory = "needs-specificity"

    return SeedGateResult(
        passed=passed, reasons=tuple(reasons), score=specificity,
        verb_found=verb or None, target_found=target or None, junk=False,
        advisory=advisory, all_verbs=all_verbs, all_targets=all_targets,
    )


def validate(text: str, tags: list = None, mode: str = "admission") -> dict:
    """Dict API -- the shape expected by propose_seed.py."""
    return validate_seed(text, tags, mode=mode).to_dict()


def passes_gate(text: str, tags: list = None, mode: str = "admission") -> bool:
    """Convenience: return True iff the proposal passes the gate."""
    return validate_seed(text, tags, mode=mode).passed


def validate_batch(
    proposals: list[str],
    tags: list = None,
    mode: str = "admission",
) -> BatchResult:
    """Validate a batch of proposals; separate junk from merely-failed.

    Returns a BatchResult with stats + categorized items so callers
    (like propose_seed.purge_junk) can treat junk and vague-but-salvageable
    proposals differently.
    """
    passed_items: list[tuple[str, dict]] = []
    failed_items: list[tuple[str, dict]] = []
    junk_items: list[tuple[str, dict]] = []

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



def suggest(text: str, tags: list = None) -> list[str]:
    """Return actionable hints for improving a proposal.

    If the proposal already passes the gate, returns an empty list.
    Otherwise, returns 1-3 short suggestions.
    """
    result = validate_seed(text, tags or [])
    if result.passed:
        return []
    hints: list[str] = []
    if not result.verb_found:
        hints.append("Add an action verb (build, test, refactor, deploy, ...)")
    if not result.target_found:
        hints.append("Name a concrete target (file.py, tool_name, #1234, ...)")
    if result.junk:
        hints.append("Rewrite as a clear, capitalized sentence")
    return hints


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
