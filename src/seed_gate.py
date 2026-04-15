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

    # Diagnostic APIs (new in PR #272)
    explanation = explain(text, tags)   # human-readable string
    breakdown = score_breakdown(text, tags)  # per-factor dict
    verb, pos = find_verb_with_position(text)  # verb + word index

    # Composable helpers
    verb   = find_verb(text)          # str | None
    target = find_target(text)        # (str, str) pair
    junk   = is_junk(text)            # str (reason) or empty
    score  = compute_score(text, verb, target, kind)  # float
    clean  = normalize_proposal(text) # strip commit prefixes

Evolution log:
    PR #237  -- initial canonical validator (165 tests)
    PR #242  -- contract alignment with propose_seed.py
    PR #245  -- auto-discovered modules, two-tier artifacts, propose_seed wiring
    PR #248  -- consolidated #245/#246/#247: false-file filter, special
                  files, known tools, question stems, batch API, smart
                  lowercase, substring dedup.
    PR #253  -- phrasal verbs, tag-implied verbs (#12530), advisory
                  labeling (#12507), rich match lists (#12521), case-
                  insensitive module matching, CONST_RE in find_target.
    PR #256  -- inflected verb normalization via generated inflection
                  map (builds->build, creating->create, deployed->deploy);
                  inflected phrasal verbs (setting up->set up); version-
                  string false-positive filter; confidence property;
                  suggest() API; expanded KNOWN_TOOLS; env var targets;
                  imperative scoring bonus.
    PR #272  -- commit-prefix normalization (fix: / feat: / etc.);
                  is_imperative property; explain() diagnostic API;
                  score_breakdown() transparency; find_verb_with_position();
                  batch CLI (--batch, JSONL support); 550+ tests.
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
# Inflected verb map -- generated from the closed verb set at import time.
# Maps inflected forms (builds, creating, deployed) to their base verb.
# Only verbs in ACTION_VERBS get inflections -- no accidental stemming.
# ---------------------------------------------------------------------------

# Verbs whose final consonant doubles before -ed/-ing
_DOUBLE_FINAL: frozenset[str] = frozenset({
    "debug", "log", "plan", "map", "ship", "wrap", "scan", "stub",
    "mock", "run", "plug", "ramp", "swap", "spin", "opt", "set",
})

# Irregular past tenses (not derivable by rules)
_IRREGULAR_PAST: dict[str, str] = {
    "built": "build", "wrote": "write", "ran": "run",
}


def _generate_inflections(base: str) -> dict[str, str]:
    """Generate {inflected: base} for -s, -es, -ed, -d, -ing forms."""
    forms: dict[str, str] = {}
    # -s / -es
    if base.endswith(("s", "sh", "ch", "x", "z")):
        forms[base + "es"] = base
    elif base.endswith("y") and len(base) > 1 and base[-2] not in "aeiou":
        forms[base[:-1] + "ies"] = base
    else:
        forms[base + "s"] = base
    # -ed (past tense)
    if base.endswith("e"):
        forms[base + "d"] = base
    elif base in _DOUBLE_FINAL:
        forms[base + base[-1] + "ed"] = base
    elif base.endswith("y") and len(base) > 1 and base[-2] not in "aeiou":
        forms[base[:-1] + "ied"] = base
    else:
        forms[base + "ed"] = base
    # -ing (present participle)
    if base.endswith("e") and not base.endswith("ee"):
        forms[base[:-1] + "ing"] = base
    elif base in _DOUBLE_FINAL:
        forms[base + base[-1] + "ing"] = base
    else:
        forms[base + "ing"] = base
    return forms


def _build_inflection_map() -> dict[str, str]:
    """Build a complete {inflected_form: base_verb} lookup at import time."""
    result: dict[str, str] = {}
    # Irregular past tenses
    for form, base in _IRREGULAR_PAST.items():
        if base in ACTION_VERBS and form not in ACTION_VERBS:
            result[form] = base
    # Regular inflections of single-word verbs
    for verb in ACTION_VERBS:
        for form, base in _generate_inflections(verb).items():
            if form not in ACTION_VERBS:
                result[form] = base
    # Phrasal verb inflections -- inflect only the head word
    for phrase, canonical in PHRASAL_VERBS.items():
        head, particle = phrase.split()
        for form, _base in _generate_inflections(head).items():
            inflected_phrase = form + " " + particle
            if inflected_phrase not in PHRASAL_VERBS:
                result[inflected_phrase] = canonical
    return result


_INFLECTION_MAP: dict[str, str] = _build_inflection_map()

# Phrasal inflection lookup: {inflected_head: {particle: canonical}}
_PHRASAL_INFLECTED: dict[str, dict[str, str]] = {}
for _inf_phrase, _inf_canonical in _INFLECTION_MAP.items():
    if " " in _inf_phrase:
        _inf_head, _inf_particle = _inf_phrase.split(maxsplit=1)
        _PHRASAL_INFLECTED.setdefault(_inf_head, {})[_inf_particle] = _inf_canonical

# SCREAMING_SNAKE_CASE constants -- e.g. ACTION_VERBS, MAX_RETRIES
CONST_RE = re.compile(r"\b[A-Z][A-Z0-9_]{2,}\b")

# ---------------------------------------------------------------------------
# Target regex patterns (compiled once)
# ---------------------------------------------------------------------------

# File-like: foo.py, bar_baz.rs, my-lib.js, state/agents.json
FILE_RE = re.compile(r"\b[\w./-]*\w+\.\w{1,8}\b")

# False positives that FILE_RE catches (abbreviations with periods)
_FALSE_FILE_MATCHES: frozenset[str] = frozenset({
    "e.g", "i.e", "a.m", "p.m", "vs.",
})

# Version strings: 2.0.1, v1.2.3, 1.0 -- should NOT match as file targets
_VERSION_RE = re.compile(r"^v?\d+\.\d+(?:\.\d+)?(?:[+.-]\w+)?$")

# Environment variable references: $STATE_DIR, ${GITHUB_TOKEN}
ENV_VAR_RE = re.compile(r"\$\{?[A-Z][A-Z0-9_]+\}?")

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
    "inject_seed", "tally_votes", "steer", "reconcile_state",
    "run_proof", "run_python", "vlink",
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
# Conventional-commit prefix normalization (PR #272)
# ---------------------------------------------------------------------------

# Matches: "fix:", "feat(auth):", "refactor!:", "docs(readme)!:"
_COMMIT_PREFIX_RE = re.compile(
    r"^(?:fix|feat|chore|docs|refactor|test|build|ci|perf|style|revert)"
    r"(?:\([^)]*\))?"  # optional scope
    r"!?"               # optional breaking-change marker
    r":\s*",
    re.I,
)


def normalize_proposal(text: str) -> str:
    """Strip conventional-commit prefixes before analysis.

    Handles: "fix: align seed_gate.py" -> "Align seed_gate.py"
             "feat(auth): add OAuth" -> "Add OAuth"
             "refactor!: rewrite parser" -> "Rewrite parser"

    The stripped text gets its first letter capitalized to avoid
    false lowercase-junk detection.
    """
    m = _COMMIT_PREFIX_RE.match(text)
    if not m:
        return text
    remainder = text[m.end():]
    if not remainder:
        return text
    return remainder[0].upper() + remainder[1:]


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
    _original_text: str = ""

    @property
    def verb(self) -> str:
        return self.verb_found or ""

    @property
    def target(self) -> str:
        return self.target_found or ""

    @property
    def confidence(self) -> str | None:
        """Derive confidence band from score: high/medium/low or None."""
        if not self.passed:
            return None
        if self.score >= 0.65:
            return "high"
        if self.score >= 0.35:
            return "medium"
        return "low"

    @property
    def is_imperative(self) -> bool:
        """True if the proposal starts with an action verb (imperative mood)."""
        if not self._original_text or not self.verb_found:
            return False
        return _starts_with_verb(self._original_text)

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "reasons": list(self.reasons),
            "score": self.score,
            "verb_found": self.verb_found,
            "target_found": self.target_found,
            "junk": self.junk,
            "advisory": self.advisory,
            "confidence": self.confidence,
            "is_imperative": self.is_imperative,
            "all_verbs": list(self.all_verbs),
            "all_targets": [list(t) for t in self.all_targets],
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
    """Return True if a FILE_RE match is actually a false positive.

    Catches abbreviations (e.g., i.e.) and bare version strings (2.0.1).
    """
    stripped = match_text.lower().rstrip(".")
    if stripped in _FALSE_FILE_MATCHES:
        return True
    if _VERSION_RE.match(match_text):
        return True
    return False


def _starts_with_verb(text: str) -> bool:
    """Return True if text starts with an action verb (single, phrasal, or inflected)."""
    words = text.split()
    if not words:
        return False
    first = words[0].lower()
    if len(words) > 1:
        second = words[1].lower()
        if first in _PHRASAL_FIRST and second in _PHRASAL_FIRST[first]:
            return True
        if first in _PHRASAL_INFLECTED and second in _PHRASAL_INFLECTED[first]:
            return True
    if first in ACTION_VERBS:
        return True
    if first in _INFLECTION_MAP:
        return True
    return False


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

    Checks base forms first, then inflected forms (builds->build,
    creating->create, deployed->deploy).  Phrasal verbs and their
    inflected forms (setting up->set up) are also recognized.
    """
    search_text = text[:limit] if limit else text
    words = re.findall(r"[a-zA-Z]+", search_text.lower())
    for i, word in enumerate(words):
        if i + 1 < len(words):
            next_word = words[i + 1]
            if word in _PHRASAL_FIRST and next_word in _PHRASAL_FIRST[word]:
                return _PHRASAL_FIRST[word][next_word]
            if word in _PHRASAL_INFLECTED and next_word in _PHRASAL_INFLECTED[word]:
                return _PHRASAL_INFLECTED[word][next_word]
        if word in ACTION_VERBS:
            return word
        if word in _INFLECTION_MAP:
            return _INFLECTION_MAP[word]
    return None


def find_all_verbs(text: str) -> list[str]:
    """Return all action verbs in *text* (deduped, order-preserving).

    Recognizes base forms, inflected forms, and phrasal verbs.
    Always returns the canonical (base) form.
    """
    words = re.findall(r"[a-zA-Z]+", text.lower())
    seen: set[str] = set()
    result: list[str] = []
    skip_next = False
    for i, word in enumerate(words):
        if skip_next:
            skip_next = False
            continue
        if i + 1 < len(words):
            next_word = words[i + 1]
            if word in _PHRASAL_FIRST and next_word in _PHRASAL_FIRST[word]:
                v = _PHRASAL_FIRST[word][next_word]
                if v not in seen:
                    seen.add(v)
                    result.append(v)
                skip_next = True
                continue
            if word in _PHRASAL_INFLECTED and next_word in _PHRASAL_INFLECTED[word]:
                v = _PHRASAL_INFLECTED[word][next_word]
                if v not in seen:
                    seen.add(v)
                    result.append(v)
                skip_next = True
                continue
        if word in ACTION_VERBS:
            if word not in seen:
                seen.add(word)
                result.append(word)
        elif word in _INFLECTION_MAP:
            base = _INFLECTION_MAP[word]
            if base not in seen:
                seen.add(base)
                result.append(base)
    return result


def find_verb_with_position(text: str) -> tuple[str | None, int]:
    """Return (canonical_verb, word_index) or (None, -1).

    *word_index* is the 0-based position in whitespace-split words
    where the verb (or its first word for phrasals) was found.
    Recognizes base, inflected, and phrasal forms.
    """
    words = re.findall(r"[a-zA-Z]+", text.lower())
    for i, word in enumerate(words):
        if i + 1 < len(words):
            next_word = words[i + 1]
            if word in _PHRASAL_FIRST and next_word in _PHRASAL_FIRST[word]:
                return _PHRASAL_FIRST[word][next_word], i
            if word in _PHRASAL_INFLECTED and next_word in _PHRASAL_INFLECTED[word]:
                return _PHRASAL_INFLECTED[word][next_word], i
        if word in ACTION_VERBS:
            return word, i
        if word in _INFLECTION_MAP:
            return _INFLECTION_MAP[word], i
    return None, -1


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
    # Environment variables: $STATE_DIR, ${GITHUB_TOKEN}
    m = ENV_VAR_RE.search(text)
    if m:
        return m.group(), "env"
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
    for m in ENV_VAR_RE.finditer(text):
        _add(m.group(), "env")
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
    for pattern in (FILE_RE, PATH_RE, ENV_VAR_RE, CONST_RE, TOOL_RE, CLI_RE, DISCUSSION_RE, CHANNEL_RE):
        for m in pattern.finditer(text):
            raw_targets.append(m.group())
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


_TARGET_KIND_SCORES: dict[str, float] = {
    "file": 4.0, "path": 3.5, "func": 3.0, "module": 3.0,
    "tool": 3.0, "cli": 3.0, "env": 3.0, "const": 2.5,
    "discussion": 2.0, "channel": 2.0, "quoted": 1.5,
}


def _compute_score_factors(
    text: str,
    verb: str | None,
    target: str | None,
    target_kind: str,
) -> dict:
    """Internal: compute per-factor score contributions.

    Returns a dict with individual factor scores, raw total, and
    the normalized 0.0-1.0 score.  Single source of truth for both
    compute_score() and score_breakdown().
    """
    factors: dict = {}
    factors["verb"] = 2.5 if verb else 0.0
    factors["target"] = _TARGET_KIND_SCORES.get(target_kind, 1.5) if target else 0.0
    factors["target_kind"] = target_kind if target else ""
    word_count = len(text.split())
    length_bonus = 0.0
    if word_count >= 8:
        length_bonus += 0.5
    if word_count >= 15:
        length_bonus += 1.0
    factors["length"] = length_bonus
    factors["word_count"] = word_count
    unique = count_unique_targets(text)
    factors["multi_target"] = 1.0 if unique >= 2 else 0.0
    factors["unique_targets"] = unique
    imperative = bool(verb and _starts_with_verb(text))
    factors["imperative"] = 0.5 if imperative else 0.0
    raw = factors["verb"] + factors["target"] + factors["length"] + factors["multi_target"] + factors["imperative"]
    factors["raw"] = raw
    factors["normalized"] = min(raw / 10.0, 1.0)
    return factors


def compute_score(
    text: str,
    verb: str | None,
    target: str | None,
    target_kind: str,
) -> float:
    """Compute a 0.0-1.0 specificity score."""
    return _compute_score_factors(text, verb, target, target_kind)["normalized"]


def score_breakdown(text: str, tags: list = None) -> dict:
    """Return per-factor score breakdown for a proposal.

    Runs the full validation pipeline, then returns the scoring
    factors as a transparent dict.  Useful for debugging scores.

    Keys: verb, target, target_kind, length, word_count,
          multi_target, unique_targets, imperative, raw, normalized.
    """
    result = validate_seed(text, tags)
    _, target_kind = find_target(normalize_proposal(text))
    return _compute_score_factors(
        normalize_proposal(text), result.verb_found, result.target_found, target_kind,
    )


def suggest(text: str, tags: list = None) -> list[str]:
    """Return actionable suggestions for improving a rejected proposal.

    Thin helper over validate_seed() -- no contract change, just
    human-readable feedback for the rejection reasons.
    """
    result = validate_seed(text, tags)
    if result.passed:
        return []
    suggestions: list[str] = []
    if not result.verb_found:
        suggestions.append(
            "Start with an action verb (build, fix, test, deploy, refactor, ...)"
        )
    if not result.target_found and not (frozenset(t.lower() for t in (tags or [])) & EXEMPT_TAGS):
        suggestions.append(
            "Name a concrete target: a filename (auth.py), tool (state_io), "
            "path (src/thermal/), or reference (#12345)"
        )
    if result.junk:
        suggestions.append(
            "Rewrite as a complete sentence starting with a capital letter"
        )
    return suggestions


def explain(text: str, tags: list = None) -> str:
    """Return a human-readable diagnostic for a seed proposal.

    Intended for CLI debugging and agent feedback.  The format
    is explicitly unstable -- do not parse it programmatically.
    """
    result = validate_seed(text, tags)
    normalized = normalize_proposal(text)
    was_normalized = normalized != text

    lines: list[str] = []
    status = "PASSED" if result.passed else "REJECTED"
    if result.junk:
        status = "JUNK"
    score_str = "%.2f" % result.score
    conf = result.confidence or "n/a"
    lines.append("%s %s (score: %s, confidence: %s)" % (
        "\u2713" if result.passed else "\u2717", status, score_str, conf))

    if was_normalized:
        lines.append("  Normalized: %r" % normalized)

    verb_desc = repr(result.verb_found) if result.verb_found else "none"
    if result.is_imperative:
        verb_desc += " (imperative)"
    lines.append("  Verb: %s" % verb_desc)

    target_desc = repr(result.target_found) if result.target_found else "none"
    if result.target_found:
        _, kind = find_target(normalized)
        target_desc += " (%s)" % kind
    lines.append("  Target: %s" % target_desc)

    if result.advisory:
        lines.append("  Advisory: %s" % result.advisory)
    if result.reasons:
        lines.append("  Reasons: %s" % "; ".join(result.reasons))
    if result.all_verbs:
        lines.append("  All verbs: %s" % ", ".join(result.all_verbs))
    if result.all_targets:
        targets_str = ", ".join("%s (%s)" % (t, k) for t, k in result.all_targets)
        lines.append("  All targets: %s" % targets_str)

    return "\n".join(lines)


# Backward-compat aliases
_detect_verb = find_verb
_detect_target = find_target
_is_junk = is_junk
_is_soft_artifact = is_soft_artifact
_canonicalize_target = canonicalize_target
_count_unique_targets = count_unique_targets
_score = lambda text, verb, target, kind: compute_score(text, verb, target, kind)
_normalize_verb = lambda word: _INFLECTION_MAP.get(word)


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

    # -- Normalize (strip commit prefixes) ---
    analysis_text = normalize_proposal(text)

    # -- Junk check (hard fail) ---
    junk_limit = 60 if mode == "purge" else 0
    junk_reason = is_junk(analysis_text, limit=junk_limit)
    if junk_reason:
        return SeedGateResult(
            passed=False, reasons=(junk_reason,), score=0.0,
            verb_found=None, target_found=None, junk=True,
            _original_text=analysis_text,
        )

    # -- Verb + target ---
    verb_limit = 200 if mode == "purge" else 0
    verb = find_verb(analysis_text, limit=verb_limit)
    target, target_kind = find_target(analysis_text)

    # -- Tag-implied verb inference (#12530) ---
    if not verb:
        for tag in tag_set:
            if tag in TAG_IMPLIED_VERBS:
                verb = TAG_IMPLIED_VERBS[tag]
                break

    # -- Question stem inference (exempt tags only) ---
    if not verb and is_exempt:
        m = _QUESTION_STEM_RE.match(analysis_text.strip())
        if m:
            verb = QUESTION_STEMS.get(m.group().lower())

    # -- Rich match info (#12521) ---
    all_verbs = tuple(find_all_verbs(analysis_text))
    all_targets = _find_all_targets(analysis_text)

    # -- Soft artifact check ---
    if is_soft_artifact(analysis_text) and not (verb and target) and not is_exempt:
        return SeedGateResult(
            passed=False,
            reasons=("soft artifact signal without redeeming verb+target",),
            score=0.0, verb_found=verb, target_found=target or None,
            junk=True, all_verbs=all_verbs, all_targets=all_targets,
            _original_text=analysis_text,
        )

    # -- Decision ---
    if mode == "purge":
        passed = True
        specificity = 0.5
    else:
        passed = bool(verb) and (bool(target) or is_exempt)
        specificity = compute_score(analysis_text, verb, target, target_kind)

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
        _original_text=analysis_text,
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


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def _cli() -> None:  # pragma: no cover
    """CLI entry-point with single and batch modes.

    Single:  python -m seed_gate 'Build auth.py' code test
    Batch:   cat proposals.txt | python -m seed_gate --batch
    Explain: python -m seed_gate --explain 'Build auth.py'
    """
    import json as _json

    args = sys.argv[1:]
    if not args:
        print("Usage: python -m seed_gate '<proposal>' [tags...]")
        print("       python -m seed_gate --batch [< proposals.txt]")
        print("       python -m seed_gate --explain '<proposal>' [tags...]")
        sys.exit(1)

    if args[0] == "--batch":
        # Batch mode: read from stdin, one per line.
        # Supports plain text or JSONL {"text": ..., "tags": [...]}
        results: list[dict] = []
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            tags: list[str] = []
            text = line
            if line.startswith("{"):
                try:
                    obj = _json.loads(line)
                    text = obj.get("text", line)
                    tags = obj.get("tags", [])
                except _json.JSONDecodeError:
                    pass
            result = validate(text, tags)
            results.append({"text": text, "result": result})
        print(_json.dumps(results, indent=2))
        sys.exit(0)

    if args[0] == "--explain":
        text = args[1] if len(args) > 1 else ""
        tags = args[2:] if len(args) > 2 else []
        print(explain(text, tags))
        sys.exit(0)

    text = args[0]
    tags = args[1:] if len(args) > 1 else []
    result = validate(text, tags)
    print(_json.dumps(result, indent=2))
    sys.exit(0 if result["passed"] else 1)


if __name__ == "__main__":
    _cli()
