"""seed_gate.py -- canonical specificity validator for seed proposals.

Consolidates ideas from 6 independent implementations (#12503, #12505,
#12507, #12511, #12521, #12530) and PRs #245–#287 into one canonical
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

    # Diagnostic APIs (consolidated from PRs #279-#287)
    match = find_verb_with_position(text)  # -> VerbMatch | None
    parts = score_breakdown(text, tags)    # -> dict[str, float]
    diag  = explain(text, tags)            # -> str

    # Composable helpers
    verb   = find_verb(text)          # str | None
    target = find_target(text)        # (str, str) pair
    junk   = is_junk(text)            # str (reason) or empty
    score  = compute_score(text, verb, target, kind)  # float
    norm   = normalize_proposal(text) # str (commit prefix stripped)

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
    PR #271  -- inflected verb normalization, version-string filter,
                  confidence, suggest(), expanded tools, env var targets,
                  imperative scoring bonus.
    PR #288  -- CONSOLIDATION of PRs #279-#287 (6 agent implementations):
                  VerbMatch dataclass, find_verb_with_position(),
                  negation detection, commit-prefix normalization,
                  score_breakdown(), explain(), enriched SeedGateResult
                  (target_kind, verb_source, verb_position, negated,
                  is_imperative, score_parts), numbered-ref filter,
                  expanded abbreviation filter, --batch/--explain CLI.
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
    "annotate", "version", "backup", "package", "post",
    "prevent", "avoid",
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

_DOUBLE_FINAL: frozenset[str] = frozenset({
    "debug", "log", "plan", "map", "ship", "wrap", "scan", "stub",
    "mock", "run", "plug", "ramp", "swap", "spin", "opt", "set",
})

_IRREGULAR_PAST: dict[str, str] = {
    "built": "build", "wrote": "write", "ran": "run",
}


def _generate_inflections(base: str) -> dict[str, str]:
    """Generate {inflected: base} for -s, -es, -ed, -d, -ing forms."""
    forms: dict[str, str] = {}
    if base.endswith(("s", "sh", "ch", "x", "z")):
        forms[base + "es"] = base
    elif base.endswith("y") and len(base) > 1 and base[-2] not in "aeiou":
        forms[base[:-1] + "ies"] = base
    else:
        forms[base + "s"] = base
    if base.endswith("e"):
        forms[base + "d"] = base
    elif base in _DOUBLE_FINAL:
        forms[base + base[-1] + "ed"] = base
    elif base.endswith("y") and len(base) > 1 and base[-2] not in "aeiou":
        forms[base[:-1] + "ied"] = base
    else:
        forms[base + "ed"] = base
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
    for form, base in _IRREGULAR_PAST.items():
        if base in ACTION_VERBS and form not in ACTION_VERBS:
            result[form] = base
    for verb in ACTION_VERBS:
        for form, base in _generate_inflections(verb).items():
            if form not in ACTION_VERBS:
                result[form] = base
    for phrase, canonical in PHRASAL_VERBS.items():
        head, particle = phrase.split()
        for form, _base in _generate_inflections(head).items():
            inflected_phrase = form + " " + particle
            if inflected_phrase not in PHRASAL_VERBS:
                result[inflected_phrase] = canonical
    return result


_INFLECTION_MAP: dict[str, str] = _build_inflection_map()

_PHRASAL_INFLECTED: dict[str, dict[str, str]] = {}
for _inf_phrase, _inf_canonical in _INFLECTION_MAP.items():
    if " " in _inf_phrase:
        _inf_head, _inf_particle = _inf_phrase.split(maxsplit=1)
        _PHRASAL_INFLECTED.setdefault(_inf_head, {})[_inf_particle] = _inf_canonical

CONST_RE = re.compile(r"\b[A-Z][A-Z0-9_]{2,}\b")

# ---------------------------------------------------------------------------
# Commit-prefix normalization (consolidated from PRs #281-#283)
# ---------------------------------------------------------------------------

_COMMIT_PREFIX_RE = re.compile(
    r"^(?:feat|fix|refactor|chore|docs|test|ci|build|perf|style|revert)"
    r"(?:\([^)]*\))?:\s*",
    re.IGNORECASE,
)

_COMMIT_PREFIX_VERBS: dict[str, str] = {
    "feat": "build", "fix": "fix", "refactor": "refactor",
    "docs": "document", "test": "test", "build": "build",
    "perf": "optimize", "ci": "configure", "revert": "roll back",
}


def normalize_proposal(text: str) -> str:
    """Strip conventional commit prefix from proposal text.

    Returns the stripped text if a prefix was found, or the original.
    Examples: 'feat: Build auth.py' -> 'Build auth.py'
              'fix(gate): Resolve crash' -> 'Resolve crash'
    """
    return _COMMIT_PREFIX_RE.sub("", text)


def _extract_commit_prefix_verb(text: str) -> str | None:
    """Return the implied verb from a commit prefix, or None."""
    m = _COMMIT_PREFIX_RE.match(text)
    if not m:
        return None
    prefix_word = m.group().split("(")[0].split(":")[0].strip().lower()
    return _COMMIT_PREFIX_VERBS.get(prefix_word)


# ---------------------------------------------------------------------------
# Target regex patterns (compiled once)
# ---------------------------------------------------------------------------

FILE_RE = re.compile(r"\b[\w./-]*\w+\.\w{1,8}\b")

_FALSE_FILE_MATCHES: frozenset[str] = frozenset({
    "e.g", "i.e", "a.m", "p.m", "vs.",
    "ph.d", "u.s", "u.k",
})

_VERSION_RE = re.compile(r"^v?\d+\.\d+(?:\.\d+)?(?:[+.-]\w+)?$")

# Numbered references: fig.1, ch.3, vol.2, eq.1, sec.4 (PR #281)
_NUMBERED_REF_RE = re.compile(r"^(?:fig|ch|vol|eq|sec|pt|no|pg)\.\d+$", re.I)

ENV_VAR_RE = re.compile(r"\$\{?[A-Z][A-Z0-9_]+\}?")

SPECIAL_FILE_RE = re.compile(
    r"\b(?:Dockerfile|Makefile|Vagrantfile|Procfile|Gemfile|Rakefile"
    r"|README|AGENTS|CLAUDE|CONSTITUTION|CONTRIBUTING|LICENSE"
    r"|CHANGELOG|ROADMAP|MANIFEST|FEATURE_FREEZE)\b"
)

PATH_RE = re.compile(
    r"(?:(?:src|tests|engine|state|docs|api|scripts|sdk|zion)/[\w_./-]+)"
)

FUNC_RE = re.compile(r"\b[\w_]+\(\)")

TOOL_RE = re.compile(r"\b[a-z][a-z0-9]*(?:[_-][a-z0-9]+)+\b")

CLI_RE = re.compile(r"(?:`[^`]+`|--[a-z][\w-]+\b|-[a-zA-Z]\b)")

DISCUSSION_RE = re.compile(r"#(\d{3,})\b")

CHANNEL_RE = re.compile(r"\b[rc]/[a-z][a-z0-9_-]+\b")

QUOTED_RE = re.compile(r"""(?:"[^"]{3,60}"|'[^']{3,60}')""")

MODULE_CONTEXT_RE = re.compile(r"(?:`[\w_]+`|import\s+[\w_]+|from\s+[\w_]+)")

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

EXEMPT_TAGS: frozenset[str] = frozenset({
    "theme", "philosophy", "debate", "exploration", "story", "lore",
})

TAG_IMPLIED_VERBS: dict[str, str] = {
    "code": "build", "build": "build", "test": "test", "debug": "debug",
    "docs": "document", "refactor": "refactor", "security": "secure",
    "deploy": "deploy", "monitor": "monitor",
}

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
# Negation detection (consolidated from PRs #279-#287)
#
# Scope: only hard-fail when negation binds to the PRIMARY verb (first
# verb in the proposal, typically position 0-2). Negation in purpose
# clauses, subordinate clauses, or after clause boundaries is ignored.
# ---------------------------------------------------------------------------

# Unicode-aware apostrophe normalization
_APOSTROPHE_CHARS = "'\u2019\u2018\u0060"

_NEGATION_CONTRACTIONS = re.compile(
    r"\b(?:don|doesn|didn|won|wouldn|shouldn|couldn|can|isn|aren|hasn|haven|wasn|weren)"
    r"[" + _APOSTROPHE_CHARS + r"]t\b",
    re.IGNORECASE,
)

_NEGATION_SPLIT = re.compile(
    r"\b(?:do|does|did|will|shall|would|should|could)\s+not\b",
    re.IGNORECASE,
)

_NEGATION_NEVER = re.compile(r"\bnever\b", re.IGNORECASE)

# Clause boundaries that reset negation scope
_CLAUSE_BOUNDARY_RE = re.compile(
    r"\b(?:that|which|who|where|when|because|so\s+that|so|to|if|even\s+though|although|but|instead|however)\b",
    re.IGNORECASE,
)

# Purpose/subordinate clause starters (negation after these is scoped)
_PURPOSE_CLAUSE_RE = re.compile(
    r"\b(?:so\s+that|so|to|in\s+order\s+to|when|if|unless|even\s+though|although)\b",
    re.IGNORECASE,
)


def detect_negation(text: str) -> bool:
    """Return True if the primary verb in *text* is negated.

    Only flags negation that binds to the MAIN clause. Negation in
    subordinate/purpose clauses (so, to, that, which, when, if) or
    after redemption words (but, instead, however) is ignored.
    """
    # Find the first clause boundary -- negation after it is scoped
    boundary_match = _CLAUSE_BOUNDARY_RE.search(text)
    # Check zone = everything before the first clause boundary
    check_zone = text[:boundary_match.start()] if boundary_match else text

    # Skip if check_zone is a question (starts with question word)
    stripped = check_zone.strip()
    if stripped and stripped[0:1].lower() in ("w", "h", "c", "s", "d"):
        if _QUESTION_STEM_RE.match(stripped):
            return False

    for pat in (_NEGATION_CONTRACTIONS, _NEGATION_SPLIT, _NEGATION_NEVER):
        if pat.search(check_zone):
            return True
    return False


# ---------------------------------------------------------------------------
# Junk / artifact detection (#12507 + main repo consolidation)
# ---------------------------------------------------------------------------

_JUNK_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^\s*[`|,()\-]"),
    re.compile(r"^\d+\.\s"),
    re.compile(r"^https?://"),
    re.compile(r"(?:TODO|FIXME|HACK)\b", re.I),
    re.compile(r"^\s*$"),
]

_LOWERCASE_START_RE = re.compile(r"^[a-z]")
_FILE_START_RE = re.compile(r"^[\w./-]*\w+\.\w{1,8}\b")

_JUNK_EXCEPTION_RE = re.compile(r"^run_\w")

_HARD_ARTIFACT_SIGNALS: tuple[str, ...] = (
    "parser grabbed", "parsing artifact", "outside that grammar",
    "the fragment was", "`) and ", "` has `",
)

_SOFT_ARTIFACT_SIGNALS: tuple[str, ...] = (
    "the regex", "the parser", "the fragment", "substring",
    "` and `", "` is ",
)

ARTIFACT_SIGNALS: tuple[str, ...] = _HARD_ARTIFACT_SIGNALS + _SOFT_ARTIFACT_SIGNALS

# ---------------------------------------------------------------------------
# Auto-discovered modules from src/*.py
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
# VerbMatch dataclass (consolidated from PRs #284, #287)
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class VerbMatch:
    """Structured result from find_verb_with_position().

    Attributes:
        verb: canonical base form of the detected verb
        token_index: 0-based position in whitespace-split tokens
        source: how the verb was found -- "text", "phrasal", "inflected"
    """
    verb: str
    token_index: int
    source: str  # "text" | "phrasal" | "inflected"

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
    target_kind: str = ""
    verb_source: str = ""
    verb_position: int | None = None
    score_parts: tuple = ()
    negated: bool = False

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
        """True when the verb is at position 0 (imperative mood)."""
        return self.verb_position == 0

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
            "all_verbs": list(self.all_verbs),
            "all_targets": [list(t) for t in self.all_targets],
            "target_kind": self.target_kind,
            "verb_source": self.verb_source,
            "verb_position": self.verb_position,
            "score_parts": [list(p) for p in self.score_parts],
            "negated": self.negated,
            "is_imperative": self.is_imperative,
        }


@dataclasses.dataclass(frozen=True)
class BatchStats:
    """Aggregate stats for a batch validation run."""
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
    stripped = match_text.lower().rstrip(".")
    if stripped in _FALSE_FILE_MATCHES:
        return True
    if _VERSION_RE.match(match_text):
        return True
    if _NUMBERED_REF_RE.match(match_text):
        return True
    return False


def _starts_with_verb(text: str) -> bool:
    """Return True if text starts with an action verb."""
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

def find_verb_with_position(text: str, limit: int = 0) -> VerbMatch | None:
    """Return a VerbMatch for the first action verb in *text*, or None.

    Checks phrasal verbs first, then base forms, then inflected forms.
    The source field indicates how the verb was matched:
      "text"      — direct base-form match
      "phrasal"   — two-word phrasal verb
      "inflected" — inflected form normalized to base
    """
    search_text = text[:limit] if limit else text
    words = re.findall(r"[a-zA-Z]+", search_text.lower())
    for i, word in enumerate(words):
        # Phrasal: base form
        if i + 1 < len(words):
            next_word = words[i + 1]
            if word in _PHRASAL_FIRST and next_word in _PHRASAL_FIRST[word]:
                return VerbMatch(
                    verb=_PHRASAL_FIRST[word][next_word],
                    token_index=i,
                    source="phrasal",
                )
            # Phrasal: inflected head
            if word in _PHRASAL_INFLECTED and next_word in _PHRASAL_INFLECTED[word]:
                return VerbMatch(
                    verb=_PHRASAL_INFLECTED[word][next_word],
                    token_index=i,
                    source="inflected",
                )
        # Base form
        if word in ACTION_VERBS:
            return VerbMatch(verb=word, token_index=i, source="text")
        # Inflected single-word
        if word in _INFLECTION_MAP:
            return VerbMatch(
                verb=_INFLECTION_MAP[word],
                token_index=i,
                source="inflected",
            )
    return None


def find_verb(text: str, limit: int = 0) -> str | None:
    """Return the first action verb found in *text*, or None.

    Thin wrapper over find_verb_with_position().
    """
    match = find_verb_with_position(text, limit=limit)
    return match.verb if match else None


def find_all_verbs(text: str) -> list[str]:
    """Return all action verbs in *text* (deduped, order-preserving)."""
    words = re.findall(r"[a-zA-Z]+", text.lower())
    result: list[str] = []
    seen: set[str] = set()
    skip_next = False
    for i, word in enumerate(words):
        if skip_next:
            skip_next = False
            continue
        if i + 1 < len(words):
            nw = words[i + 1]
            v = None
            if word in _PHRASAL_FIRST and nw in _PHRASAL_FIRST[word]:
                v = _PHRASAL_FIRST[word][nw]
            elif word in _PHRASAL_INFLECTED and nw in _PHRASAL_INFLECTED[word]:
                v = _PHRASAL_INFLECTED[word][nw]
            if v is not None:
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


def find_target(text: str) -> tuple[str, str]:
    """Return (target_string, target_kind) or ('', '').

    Checks patterns in priority order -- most specific first.
    """
    for m in FILE_RE.finditer(text):
        if not _is_false_file_match(m.group()):
            return m.group(), "file"
    m = SPECIAL_FILE_RE.search(text)
    if m:
        return m.group(), "file"
    m = PATH_RE.search(text)
    if m:
        return m.group(), "path"
    m = FUNC_RE.search(text)
    if m:
        return m.group(), "func"
    m = CHANNEL_RE.search(text)
    if m:
        return m.group(), "channel"
    m = ENV_VAR_RE.search(text)
    if m:
        return m.group(), "env"
    m = CONST_RE.search(text)
    if m:
        return m.group(), "const"
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
    if _LOWERCASE_START_RE.match(stripped):
        if not _starts_with_verb(stripped) and not _starts_with_file(stripped):
            return "starts lowercase (not a verb or file)"
    head = stripped[:80].lower()
    for signal in _HARD_ARTIFACT_SIGNALS:
        if signal in head:
            return "artifact detected: %r" % signal
    return ""


def _find_all_targets(text: str) -> tuple[tuple[str, str], ...]:
    """Return all targets in *text* as ((target, kind), ...)."""
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
    """Normalize a target string for dedup."""
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
    """Count distinct concrete targets in *text* after canonicalization."""
    raw_targets: list[str] = []
    for pattern in (FILE_RE, PATH_RE, ENV_VAR_RE, CONST_RE, TOOL_RE, CLI_RE, DISCUSSION_RE, CHANNEL_RE):
        for m in pattern.finditer(text):
            raw_targets.append(m.group())
    canonical: list[str] = []
    for t in raw_targets:
        c = canonicalize_target(t)
        if c:
            canonical.append(c)
    canonical_sorted = sorted(set(canonical), key=len, reverse=True)
    unique: list[str] = []
    for c in canonical_sorted:
        if not any(c in u for u in unique):
            unique.append(c)
    return len(unique)


# ---------------------------------------------------------------------------
# Scoring decomposition (consolidated from PRs #279-#284)
# ---------------------------------------------------------------------------

def _compute_score_parts(
    text: str,
    verb: str | None,
    target: str | None,
    target_kind: str,
) -> list[tuple[str, float]]:
    """Single source of truth for score decomposition.

    Returns named (component, value) pairs that sum to the raw score.
    Both compute_score() and score_breakdown() use this.
    """
    parts: list[tuple[str, float]] = []
    if verb:
        parts.append(("verb", 2.5))
    if target:
        kind_scores = {
            "file": 4.0, "path": 3.5, "func": 3.0, "module": 3.0,
            "tool": 3.0, "cli": 3.0, "env": 3.0, "const": 2.5,
            "discussion": 2.0, "channel": 2.0, "quoted": 1.5,
        }
        parts.append(("target", kind_scores.get(target_kind, 1.5)))
    words = text.split()
    if len(words) >= 15:
        parts.append(("length", 1.5))
    elif len(words) >= 8:
        parts.append(("length", 0.5))
    unique = count_unique_targets(text)
    if unique >= 2:
        parts.append(("multi_target", 1.0))
    if verb and _starts_with_verb(text):
        parts.append(("imperative", 0.5))
    return parts


def compute_score(
    text: str,
    verb: str | None,
    target: str | None,
    target_kind: str,
) -> float:
    """Compute a 0.0-1.0 specificity score."""
    parts = _compute_score_parts(text, verb, target, target_kind)
    raw = sum(v for _, v in parts)
    return min(raw / 10.0, 1.0)


def score_breakdown(text: str, tags: list = None) -> dict:
    """Return a decomposed scoring dict for diagnostic use.

    Keys: each named component, plus 'total_raw' and 'score' (normalized).
    """
    tags = tags or []
    # Run through the same analysis pipeline as validate_seed
    normalized = normalize_proposal(text)
    analysis_text = normalized if normalized != text else text
    verb = find_verb(analysis_text)
    target, target_kind = find_target(analysis_text)
    # Fallback: if normalization stripped the verb, check raw
    if not verb and normalized != text:
        verb = find_verb(text)
    # Tag-implied verb
    if not verb:
        tag_set = frozenset(t.lower().strip() for t in tags)
        for tag in tag_set:
            if tag in TAG_IMPLIED_VERBS:
                verb = TAG_IMPLIED_VERBS[tag]
                break
    parts = _compute_score_parts(text, verb, target, target_kind)
    result: dict[str, float] = {}
    for name, value in parts:
        result[name] = value
    result["total_raw"] = sum(v for _, v in parts)
    result["score"] = min(result["total_raw"] / 10.0, 1.0)
    return result


def explain(text: str, tags: list = None) -> str:
    """Return a human-readable diagnostic string for a proposal.

    Format is intentionally unstable -- do not parse programmatically.
    Use validate_seed() or score_breakdown() for structured data.
    """
    result = validate_seed(text, tags)
    parts_dict = score_breakdown(text, tags)

    status = "PASS" if result.passed else "FAIL"
    if result.junk:
        status = "JUNK"

    lines: list[str] = []
    lines.append(f"{status} ({result.score:.2f} {result.confidence or 'n/a'})")

    if result.verb_found:
        pos_str = f" at position {result.verb_position}" if result.verb_position is not None else ""
        src_str = f" [{result.verb_source}]" if result.verb_source else ""
        lines.append(f"  verb: '{result.verb_found}'{pos_str}{src_str}")
    else:
        lines.append("  verb: none")

    if result.target_found:
        lines.append(f"  target: '{result.target_found}' ({result.target_kind})")
    else:
        lines.append("  target: none")

    if result.negated:
        lines.append("  negated: yes (primary verb negated)")

    # Score components
    comp_strs = []
    for name in ("verb", "target", "length", "multi_target", "imperative"):
        if name in parts_dict:
            comp_strs.append(f"{name}={parts_dict[name]:.1f}")
    if comp_strs:
        lines.append(f"  score: [{', '.join(comp_strs)}]")

    if result.reasons:
        lines.append(f"  reasons: {'; '.join(result.reasons)}")
    if result.advisory:
        lines.append(f"  advisory: {result.advisory}")

    return "\n".join(lines)


def suggest(text: str, tags: list = None) -> list[str]:
    """Return actionable suggestions for improving a rejected proposal."""
    result = validate_seed(text, tags)
    if result.passed:
        return []
    suggestions: list[str] = []
    if result.negated:
        suggestions.append(
            "Rephrase as a positive action (what TO do, not what NOT to do)"
        )
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

    # -- Commit-prefix normalization ---
    normalized = normalize_proposal(text)
    commit_prefix_verb = _extract_commit_prefix_verb(text)
    analysis_text = normalized if normalized != text else text

    # -- Junk check (on normalized text) ---
    junk_limit = 60 if mode == "purge" else 0
    junk_reason = is_junk(analysis_text, limit=junk_limit)
    if junk_reason:
        return SeedGateResult(
            passed=False, reasons=(junk_reason,), score=0.0,
            verb_found=None, target_found=None, junk=True,
        )

    # -- Verb + target (on normalized text, fallback to raw) ---
    verb_limit = 200 if mode == "purge" else 0
    verb_match = find_verb_with_position(analysis_text, limit=verb_limit)
    verb = verb_match.verb if verb_match else None
    verb_source = verb_match.source if verb_match else ""
    verb_position = verb_match.token_index if verb_match else None

    # Fallback: if normalization killed the verb, try raw text
    if not verb and normalized != text:
        raw_match = find_verb_with_position(text, limit=verb_limit)
        if raw_match:
            verb_match = raw_match
            verb = raw_match.verb
            verb_source = raw_match.source
            verb_position = raw_match.token_index

    # Commit-prefix verb as last-resort fallback
    if not verb and commit_prefix_verb:
        verb = commit_prefix_verb
        verb_source = "commit_prefix"
        verb_position = None

    target, target_kind = find_target(analysis_text)
    # Fallback target from raw text
    if not target and normalized != text:
        target, target_kind = find_target(text)

    # -- Tag-implied verb inference (#12530) ---
    if not verb:
        for tag in tag_set:
            if tag in TAG_IMPLIED_VERBS:
                verb = TAG_IMPLIED_VERBS[tag]
                verb_source = "tag"
                verb_position = None
                break

    # -- Question stem inference (exempt tags only) ---
    if not verb and is_exempt:
        m = _QUESTION_STEM_RE.match(analysis_text.strip())
        if m:
            verb = QUESTION_STEMS.get(m.group().lower())
            if verb:
                verb_source = "question"
                verb_position = None

    # -- Negation check (on original text, main clause only) ---
    negated = detect_negation(text)

    # -- Rich match info ---
    all_verbs = tuple(find_all_verbs(analysis_text))
    all_targets = _find_all_targets(analysis_text)

    # -- Soft artifact check ---
    if is_soft_artifact(analysis_text) and not (verb and target) and not is_exempt:
        return SeedGateResult(
            passed=False,
            reasons=("soft artifact signal without redeeming verb+target",),
            score=0.0, verb_found=verb, target_found=target or None,
            junk=True, all_verbs=all_verbs, all_targets=all_targets,
            target_kind=target_kind, verb_source=verb_source,
            verb_position=verb_position, negated=negated,
        )

    # -- Score parts (single source of truth) ---
    score_parts = _compute_score_parts(text, verb, target, target_kind)

    # -- Decision ---
    if mode == "purge":
        passed = True
        specificity = 0.5
    else:
        passed = bool(verb) and (bool(target) or is_exempt)
        specificity = compute_score(text, verb, target, target_kind)
        # Negation hard-fail: only when primary verb is negated
        if passed and negated:
            passed = False

    reasons: list[str] = []
    advisory = ""
    if not passed:
        if negated and verb:
            reasons.append("Primary verb is negated")
        if not verb:
            reasons.append("No action verb found")
        if not target and not is_exempt:
            reasons.append("No concrete target (filename, tool, or reference)")
        if verb and not target and not is_exempt and not negated:
            advisory = "needs-specificity"
    elif verb and not target:
        advisory = "needs-specificity"

    return SeedGateResult(
        passed=passed, reasons=tuple(reasons), score=specificity,
        verb_found=verb or None, target_found=target or None, junk=False,
        advisory=advisory, all_verbs=all_verbs, all_targets=all_targets,
        target_kind=target_kind, verb_source=verb_source,
        verb_position=verb_position,
        score_parts=tuple(score_parts),
        negated=negated,
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
    """Validate a batch of proposals; separate junk from merely-failed."""
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
    args = sys.argv[1:]
    if not args:
        print("Usage: python -m seed_gate '<proposal text>' [tag1 tag2 ...]")
        print("       python -m seed_gate --explain '<proposal text>'")
        print("       python -m seed_gate --batch < proposals.txt")
        sys.exit(1)

    import json as _json

    # --explain mode
    if args[0] == "--explain":
        text = args[1] if len(args) > 1 else ""
        tags = args[2:] if len(args) > 2 else []
        print(explain(text, tags))
        sys.exit(0)

    # --batch mode: read from stdin
    if args[0] == "--batch":
        import sys as _sys
        results = []
        for line in _sys.stdin:
            line = line.strip()
            if not line:
                continue
            # Support JSONL: {"text": "...", "tags": [...]}
            try:
                obj = _json.loads(line)
                text = obj.get("text", line)
                tags = obj.get("tags", [])
            except (ValueError, _json.JSONDecodeError):
                text = line
                tags = []
            results.append({"text": text, "result": validate(text, tags)})
        print(_json.dumps(results, indent=2))
        sys.exit(0)

    text = args[0]
    tags = args[1:] if len(args) > 1 else []
    result = validate(text, tags)
    print(_json.dumps(result, indent=2))
    sys.exit(0 if result["passed"] else 1)


if __name__ == "__main__":
    _cli()
