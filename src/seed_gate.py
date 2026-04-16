"""seed_gate.py -- canonical specificity validator for seed proposals.

Consolidates ideas from 6 independent implementations (#12503, #12505,
#12507, #12511, #12521, #12530) and PRs #245, #246, #247, #253, #256,
#272, #273 into one validator that checks for an *action verb* plus a
*concrete target* (filename, tool name, path, or discussion reference).

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
    verb   = find_verb(text)          # str | None
    target = find_target(text)        # (str, str) pair
    junk   = is_junk(text)            # str (reason) or empty
    score  = compute_score(text, verb, target, kind)  # float

    # Diagnostic APIs
    explanation = explain(text, tags)           # str
    breakdown   = score_breakdown(text, tags)   # dict
    vm          = find_verb_with_position(text)  # VerbMatch | None

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
    PR #272  -- diagnostic APIs: explain(), score_breakdown(),
                  find_verb_with_position(); commit-prefix handling
                  (fix: build → accepted); _KIND_SCORES module constant;
                  "redesign" verb.
    PR #289  -- 6-agent consolidation: negation awareness (#12503),
                  VerbMatch dataclass (#12507), proper noun targets
                  (#12503), placeholder penalty (#12530), abbreviated
                  ref filter (#12505), verb weights (#12511/#12530),
                  BatchResult.summary() (#12521), shared _analyze() spine.
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
    "redesign",
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
# Negation detection (#12503 consolidation)
# ---------------------------------------------------------------------------

_NEGATION_WORDS: frozenset[str] = frozenset({
    "don't", "dont", "do not",
    "never", "avoid", "stop", "cease", "halt",
    "prevent", "prohibit",
    "not", "no",
    "can't", "cant", "cannot",
    "won't", "wont", "will not",
    "shouldn't", "shouldnt", "should not",
    "doesn't", "doesnt", "does not",
    "didn't", "didnt", "did not",
    "isn't", "isnt", "is not",
    "aren't", "arent", "are not",
})

# Two-word negation phrases that need special handling
_TWO_WORD_NEGATIONS: frozenset[str] = frozenset({
    "do not", "will not", "should not", "does not",
    "did not", "is not", "are not",
})

# Idioms where negation does NOT negate the following verb
_NEGATION_EXCEPTIONS: tuple[str, ...] = (
    "not only", "not just", "not merely",
)

# Clause boundary words -- negation doesn't cross these
_CLAUSE_BOUNDARIES: frozenset[str] = frozenset({
    "but", "however", "although", "though", "yet",
    "while", "whereas", "instead", "rather",
})

# Clause boundary punctuation
_CLAUSE_BOUNDARY_PUNCT: frozenset[str] = frozenset({",", ";", "—", "–", ":"})


def _is_negated(text: str, verb_position: int) -> bool:
    """Check if the verb at word position is negated.

    Scans backward from verb_position within a 3-word window.
    Clause boundaries (punctuation or conjunctions) reset the window.
    Respects idioms like 'not only' that don't negate.
    """
    words = text.split()
    if verb_position <= 0 or verb_position >= len(words):
        return False

    # Check for "not only"/"not just" idioms around the verb
    text_lower = text.lower()
    for exc in _NEGATION_EXCEPTIONS:
        if exc in text_lower:
            # If the exception phrase is near the verb, don't treat as negation
            exc_pos = text_lower.find(exc)
            verb_word = words[verb_position].lower()
            verb_text_pos = text_lower.find(verb_word, exc_pos)
            if verb_text_pos != -1 and verb_text_pos - exc_pos < 30:
                return False

    # Scan backward up to 3 non-filler words (or clause boundary)
    words_checked = 0
    for i in range(verb_position - 1, -1, -1):
        if words_checked >= 3:
            break
        w = words[i].lower().rstrip(":,;!?.")
        # Check for clause boundary punctuation in the word
        if any(p in words[i] for p in _CLAUSE_BOUNDARY_PUNCT):
            break
        if w in _CLAUSE_BOUNDARIES:
            break
        # Check two-word negations (e.g., "do not")
        if i > 0:
            two_word = words[i - 1].lower().rstrip(":,;!?.") + " " + w
            if two_word in _TWO_WORD_NEGATIONS:
                return True
        if w in _NEGATION_WORDS:
            return True
        words_checked += 1

    return False


# ---------------------------------------------------------------------------
# Verb weight categories (#12511, #12530 -- diagnostic metadata only)
# ---------------------------------------------------------------------------

_VERB_WEIGHTS: dict[str, str] = {}
_HIGH_ACTIVATION: frozenset[str] = frozenset({
    "build", "create", "implement", "write", "design", "develop",
    "deploy", "launch", "ship", "release", "containerize", "scaffold",
    "bootstrap", "provision", "rewrite", "redesign",
})
_LOW_ACTIVATION: frozenset[str] = frozenset({
    "explore", "investigate", "consider", "debate", "discuss",
    "propose", "plan", "evaluate", "assess", "review", "document",
    "map", "diagram",
})
for _v in ACTION_VERBS:
    if _v in _HIGH_ACTIVATION:
        _VERB_WEIGHTS[_v] = "high"
    elif _v in _LOW_ACTIVATION:
        _VERB_WEIGHTS[_v] = "low"
    else:
        _VERB_WEIGHTS[_v] = "medium"


def verb_weight(verb: str) -> str:
    """Return the activation weight category for a verb: high/medium/low."""
    return _VERB_WEIGHTS.get(verb, "medium")


# ---------------------------------------------------------------------------
# Target regex patterns (compiled once)
# ---------------------------------------------------------------------------

# File-like: foo.py, bar_baz.rs, my-lib.js, state/agents.json
FILE_RE = re.compile(r"\b[\w./-]*\w+\.\w{1,8}\b")

# False positives that FILE_RE catches (abbreviations with periods)
_FALSE_FILE_MATCHES: frozenset[str] = frozenset({
    "e.g", "i.e", "a.m", "p.m", "vs.",
})

# Abbreviated references: fig.1, sec.2, vol.3, no.5 (#12505)
_ABBREV_REF_RE = re.compile(r"^(?:fig|sec|vol|no|eq|ch|pt|ex|app)\.\d+$", re.I)

# Version strings: 2.0.1, v1.2.3, 1.0 -- should NOT match as file targets
_VERSION_RE = re.compile(r"^v?\d+\.\d+(?:\.\d+)?(?:[+.-]\w+)?$")

# Environment variable references: $STATE_DIR, ${GITHUB_TOKEN}
ENV_VAR_RE = re.compile(r"\$\{?[A-Z][A-Z0-9_]+\}?")

# Conventional commit prefix: fix:, feat(scope):, chore!: etc.
_COMMIT_PREFIX_RE = re.compile(
    r"^(?:fix|feat|chore|docs|style|refactor|perf|test|build|ci|revert)"
    r"(?:\([^)]*\))?!?:\s*"
)

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

# Proper noun targets -- last-resort detection (#12503)
PROPER_NOUN_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,4}\b")
_PROPER_NOUN_STOPWORDS: frozenset[str] = frozenset({
    "The", "A", "An", "This", "That", "Some", "Any", "Each", "Every",
    "My", "Your", "Our", "Their", "Its", "His", "Her",
    "What", "How", "Why", "When", "Where", "Which",
    "I", "We", "You", "They", "He", "She", "It",
})
_PROPER_NOUN_SUFFIXES: frozenset[str] = frozenset({
    "protocol", "model", "framework", "engine", "system", "pattern",
    "algorithm", "module", "service", "handler", "controller",
    "processor", "manager", "layer", "interface", "pipeline",
    "doctrine", "amendment", "principle", "architecture",
})

# Placeholder filenames -- soft score penalty (#12530)
_PLACEHOLDER_FILES: frozenset[str] = frozenset({
    "test.py", "foo.py", "bar.py", "baz.py", "example.json",
    "main.py", "index.js", "index.html", "app.py", "temp.py",
    "tmp.py", "sample.py", "demo.py", "hello.py", "scratch.py",
    "untitled.py", "test.js", "foo.js", "bar.js", "example.py",
})

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

# Two-tier artifact signals (rubber-duck advised split):
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
# Dataclass results
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class VerbMatch:
    """Structured result from verb detection (#12507).

    Attributes:
        verb: The canonical (base) verb form.
        position: 0-based word index in the original text.
        origin: Where the verb came from: "text", "tag", or "question".
        match_kind: How the verb was matched: "explicit", "inflected",
                    "phrasal", or "implied".
    """
    verb: str
    position: int
    origin: str     # "text" | "tag" | "question"
    match_kind: str  # "explicit" | "inflected" | "phrasal" | "implied"


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
    negated: bool = False
    verb_match: object = None  # VerbMatch | None

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
    def strength(self) -> str:
        """Derive strength tier from score: strong/moderate/weak/none."""
        if self.score >= 0.65:
            return "strong"
        if self.score >= 0.35:
            return "moderate"
        if self.score > 0.0:
            return "weak"
        return "none"

    def to_dict(self) -> dict:
        d = {
            "passed": self.passed,
            "reasons": list(self.reasons),
            "score": self.score,
            "verb_found": self.verb_found,
            "target_found": self.target_found,
            "junk": self.junk,
            "advisory": self.advisory,
            "confidence": self.confidence,
            "strength": self.strength,
            "negated": self.negated,
            "all_verbs": list(self.all_verbs),
            "all_targets": [list(t) for t in self.all_targets],
        }
        if self.verb_match is not None:
            d["verb_match"] = dataclasses.asdict(self.verb_match)
        else:
            d["verb_match"] = None
        if self.verb_found:
            d["verb_weight"] = verb_weight(self.verb_found)
        else:
            d["verb_weight"] = None
        return d


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

    def summary(self) -> str:
        """Human-readable batch diagnostics (#12521)."""
        s = self.stats
        return (
            f"batch: {s.total} total, {s.passed} passed ({s.pass_rate:.0%}), "
            f"{s.failed} failed, {s.junk} junk ({s.junk_rate:.0%})"
        )


# Module-level score weights (shared by compute_score and score_breakdown)
_KIND_SCORES: dict[str, float] = {
    "file": 4.0, "path": 3.5, "func": 3.0, "module": 3.0,
    "tool": 3.0, "cli": 3.0, "env": 3.0, "const": 2.5,
    "discussion": 2.0, "channel": 2.0, "quoted": 1.5,
    "proper_noun": 1.0,
}

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
    if _ABBREV_REF_RE.match(match_text):
        return True
    return False


def _is_placeholder_file(match_text: str) -> bool:
    """Return True if a file match is a generic placeholder."""
    return match_text.lower() in _PLACEHOLDER_FILES


def _starts_with_verb(text: str) -> bool:
    """Return True if text starts with an action verb."""
    words = text.split()
    if not words:
        return False
    first = words[0].lower().rstrip(":,;!?.")
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


def _strip_commit_prefix(text: str) -> str:
    """Strip a conventional commit prefix (fix:, feat(scope):, etc.)."""
    return _COMMIT_PREFIX_RE.sub("", text)


def _starts_with_file(text: str) -> bool:
    """Return True if text starts with a file-like pattern."""
    m = _FILE_START_RE.match(text.strip())
    if m:
        return not _is_false_file_match(m.group())
    return False


def _detect_proper_noun(text: str) -> tuple[str, str]:
    """Detect a proper noun target as last resort.

    Only matches capitalized multi-word phrases with a recognized
    substance suffix (Protocol, Model, etc.) or 3+ capitalized words.
    Skips stopwords and action verbs at the start of the phrase.
    Returns (target, 'proper_noun') or ('', '').
    """
    for m in PROPER_NOUN_RE.finditer(text):
        phrase = m.group()
        words = phrase.split()
        # Strip leading stopwords and action verbs
        while words and (words[0] in _PROPER_NOUN_STOPWORDS or words[0].lower() in ACTION_VERBS):
            words = words[1:]
        if len(words) < 2:
            continue
        phrase = " ".join(words)
        # Accept if last word is a substance suffix
        if words[-1].lower() in _PROPER_NOUN_SUFFIXES:
            return phrase, "proper_noun"
        # Accept if 3+ capitalized words (high specificity)
        if len(words) >= 3:
            return phrase, "proper_noun"
    return "", ""


# ---------------------------------------------------------------------------
# Public composable helpers
# ---------------------------------------------------------------------------

def find_verb(text: str, limit: int = 0) -> str | None:
    """Return the first action verb found in *text*, or None."""
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
    """Return all action verbs in *text* (deduped, order-preserving)."""
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


def find_verb_with_position(text: str, limit: int = 0) -> VerbMatch | None:
    """Return a VerbMatch or None.

    Checks phrasal verbs first (highest specificity), then single-word
    verbs (explicit), then inflected forms.
    """
    check = text[:limit] if limit else text
    words = check.split()
    if not words:
        return None
    # Pass 1: phrasal verbs
    for i, w in enumerate(words[:-1]):
        lo = w.lower().rstrip(":,;!?.")
        nxt = words[i + 1].lower()
        if lo in _PHRASAL_FIRST and nxt in _PHRASAL_FIRST[lo]:
            return VerbMatch(
                verb=_PHRASAL_FIRST[lo][nxt], position=i,
                origin="text", match_kind="phrasal",
            )
        if lo in _PHRASAL_INFLECTED and nxt in _PHRASAL_INFLECTED[lo]:
            return VerbMatch(
                verb=_PHRASAL_INFLECTED[lo][nxt], position=i,
                origin="text", match_kind="phrasal",
            )
    # Pass 2: single-word verbs
    for i, w in enumerate(words):
        lo = w.lower().rstrip(":,;!?.")
        if lo in ACTION_VERBS:
            return VerbMatch(
                verb=lo, position=i,
                origin="text", match_kind="explicit",
            )
        base = _INFLECTION_MAP.get(lo)
        if base:
            return VerbMatch(
                verb=base, position=i,
                origin="text", match_kind="inflected",
            )
    return None


def find_target(text: str) -> tuple[str, str]:
    """Return (target_string, target_kind) or ('', '').

    Checks patterns in priority order -- most specific first.
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
    # Channel before tool
    m = CHANNEL_RE.search(text)
    if m:
        return m.group(), "channel"
    # Environment variables
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
    # Last resort: proper noun detection (#12503)
    proper, kind = _detect_proper_noun(text)
    if proper:
        return proper, kind
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
    # Hard artifact signals -- always fail (first 80 chars) -- checked BEFORE lowercase
    head = stripped[:80].lower()
    for signal in _HARD_ARTIFACT_SIGNALS:
        if signal in head:
            return "artifact detected: %r" % signal
    # Smart lowercase handling
    if _LOWERCASE_START_RE.match(stripped):
        if not _starts_with_verb(stripped) and not _starts_with_file(stripped):
            without_prefix = _strip_commit_prefix(stripped)
            if without_prefix == stripped or (
                not _starts_with_verb(without_prefix) and not _starts_with_file(without_prefix)
            ):
                return "starts lowercase (not a verb or file)"
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
    # Proper noun targets as last resort
    proper, kind = _detect_proper_noun(text)
    if proper:
        _add(proper, kind)
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
        raw += _KIND_SCORES.get(target_kind, 1.5)
    words = text.split()
    if len(words) >= 8:
        raw += 0.5
    if len(words) >= 15:
        raw += 1.0
    unique = count_unique_targets(text)
    if unique >= 2:
        raw += 1.0
    # Imperative bonus
    if verb and _starts_with_verb(text):
        raw += 0.5
    # Placeholder penalty (#12530) -- soft, never flips pass to fail
    if target and _is_placeholder_file(target):
        raw -= 0.5
    return min(max(raw / 10.0, 0.0), 1.0)


def suggest(text: str, tags: list = None) -> list[str]:
    """Return actionable suggestions for improving a rejected proposal."""
    result = validate_seed(text, tags)
    if result.passed:
        return []
    suggestions: list[str] = []
    if result.negated:
        suggestions.append(
            "Remove the negation — proposals should describe what TO do, not what to avoid"
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
# Shared analysis spine (consolidation from PR #272, #273)
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class _Analysis:
    """Internal analysis result -- single source of truth."""
    text: str
    junk_reason: str
    verb: str | None
    verb_match: VerbMatch | None
    target: str
    target_kind: str
    negated: bool
    is_exempt: bool
    tag_set: frozenset
    all_verbs: tuple
    all_targets: tuple


def _analyze(text: str, tags: list = None, *, mode: str = "admission") -> _Analysis:
    """Run the full analysis pipeline once. All public APIs render from this."""
    tags = tags or []
    tag_set = frozenset(t.lower().strip() for t in tags)
    is_exempt = bool(tag_set & EXEMPT_TAGS)

    # Junk check
    junk_limit = 60 if mode == "purge" else 0
    junk_reason = is_junk(text, limit=junk_limit)

    # Strip commit prefix for verb detection
    stripped = _strip_commit_prefix(text)
    search_text = stripped if stripped != text else text

    # Verb detection
    vm = find_verb_with_position(search_text)
    verb = vm.verb if vm else None

    # Tag-implied verb inference
    if not verb:
        for tag in tag_set:
            if tag in TAG_IMPLIED_VERBS:
                verb = TAG_IMPLIED_VERBS[tag]
                vm = VerbMatch(verb=verb, position=-1, origin="tag", match_kind="implied")
                break

    # Question stem inference (exempt tags only)
    if not verb and is_exempt:
        m = _QUESTION_STEM_RE.match(text.strip())
        if m:
            verb = QUESTION_STEMS.get(m.group().lower())
            if verb:
                vm = VerbMatch(verb=verb, position=-1, origin="question", match_kind="implied")

    # Negation detection -- only for explicit text verbs
    negated = False
    if vm and vm.origin == "text" and vm.position >= 0:
        negated = _is_negated(search_text, vm.position)

    # Target detection
    target, target_kind = find_target(text)

    # Rich match info
    all_verbs = tuple(find_all_verbs(text))
    all_targets = _find_all_targets(text)

    return _Analysis(
        text=text, junk_reason=junk_reason,
        verb=verb, verb_match=vm,
        target=target, target_kind=target_kind,
        negated=negated, is_exempt=is_exempt, tag_set=tag_set,
        all_verbs=all_verbs, all_targets=all_targets,
    )


# ---------------------------------------------------------------------------
# Diagnostic APIs (PR #272 + consolidation)
# ---------------------------------------------------------------------------

def score_breakdown(text: str, tags: list = None) -> dict[str, float]:
    """Return component-by-component score decomposition."""
    a = _analyze(text, tags)
    components: dict[str, float] = {
        "verb": 0.0, "target": 0.0, "length": 0.0,
        "multi_target": 0.0, "imperative": 0.0, "placeholder": 0.0,
    }
    if a.verb:
        components["verb"] = 2.5
    if a.target:
        components["target"] = _KIND_SCORES.get(a.target_kind, 1.5)
    words = text.split()
    if len(words) >= 8:
        components["length"] += 0.5
    if len(words) >= 15:
        components["length"] += 1.0
    unique = count_unique_targets(text)
    if unique >= 2:
        components["multi_target"] = 1.0
    if a.verb and _starts_with_verb(text):
        components["imperative"] = 0.5
    if a.target and _is_placeholder_file(a.target):
        components["placeholder"] = -0.5
    components["total"] = sum(components.values())
    return components


def explain(text: str, tags: list = None) -> str:
    """Return a human-readable diagnostic string for a proposal."""
    result = validate_seed(text, tags)
    parts: list[str] = []
    parts.append("PASS" if result.passed else "FAIL")
    parts.append("verb=%s" % (result.verb_found or "none"))
    if result.verb_found:
        parts.append("weight=%s" % verb_weight(result.verb_found))
    parts.append("target=%s" % (result.target_found or "none"))
    parts.append("score=%.2f" % result.score)
    if result.confidence:
        parts.append("confidence=%s" % result.confidence)
    if result.negated:
        parts.append("negated=true")
    if result.junk:
        parts.append("junk=true")
    if result.advisory:
        parts.append("advisory=%s" % result.advisory)
    if result.reasons:
        parts.append("reasons=[%s]" % "; ".join(result.reasons))
    tips = suggest(text, tags)
    if tips:
        parts.append("suggestions=[%s]" % "; ".join(tips))
    return " | ".join(parts)


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
    a = _analyze(text, tags, mode=mode)

    # -- Junk check (hard fail) ---
    if a.junk_reason:
        return SeedGateResult(
            passed=False, reasons=(a.junk_reason,), score=0.0,
            verb_found=None, target_found=None, junk=True,
            negated=False, verb_match=None,
        )

    # -- Negation check (hard fail for text verbs) ---
    if a.negated:
        return SeedGateResult(
            passed=False, reasons=("Primary verb is negated",),
            score=0.0, verb_found=a.verb, target_found=a.target or None,
            junk=False, negated=True,
            all_verbs=a.all_verbs, all_targets=a.all_targets,
            verb_match=a.verb_match,
        )

    # -- Soft artifact check ---
    if is_soft_artifact(a.text) and not (a.verb and a.target) and not a.is_exempt:
        return SeedGateResult(
            passed=False,
            reasons=("soft artifact signal without redeeming verb+target",),
            score=0.0, verb_found=a.verb, target_found=a.target or None,
            junk=True, all_verbs=a.all_verbs, all_targets=a.all_targets,
            verb_match=a.verb_match,
        )

    # -- Decision ---
    if mode == "purge":
        passed = True
        specificity = 0.5
    else:
        passed = bool(a.verb) and (bool(a.target) or a.is_exempt)
        specificity = compute_score(a.text, a.verb, a.target, a.target_kind)

    reasons: list[str] = []
    advisory = ""
    if not passed:
        if not a.verb:
            reasons.append("No action verb found")
        if not a.target and not a.is_exempt:
            reasons.append("No concrete target (filename, tool, or reference)")
        if a.verb and not a.target and not a.is_exempt:
            advisory = "needs-specificity"
    elif a.verb and not a.target:
        advisory = "needs-specificity"

    return SeedGateResult(
        passed=passed, reasons=tuple(reasons), score=specificity,
        verb_found=a.verb or None, target_found=a.target or None, junk=False,
        advisory=advisory, all_verbs=a.all_verbs, all_targets=a.all_targets,
        negated=False, verb_match=a.verb_match,
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
    if len(sys.argv) < 2:
        print("Usage: python -m seed_gate '<proposal text>' [tag1 tag2 ...]")
        print("       python -m seed_gate --explain '<proposal text>' [tag1 tag2 ...]")
        print("       python -m seed_gate --batch '<text1>' '<text2>' ...")
        sys.exit(1)
    import json as _json
    if sys.argv[1] == "--explain":
        text = sys.argv[2] if len(sys.argv) > 2 else ""
        tags = sys.argv[3:] if len(sys.argv) > 3 else []
        print(explain(text, tags))
        sys.exit(0)
    if sys.argv[1] == "--batch":
        texts = sys.argv[2:]
        br = validate_batch(texts)
        print(br.summary())
        for text, result in br.passed_items:
            print("  PASS: %s" % text[:60])
        for text, result in br.failed_items:
            print("  FAIL: %s" % text[:60])
        for text, result in br.junk_items:
            print("  JUNK: %s" % text[:60])
        sys.exit(0)
    text = sys.argv[1]
    tags = sys.argv[2:] if len(sys.argv) > 2 else []
    result = validate(text, tags)
    print(_json.dumps(result, indent=2))
    sys.exit(0 if result["passed"] else 1)


if __name__ == "__main__":
    _cli()
