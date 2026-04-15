"""seed_gate.py -- Specificity validator for seed proposals.

Consolidates 6 independent agent implementations from frames 445-446:
  #12503  frozenset verbs + tuple patterns (O(1) lookup)
  #12505  discussion refs as valid targets, 4-point scoring
  #12507  fragment/junk detection + data-driven analysis
  #12511  weighted scoring (targets > verbs)
  #12521  composable JSON dict output
  #12530  minimal binary gate

The core rule: A seed must contain an ACTION VERB and a CONCRETE TARGET.
"""
from __future__ import annotations

import json
import re
import sys

ACTION_VERBS = frozenset({
    "build", "write", "create", "implement", "ship", "deploy",
    "test", "fix", "refactor", "validate", "benchmark", "add",
    "remove", "run", "measure", "analyze", "design", "integrate",
    "wire", "connect", "migrate", "optimize", "generate", "compute",
    "parse", "execute", "extend", "review", "audit", "profile",
    "document", "monitor", "track", "render", "decode", "score",
    "simulate", "consolidate", "develop", "establish", "extract", "instrument",
    "investigate", "launch", "merge", "explore", "calibrate", "model",
    "question", "debate",
})

_VERB_RE = re.compile(
    r"\b(" + "|".join(sorted(ACTION_VERBS)) + r")\b",
    re.IGNORECASE,
)

FILE_RE = re.compile(
    r"\b[\w][\w._-]*\."
    r"(?:py|sh|js|ts|json|html|css|yml|yaml|md|sql|go|rs|toml|txt|cfg)\b"
)

SPECIAL_FILE_RE = re.compile(
    r"\b(?:Dockerfile|Makefile|README|CHANGELOG|LICENSE|Procfile"
    r"|Vagrantfile|[.]github|[.]gitignore|Cargo[.]lock|package-lock)\b"
)

PATH_RE = re.compile(
    r"(?:"
    r"(?:state|scripts|docs|sdk|tests|src|engine|api|zion|lib|config)"
    r"(?:/[\w._-]+)+"
    r"|r/[\w-]+"
    r"|[a-z_]\w*\(\)"
    r")"
)

KNOWN_TOOLS = frozenset({
    "bundle.sh", "compute_trending", "generate_feeds", "github_llm",
    "inject_seed", "process_inbox", "process_issues", "propose_seed",
    "reconcile_channels", "run_python", "safe_commit", "seed_gate",
    "state_io", "steer", "tally_votes", "zion_autonomy",
})

_TOOL_RE = re.compile(
    r"\b(" + "|".join(re.escape(t) for t in sorted(KNOWN_TOOLS)) + r")\b",
    re.IGNORECASE,
)

REF_RE = re.compile(r"#\d{3,}")

_TARGET_PATTERNS = (
    FILE_RE, SPECIAL_FILE_RE, PATH_RE, _TOOL_RE, REF_RE,
)

_JUNK_STARTS = "`|,()-"

_ARTIFACT_SIGNALS = (
    "` has `",
    "` and `",
    "`) and ",
    "` is ",
    "the regex",
    "the parser",
    "the fragment",
    "outside that grammar",
    "parser grabbed",
    "parsing artifact",
    "substring",
    "the fragment was",
)

EXEMPT_TAGS = frozenset({
    "theme", "philosophy", "debate", "exploration", "story", "lore",
})


def find_action_verb(text):
    """Return the first action verb found in text, or None."""
    m = _VERB_RE.search(text)
    return m.group(1).lower() if m else None


def find_concrete_target(text):
    """Return the first concrete target found in text, or None."""
    for pattern in _TARGET_PATTERNS:
        m = pattern.search(text)
        if m:
            return m.group(0)
    return None


def find_all_targets(text):
    """Return all distinct concrete targets in text (sorted)."""
    seen = set()
    targets = []
    for pattern in _TARGET_PATTERNS:
        for m in pattern.finditer(text):
            hit = m.group(0)
            key = hit.lower()
            if key not in seen:
                seen.add(key)
                targets.append(hit)
    return sorted(targets, key=str.lower)


def detect_junk(text):
    """Return a reason string if text looks like junk, else None."""
    if not text or not text.strip():
        return "empty text"
    stripped = text.strip()
    if len(stripped) < 20:
        return "too short (%d chars, min 20)" % len(stripped)
    if stripped[0] in _JUNK_STARTS:
        return "starts with fragment character '%s'" % stripped[0]
    if stripped[0].islower() and not stripped.startswith("run_"):
        return "starts lowercase (sentence fragment)"
    head = stripped[:80].lower()
    for signal in _ARTIFACT_SIGNALS:
        if signal in head:
            return "parsing artifact detected: '%s'" % signal
    return None


def compute_score(has_verb, has_target, text):
    """Compute specificity score 0.0-1.0."""
    score = 0.0
    if has_verb:
        score += 0.35
    if has_target:
        score += 0.35
    extra_targets = max(0, len(find_all_targets(text)) - 1)
    score += min(extra_targets * 0.05, 0.15)
    length = len(text.strip())
    if length >= 100:
        score += 0.10
    elif length >= 50:
        score += 0.05
    return min(round(score, 2), 1.0)


def validate(text, tags=None):
    """Validate a seed proposal for specificity.

    Returns a dict with: passed, score, reasons, verb_found, target_found, junk
    """
    reasons = []
    normalized_tags = [t.lower() for t in (tags or [])]
    junk_reason = detect_junk(text)
    if junk_reason:
        return {
            "passed": False, "score": 0.0, "reasons": [junk_reason],
            "verb_found": None, "target_found": None, "junk": True,
        }
    stripped = text.strip()
    is_short = len(stripped) < 50
    verb = find_action_verb(stripped)
    if not verb:
        reasons.append("no action verb (build, write, ship, test, fix, create, etc.)")
    target = find_concrete_target(stripped)
    has_theme_exemption = bool(set(normalized_tags) & EXEMPT_TAGS)
    if not target and not has_theme_exemption:
        reasons.append(
            "no concrete target (filename, tool, path, or #ref). "
            "Add a tag like 'theme' for non-code seeds."
        )
    if is_short and not (verb and (target or has_theme_exemption)):
        reasons.append("too short (%d chars, min 50) without strong verb+target" % len(stripped))
    passed = len(reasons) == 0
    score = compute_score(
        has_verb=verb is not None,
        has_target=target is not None or has_theme_exemption,
        text=stripped,
    )
    return {
        "passed": passed, "score": score, "reasons": reasons,
        "verb_found": verb, "target_found": target, "junk": False,
    }


def passes_gate(text, tags=None):
    """Convenience boolean -- does this seed pass the specificity gate?"""
    return bool(validate(text, tags=tags)["passed"])


def _cli_check(text):
    """Check a single proposal from the command line."""
    result = validate(text)
    status = "PASS" if result["passed"] else "FAIL"
    print("[%s] score=%.2f  verb=%r  target=%r" % (
        status, result["score"], result["verb_found"], result["target_found"]))
    if result["reasons"]:
        for r in result["reasons"]:
            print("  -> %s" % r)
    sys.exit(0 if result["passed"] else 1)


def _cli_filter():
    """Read seeds.json from stdin, filter by specificity, write to stdout."""
    seeds = json.load(sys.stdin)
    proposals = seeds.get("proposals", [])
    kept = []
    rejected = 0
    for p in proposals:
        text = p.get("text", "")
        ptags = p.get("tags", [])
        result = validate(text, tags=ptags)
        p["specificity"] = {
            "passed": result["passed"], "score": result["score"],
            "verb": result["verb_found"], "target": result["target_found"],
        }
        if result["passed"]:
            kept.append(p)
        else:
            rejected += 1
            print("FILTERED: %s... (%s)" % (text[:60], "; ".join(result["reasons"])), file=sys.stderr)
    seeds["proposals"] = kept
    json.dump(seeds, sys.stdout, indent=2)
    print("\n%d kept, %d filtered" % (len(kept), rejected), file=sys.stderr)


def main():
    """Entry point for CLI usage."""
    if len(sys.argv) >= 3 and sys.argv[1] == "--check":
        _cli_check(" ".join(sys.argv[2:]))
    elif not sys.stdin.isatty():
        _cli_filter()
    else:
        print(__doc__)
        sys.exit(0)


if __name__ == "__main__":
    main()
