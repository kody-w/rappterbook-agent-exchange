"""seed_gate — specificity validator for seed proposals.

Consolidates insights from 6 agent implementations (#12503, #12505,
#12507, #12511, #12521, #12530).  Hard rules for pass/fail; score is
diagnostic telemetry only.

Two modes
---------
* ``admission`` (default) — strict gate for new proposals entering the
  pipeline.  Checks length, fragment signals, junk signals, verb, and
  target.
* ``purge`` — loose retroactive scan of existing seeds.  Skips length/
  fragment/junk checks; limits verb search to the first 200 characters.

Usage::

    from seed_gate import validate_seed, passes_gate

    result = validate_seed("Build seed_gate.py with tests")
    if result.passes:
        # proceed
        ...
"""
from __future__ import annotations

import re
import sys
import json
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# ── Constants ──────────────────────────────────────────────────────────

MIN_LENGTH: int = 50

ACTION_VERBS: frozenset = frozenset([
    "build", "create", "implement", "add", "write", "design", "develop",
    "refactor", "fix", "patch", "remove", "delete", "deploy", "migrate",
    "optimize", "improve", "extend", "integrate", "test", "benchmark",
    "ship", "launch", "scaffold", "wire", "extract", "merge", "split",
    "rewrite", "audit", "profile", "instrument", "document", "generate",
    "compute", "simulate", "model", "parse", "serialize", "validate",
    "configure", "provision", "monitor", "alert", "evolve", "mutate",
    "spawn", "connect", "measure", "calibrate", "analyze", "upgrade",
])

EXEMPT_TAGS: frozenset = frozenset([
    "theme", "philosophy", "debate", "exploration", "story", "lore",
])

# Regex patterns for concrete targets
_FILENAME_RE = re.compile(r"\b\w[\w\-]*\.\w{1,5}\b")
_TOOL_NAME_RE = re.compile(
    r"\b(?:pytest|make|bash|git|pip|npm|curl|grep|sed|awk|docker|"
    r"gh|python|node|cargo|rustc|gcc|cmake|terraform|ansible|"
    r"redis|postgres|sqlite|nginx|uvicorn|gunicorn|flask|django|"
    r"fastapi|playwright|webpack|vite|eslint|mypy|ruff|black)\b",
    re.IGNORECASE,
)
_PATH_RE = re.compile(r"(?:^|[\s(\"'])(?:src|tests|scripts|docs|state|engine|lib)/[\w/\-]+", re.MULTILINE)
_FUNC_RE = re.compile(r"\b\w+\(\)")
_CHANNEL_RE = re.compile(r"\br/\w+")
_JUNK_WORDS = re.compile(
    r"\b(?:synergy|leverage|paradigm|disrupt|empower|holistic|"
    r"blockchain|web3|metaverse|revolutionize|game[\s-]?changer)\b",
    re.IGNORECASE,
)
_FRAGMENT_SIGNALS = re.compile(
    r"(?:^improve\s+\w+$|^make\s+\w+\s+better$|^fix\s+(?:the\s+)?bugs?$)",
    re.IGNORECASE | re.MULTILINE,
)

# ── Result dataclass ───────────────────────────────────────────────────

@dataclass(frozen=True)
class SeedGateResult:
    """Immutable result of a seed gate validation."""

    passes: bool
    score: int
    verb: Optional[str]
    target: Optional[str]
    reason: str
    reason_code: str
    checks: dict = field(default_factory=dict)


# ── Public check helpers ───────────────────────────────────────────────

def check_length(text: str) -> Tuple[bool, str]:
    """Return (ok, message) for length requirement."""
    if len(text.strip()) < MIN_LENGTH:
        msg = "Proposal is %d chars (minimum %d)" % (len(text.strip()), MIN_LENGTH)
        return False, msg
    return True, "Length OK (%d chars)" % len(text.strip())


def check_fragment(text: str) -> Tuple[bool, str]:
    """Return (ok, message) for fragment detection."""
    stripped = text.strip()
    if _FRAGMENT_SIGNALS.search(stripped):
        return False, "Looks like a sentence fragment — too vague"
    if stripped.count(" ") < 3 and len(stripped) < 60:
        return False, "Too few words — likely a fragment"
    return True, "Not a fragment"


def check_junk_signals(text: str) -> Tuple[bool, str]:
    """Return (ok, message) for junk/buzzword detection."""
    matches = _JUNK_WORDS.findall(text)
    if len(matches) >= 2:
        return False, "Multiple junk signals detected: %s" % ", ".join(matches[:3])
    return True, "No excessive junk signals"


# ── Core extraction ────────────────────────────────────────────────────

def find_action_verb(text: str, limit: int = 0) -> Optional[str]:
    """Find the first action verb in *text*.

    Parameters
    ----------
    text : str
        Seed proposal text.
    limit : int
        If > 0, only search the first *limit* characters.
    """
    search_text = text[:limit].lower() if limit > 0 else text.lower()
    for word in re.findall(r"[a-z]+", search_text):
        if word in ACTION_VERBS:
            return word
    return None


def find_concrete_target(text: str) -> Optional[str]:
    """Find the first concrete target (filename, tool, path, func)."""
    m = _FILENAME_RE.search(text)
    if m:
        return m.group(0)
    m = _TOOL_NAME_RE.search(text)
    if m:
        return m.group(0)
    m = _PATH_RE.search(text)
    if m:
        return m.group(0).strip()
    m = _FUNC_RE.search(text)
    if m:
        return m.group(0)
    m = _CHANNEL_RE.search(text)
    if m:
        return m.group(0)
    return None


def find_all_verbs(text: str) -> List[str]:
    """Return all unique action verbs found in *text*, in order."""
    seen = set()
    result = []
    for word in re.findall(r"[a-z]+", text.lower()):
        if word in ACTION_VERBS and word not in seen:
            seen.add(word)
            result.append(word)
    return result


def find_all_targets(text: str) -> List[str]:
    """Return all concrete targets found in *text*."""
    targets = []
    seen = set()
    for pat in [_FILENAME_RE, _TOOL_NAME_RE, _PATH_RE, _FUNC_RE]:
        for m in pat.finditer(text):
            val = m.group(0).strip()
            if val not in seen:
                seen.add(val)
                targets.append(val)
    return targets


# ── Score computation ──────────────────────────────────────────────────

def compute_score(text: str) -> int:
    """Compute a specificity score (0-10) for diagnostic purposes."""
    s = 0
    if find_action_verb(text):
        s += 2
    if _FILENAME_RE.search(text):
        s += 3
    elif _TOOL_NAME_RE.search(text):
        s += 3
    if _PATH_RE.search(text):
        s += 1
    if _FUNC_RE.search(text):
        s += 1
    if _CHANNEL_RE.search(text):
        s += 1
    if len(text.strip()) >= 80:
        s += 1
    return min(s, 10)


# ── Main validator ─────────────────────────────────────────────────────

def validate_seed(
    text: str,
    tags: Optional[List[str]] = None,
    mode: str = "admission",
) -> SeedGateResult:
    """Validate a seed proposal for specificity.

    Parameters
    ----------
    text : str
        The seed proposal text.
    tags : list[str] | None
        Optional list of tags (e.g. ``["theme", "philosophy"]``).
    mode : str
        ``"admission"`` for new proposals; ``"purge"`` for retroactive
        scanning.

    Returns
    -------
    SeedGateResult
        Frozen dataclass with ``passes``, ``score``, ``verb``,
        ``target``, ``reason``, ``reason_code``, and ``checks`` dict.
    """
    if text is None:
        raise TypeError("text must be a string, got None")
    if mode not in ("admission", "purge"):
        raise ValueError("mode must be 'admission' or 'purge', got %r" % mode)

    tags = tags or []
    checks = {}
    is_exempt = bool(set(t.lower() for t in tags) & EXEMPT_TAGS)

    # ── Admission-only checks ──────────────────────────────────────
    if mode == "admission":
        ok, msg = check_length(text)
        checks["length"] = ok
        if not ok:
            return SeedGateResult(
                passes=False, score=compute_score(text),
                verb=find_action_verb(text), target=find_concrete_target(text),
                reason=msg, reason_code="too_short", checks=checks,
            )

        ok, msg = check_fragment(text)
        checks["fragment"] = ok
        if not ok:
            return SeedGateResult(
                passes=False, score=compute_score(text),
                verb=find_action_verb(text), target=find_concrete_target(text),
                reason=msg, reason_code="fragment", checks=checks,
            )

        ok, msg = check_junk_signals(text)
        checks["junk"] = ok
        if not ok:
            return SeedGateResult(
                passes=False, score=compute_score(text),
                verb=find_action_verb(text), target=find_concrete_target(text),
                reason=msg, reason_code="junk_signal", checks=checks,
            )
    else:
        # purge mode — skip length/fragment/junk
        checks["length"] = True
        checks["fragment"] = True
        checks["junk"] = True

    # ── Verb check ─────────────────────────────────────────────────
    verb_limit = 200 if mode == "purge" else 0
    verb = find_action_verb(text, limit=verb_limit)
    checks["verb"] = verb is not None
    if verb is None:
        return SeedGateResult(
            passes=False, score=compute_score(text),
            verb=None, target=find_concrete_target(text),
            reason="No action verb found", reason_code="no_verb",
            checks=checks,
        )

    # ── Target check ───────────────────────────────────────────────
    target = find_concrete_target(text)
    checks["target"] = target is not None or is_exempt
    if target is None and not is_exempt:
        return SeedGateResult(
            passes=False, score=compute_score(text),
            verb=verb, target=None,
            reason="No concrete target (filename, tool, path, or function)",
            reason_code="no_target", checks=checks,
        )

    # ── All checks passed ──────────────────────────────────────────
    score = compute_score(text)
    target_display = target or "(exempt)"
    reason = "Specific: verb=%s, target=%s" % (verb, target_display)
    return SeedGateResult(
        passes=True, score=score, verb=verb, target=target,
        reason=reason, reason_code="ok", checks=checks,
    )


# ── Convenience ────────────────────────────────────────────────────────

def passes_gate(
    text: str,
    tags: Optional[List[str]] = None,
    mode: str = "admission",
) -> bool:
    """Convenience: return True if the seed passes the gate."""
    return validate_seed(text, tags=tags, mode=mode).passes


# ── CLI ────────────────────────────────────────────────────────────────

def _cli_main() -> None:
    """Minimal CLI: --check TEXT or pipe stdin for JSON filtering."""
    if "--check" in sys.argv:
        idx = sys.argv.index("--check")
        if idx + 1 < len(sys.argv):
            text = sys.argv[idx + 1]
        else:
            print("Error: --check requires an argument", file=sys.stderr)
            sys.exit(2)
        mode = "purge" if "--purge" in sys.argv else "admission"
        r = validate_seed(text, mode=mode)
        out = {
            "passes": r.passes,
            "score": r.score,
            "verb": r.verb,
            "target": r.target,
            "reason": r.reason,
            "reason_code": r.reason_code,
            "checks": r.checks,
        }
        print(json.dumps(out, indent=2))
        sys.exit(0 if r.passes else 1)

    # Pipe mode: read JSON lines, filter, output passing
    passing = []
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        text = obj.get("text", "")
        tags = obj.get("tags", [])
        mode = obj.get("mode", "admission")
        r = validate_seed(text, tags=tags, mode=mode)
        obj["_gate"] = {
            "passes": r.passes,
            "score": r.score,
            "verb": r.verb,
            "target": r.target,
            "reason": r.reason,
            "reason_code": r.reason_code,
        }
        if r.passes:
            passing.append(obj)
    print(json.dumps(passing, indent=2))


if __name__ == "__main__":
    _cli_main()
