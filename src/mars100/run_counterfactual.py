"""
Runner: generate counterfactual analysis from emergence data.

Reads  docs/mars-100/emergence.json
Writes docs/mars-100/counterfactuals.json
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Allow running from repo root: python -m src.mars100.run_counterfactual
ROOT = Path(__file__).resolve().parent.parent.parent


def main() -> None:
    """Generate counterfactuals and write output."""
    from src.mars100.counterfactual import (
        generate_counterfactuals,
        run_all_counterfactuals,
    )

    emergence_path = ROOT / "docs" / "mars-100" / "emergence.json"
    if not emergence_path.exists():
        print(f"ERROR: {emergence_path} not found -- run archaeology first", file=sys.stderr)
        sys.exit(1)

    emergence = json.loads(emergence_path.read_text())
    print(f"Loaded emergence data: {len(emergence)} keys")

    scenarios = generate_counterfactuals(emergence)
    print(f"Generated {len(scenarios)} counterfactual scenarios")

    results = run_all_counterfactuals(emergence, seed=42, total_years=100)
    print(f"Ran {len(results)} counterfactuals")

    output = {
        "_meta": {
            "engine": "mars-100-counterfactual",
            "version": "1.0",
            "generated": datetime.now(timezone.utc).isoformat(),
            "total_scenarios": len(results),
            "seed": 42,
        },
        "counterfactuals": results,
    }

    out_path = ROOT / "docs" / "mars-100" / "counterfactuals.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(output, indent=2))
    tmp.rename(out_path)
    print(f"Wrote {out_path} ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
