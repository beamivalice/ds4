#!/usr/bin/env python3
"""Summarize prefill ablation CSV (enable-one + leave-one-out)."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def load(path: Path) -> dict[str, dict[int, tuple[float, float]]]:
    rows: dict[str, dict[int, tuple[float, float]]] = {}
    with path.open(newline="") as f:
        for r in csv.DictReader(f):
            name = r["config"]
            ctx = int(r["ctx_tokens"])
            rows.setdefault(name, {})[ctx] = (float(r["prefill_tps"]), float(r["gen_tps"]))
    return rows


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("csv")
    p.add_argument("--ref-prefill-2048", type=float, default=421.0)
    p.add_argument("--ref-prefill-8192", type=float, default=401.0)
    p.add_argument("--out-matrix", type=Path, default=None)
    args = p.parse_args()

    data = load(Path(args.csv))
    baseline = data.get("baseline", {})
    full = data.get("full", {})

    print("\n=== Enable-one vs baseline (PREFILL_BASELINE=1) ===\n")
    b2048 = baseline.get(2048, (float("nan"),))[0]
    b8192 = baseline.get(8192, (float("nan"),))[0]
    for name in ["baseline", "only_moe_wide_tiles", "only_moe_gate_up_pair", "only_flash_nwg64", "full"]:
        if name not in data:
            continue
        p2 = data[name].get(2048, (float("nan"),))[0]
        p8 = data[name].get(8192, (float("nan"),))[0]
        print(f"  {name:<24} {p2:7.1f} ({p2 - b2048:+.1f} @2048)  {p8:7.1f} ({p8 - b8192:+.1f} @8192)")

    if full:
        print("\n=== Leave-one-out from full (Δ when feature disabled) ===\n")
        f2048 = full.get(2048, (0.0,))[0]
        f8192 = full.get(8192, (0.0,))[0]
        for key, label in [
            ("no_moe_wide_tiles", "MoE wide tiles"),
            ("no_moe_gate_up_pair", "Q4_K gate+up pair"),
            ("no_flash_nwg64", "flash nwg=64"),
        ]:
            if key not in data:
                continue
            o2 = data[key].get(2048, (f2048,))[0]
            o8 = data[key].get(8192, (f8192,))[0]
            print(f"  {label:<22} {f2048 - o2:+7.1f} @2048   {f8192 - o8:+7.1f} @8192")

    if args.out_matrix and full and baseline:
        lines = ["config,ctx_tokens,prefill_tps,gen_tps,delta_prefill_vs_baseline"]
        for name, d in sorted(data.items()):
            for ctx in (2048, 8192):
                if ctx not in d:
                    continue
                p, g = d[ctx]
                ref = baseline.get(ctx, (0.0,))[0]
                lines.append(f"{name},{ctx},{p:.2f},{g:.2f},{p - ref:.2f}")
        args.out_matrix.write_text("\n".join(lines) + "\n")
        print(f"\nWrote {args.out_matrix}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
