#!/usr/bin/env bash
# Enable-one-at-a-time prefill A/B vs baseline (all new opts off).
# Production default: unset DS4_METAL_PREFILL_BASELINE (all opts on).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BENCH="${DS4_BENCH:-$ROOT/ds4-bench}"
MODEL="${DS4_BENCH_MODEL:-$ROOT/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf}"
PROMPT="${DS4_BENCH_PROMPT:-$ROOT/speed-bench/promessi_sposi.txt}"
OUT_DIR="${DS4_ABLATE_OUT:-$ROOT/speed-bench}"
CSV_OUT="$OUT_DIR/prefill_ablation.csv"
GEN_N="${DS4_BENCH_GEN_TOKENS:-128}"

if [[ ! -x "$BENCH" ]]; then
  echo "ablate: build ds4-bench first (make ds4-bench)" >&2
  exit 1
fi
if [[ ! -f "$MODEL" ]]; then
  echo "ablate: model not found: $MODEL" >&2
  exit 1
fi

# name|extra env (after BASELINE=1 for per-feature runs; full has no baseline flag)
configs=(
  "baseline|DS4_METAL_PREFILL_BASELINE=1"
  "only_moe_wide_tiles|DS4_METAL_PREFILL_BASELINE=1 DS4_METAL_ENABLE_PREFILL_MOE_WIDE_TILES=1"
  "only_moe_gate_up_pair|DS4_METAL_PREFILL_BASELINE=1 DS4_METAL_ENABLE_PREFILL_MOE_GATE_UP_PAIR=1"
  "only_flash_nwg64|DS4_METAL_PREFILL_BASELINE=1 DS4_METAL_ENABLE_PREFILL_FLASH_NWG64=1"
  "full|"
)

run_config() {
  local name="$1"
  local extra="$2"
  local csv="$OUT_DIR/.ablate_${name}.csv"
  echo "ablate: === $name ===" >&2
  # shellcheck disable=SC2086
  env $extra "$BENCH" \
    -m "$MODEL" \
    --prompt-file "$PROMPT" \
    --ctx-start 2048 \
    --ctx-max 8192 \
    --step-incr 2048 \
    --gen-tokens "$GEN_N" \
    --warm-weights \
    --csv "$csv" \
    2>"$OUT_DIR/.ablate_${name}.log"
  awk -F, -v n="$name" '
    NR == 1 { next }
    $1 == 2048 || $1 == 8192 {
      printf "%s,%s,%s,%s,%s\n", n, $1, $3, $5, $6
    }
  ' "$csv"
}

echo "config,ctx_tokens,prefill_tps,gen_tps,kvcache_bytes" >"$CSV_OUT"
for entry in "${configs[@]}"; do
  name="${entry%%|*}"
  extra="${entry#*|}"
  while IFS= read -r row; do
    [[ -n "$row" ]] && echo "$row" >>"$CSV_OUT"
  done < <(run_config "$name" "$extra")
done

python3 - "$CSV_OUT" <<'PY'
import csv, sys
from pathlib import Path
rows = {}
with Path(sys.argv[1]).open() as f:
    for r in csv.DictReader(f):
        rows.setdefault(r["config"], {})[int(r["ctx_tokens"])] = (
            float(r["prefill_tps"]), float(r["gen_tps"]))
b = rows.get("baseline", {})
print("\n=== Prefill ablation (Δ vs baseline, enable-one-at-a-time) ===\n")
print(f"{'config':<24} {'2048 prefill':>12} {'Δ2048':>8} {'gen':>6}  {'8192 prefill':>12} {'Δ8192':>8} {'gen':>6}")
print("-" * 78)
for name in ["baseline", "only_moe_wide_tiles", "only_moe_gate_up_pair", "only_flash_nwg64", "full"]:
    if name not in rows:
        continue
    p2, g2 = rows[name].get(2048, (float("nan"), float("nan")))
    p8, g8 = rows[name].get(8192, (float("nan"), float("nan")))
    b2 = b.get(2048, (p2,))[0]
    b8 = b.get(8192, (p8,))[0]
    d2 = p2 - b2 if name != "baseline" else 0.0
    d8 = p8 - b8 if name != "baseline" else 0.0
    print(f"{name:<24} {p2:12.1f} {d2:+8.1f} {g2:6.1f}  {p8:12.1f} {d8:+8.1f} {g8:6.1f}")
if "full" in rows and b:
    f2 = rows["full"].get(2048, (0,))[0]
    f8 = rows["full"].get(8192, (0,))[0]
    print(f"\nfull − baseline @2048: {f2 - b.get(2048,(0,))[0]:+.1f}  @8192: {f8 - b.get(8192,(0,))[0]:+.1f}")
PY

echo "ablate: wrote $CSV_OUT" >&2
