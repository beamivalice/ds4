#!/usr/bin/env bash
# Prefill A/B: enable-one-from-baseline + leave-one-out-from-full.
# Production: unset DS4_METAL_PREFILL_BASELINE (all opts on).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BENCH="${DS4_BENCH:-$ROOT/ds4-bench}"
MODEL="${DS4_BENCH_MODEL:-$ROOT/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf}"
PROMPT="${DS4_BENCH_PROMPT:-$ROOT/speed-bench/promessi_sposi.txt}"
OUT_DIR="${DS4_ABLATE_OUT:-$ROOT/speed-bench}"
CSV_OUT="$OUT_DIR/prefill_ablation.csv"
MATRIX_OUT="$OUT_DIR/port_matrix.csv"
GEN_N="${DS4_BENCH_GEN_TOKENS:-128}"

if [[ ! -x "$BENCH" ]]; then
  echo "ablate: build ds4-bench first (make ds4-bench)" >&2
  exit 1
fi
if [[ ! -f "$MODEL" ]]; then
  echo "ablate: model not found: $MODEL" >&2
  exit 1
fi

configs=(
  "baseline|DS4_METAL_PREFILL_BASELINE=1"
  "only_moe_wide_tiles|DS4_METAL_PREFILL_BASELINE=1 DS4_METAL_ENABLE_PREFILL_MOE_WIDE_TILES=1"
  "only_moe_gate_up_pair|DS4_METAL_PREFILL_BASELINE=1 DS4_METAL_ENABLE_PREFILL_MOE_GATE_UP_PAIR=1"
  "only_flash_nwg64|DS4_METAL_PREFILL_BASELINE=1 DS4_METAL_ENABLE_PREFILL_FLASH_NWG64=1"
  "full|"
  "no_moe_wide_tiles|DS4_METAL_DISABLE_PREFILL_MOE_WIDE_TILES=1"
  "no_moe_gate_up_pair|DS4_METAL_DISABLE_MOE_GATE_UP_PAIR=1"
  "no_flash_nwg64|DS4_METAL_DISABLE_PREFILL_FLASH_NWG64=1"
  "baseline_legacy_moe_flash|DS4_METAL_PREFILL_BASELINE=1"
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

cp "$CSV_OUT" "$OUT_DIR/port_baseline.csv"
python3 "$ROOT/speed-bench/plot_ablation_prefill.py" "$CSV_OUT" \
  --out-matrix "$MATRIX_OUT" 2>&1

echo "ablate: wrote $CSV_OUT and $MATRIX_OUT" >&2
