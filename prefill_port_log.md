# Prefill port log (ds4cursor → ds4main)

Bench model: `gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf`

## Toggle semantics

| Env | Effect |
|-----|--------|
| *(unset)* | **All ported opts ON** (production default) |
| `DS4_METAL_PREFILL_BASELINE=1` | All ported opts **OFF** (matches pre-port MoE/flash behavior) |
| `DS4_METAL_ENABLE_PREFILL_MOE_WIDE_TILES=1` | With baseline: enable MoE n64/n128 tiles only |
| `DS4_METAL_ENABLE_PREFILL_MOE_GATE_UP_PAIR=1` | With baseline: enable Q4_K fused gate+up only |
| `DS4_METAL_ENABLE_PREFILL_FLASH_NWG64=1` | With baseline: enable flash nwg=64 when n_keys≥1536 |
| `DS4_METAL_FLASH_ATTN_NWG` | Force flash nwg (overrides heuristic) |

## Run ablation

```bash
make ds4-bench
./speed-bench/ablate_prefill.sh
```

Results: `speed-bench/prefill_ablation.csv`

## Documented ds4main reference (pre-port)

~421 prefill tok/s @ 2048, ~401 @ 8192 (promessi, gen 128, incremental).

Fill in measured numbers after running `ablate_prefill.sh`:

| config | 2048 prefill | 8192 prefill | Δ2048 vs baseline | Δ8192 vs baseline |
|--------|-------------:|-------------:|------------------:|------------------:|
| baseline | | | 0 | 0 |
| only_moe_wide_tiles | | | | |
| only_moe_gate_up_pair | | | | |
| only_flash_nwg64 | | | | |
| full | | | | |
