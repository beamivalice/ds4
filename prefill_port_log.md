# Prefill port log (ds4cursor → ds4main)

Branch: `prefill-port`  
Model: `gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf`  
Prompt: `speed-bench/promessi_sposi.txt`, gen 128, warm weights

## Toggle semantics

| Env | Effect |
|-----|--------|
| *(unset)* | **All ported opts ON** (production) |
| `DS4_METAL_PREFILL_BASELINE=1` | All ported opts **OFF** |
| `DS4_METAL_ENABLE_PREFILL_MOE_WIDE_TILES=1` | With baseline: wide tiles only |
| `DS4_METAL_ENABLE_PREFILL_MOE_GATE_UP_PAIR=1` | With baseline: Q4_K fused gate+up only |
| `DS4_METAL_ENABLE_PREFILL_FLASH_NWG64=1` | With baseline: flash nwg=64 heuristic only |
| `DS4_METAL_DISABLE_PREFILL_MOE_WIDE_TILES=1` | With full: disable wide tiles |
| `DS4_METAL_DISABLE_MOE_GATE_UP_PAIR=1` | With full: disable fused gate+up |
| `DS4_METAL_DISABLE_PREFILL_FLASH_NWG64=1` | With full: force flash nwg=32 |
| `DS4_METAL_FLASH_ATTN_NWG` | Force flash nwg (numeric override) |

## Measured (M5 Max, May 2026)

### Enable-one vs baseline (`PREFILL_BASELINE=1`)

| config | 2048 prefill | 8192 prefill | Δ2048 | Δ8192 |
|--------|-------------:|-------------:|------:|------:|
| baseline | 456.2 | 405.2 | 0 | 0 |
| only_moe_wide_tiles | 660.7 | 489.6 | +204.5 | +84.4 |
| only_moe_gate_up_pair | 436.3 | 345.8 | −19.9 | −59.4 |
| only_flash_nwg64 | 467.1 | 316.5 | +10.9 | −88.6 |
| full | 663.1 | 435.3 | +206.9 | +30.2 |

### Leave-one-out from full

| feature removed | Δ2048 | Δ8192 |
|-----------------|------:|------:|
| MoE wide tiles | +216.3 | +108.8 |
| Q4_K gate+up pair | +1.8 | −9.1 |
| flash nwg=64 | −0.3 | −6.0 |

**Takeaway:** MoE n64/n128 tiles dominate (~+200 tok/s @ 2048). Fused gate+up and flash nwg are small on this IQ2 model when wide tiles are already on; gate+up helps only in combination with wide tiles.

Doc reference (pre-port ds4main): ~421 @ 2048, ~401 @ 8192.  
`baseline` here: 456 / 405 (same binary, `PREFILL_BASELINE=1`).

## Artifacts

- `speed-bench/prefill_ablation.csv`
- `speed-bench/port_matrix.csv`
- `speed-bench/port_baseline.csv`
- `./speed-bench/ablate_prefill.sh`

## Quality

- `make ds4_test` — run on quiet machine (not while `ds4-bench` holds the process lock).
