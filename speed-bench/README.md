## Benchmarking

Here we collect prefill and generation speed obtained with different hardware.

Run `ds4-bench` as:

```
./ds4-bench \
  -m ds4flash.gguf \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 2048 \
  --ctx-max 65536 \
  --step-incr 2048 \
  --gen-tokens 128
```

Provide PR including your numbers if your hardware was not already tested.
Call the benchmark csv file something like `m3_max.csv` or alike, so that
it is clear what hardware was used for the benchmark.

To generate an SVG graph from a CSV file:

```
python3 speed-bench/plot_speed.py speed-bench/m3_max.csv --title "M3 Max t/s"
```

The script uses only the Python standard library. By default it writes a file
next to the CSV using the `_ts.svg` suffix, such as `speed-bench/m3_max_ts.svg`.

## Prefill tuning (Metal / q4half)

On macOS 26+ with Metal 4 tensor matmul enabled (default), prefill throughput
benefits from 64-aligned chunking (`metal_graph_prefill_chunked_range` in `ds4.c`)
and the simdgroup MoE paths below.

**M5 Max, `ds4flash-q4half.gguf`, promessi_sposi, gen 128** (representative):

| ctx | prefill tok/s | gen tok/s |
|-----|---------------|-----------|
| 2048 | ~458 | ~35 |
| 8096 (1952 tok step) | ~377 | ~29.5 |

Disable Metal 4 matmul for A/B: `DS4_METAL_DISABLE_METAL4=1`.

### Environment variables

| Variable | Effect |
|----------|--------|
| `DS4_METAL_DISABLE_METAL4` | Force legacy simdgroup dense/attn-out matmul |
| `DS4_METAL_DISABLE_MOE_GATE_UP_PAIR` | Use separate gate/up MoE encodes instead of fused pair kernel |
| `DS4_METAL_MOE_MPP` | Reserved; logs once and uses simdgroup MoE (TensorOps MoE not wired) |
| `DS4_METAL_FLASH_ATTN_NWG` | Override prefill flash-attn `nwg` (default 64 when n_keys ≥ 1536) |
| `DS4_METAL_FLASH_ATTN_NSG` | Override prefill flash-attn simdgroup count |
| `DS4_METAL_Q8_PATH_LOG` | Log Q8_0 matmul path (`nax` vs `legacy`) per call |
| `DS4_METAL_F16_PATH_LOG` | Log F16 matmul path (`nax` vs `legacy`) per call |
| `DS4_METAL_PREFILL_CHUNK` | Max tokens per prefill graph chunk (default 4096 for long prompts) |

### Stage profiling

```bash
DS4_METAL_LAYER_STAGE_PROFILE=1 \
DS4_METAL_MOE_STAGE_PROFILE=1 \
DS4_METAL_FLASH_ATTN_STAGE_PROFILE=1 \
./ds4-bench -m MODEL --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 2048 --ctx-max 2048 --gen-tokens 1
```
