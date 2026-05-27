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

## Prefill optimization toggles (ported from ds4cursor)

Production default: all optimizations **on** (do not set `DS4_METAL_PREFILL_BASELINE`).

Per-feature benchmarking vs legacy behavior:

```bash
# Baseline (all new opts off)
DS4_METAL_PREFILL_BASELINE=1 ./ds4-bench -m MODEL ...

# Enable one feature on top of baseline
DS4_METAL_PREFILL_BASELINE=1 DS4_METAL_ENABLE_PREFILL_MOE_WIDE_TILES=1 ./ds4-bench ...
DS4_METAL_PREFILL_BASELINE=1 DS4_METAL_ENABLE_PREFILL_MOE_GATE_UP_PAIR=1 ./ds4-bench ...
DS4_METAL_PREFILL_BASELINE=1 DS4_METAL_ENABLE_PREFILL_FLASH_NWG64=1 ./ds4-bench ...

# Full (default)
./ds4-bench -m MODEL ...
```

Automated sweep:

```bash
./speed-bench/ablate_prefill.sh
```

See `prefill_port_log.md` for details.
