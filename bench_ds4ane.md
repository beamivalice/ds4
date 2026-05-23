# ds4 ANE+Pipeline Benchmark

**Path A: Cross-layer pipeline (no wait between ANE commit and MTL3 dispatch)**
**Date:** 2026-05-23 17:32

```
ds4-bench: context buffers 751.71 MiB (ctx=32768, backend=metal, prefill_chunk=2048, raw_kv_rows=2304, compressed_kv_rows=8194)
ds4: Metal device Apple M5 Max, 128.00 GiB RAM
ds4: M5 Neural Accelerator tensor matmul enabled
ds4: Metal model views created in 2.078 ms, residency requested in 586.591 ms, warmup 4.263 ms (mapped 93065.67 MiB from offset 5.08 MiB)
ds4: Metal mapped mmaped model as 2 overlapping shared buffers
ds4: converted 345 Q8_0 tensors to INT8 in 4.647s
ds4: metal backend initialized for graph diagnostics
ctx_tokens,prefill_tokens,prefill_tps,gen_tokens,gen_tps,kvcache_bytes
2048,2048,506.10,128,31.05,52184460
6144,4096,504.62,128,30.46,108561804
10240,4096,486.28,128,30.11,164939148
14336,4096,468.57,128,29.82,221316492
16384,2048,457.95,128,29.74,249505164
```

