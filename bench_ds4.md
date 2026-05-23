# ds4 Baseline Benchmark (SIMD Q8_0, no ANE)

**Hardware:** Apple M5 Max, 128 GB unified memory, 614 GB/s bandwidth  
**Model:** DeepSeek V4 Flash (93 GB GGUF, Q8_0 attention/FFN)  
**Date:** 2026-05-23 17:10

```
ds4-bench: context buffers 751.71 MiB (ctx=32768, backend=metal, prefill_chunk=2048, raw_kv_rows=2304, compressed_kv_rows=8194)
ds4: Metal device Apple M5 Max, 128.00 GiB RAM
ds4: M5 Neural Accelerator tensor matmul enabled
ds4: Metal model views created in 2.225 ms, residency requested in 575.990 ms, warmup 3.934 ms (mapped 93065.67 MiB from offset 5.08 MiB)
ds4: Metal mapped mmaped model as 2 overlapping shared buffers
ds4: metal backend initialized for graph diagnostics
ctx_tokens,prefill_tokens,prefill_tps,gen_tokens,gen_tps,kvcache_bytes
2048,2048,437.41,128,31.01,52184460
6144,4096,388.36,128,30.39,108561804
10240,4096,377.33,128,30.05,164939148
14336,4096,362.91,128,29.74,221316492
16384,2048,353.08,128,29.65,249505164
```

