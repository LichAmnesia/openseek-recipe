## Results: 2026-09-21-maxq-300w

### seven

Weighted decode: **134.4 tok/s**

| Case | decode tok/s (median) | TTFT ms (median) | quality contract |
|---|---:|---:|:-:|
| math | 186.1 | 100 | 3/3 |
| code | 181.0 | 96 | 3/3 |
| structured-json | 170.4 | 102 | 3/3 |
| hello | 162.4 | 84 | 3/3 |
| topic | 126.2 | 99 | 3/3 |
| fable | 106.5 | 101 | 1/3 |
| multilingual | 102.3 | 99 | 2/3 |

MTP acceptance: **52.5%** (2013/3831 draft tokens, mean acceptance length 2.58)

### orchid

Weighted decode: **217.0 tok/s**

| Case | decode tok/s (median) | TTFT ms (median) | quality contract |
|---|---:|---:|:-:|
| orchid | 215.5 | 97 | 2/5 |

MTP acceptance: **97.1%** (2686/2766 draft tokens, mean acceptance length 3.91)

### Decode scaling (clients)

256 output tokens per request, MTP=3

| Concurrency | aggregate tok/s (median) | min | max | per-stream | TTFT ms (median) |
|---:|---:|---:|---:|---:|---:|
| 1 | 104.5 | 103.7 | 111.9 | 104.5 | 85 |
| 2 | 170.9 | 138.2 | 173.2 | 85.5 | 99 |
| 4 | 205.0 | 52.1 | 256.4 | 51.3 | 207 |
| 8 | 431.1 | 423.5 | 448.4 | 53.9 | 152 |
| 16 | 595.2 | 581.2 | 596.3 | 37.2 | 317 |

API tool constraints: **16/16**
Vision: 1 img pass, 4 img pass, 16 img pass
