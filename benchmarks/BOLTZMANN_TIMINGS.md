# Boltzmann timings

Measured 2026-09-28 with `uv run python benchmarks/boltzmann.py --write`: `conductivity_tensor`, numba backend, warm, best of 20 calls, on a fourfold pocket with anisotropic τ and fields from 0.1 to 100 T.

Machine: Intel64 Family 6 Model 158 Stepping 10, GenuineIntel, 12 logical CPUs, numba threads 12 (omp layer); Python 3.9.25, numpy 2.0.2, numba 0.60.0.

| N | nB | symmetrised | time (ms) | ns per node-field |
|---:|---:|:---:|---:|---:|
| 256 | 100 | yes | 1.36 | 53.30 |
| 512 | 1000 | yes | 4.64 | 9.06 |
| 1024 | 1000 | yes | 7.72 | 7.54 |
| 2048 | 1000 | yes | 14.10 | 6.89 |
| 1024 | 2000 | yes | 13.74 | 6.71 |
| 1024 | 1000 | no | 6.34 | 6.19 |
| 4096 | 1000 | yes | 29.48 | 7.20 |

- ok: typical case (N=1024, nB=1000, symmetrised) <= 10 ms: 7.72 ms
- ok: doubling N scales time by 1.6-2.5x: 1.83x
- ok: doubling nB scales time by 1.6-2.5x: 1.78x
