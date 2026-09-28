# Boltzmann timings

Measured 2026-09-28 with `uv run python benchmarks/boltzmann.py --write`: `conductivity_tensor`, numba backend, warm, best of 20 calls, on a fourfold pocket with anisotropic τ and fields from 0.1 to 100 T.

Machine: Intel64 Family 6 Model 158 Stepping 10, GenuineIntel, 12 logical CPUs, numba threads 12 (omp layer); Python 3.14.5, numpy 2.5.3, numba 0.67.0.

| N | nB | symmetrised | time (ms) | ns per node-field |
|---:|---:|:---:|---:|---:|
| 256 | 100 | yes | 1.36 | 53.24 |
| 512 | 1000 | yes | 4.80 | 9.37 |
| 1024 | 1000 | yes | 7.78 | 7.60 |
| 2048 | 1000 | yes | 14.53 | 7.09 |
| 1024 | 2000 | yes | 13.67 | 6.68 |
| 1024 | 1000 | no | 6.20 | 6.06 |
| 4096 | 1000 | yes | 30.81 | 7.52 |

- ok: typical case (N=1024, nB=1000, symmetrised) <= 10 ms: 7.78 ms
- ok: doubling N scales time by 1.6-2.5x: 1.87x
- ok: doubling nB scales time by 1.6-2.5x: 1.76x
