# Boltzmann timings

Measured 2026-09-27 with `uv run python benchmarks/boltzmann.py --write`: `conductivity_tensor`, numba backend, warm, best of 20 calls, on a fourfold pocket with anisotropic τ and fields from 0.1 to 100 T.

Machine: Intel64 Family 6 Model 158 Stepping 10, GenuineIntel, 12 logical CPUs, numba threads 12 (omp layer); Python 3.9.25, numpy 2.0.2, numba 0.60.0.

| N | nB | symmetrised | time (ms) | ns per node-field |
|---:|---:|:---:|---:|---:|
| 256 | 100 | yes | 1.33 | 51.77 |
| 512 | 1000 | yes | 4.74 | 9.25 |
| 1024 | 1000 | yes | 8.00 | 7.81 |
| 2048 | 1000 | yes | 14.55 | 7.10 |
| 1024 | 2000 | yes | 14.78 | 7.21 |
| 1024 | 1000 | no | 5.92 | 5.78 |
| 4096 | 1000 | yes | 30.16 | 7.36 |

- ok: typical case (N=1024, nB=1000, symmetrised) <= 10 ms: 8.00 ms
- ok: doubling N scales time by 1.6-2.5x: 1.82x
- ok: doubling nB scales time by 1.6-2.5x: 1.85x
