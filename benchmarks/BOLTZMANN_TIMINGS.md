# Boltzmann timings

Measured 2026-09-29 with `uv run python benchmarks/boltzmann.py --write`: `conductivity_tensor`, numba backend, warm, best of 40 calls, on a fourfold pocket with anisotropic τ and fields from 0.1 to 100 T.

Machine: Intel64 Family 6 Model 158 Stepping 10, GenuineIntel, 12 logical CPUs, numba threads 12 (omp layer); Python 3.14.5, numpy 2.5.3, numba 0.67.0.

| N | nB | symmetrised | time (ms) | ns per node-field |
|---:|---:|:---:|---:|---:|
| 256 | 100 | yes | 1.10 | 42.92 |
| 512 | 1000 | yes | 4.44 | 8.67 |
| 1024 | 1000 | yes | 7.41 | 7.24 |
| 2048 | 1000 | yes | 14.78 | 7.22 |
| 1024 | 2000 | yes | 13.82 | 6.75 |
| 1024 | 1000 | no | 6.14 | 6.00 |
| 4096 | 1000 | yes | 31.46 | 7.68 |

- ok: typical case (N=1024, nB=1000, symmetrised) <= 10 ms: 7.41 ms
- ok: doubling N scales time by 1.6-2.5x: 1.99x
- ok: doubling nB scales time by 1.6-2.5x: 1.86x

## With a scattering kernel

`conductivity(..., scattering_kernel=...)` on a pair of warped open sheets coupled by a smooth kernel, 31 fields, best of 3 calls. It is a dense solve per field on the nodes the kernel couples, so the time grows as their number cubed. Reported, not gated.

| nodes coupled | fields | time (ms) | ms per field |
|---:|---:|---:|---:|
| 512 | 31 | 1262 | 40.7 |
| 1024 | 31 | 3321 | 107.1 |
| 2048 | 31 | 15967 | 515.1 |
