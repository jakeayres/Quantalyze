"""Timing benchmark for quantalyze.beta.boltzmann (local; not part of the test suite).

Times `conductivity_tensor` (numba backend, warm) on a fourfold pocket with anisotropic
τ, over a grid of contour sizes N and field counts nB, and checks the performance goals:
the typical case (N = 1024, nB = 1000, symmetrised) in at most 10 ms, and linear
scaling (doubling N or nB multiplies the time by 1.6–2.5).

    uv run python benchmarks/boltzmann.py                     # print the table and checks
    uv run python benchmarks/boltzmann.py --write             # also update BOLTZMANN_TIMINGS.md
    uv run python benchmarks/boltzmann.py --save-baseline benchmarks/boltzmann_baseline.json
    uv run python benchmarks/boltzmann.py --check benchmarks/boltzmann_baseline.json

`--check` exits with status 1 if any case is more than 25% slower than the baseline.
Timings depend on the machine, so keep the baseline from the machine you compare on.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import sys
import time
from pathlib import Path

import numba
import numpy as np

from quantalyze.beta.boltzmann import generators, scattering
from quantalyze.beta.boltzmann._response import conductivity_tensor
from quantalyze.core.constants import ELECTRON_MASS

HERE = Path(__file__).resolve().parent
TIMINGS = HERE / "BOLTZMANN_TIMINGS.md"
TYPICAL = (1024, 1000, True)
TARGET_MS = 10.0
SCALING = (1.6, 2.5)
REGRESSION = 1.25

# (N, nB, symmetrize)
CASES = [
    (256, 100, True),
    (512, 1000, True),
    (1024, 1000, True),
    (2048, 1000, True),
    (1024, 2000, True),
    (1024, 1000, False),
    (4096, 1000, True),
]


def contour(n):
    return generators.polar(n, k_fermi=lambda p: 7.35e9 - 0.25e9 * np.cos(4 * p), mass=5 * ELECTRON_MASS,
                            tau=lambda p: scattering.cos4phi(p, 1e-13, anisotropy=0.3), carrier="hole")


def time_case(n, n_fields, symmetrize, repeats):
    df = contour(n)
    arrays = [df[c].to_numpy() for c in ("kx", "ky", "vx", "vy", "tau")]
    fields = np.linspace(0.1, 100, n_fields)  # ω_cτ from about 0.004 to 4
    call = lambda: conductivity_tensor(*arrays, fields, layer_spacing=1e-9, symmetrize=symmetrize)  # noqa: E731
    call()  # warm: compile or load from cache
    best = np.inf
    for _ in range(repeats):
        start = time.perf_counter()
        call()
        best = min(best, time.perf_counter() - start)
    return 1e3 * best


def key(case):
    n, n_fields, symmetrize = case
    return f"N={n} nB={n_fields} {'sym' if symmetrize else 'unsym'}"


def machine():
    return (f"{platform.processor() or platform.machine()}, {os.cpu_count()} logical CPUs, "
            f"numba threads {numba.get_num_threads()} ({numba.threading_layer()} layer); "
            f"Python {platform.python_version()}, numpy {np.__version__}, numba {numba.__version__}")


def checks(ms):
    typical = ms[key(TYPICAL)]
    n_ratio = ms[key((2048, 1000, True))] / typical
    b_ratio = ms[key((1024, 2000, True))] / typical
    return [
        (f"typical case (N=1024, nB=1000, symmetrised) <= {TARGET_MS:g} ms", f"{typical:.2f} ms", typical <= TARGET_MS),
        (f"doubling N scales time by {SCALING[0]}-{SCALING[1]}x", f"{n_ratio:.2f}x", SCALING[0] <= n_ratio <= SCALING[1]),
        (f"doubling nB scales time by {SCALING[0]}-{SCALING[1]}x", f"{b_ratio:.2f}x", SCALING[0] <= b_ratio <= SCALING[1]),
    ]


def table(ms):
    lines = ["| N | nB | symmetrised | time (ms) | ns per node-field |", "|---:|---:|:---:|---:|---:|"]
    for case in CASES:
        n, n_fields, symmetrize = case
        t = ms[key(case)]
        lines.append(f"| {n} | {n_fields} | {'yes' if symmetrize else 'no'} | {t:.2f} | {1e6 * t / (n * n_fields):.2f} |")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repeats", type=int, default=20, help="timed calls per case; the best is kept")
    parser.add_argument("--write", action="store_true", help=f"write the table to {TIMINGS.name}")
    parser.add_argument("--save-baseline", type=Path, help="write the timings to this JSON file")
    parser.add_argument("--check", type=Path, help="compare with this baseline JSON; exit 1 on a >25%% regression")
    args = parser.parse_args(argv)

    ms = {key(case): time_case(*case, repeats=args.repeats) for case in CASES}
    print(machine())
    print(table(ms))
    results = checks(ms)
    for label, value, ok in results:
        print(f"[{'ok' if ok else 'FAIL'}] {label}: {value}")

    status = 0 if all(ok for _, _, ok in results) else 1
    if args.check:
        baseline = json.loads(args.check.read_text())["ms"]
        for name, reference in baseline.items():
            if name in ms and ms[name] > REGRESSION * reference:
                print(f"REGRESSION {name}: {ms[name]:.2f} ms vs baseline {reference:.2f} ms "
                      f"({ms[name] / reference:.2f}x)")
                status = 1
        if status == 0:
            print(f"no case more than {100 * (REGRESSION - 1):.0f}% slower than {args.check.name}")
    if args.save_baseline:
        args.save_baseline.write_text(json.dumps({"machine": machine(), "ms": ms}, indent=1) + "\n")
    if args.write:
        TIMINGS.write_text(
            "# Boltzmann timings\n\n"
            f"Measured {datetime.date.today().isoformat()} with `uv run python benchmarks/boltzmann.py --write`: "
            "`conductivity_tensor`, numba backend, warm, best of "
            f"{args.repeats} calls, on a fourfold pocket with anisotropic τ and fields from 0.1 to 100 T.\n\n"
            f"Machine: {machine()}.\n\n{table(ms)}\n\n"
            + "\n".join(f"- {'ok' if ok else 'FAIL'}: {label}: {value}" for label, value, ok in results) + "\n",
            encoding="utf-8",
        )
    return status


if __name__ == "__main__":
    sys.exit(main())
