"""CLI entrypoint for the S1-S6 Jev probe.

    uv run python -m jev.probe.run_probe --backend live --suites S1,S2,S3 --n 200
    uv run python -m jev.probe.run_probe --backend fixture --suites S1 --n 5   # harness dev only

Backends:
  fixture -- hand-written example responses (probe/fixtures/*.jsonl); exercises
             client/suite code, NOT a measurement of Jev. Will typically fail
             ("no cached response") for anything but the exact fixture requests.
  replay  -- JSONL cache(s) from previous live runs; --cache to point at one.
  live    -- real API calls (needs JEV_API_KEY); every response is appended to
             --cache automatically via CachingJevClient, so repeat runs re-use
             it via `replay` instead of paying twice.

Every result records the exact `model` version string(s) seen -- Jev is four
days old as of this writing and will move under you; never report a number
without it.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

from jev.probe.client import BaseJevClient, CachingJevClient, FixtureJevClient, LiveJevClient, ReplayJevClient
from jev.probe.suites import (
    HYPOTHESES,
    run_s1_resolvable,
    run_s2_known_ambiguous,
    run_s3_unanswerable,
    run_s4_type_consistency,
    run_s5_confidence_vs_probabilities,
    run_s6_prior_sensitivity,
)

SUITE_RUNNERS = {
    "S1": lambda client, n, seed: run_s1_resolvable(client, n=n, seed=seed),
    "S2": lambda client, n, seed: run_s2_known_ambiguous(client, n_per_level=max(n // 4, 5), seed=seed),
    "S3": lambda client, n, seed: run_s3_unanswerable(client, n=n, seed=seed),
    "S4": lambda client, n, seed: run_s4_type_consistency(client, n=n, seed=seed),
    "S5": lambda client, n, seed: run_s5_confidence_vs_probabilities(client, n=n, seed=seed),
    "S6": lambda client, n, seed: run_s6_prior_sensitivity(client, n=n, seed=seed),
}


def build_client(args) -> BaseJevClient:
    if args.backend == "fixture":
        return FixtureJevClient()
    if args.backend == "replay":
        if not args.cache:
            sys.exit("--cache is required for --backend replay")
        return ReplayJevClient([args.cache])
    if args.backend == "live":
        cache = args.cache or "outputs/jev_cache.jsonl"
        return CachingJevClient(LiveJevClient(), cache)
    raise ValueError(args.backend)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=["fixture", "replay", "live"], default="fixture")
    p.add_argument("--cache", default=None, help="JSONL cache path (required for replay; default for live)")
    p.add_argument("--suites", default="S1,S2,S3,S4,S5,S6")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--out", default="outputs/probe_results.json")
    args = p.parse_args()

    client = build_client(args)
    suite_ids = [s.strip().upper() for s in args.suites.split(",")]

    results = {}
    for sid in suite_ids:
        if sid not in SUITE_RUNNERS:
            print(f"skipping unknown suite '{sid}'")
            continue
        print(f"\n=== {sid} ===")
        print(f"hypothesis: {HYPOTHESES[sid].prediction}")
        result = SUITE_RUNNERS[sid](client, args.n, args.seed)
        print(f"model version(s) seen: {result.model_versions}")
        print(f"n_items: {result.n_items}")
        for k, v in result.metrics.items():
            print(f"  {k}: {v}")
        results[sid] = dataclasses.asdict(result)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
