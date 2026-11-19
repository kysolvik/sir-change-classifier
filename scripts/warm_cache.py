"""Pre-warm the shared cache for the preset study areas.

Run once at deploy so students hitting a preset never pay a cold read. In
production set GCS_CACHE_BUCKET first so warmed windows land in GCS:

    # AEF embeddings (~10 s/window)
    GCS_CACHE_BUCKET=my-bucket uv run python -m scripts.warm_cache
    GCS_CACHE_BUCKET=my-bucket uv run python -m scripts.warm_cache --all-years

    # Raw Landsat composites, 1984-2025 (~15-25 s each; 6 presets x 42 years).
    # Warm these before students use Landsat, since each cold (preset, year)
    # otherwise composites on the request path. Each composite already reads its
    # scenes in parallel (LANDSAT_LOAD_THREADS); running several composites at
    # once didn't speed up a bulk warm (it's bandwidth-bound), so this is serial.
    GCS_CACHE_BUCKET=my-bucket uv run python -m scripts.warm_cache --source landsat --all-years

    # A slice, e.g. one preset's 1980s:
    uv run python -m scripts.warm_cache --source landsat --years 1984-1989 --preset 3

Already-cached windows are skipped quickly (a cache hit), so re-running after a
partial failure only fills the gaps.
"""
from __future__ import annotations

import argparse
import asyncio
import time

from app import config, geo
from app.cache import cache


async def warm(jobs: list[tuple[str, config.Preset, int]]) -> list:
    failures: list = []
    for source, preset, year in jobs:
        box = geo.build_box(preset.lon, preset.lat)
        t0 = time.time()
        try:
            await cache.get_or_read(box, year, source)
            print(f"  ok   [{source}] {preset.name} {year}  ({time.time() - t0:.1f}s)", flush=True)
            cache.clear_hot()  # warming only needs the persistent copy
        except Exception as exc:  # keep going; report failures
            failures.append((source, preset.name, year, str(exc)))
            print(f"  FAIL [{source}] {preset.name} {year}: {exc}", flush=True)
    return failures


def parse_years(text: str) -> list[int]:
    lo, _, hi = text.partition("-")
    return list(range(int(lo), int(hi or lo) + 1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all-years", action="store_true", help="every year the source covers")
    ap.add_argument("--years", help="a year or range, e.g. 2020 or 1984-1999")
    ap.add_argument(
        "--source", choices=[*config.FEATURE_SOURCES, "all"], default="aef",
        help="feature source(s) to warm",
    )
    ap.add_argument("--preset", type=int, action="append", help="preset index (repeatable)")
    args = ap.parse_args()

    sources = list(config.FEATURE_SOURCES) if args.source == "all" else [args.source]
    presets = [config.PRESETS[i] for i in args.preset] if args.preset else config.PRESETS
    jobs = []
    for sid in sources:
        src = config.FEATURE_SOURCES[sid]
        if args.all_years:
            years = src.years
        elif args.years:
            years = [y for y in parse_years(args.years) if y in src.years]
        else:
            years = [config.DEFAULT_YEAR]
        jobs += [(sid, p, y) for p in presets for y in years]

    print(f"Warming {len(jobs)} windows ({sources}, {len(presets)} presets)…")
    t0 = time.time()
    failures = asyncio.run(warm(jobs))
    print(f"Done in {(time.time() - t0) / 60:.1f} min. Source reads: {cache.source_reads}.")
    if failures:
        print(f"{len(failures)} failed (often a year with no usable scenes there):")
        for f in failures:
            print("   ", *f)


if __name__ == "__main__":
    main()
