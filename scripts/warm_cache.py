"""Pre-warm the shared embedding cache for the preset study areas.

Run once at deploy so students hitting a preset never pay the ~10 s cold read.
In production set GCS_CACHE_BUCKET first so the warmed windows land in GCS:

    GCS_CACHE_BUCKET=my-bucket uv run python -m scripts.warm_cache            # default year
    GCS_CACHE_BUCKET=my-bucket uv run python -m scripts.warm_cache --all-years
"""
from __future__ import annotations

import argparse
import asyncio
import time

from app import config, geo
from app.cache import cache


async def warm(years: list[int]) -> None:
    for preset in config.PRESETS:
        box = geo.build_box(preset.lon, preset.lat)
        for year in years:
            t0 = time.time()
            try:
                await cache.get_or_read(box, year)
                print(f"  ok   {preset.name} {year}  ({time.time() - t0:.1f}s)")
            except Exception as exc:  # keep going; report failures
                print(f"  FAIL {preset.name} {year}: {exc}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all-years", action="store_true", help="warm every year, not just the default")
    args = ap.parse_args()
    years = config.YEARS if args.all_years else [config.DEFAULT_YEAR]
    print(f"Warming {len(config.PRESETS)} presets x {len(years)} year(s)…")
    asyncio.run(warm(years))
    print(f"Done. Source reads performed: {cache.source_reads}")


if __name__ == "__main__":
    main()
