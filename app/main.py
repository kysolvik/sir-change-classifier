"""FastAPI application: config + classify endpoints, static frontend.

Stateless by design — every /classify call carries its points and both years, so
any worker can serve any request. The only shared state is the embedding cache.
"""
from __future__ import annotations

import asyncio
import base64
import contextlib
import pathlib
import time
from collections import deque
from typing import ClassVar, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator, model_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from limits import parse as parse_limit
import numpy as np

from . import classify, colorize, config, embeddings, geo
from .cache import cache

WEB_DIR = pathlib.Path(__file__).resolve().parent.parent / "web"



def client_ip(request: Request) -> str:
    """The real client IP for rate limiting.

    Behind Cloud Run, ``request.client.host`` is the front end's address, so every
    user would share one bucket. Cloud Run appends the caller's IP as the
    *rightmost* X-Forwarded-For entry; anything left of it is client-supplied and
    spoofable, so only the rightmost entry is trusted.
    """
    xff = request.headers.get("x-forwarded-for", "")
    if xff.strip():
        return xff.split(",")[-1].strip()
    return request.client.host if request.client else "unknown"


limiter = Limiter(key_func=client_ip)
_default_limit = parse_limit(config.RATE_LIMIT_DEFAULT)


class HourlyCap:
    """Sliding one-hour window on expensive calls, shared by all IPs in-process."""

    def __init__(self, per_hour: int):
        self.per_hour = per_hour
        self._stamps: deque[float] = deque()

    def try_acquire(self) -> bool:
        now = time.monotonic()
        while self._stamps and now - self._stamps[0] > 3600:
            self._stamps.popleft()
        if len(self._stamps) >= self.per_hour:
            return False
        self._stamps.append(now)
        return True


classify_cap = HourlyCap(config.GLOBAL_CLASSIFY_PER_HOUR)


def _require_capacity() -> None:
    if not classify_cap.try_acquire():
        raise HTTPException(
            503, "The classifier is busy right now. Please try again in a few minutes."
        )


# --- request/response models ----------------------------------------------
class ClassSpec(BaseModel):
    name: str
    color: str = "#ff0000"


class PointSpec(BaseModel):
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    cls: str = Field(alias="class")

    model_config = {"populate_by_name": True}


FeatureSourceId = Literal[tuple(config.FEATURE_SOURCES)]


class _ClassifyBase(BaseModel):
    """Fields + validation shared by /api/classify and /api/compare."""

    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    classifier: Literal["rf", "knn"] = "rf"
    feature_source: FeatureSourceId = config.DEFAULT_FEATURE_SOURCE
    classes: list[ClassSpec]
    points: list[PointSpec]

    # Year fields checked against the chosen feature source's range.
    _year_fields: ClassVar[tuple[str, ...]] = ()

    @model_validator(mode="after")
    def _years_in_source_range(self):
        src = config.FEATURE_SOURCES[self.feature_source]
        for name in self._year_fields:
            if getattr(self, name) not in src.years:
                raise ValueError(
                    f"{name}: {src.label} cover {src.min_year}-{src.max_year}"
                )
        return self

    @field_validator("classes")
    @classmethod
    def _classes_ok(cls, v: list[ClassSpec]) -> list[ClassSpec]:
        if not (1 <= len(v) <= config.MAX_CLASSES):
            raise ValueError(f"need 1-{config.MAX_CLASSES} classes")
        names = [c.name for c in v]
        if len(set(names)) != len(names):
            raise ValueError("class names must be unique")
        return v

    @field_validator("points")
    @classmethod
    def _points_ok(cls, v: list[PointSpec]) -> list[PointSpec]:
        if len(v) > config.MAX_POINTS:
            raise ValueError(f"at most {config.MAX_POINTS} points")
        return v


class ClassifyRequest(_ClassifyBase):
    training_year: int
    target_year: int
    _year_fields: ClassVar[tuple[str, ...]] = ("training_year", "target_year")


class CompareRequest(_ClassifyBase):
    training_year: int
    year_a: int
    year_b: int
    _year_fields: ClassVar[tuple[str, ...]] = ("training_year", "year_a", "year_b")


class EvaluateRequest(_ClassifyBase):
    """Score the same points on several feature sources (``feature_source`` unused).

    ``sources`` defaults to every source; ones that don't cover ``training_year``
    or this area are reported as skipped rather than failing the request.
    """

    training_year: int
    sources: list[FeatureSourceId] | None = None


def _require_known_point_classes(body: _ClassifyBase) -> None:
    names = {c.name for c in body.classes}
    for p in body.points:
        if p.cls not in names:
            raise HTTPException(400, f"point references unknown class '{p.cls}'")


# Slow sources (raw Landsat) are pre-warmed for presets only; a box "is a preset"
# when it snaps to the same grid cell as one (nearby free-entry coords collapse too).
_PRESET_BOX_KEYS = {geo.build_box(p.lon, p.lat).key() for p in config.PRESETS}


def _source_blocked(box: geo.Box, source: str) -> str | None:
    """Why ``source`` can't be used for ``box`` (None if it can)."""
    src = config.FEATURE_SOURCES[source]
    if src.preset_only and box.key() not in _PRESET_BOX_KEYS:
        return f"{src.label} are available for the example places only, for now."
    return None


def _require_source_allowed(box: geo.Box, source: str) -> None:
    reason = _source_blocked(box, source)
    if reason:
        raise HTTPException(400, reason)


def _png_data_url(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm the tile index in the background so the first request is fast without
    # blocking readiness.
    task = asyncio.create_task(embeddings.get_index())
    yield
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError, Exception):
        await task


def create_app() -> FastAPI:
    app = FastAPI(title="SIR Change Classifier", lifespan=lifespan)
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    @app.middleware("http")
    async def flood_guard(request: Request, call_next):
        # slowapi's own middleware skips mounts, so the per-IP default limit for
        # every path (static files included) is applied here.
        if not limiter.limiter.hit(_default_limit, "default", client_ip(request)):
            return JSONResponse({"error": "Too many requests"}, status_code=429)
        response = await call_next(request)
        if not request.url.path.startswith("/api/") and response.status_code == 200:
            response.headers.setdefault("Cache-Control", "public, max-age=600")
        return response

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"status": "ok"}

    @app.get("/api/config")
    async def get_config() -> dict:
        return {
            "box_size_km": config.BOX_SIZE_KM,
            "years": config.YEARS,
            "default_year": config.DEFAULT_YEAR,
            "min_year": config.MIN_YEAR,
            "max_year": config.MAX_YEAR,
            "max_classes": config.MAX_CLASSES,
            "max_points": config.MAX_POINTS,
            "default_classifier": config.DEFAULT_CLASSIFIER,
            "default_feature_source": config.DEFAULT_FEATURE_SOURCE,
            "feature_sources": [
                {
                    "id": s.id,
                    "label": s.label,
                    "min_year": s.min_year,
                    "max_year": s.max_year,
                    "res_m": s.res_m,
                    "preset_only": s.preset_only,
                    "imagery": s.imagery,
                }
                for s in config.FEATURE_SOURCES.values()
            ],
            "imagery_styles": [
                {"id": k, "label": v["label"]} for k, v in colorize.IMAGERY_STYLES.items()
            ],
            "presets": [
                {"name": p.name, "lat": p.lat, "lon": p.lon, "blurb": p.blurb}
                for p in config.PRESETS
            ],
        }

    @app.post("/api/classify")
    @limiter.limit(config.RATE_LIMIT)
    async def do_classify(request: Request, body: ClassifyRequest) -> JSONResponse:
        _require_known_point_classes(body)
        _require_capacity()

        box = geo.build_box(body.lon, body.lat)
        _require_source_allowed(box, body.feature_source)
        src = body.feature_source
        try:
            train_win = await cache.get_or_read(box, body.training_year, src)
            target_win = (
                train_win
                if body.target_year == body.training_year
                else await cache.get_or_read(box, body.target_year, src)
            )
        except embeddings.NoDataError as exc:
            raise HTTPException(422, str(exc))

        class_names = [c.name for c in body.classes]
        colors = [c.color for c in body.classes]
        try:
            result = await asyncio.to_thread(
                classify.train_and_predict,
                train_win,
                target_win,
                body.points,
                class_names,
                body.classifier,
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc))

        png = await asyncio.to_thread(colorize.labels_to_png, result.labels, colors)
        min_lon, min_lat, max_lon, max_lat = box.lonlat_bounds()

        return JSONResponse(
            {
                "image": _png_data_url(png),
                "bounds": [[min_lat, min_lon], [max_lat, max_lon]],
                "training_year": body.training_year,
                "target_year": body.target_year,
                "classifier": body.classifier,
                "feature_source": body.feature_source,
                "accuracy": result.accuracy,
                "per_class_accuracy": {
                    class_names[i]: a for i, a in result.per_class_accuracy.items()
                },
                "n_points_used": result.n_points_used,
                "n_points_skipped": result.n_points_skipped,
                "used_flags": result.used_flags,
                "class_pixel_counts": {
                    class_names[i]: c for i, c in result.class_pixel_counts.items()
                },
            }
        )

    @app.post("/api/compare")
    @limiter.limit(config.RATE_LIMIT)
    async def do_compare(request: Request, body: CompareRequest) -> JSONResponse:
        _require_known_point_classes(body)
        _require_capacity()

        box = geo.build_box(body.lon, body.lat)
        _require_source_allowed(box, body.feature_source)
        src = body.feature_source
        try:
            train_win = await cache.get_or_read(box, body.training_year, src)
            win_a = (
                train_win
                if body.year_a == body.training_year
                else await cache.get_or_read(box, body.year_a, src)
            )
            if body.year_b == body.training_year:
                win_b = train_win
            elif body.year_b == body.year_a:
                win_b = win_a
            else:
                win_b = await cache.get_or_read(box, body.year_b, src)
        except embeddings.NoDataError as exc:
            raise HTTPException(422, str(exc))

        class_names = [c.name for c in body.classes]
        try:
            trained = await asyncio.to_thread(
                classify.train_model, train_win, body.points, class_names, body.classifier
            )
            labels_a = await asyncio.to_thread(classify.predict_window, trained, win_a)
            labels_b = await asyncio.to_thread(classify.predict_window, trained, win_b)
        except ValueError as exc:
            raise HTTPException(400, str(exc))

        ids, observed, compared_pixels = classify.transition_map(
            labels_a, labels_b, len(class_names)
        )
        palette = colorize.transition_palette(len(observed))
        color_by_id = {}
        transitions = []
        for (frm, to, pixels), color in zip(observed, palette):
            color_by_id[frm * len(class_names) + to] = color
            transitions.append(
                {
                    "from": class_names[frm],
                    "to": class_names[to],
                    "color": color,
                    "pixels": pixels,
                }
            )

        png = await asyncio.to_thread(colorize.ids_to_png, ids, color_by_id)
        min_lon, min_lat, max_lon, max_lat = box.lonlat_bounds()

        return JSONResponse(
            {
                "image": _png_data_url(png),
                "bounds": [[min_lat, min_lon], [max_lat, max_lon]],
                "training_year": body.training_year,
                "year_a": body.year_a,
                "year_b": body.year_b,
                "classifier": body.classifier,
                "feature_source": body.feature_source,
                "accuracy": trained.accuracy,
                "n_points_used": trained.n_points_used,
                "n_points_skipped": trained.n_points_skipped,
                "compared_pixels": compared_pixels,
                "transitions": transitions,
            }
        )

    @app.post("/api/evaluate")
    @limiter.limit(config.RATE_LIMIT)
    async def do_evaluate(request: Request, body: EvaluateRequest) -> JSONResponse:
        """Compare feature sources: cross-validated accuracy on the same points."""
        _require_known_point_classes(body)
        box = geo.build_box(body.lon, body.lat)

        wanted = body.sources or list(config.FEATURE_SOURCES)
        skipped, candidates = [], []
        for sid in dict.fromkeys(wanted):  # de-dupe, keep order
            src = config.FEATURE_SOURCES[sid]
            if body.training_year not in src.years:
                reason = f"Only covers {src.min_year}-{src.max_year}."
            else:
                reason = _source_blocked(box, sid)
            if reason:
                skipped.append({"source": sid, "label": src.label, "reason": reason})
            else:
                candidates.append(sid)
        if not candidates:
            raise HTTPException(400, "No feature source covers this year and area.")
        _require_capacity()

        wins = {}
        for sid in candidates:
            try:
                wins[sid] = await cache.get_or_read(box, body.training_year, sid)
            except embeddings.NoDataError as exc:
                label = config.FEATURE_SOURCES[sid].label
                skipped.append({"source": sid, "label": label, "reason": str(exc)})
        if not wins:
            raise HTTPException(422, "No feature source has data here for that year.")

        class_names = [c.name for c in body.classes]
        try:
            results, n_shared = await asyncio.to_thread(
                classify.compare_sources, wins, body.points, class_names, body.classifier
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc))

        return JSONResponse(
            {
                "training_year": body.training_year,
                "classifier": body.classifier,
                "n_points_total": len(body.points),
                "n_points_shared": n_shared,
                "results": [
                    {
                        "source": sid,
                        "label": config.FEATURE_SOURCES[sid].label,
                        "res_m": config.FEATURE_SOURCES[sid].res_m,
                        "accuracy": r.evaluation.accuracy,
                        "per_class": {
                            class_names[i]: a for i, a in r.evaluation.per_class.items()
                        },
                        "n_points_usable": r.n_points_usable,
                    }
                    for sid, r in results.items()
                ],
                "skipped": skipped,
            }
        )

    @app.get("/api/imagery")
    @limiter.limit(config.IMAGERY_RATE_LIMIT)
    async def get_imagery(
        request: Request,
        lat: float,
        lon: float,
        year: int,
        style: str = "natural",
        source: str = "landsat",
    ) -> JSONResponse:
        """A picture of the raw imagery behind a feature source, for one year."""
        src = config.FEATURE_SOURCES.get(source)
        if src is None or not src.imagery:
            raise HTTPException(400, f"no imagery for source '{source}'")
        if style not in colorize.IMAGERY_STYLES:
            raise HTTPException(400, f"unknown style '{style}'")
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            raise HTTPException(422, "lat/lon out of range")
        if year not in src.years:
            raise HTTPException(422, f"{src.label} cover {src.min_year}-{src.max_year}")
        box = geo.build_box(lon, lat)
        _require_source_allowed(box, source)
        try:
            win = await cache.get_or_read(box, year, source)
        except embeddings.NoDataError as exc:
            raise HTTPException(422, str(exc))

        png = await asyncio.to_thread(
            colorize.bands_to_png, win.array, config.LANDSAT_BANDS, style
        )
        min_lon, min_lat, max_lon, max_lat = box.lonlat_bounds()
        coverage = float(np.isfinite(win.array).all(axis=-1).mean())
        return JSONResponse(
            {
                "image": _png_data_url(png),
                "bounds": [[min_lat, min_lon], [max_lat, max_lon]],
                "year": year,
                "style": style,
                "source": source,
                "res_m": win.pix,
                "coverage": coverage,
            }
        )

    if WEB_DIR.is_dir():
        app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")

    return app


app = create_app()
