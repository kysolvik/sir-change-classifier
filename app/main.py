"""FastAPI application: config + classify endpoints, static frontend.

Stateless by design — every /classify call carries its points and both years, so
any worker can serve any request. The only shared state is the embedding cache.
"""
from __future__ import annotations

import asyncio
import base64
import contextlib
import pathlib
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from . import classify, colorize, config, embeddings, geo
from .cache import cache

WEB_DIR = pathlib.Path(__file__).resolve().parent.parent / "web"

limiter = Limiter(key_func=get_remote_address)


# --- request/response models ----------------------------------------------
class ClassSpec(BaseModel):
    name: str
    color: str = "#ff0000"


class PointSpec(BaseModel):
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    cls: str = Field(alias="class")

    model_config = {"populate_by_name": True}


class _ClassifyBase(BaseModel):
    """Fields + validation shared by /api/classify and /api/compare."""

    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    classifier: Literal["rf", "knn"] = "rf"
    classes: list[ClassSpec]
    points: list[PointSpec]

    @staticmethod
    def _check_year(v: int) -> int:
        if v not in config.YEARS:
            raise ValueError(f"year must be in {config.MIN_YEAR}-{config.MAX_YEAR}")
        return v

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

    @field_validator("training_year", "target_year")
    @classmethod
    def _years(cls, v: int) -> int:
        return cls._check_year(v)


class CompareRequest(_ClassifyBase):
    training_year: int
    year_a: int
    year_b: int

    @field_validator("training_year", "year_a", "year_b")
    @classmethod
    def _years(cls, v: int) -> int:
        return cls._check_year(v)


def _require_known_point_classes(body: _ClassifyBase) -> None:
    names = {c.name for c in body.classes}
    for p in body.points:
        if p.cls not in names:
            raise HTTPException(400, f"point references unknown class '{p.cls}'")


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
            "presets": [
                {"name": p.name, "lat": p.lat, "lon": p.lon, "blurb": p.blurb}
                for p in config.PRESETS
            ],
        }

    @app.post("/api/classify")
    @limiter.limit(config.RATE_LIMIT)
    async def do_classify(request: Request, body: ClassifyRequest) -> JSONResponse:
        _require_known_point_classes(body)

        box = geo.build_box(body.lon, body.lat)
        try:
            train_win = await cache.get_or_read(box, body.training_year)
            target_win = (
                train_win
                if body.target_year == body.training_year
                else await cache.get_or_read(box, body.target_year)
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
        data_url = "data:image/png;base64," + base64.b64encode(png).decode("ascii")
        min_lon, min_lat, max_lon, max_lat = box.lonlat_bounds()

        return JSONResponse(
            {
                "image": data_url,
                "bounds": [[min_lat, min_lon], [max_lat, max_lon]],
                "training_year": body.training_year,
                "target_year": body.target_year,
                "classifier": body.classifier,
                "accuracy": result.accuracy,
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

        box = geo.build_box(body.lon, body.lat)
        try:
            train_win = await cache.get_or_read(box, body.training_year)
            win_a = (
                train_win
                if body.year_a == body.training_year
                else await cache.get_or_read(box, body.year_a)
            )
            if body.year_b == body.training_year:
                win_b = train_win
            elif body.year_b == body.year_a:
                win_b = win_a
            else:
                win_b = await cache.get_or_read(box, body.year_b)
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
        data_url = "data:image/png;base64," + base64.b64encode(png).decode("ascii")
        min_lon, min_lat, max_lon, max_lat = box.lonlat_bounds()

        return JSONResponse(
            {
                "image": data_url,
                "bounds": [[min_lat, min_lon], [max_lat, max_lon]],
                "training_year": body.training_year,
                "year_a": body.year_a,
                "year_b": body.year_b,
                "classifier": body.classifier,
                "accuracy": trained.accuracy,
                "n_points_used": trained.n_points_used,
                "n_points_skipped": trained.n_points_skipped,
                "compared_pixels": compared_pixels,
                "transitions": transitions,
            }
        )

    if WEB_DIR.is_dir():
        app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")

    return app


app = create_app()
