"""The HTTP API the React front end talks to.

Three things only: say what the server can do, accept a capture, and report on
one that is being checked. Everything else the browser needs -- the label
names, their meanings, the score bands -- is served rather than duplicated in
JavaScript, so the front end cannot drift out of step with
``capturecheck/labeling/schema.py``.

Uploads are written to a fresh directory per job and deleted when the job ends.
Nothing from an upload is executed, and the DINOv2 cache is switched off for
them: a stranger's video is not a reason to write into ``data/features``, where
the training inputs live.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from capturecheck.api.jobs import Job, JobStore
from capturecheck.config import get_config
from capturecheck.labeling import schema
from capturecheck.presentation import QUALITY_BANDS, json_safe, quality_band
from capturecheck.utils.logging import get_logger

log = get_logger(__name__)

#: What may be uploaded. A bundle of photos arrives zipped; a video as itself.
ZIP_EXTENSIONS = {".zip"}


def allowed_extensions() -> set[str]:
    from capturecheck.io.video import VIDEO_EXTENSIONS

    return set(VIDEO_EXTENSIONS) | ZIP_EXTENSIONS


def max_upload_bytes() -> int:
    """Upload ceiling, in bytes. ``CAPTURECHECK_MAX_UPLOAD_MB`` overrides it."""
    try:
        megabytes = int(os.environ.get("CAPTURECHECK_MAX_UPLOAD_MB", "2048"))
    except ValueError:
        megabytes = 2048
    return max(1, megabytes) * 1024 * 1024


# --------------------------------------------------------------------------- #
# Turning a checked capture into what the browser draws
# --------------------------------------------------------------------------- #
def result_payload(result, prediction) -> dict:
    """The same shape as ``capturecheck check --json``, so the two agree."""
    row = result.video_row
    evidence = result.evidence
    flagged = set(result.problems)

    problems = []
    for name in schema.VIDEO_LABEL_NAMES:
        probability = prediction.problems.get(name) if prediction else None
        problems.append(
            {
                "name": name,
                "title": schema.LABEL_TITLES.get(name, name),
                "flagged": bool(int(row.get(name, 0) or 0)),
                "probability": probability,
                "evidence": evidence.get(name),
                # The rules found nothing but the model is not convinced.
                "suspected": bool(
                    probability is not None and probability >= 0.5 and name not in flagged
                ),
            }
        )

    return json_safe(
        {
            "capture_id": result.capture_id,
            "input_type": str(row.get("input_type")),
            "n_frames": int(row.get("n_frames") or 0),
            "score": prediction.score if prediction else None,
            "band": quality_band(prediction.score) if prediction else None,
            "score_per_seed": list(prediction.score_per_seed) if prediction else [],
            "model": prediction.run_id if prediction else None,
            "backend": prediction.backend if prediction else None,
            "problems": problems,
            "stats": {k: row.get(k) for k in schema.VIDEO_STAT_COLUMNS},
            "review": bool(int(row.get("review", 0) or 0)),
            "review_reason": str(row.get("review_reason") or ""),
            "time_s": row.get("time_s_total"),
            "budget_s": get_config().runtime.time_budget_s,
        }
    )


def check_upload(path: Path, run_dir: str | Path | None, use_dino: bool, stage) -> dict:
    """Measure and score one uploaded capture. Runs on a job thread."""
    from capturecheck.labeling.pipeline import label_capture
    from capturecheck.model.predict import predict_capture

    stage("measuring the capture")
    result = label_capture(
        path,
        use_dino=use_dino,
        keep_measurements=True,
        # An upload is a guest, not part of the dataset: it must not write into
        # the feature cache the model is trained from.
        use_cache=False,
        cache_measurements=False,
    )
    stage("scoring with the model")
    prediction = predict_capture(result, run_dir=run_dir)
    stage("writing the report")
    return result_payload(result, prediction)


# --------------------------------------------------------------------------- #
# The app
# --------------------------------------------------------------------------- #
def create_app(
    run_dir: str | Path | None = None,
    use_dino: bool = True,
    static_dir: str | Path | None = None,
    allow_origins: list[str] | None = None,
) -> FastAPI:
    store = JobStore()
    app = FastAPI(
        title="CaptureCheck",
        description="Predict 3D Gaussian Splatting quality before you train.",
        version=_version(),
    )
    app.state.store = store
    app.state.run_dir = run_dir
    app.state.use_dino = use_dino

    # The dev server runs on another port, so the browser calls this one
    # cross-origin. In production the front end is served from here and this
    # never applies.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins or ["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health() -> dict:
        """What this server can do right now, including whether it can score."""
        from capturecheck.model.train import latest_run

        run = Path(run_dir) if run_dir else latest_run()
        return {
            "ok": True,
            "version": _version(),
            "model": run.name if run else None,
            "can_score": run is not None,
            "budget_s": get_config().runtime.time_budget_s,
            "max_upload_mb": max_upload_bytes() // (1024 * 1024),
            "accepts": sorted(allowed_extensions()),
            "bands": [{"at_least": c, "name": n} for c, n in QUALITY_BANDS],
        }

    @app.get("/api/labels")
    def labels() -> list[dict]:
        """Every capture problem and what it means, straight from the schema."""
        return [
            {
                "name": d.name,
                "title": schema.LABEL_TITLES.get(d.name, d.name),
                "meaning": d.meaning,
            }
            for d in schema.VIDEO_LABELS
        ]

    @app.post("/api/captures", status_code=202)
    async def upload(file: UploadFile = File(...)) -> dict:
        """Accept a capture and start checking it. Returns a job to poll."""
        name = Path(file.filename or "capture").name
        suffix = Path(name).suffix.lower()
        if suffix not in allowed_extensions():
            raise HTTPException(
                status_code=415,
                detail=(
                    f"`{suffix or name}` is not a capture. Upload a video "
                    f"({', '.join(sorted(e for e in allowed_extensions() if e != '.zip'))}) "
                    "or a .zip of photos."
                ),
            )

        workdir = Path(tempfile.mkdtemp(prefix="capturecheck_upload_"))
        target = workdir / name
        limit = max_upload_bytes()
        written = 0
        try:
            with target.open("wb") as handle:
                while chunk := await file.read(1024 * 1024):
                    written += len(chunk)
                    if written > limit:
                        raise HTTPException(
                            status_code=413,
                            detail=f"That file is over the {limit // (1024 * 1024)} MB limit.",
                        )
                    handle.write(chunk)
        except HTTPException:
            shutil.rmtree(workdir, ignore_errors=True)
            raise
        except Exception:
            shutil.rmtree(workdir, ignore_errors=True)
            raise

        if written == 0:
            shutil.rmtree(workdir, ignore_errors=True)
            raise HTTPException(status_code=400, detail="That file is empty.")

        log.info("accepted %s (%.1f MB)", name, written / 1024 / 1024)
        job = store.submit(
            filename=name,
            workdir=workdir,
            work=lambda stage: check_upload(target, app.state.run_dir, app.state.use_dino, stage),
        )
        return job.as_dict()

    @app.get("/api/captures/{job_id}")
    def job_status(job_id: str) -> dict:
        job = _require(store.get(job_id))
        return job.as_dict()

    @app.get("/api/captures")
    def job_list() -> list[dict]:
        return [j.as_dict() for j in store.all()]

    @app.delete("/api/captures/{job_id}")
    def forget(job_id: str) -> dict:
        """Drop a finished job. The uploaded file is already gone."""
        _require(store.get(job_id))
        store.forget(job_id)
        return {"ok": True, "job_id": job_id}

    @app.on_event("shutdown")
    def _stop() -> None:
        store.shutdown()

    _mount_static(app, static_dir)
    return app


def _require(job: Job | None) -> Job:
    if job is None:
        raise HTTPException(status_code=404, detail="No such job. It may have been forgotten.")
    return job


def _version() -> str:
    from capturecheck import __version__

    return __version__


def _mount_static(app, static_dir: str | Path | None) -> None:
    """Serve the built front end, when there is one.

    Without a build the API still works and the dev server talks to it, so a
    missing ``frontend/dist`` is a normal state rather than an error.
    """
    root = Path(static_dir) if static_dir else _default_static()
    index = root / "index.html"
    if not index.is_file():
        log.info("no built front end at %s; serving the API only", root)

        @app.get("/")
        def _no_ui() -> dict:
            return {
                "ok": True,
                "message": (
                    "The API is running. Build the front end with "
                    "`npm --prefix frontend run build`, or start it in "
                    "development with `npm --prefix frontend run dev`."
                ),
                "docs": "/docs",
            }

        return

    app.mount("/assets", StaticFiles(directory=root / "assets"), name="assets")

    @app.get("/")
    def _index() -> FileResponse:
        return FileResponse(index)

    log.info("serving the front end from %s", root)


def _default_static() -> Path:
    return Path(__file__).resolve().parents[2] / "frontend" / "dist"
