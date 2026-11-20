"""The HTTP API: jobs, uploads, and the payload the browser draws.

Everything here runs offline. DINOv2 is switched off rather than mocked,
because the API has to work without it anyway -- that is what `--no-dino` is --
and a test that downloads weights is not a test.

The routes are called through their endpoint functions rather than over a
socket: an HTTP client is a third dependency for no extra coverage, and the
interesting behaviour is in what the handlers do, not in starlette's routing.
"""

from __future__ import annotations

import asyncio
import io
import json
import threading
import time
from pathlib import Path

import pytest

from capturecheck.api.app import (
    allowed_extensions,
    check_upload,
    create_app,
    max_upload_bytes,
    result_payload,
)
from capturecheck.api.jobs import DONE, FAILED, JobStore, wait_for
from capturecheck.labeling import schema
from capturecheck.presentation import QUALITY_BANDS, quality_band


# --------------------------------------------------------------------------- #
# Reaching the route handlers
# --------------------------------------------------------------------------- #
def endpoint(app, path: str, method: str = "GET"):
    for route in app.routes:
        if getattr(route, "path", None) == path and method in getattr(route, "methods", ()):
            return route.endpoint
    raise AssertionError(f"no {method} {path} in the app")


@pytest.fixture
def app():
    return create_app(use_dino=False)


@pytest.fixture
def store():
    s = JobStore()
    yield s
    s.shutdown()


# --------------------------------------------------------------------------- #
# The job store
# --------------------------------------------------------------------------- #
def test_a_job_runs_and_keeps_what_it_returned(store, tmp_path):
    job = store.submit("x.mp4", tmp_path, lambda stage: {"answer": 42})
    done = wait_for(store, job.job_id, timeout=10)

    assert done.status == DONE
    assert done.result == {"answer": 42}
    assert done.error is None
    assert done.elapsed_s >= 0


def test_a_failing_job_reports_a_sentence_not_a_traceback(store, tmp_path):
    def explode(stage):
        raise ValueError("that is not a capture")

    job = store.submit("x.mp4", tmp_path, explode)
    done = wait_for(store, job.job_id, timeout=10)

    assert done.status == FAILED
    assert done.error == "that is not a capture"
    assert done.result is None
    # The browser gets the message, and nothing that leaks a path.
    assert "Traceback" not in json.dumps(done.as_dict())


def test_the_stage_is_reported_while_the_job_runs(store, tmp_path):
    seen: list[str] = []
    gate = threading.Event()

    def slow(stage):
        stage("measuring the capture")
        seen.append("measuring")
        gate.wait(timeout=5)
        return {}

    job = store.submit("x.mp4", tmp_path, slow)
    for _ in range(200):
        if store.get(job.job_id).stage == "measuring the capture":
            break
        time.sleep(0.01)
    assert store.get(job.job_id).stage == "measuring the capture"
    gate.set()
    wait_for(store, job.job_id, timeout=10)


def test_the_upload_is_deleted_once_the_job_ends(store, tmp_path):
    workdir = tmp_path / "upload"
    workdir.mkdir()
    (workdir / "video.mp4").write_bytes(b"not really a video")

    job = store.submit("video.mp4", workdir, lambda stage: {})
    wait_for(store, job.job_id, timeout=10)

    # A stranger's file is not ours to keep.
    assert not workdir.exists()
    assert store.get(job.job_id).workdir is None


def test_the_upload_is_deleted_even_when_the_job_fails(store, tmp_path):
    workdir = tmp_path / "upload"
    workdir.mkdir()
    (workdir / "video.mp4").write_bytes(b"junk")

    def explode(stage):
        raise RuntimeError("no")

    job = store.submit("video.mp4", workdir, explode)
    wait_for(store, job.job_id, timeout=10)
    assert not workdir.exists()


def test_jobs_run_one_at_a_time(store, tmp_path):
    """Two at once would make both miss the 60-second budget."""
    running = []
    peak = []

    def work(stage):
        running.append(1)
        peak.append(len(running))
        time.sleep(0.05)
        running.pop()
        return {}

    jobs = [store.submit(f"{i}.mp4", tmp_path / str(i), work) for i in range(4)]
    for job in jobs:
        wait_for(store, job.job_id, timeout=20)
    assert max(peak) == 1


def test_old_jobs_are_forgotten(tmp_path):
    store = JobStore(max_jobs=3)
    try:
        jobs = [store.submit(f"{i}.mp4", tmp_path / str(i), lambda stage: {}) for i in range(6)]
        for job in jobs:
            wait_for(store, job.job_id, timeout=10)
        assert len(store.all()) <= 3
    finally:
        store.shutdown()


def test_forget_drops_one_job(store, tmp_path):
    job = store.submit("x.mp4", tmp_path, lambda stage: {})
    wait_for(store, job.job_id, timeout=10)
    store.forget(job.job_id)
    assert store.get(job.job_id) is None


# --------------------------------------------------------------------------- #
# What the server says about itself
# --------------------------------------------------------------------------- #
def test_health_describes_what_the_server_can_do(app):
    health = endpoint(app, "/api/health")()

    assert health["ok"] is True
    assert health["budget_s"] > 0
    assert ".mp4" in health["accepts"]
    assert ".zip" in health["accepts"]
    assert isinstance(health["can_score"], bool)
    # The browser must not carry its own copy of the cut-offs.
    assert [b["at_least"] for b in health["bands"]] == [c for c, _ in QUALITY_BANDS]


def test_the_labels_endpoint_is_the_schema(app):
    labels = endpoint(app, "/api/labels")()

    assert [d["name"] for d in labels] == list(schema.VIDEO_LABEL_NAMES)
    assert all(d["meaning"] for d in labels)
    assert all(d["title"] for d in labels)


def test_the_front_end_cannot_invent_its_own_bands(app):
    """Every band the API offers is one the Python side would produce."""
    for band in endpoint(app, "/api/health")()["bands"]:
        assert quality_band(band["at_least"]) == band["name"]


def test_a_zip_and_every_video_extension_are_accepted():
    from capturecheck.io.video import VIDEO_EXTENSIONS

    assert allowed_extensions() == set(VIDEO_EXTENSIONS) | {".zip"}


def test_the_upload_ceiling_can_be_changed(monkeypatch):
    monkeypatch.setenv("CAPTURECHECK_MAX_UPLOAD_MB", "7")
    assert max_upload_bytes() == 7 * 1024 * 1024
    # Nonsense falls back rather than crashing the server at import time.
    monkeypatch.setenv("CAPTURECHECK_MAX_UPLOAD_MB", "not a number")
    assert max_upload_bytes() == 2048 * 1024 * 1024


# --------------------------------------------------------------------------- #
# Uploading
# --------------------------------------------------------------------------- #
def upload_file(name: str, data: bytes):
    from starlette.datastructures import UploadFile

    return UploadFile(filename=name, file=io.BytesIO(data))


def test_a_file_that_is_not_a_capture_is_refused(app):
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as caught:
        asyncio.run(endpoint(app, "/api/captures", "POST")(upload_file("notes.txt", b"hello")))

    assert caught.value.status_code == 415
    assert ".txt" in caught.value.detail


def test_an_empty_file_is_refused(app):
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as caught:
        asyncio.run(endpoint(app, "/api/captures", "POST")(upload_file("empty.mp4", b"")))

    assert caught.value.status_code == 400


def test_a_file_over_the_ceiling_is_refused(app, monkeypatch):
    from fastapi import HTTPException

    monkeypatch.setenv("CAPTURECHECK_MAX_UPLOAD_MB", "1")
    with pytest.raises(HTTPException) as caught:
        asyncio.run(
            endpoint(app, "/api/captures", "POST")(upload_file("big.mp4", b"x" * (2 << 20)))
        )

    assert caught.value.status_code == 413


def test_a_refused_upload_leaves_nothing_behind(app, monkeypatch, tmp_path):
    from fastapi import HTTPException

    monkeypatch.setattr("tempfile.mkdtemp", lambda **kw: str(tmp_path / "work"))
    (tmp_path / "work").mkdir()
    monkeypatch.setenv("CAPTURECHECK_MAX_UPLOAD_MB", "1")

    with pytest.raises(HTTPException):
        asyncio.run(
            endpoint(app, "/api/captures", "POST")(upload_file("big.mp4", b"x" * (2 << 20)))
        )
    assert not (tmp_path / "work").exists()


def test_asking_for_a_job_that_never_existed_is_a_404(app):
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as caught:
        endpoint(app, "/api/captures/{job_id}")("nosuchjob")
    assert caught.value.status_code == 404


# --------------------------------------------------------------------------- #
# The whole way through
# --------------------------------------------------------------------------- #
def test_a_capture_is_measured_and_turned_into_a_payload(tiny_capture_dir):
    stages: list[str] = []
    payload = check_upload(tiny_capture_dir, run_dir=None, use_dino=False, stage=stages.append)

    assert stages  # it said what it was doing
    assert payload["n_frames"] > 0
    assert payload["input_type"] == "video"
    assert [p["name"] for p in payload["problems"]] == list(schema.VIDEO_LABEL_NAMES)
    assert set(payload["stats"]) == set(schema.VIDEO_STAT_COLUMNS)
    assert payload["budget_s"] > 0


def test_a_photo_bundle_goes_through_the_same_path(tiny_bundle):
    payload = check_upload(tiny_bundle, run_dir=None, use_dino=False, stage=lambda s: None)
    assert payload["input_type"] == "images"
    assert payload["n_frames"] > 0


def test_the_payload_is_strictly_json(tiny_capture_dir):
    """A missing measurement must be null, not a bare NaN no browser can read."""
    payload = check_upload(tiny_capture_dir, run_dir=None, use_dino=False, stage=lambda s: None)
    text = json.dumps(payload)  # would raise on a NaN-producing object
    assert "NaN" not in text
    assert "Infinity" not in text


def test_an_upload_never_writes_into_the_feature_cache(tiny_capture_dir, tmp_path, monkeypatch):
    """Non-negotiable 9: data/features holds training inputs, not visitors.

    A stranger's upload writing there would put a row into the training data
    that no label CSV knows about.
    """
    cache = tmp_path / "features"
    cache.mkdir()
    from capturecheck.config import get_config

    monkeypatch.setitem(get_config().paths.__dict__, "features", str(cache))

    check_upload(tiny_capture_dir, run_dir=None, use_dino=False, stage=lambda s: None)
    assert list(cache.iterdir()) == []


def test_a_capture_that_cannot_be_read_says_so(tmp_path):
    empty = tmp_path / "nothing"
    empty.mkdir()
    with pytest.raises((FileNotFoundError, ValueError)):
        check_upload(empty, run_dir=None, use_dino=False, stage=lambda s: None)


# --------------------------------------------------------------------------- #
# The payload itself
# --------------------------------------------------------------------------- #
class FakePrediction:
    score = 0.8123
    run_id = "test_run"
    backend = "torch"
    score_per_seed = [0.80, 0.82]
    problems = dict.fromkeys(schema.VIDEO_LABEL_NAMES, 0.6)


def test_the_payload_separates_what_was_found_from_what_is_suspected(tiny_capture_dir):
    from capturecheck.labeling.pipeline import label_capture

    result = label_capture(tiny_capture_dir, use_dino=False, keep_measurements=True)
    payload = result_payload(result, FakePrediction())

    flagged = {p["name"] for p in payload["problems"] if p["flagged"]}
    suspected = {p["name"] for p in payload["problems"] if p["suspected"]}

    # The model says 0.6 for everything, so anything the rules did not raise
    # is a suspicion, and nothing is ever both.
    assert flagged.isdisjoint(suspected)
    assert suspected == set(schema.VIDEO_LABEL_NAMES) - flagged
    assert payload["band"] == quality_band(0.8123)
    assert payload["model"] == "test_run"


def test_without_a_model_there_is_no_score_but_still_evidence(tiny_capture_dir):
    from capturecheck.labeling.pipeline import label_capture

    result = label_capture(tiny_capture_dir, use_dino=False, keep_measurements=True)
    payload = result_payload(result, None)

    assert payload["score"] is None
    assert payload["band"] is None
    assert payload["model"] is None
    # The measurements do not depend on the model.
    assert set(payload["stats"]) == set(schema.VIDEO_STAT_COLUMNS)
    assert len(payload["problems"]) == schema.N_PROBLEMS
    assert not any(p["suspected"] for p in payload["problems"])


# --------------------------------------------------------------------------- #
# The app as a whole
# --------------------------------------------------------------------------- #
def test_the_app_exposes_the_routes_the_front_end_calls(app):
    paths = {getattr(r, "path", None) for r in app.routes}
    assert {"/api/health", "/api/labels", "/api/captures", "/api/captures/{job_id}"} <= paths


def test_a_missing_front_end_build_is_not_an_error(tmp_path):
    """The API is useful on its own, and `npm run dev` serves its own files."""
    built = create_app(static_dir=tmp_path / "does-not-exist")
    assert endpoint(built, "/")()["ok"] is True


def test_a_built_front_end_is_served(tmp_path):
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<!doctype html><title>CaptureCheck</title>")

    built = create_app(static_dir=dist)
    response = endpoint(built, "/")()
    assert Path(response.path) == dist / "index.html"
