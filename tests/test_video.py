"""Video probing and 2 fps frame sampling."""

from __future__ import annotations

import numpy as np

from capturecheck.config import get_config
from capturecheck.io import video as video_io


def test_probe_reports_size_and_duration(tiny_video):
    info = video_io.probe(tiny_video)
    assert (info.width, info.height) == (320, 240)
    assert info.fps > 0
    assert 1.5 <= info.duration_s <= 2.5
    assert info.probe_source in {"ffprobe", "opencv"}


def test_sample_frames_takes_two_per_second(tiny_video):
    frames, times, info = video_io.sample_frames(tiny_video, fps=2, width=320)
    # A 2-second video at 2 fps gives 4 frames.
    assert len(frames) == 4
    assert len(times) == len(frames)
    assert all(f.shape[:2] == (240, 320) for f in frames)
    assert times == sorted(times)
    assert times[-1] < info.duration_s + 0.5


def test_sample_frames_resizes_to_requested_width(tiny_video):
    frames, _, _ = video_io.sample_frames(tiny_video, fps=2, width=160)
    assert frames[0].shape[1] == 160
    assert frames[0].shape[0] == 120  # aspect ratio kept


def test_sample_frames_respects_max_frames(tiny_video):
    frames, times, _ = video_io.sample_frames(tiny_video, fps=30, max_frames=3, width=320)
    assert len(frames) == 3
    assert times[0] < times[-1]


def test_target_times_counts():
    """A 60 s video gives 120 frames at 2 fps; a 10-minute one is capped at 600."""
    cfg = get_config()
    assert len(video_io._target_times(60, cfg.sampling.fps, 600)) == 120
    assert len(video_io._target_times(600, cfg.sampling.fps, 600)) == 600
    assert len(video_io._target_times(3600, cfg.sampling.fps, 600)) == 600
    # The capped samples still span the whole video.
    times = video_io._target_times(3600, 2, 600)
    assert times[0] < 10 and times[-1] > 3590


def test_resize_to_width_keeps_aspect():
    img = np.zeros((480, 960, 3), dtype=np.uint8)
    out = video_io.resize_to_width(img, 480)
    assert out.shape[:2] == (240, 480)
