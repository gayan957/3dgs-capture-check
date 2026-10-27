"""Image bundles: folders, zips, EXIF and ordering."""

from __future__ import annotations

from capturecheck.io import images as images_io
from capturecheck.io.load import load_capture


def test_read_bundle_returns_every_photo_in_order(tiny_bundle):
    read = images_io.read_bundle(tiny_bundle, width=320)
    assert len(read.frames) == 20
    assert read.names == [f"IMG_{i:04d}.jpg" for i in range(20)]
    assert read.info.order_source == "exif_time"
    assert read.times[0] == 0.0
    assert read.times[-1] > read.times[0]


def test_zip_matches_folder(tiny_bundle, tiny_bundle_zip):
    folder = images_io.read_bundle(tiny_bundle, width=320)
    zipped = images_io.read_bundle(tiny_bundle_zip, width=320)
    assert folder.names == zipped.names
    assert len(folder.frames) == len(zipped.frames)
    assert folder.info.n_cameras == zipped.info.n_cameras


def test_read_exif_fields(tiny_bundle):
    photo = sorted(tiny_bundle.glob("*.jpg"))[0]
    exif = images_io.read_exif(photo)
    assert exif.camera == "TestCam X1"
    assert exif.exposure_s == 0.008
    assert exif.iso == 200
    assert exif.focal_length_35mm == 26
    assert exif.datetime_original is not None


def test_bundle_info_counts_cameras_and_sizes(tiny_bundle):
    read = images_io.read_bundle(tiny_bundle, width=320)
    info = images_io.bundle_info(read.exif)
    assert info.n_photos == 20
    assert info.n_cameras == 1
    assert info.n_sizes == 1
    assert info.has_exif_time is True


def test_max_frames_thins_the_bundle(tiny_bundle):
    read = images_io.read_bundle(tiny_bundle, max_frames=5, width=320)
    assert len(read.frames) == 5
    assert read.names[0] == "IMG_0000.jpg"
    assert read.names[-1] == "IMG_0019.jpg"


def test_bundle_without_exif_falls_back_to_name_order(no_exif_bundle):
    read = images_io.read_bundle(no_exif_bundle, width=320)
    assert read.info.has_exif_time is False
    assert read.info.order_source in {"name", "unknown"}
    assert read.times == [None] * len(read.frames)


def test_load_capture_handles_all_three_inputs(tiny_video, tiny_bundle, tiny_bundle_zip):
    vid = load_capture(tiny_video, width=320)
    assert vid.input_type == "video" and vid.n_frames == 4
    assert vid.label_for(0) == "frame_000"

    folder = load_capture(tiny_bundle, width=320)
    assert folder.input_type == "images" and folder.n_frames == 20
    assert folder.capture_id == "kitchen_photos"
    assert folder.has_order

    zipped = load_capture(tiny_bundle_zip, width=320)
    assert zipped.input_type == "images" and zipped.n_frames == 20
    assert zipped.capture_id == "kitchen_photos"


def test_load_capture_reads_capture_folder(tiny_capture_dir):
    cap = load_capture(tiny_capture_dir, width=320)
    assert cap.input_type == "video"
    assert cap.capture_id == "kitchen_video"
    assert cap.focal_length_px is None  # videos carry no EXIF focal length
