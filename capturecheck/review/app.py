"""Streamlit screen for reviewing the captures the rules were unsure about.

Run it with::

    streamlit run capturecheck/review/app.py

It reads ``review_queue.csv`` and ``frame_labels.csv``, shows a dozen thumbnails
with the measurements beside them, and writes the reviewer's decision to
``human_labels.csv``. No database: the CSVs are the whole state.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from capturecheck.config import get_config
from capturecheck.io import dataset
from capturecheck.io.load import load_capture
from capturecheck.labeling import schema

N_THUMBNAILS = 12
THUMBNAIL_WIDTH = 320
HIGHLIGHTS: tuple[tuple[str, str, str], ...] = (
    ("label_blur", "BLURRY", "blurry"),
    ("label_view", "DUPLICATE", "duplicate"),
    ("label_light", "DARK", "dark"),
    ("label_light", "BRIGHT", "blown out"),
    ("label_texture", "LOW_TEXTURE", "low texture"),
    ("label_shiny", 1, "shiny"),
    ("label_noise", 1, "noisy"),
)


@st.cache_data(show_spinner=False)
def load_tables(labels_dir: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Queue, frame labels and capture labels, re-read when the folder changes."""
    folder = Path(labels_dir)
    queue = dataset.read_review_queue(out_dir=folder)
    frames = dataset.read_frame_labels(out_dir=folder)
    videos = dataset.read_video_labels(out_dir=folder)
    return queue, frames, videos


@st.cache_data(show_spinner=False)
def capture_paths() -> dict[str, str]:
    """capture_id -> path on disk, for both real captures and broken copies."""
    return {ref.capture_id: str(ref.path) for ref in dataset.discover_all()}


@st.cache_data(show_spinner=True, max_entries=8)
def load_thumbnails(path: str, capture_id: str, wanted: list[int]) -> dict[int, np.ndarray]:
    """RGB thumbnails for the chosen frame indices."""
    import cv2

    capture = load_capture(path, capture_id=capture_id, width=THUMBNAIL_WIDTH)
    out: dict[int, np.ndarray] = {}
    for idx in wanted:
        if 0 <= idx < capture.n_frames:
            out[idx] = cv2.cvtColor(capture.frames[idx], cv2.COLOR_BGR2RGB)
    return out


def pick_frames(frame_df: pd.DataFrame, n: int = N_THUMBNAILS) -> list[int]:
    """Show the interesting frames first, then fill up with an even spread."""
    if frame_df.empty:
        return []
    indices = frame_df["frame_idx"].tolist()
    interesting: list[int] = []
    for column, value, _ in HIGHLIGHTS:
        if column not in frame_df.columns:
            continue
        hits = frame_df.loc[frame_df[column] == value, "frame_idx"].tolist()
        interesting.extend(hits[: max(1, n // 4)])

    chosen: list[int] = []
    for idx in interesting:
        if idx not in chosen:
            chosen.append(int(idx))
        if len(chosen) >= n // 2:
            break

    spread = np.linspace(0, len(indices) - 1, num=min(n, len(indices))).round().astype(int)
    for pos in spread:
        idx = int(indices[pos])
        if idx not in chosen:
            chosen.append(idx)
        if len(chosen) >= n:
            break
    return sorted(chosen)[:n]


def frame_notes(row: pd.Series) -> list[str]:
    """Which problems a single frame has, as short words for the caption."""
    notes = []
    for column, value, word in HIGHLIGHTS:
        if column in row and row[column] == value:
            notes.append(word)
    return notes


def caption_for(row: pd.Series, is_video: bool) -> str:
    when = ""
    if is_video and pd.notna(row.get("time_s")):
        when = f"{float(row['time_s']):.1f} s · "
    name = str(row.get("frame_name", f"frame {int(row['frame_idx'])}"))
    notes = frame_notes(row)
    suffix = f"  ⚠ {', '.join(notes)}" if notes else ""
    return f"{when}{name}{suffix}"


def main() -> None:
    st.set_page_config(page_title="CaptureCheck review", layout="wide")
    cfg = get_config()

    st.sidebar.title("CaptureCheck review")
    labels_dir = st.sidebar.text_input("Labels folder", str(cfg.paths.resolve("labels")))
    reviewer = st.sidebar.text_input("Reviewer name", "")
    if st.sidebar.button("Reload CSVs"):
        st.cache_data.clear()
        st.rerun()

    queue, frames, videos = load_tables(labels_dir)
    done = dataset.read_human_labels(out_dir=Path(labels_dir))
    reviewed_ids = set(done["capture_id"]) if not done.empty else set()

    if queue.empty:
        st.info(
            "The review queue is empty. Run `capturecheck label data` first: "
            "captures whose measurements sit close to a threshold land here."
        )
        return

    pending = queue[~queue["capture_id"].isin(reviewed_ids)]
    st.sidebar.metric("Reviewed", f"{len(reviewed_ids)} of {len(queue)}")
    st.sidebar.progress(len(reviewed_ids) / max(1, len(queue)))

    if pending.empty:
        st.success(f"All {len(queue)} captures in the queue have been reviewed. Thank you!")
        st.dataframe(done.tail(20), width="stretch")
        return

    options = pending["capture_id"].tolist()
    capture_id = st.sidebar.selectbox("Capture", options, index=0)
    show_only_flagged = st.sidebar.checkbox("Only show flagged frames", value=False)

    queue_row = pending[pending["capture_id"] == capture_id].iloc[0]
    video_row = (
        videos[videos["capture_id"] == capture_id].iloc[0]
        if not videos.empty and (videos["capture_id"] == capture_id).any()
        else queue_row
    )
    frame_df = frames[frames["capture_id"] == capture_id].sort_values("frame_idx")
    is_video = str(video_row.get("input_type", "video")) == "video"

    st.title(capture_id)
    st.caption(
        f"{video_row.get('input_type', '?')} · {int(video_row.get('n_frames') or 0)} frames · "
        f"scene `{video_row.get('scene_id', '?')}` · source `{video_row.get('source', '?')}`"
    )
    st.warning(f"**Why this needs review:** {queue_row.get('review_reason', '')}")

    # --- measurements ------------------------------------------------------ #
    with st.expander("Measurements", expanded=True):
        stats = {
            key: video_row.get(key)
            for key in schema.VIDEO_STAT_COLUMNS
            if key in video_row and pd.notna(video_row.get(key))
        }
        st.dataframe(
            pd.DataFrame([stats]).T.rename(columns={0: "value"}), width="stretch", height=260
        )

    evidence: dict = {}
    raw_evidence = video_row.get("evidence")
    if isinstance(raw_evidence, str) and raw_evidence.strip():
        try:
            evidence = json.loads(raw_evidence)
        except json.JSONDecodeError:
            evidence = {}

    # --- thumbnails -------------------------------------------------------- #
    st.subheader("Frames")
    shown = frame_df
    if show_only_flagged:
        mask = np.zeros(len(frame_df), dtype=bool)
        for column, value, _ in HIGHLIGHTS:
            if column in frame_df.columns:
                mask |= (frame_df[column] == value).to_numpy()
        shown = frame_df[mask]
        if shown.empty:
            st.caption("No frame carries a problem label; showing an even spread instead.")
            shown = frame_df

    wanted = pick_frames(shown)
    paths = capture_paths()
    path = paths.get(capture_id)
    if path is None:
        st.error(f"Cannot find {capture_id} on disk any more; thumbnails unavailable.")
    else:
        thumbnails = load_thumbnails(path, capture_id, wanted)
        columns = st.columns(4)
        for position, idx in enumerate(wanted):
            image = thumbnails.get(idx)
            if image is None:
                continue
            row = frame_df[frame_df["frame_idx"] == idx]
            caption = caption_for(row.iloc[0], is_video) if not row.empty else f"frame {idx}"
            with columns[position % 4]:
                st.image(image, caption=caption, width="stretch")

    # --- the decision ------------------------------------------------------ #
    st.subheader("Labels")
    st.caption("Tick what is really wrong with this capture. The auto label is pre-ticked.")

    with st.form(key=f"review_{capture_id}"):
        decisions: dict[str, int] = {}
        columns = st.columns(2)
        for position, name in enumerate(schema.VIDEO_LABEL_NAMES):
            auto = int(pd.to_numeric(video_row.get(name, 0), errors="coerce") or 0)
            definition = schema.video_label_def(name)
            detail = evidence.get(name)
            help_text = definition.meaning + (f"\n\nAuto evidence: {detail}" if detail else "")
            with columns[position % 2]:
                decisions[name] = int(
                    st.checkbox(
                        f"{schema.LABEL_TITLES.get(name, name)}"
                        + ("  ·  auto: yes" if auto else ""),
                        value=bool(auto),
                        help=help_text,
                        key=f"{capture_id}_{name}",
                    )
                )
        note = st.text_area("Note (optional)", key=f"note_{capture_id}")
        submitted = st.form_submit_button("Save and next", type="primary")

    if submitted:
        if not reviewer.strip():
            st.error("Please put your name in the sidebar first, so labels can be attributed.")
            return
        record = {
            "capture_id": capture_id,
            "scene_id": video_row.get("scene_id", capture_id),
            "input_type": video_row.get("input_type", "video"),
            **decisions,
            "note": note.strip(),
            "reviewer": reviewer.strip(),
            "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
            "label_source": "human",
        }
        dataset.append_human_labels([record], out_dir=Path(labels_dir))

        # Drop it from the queue on disk so another reviewer does not repeat it.
        remaining = queue[queue["capture_id"] != capture_id]
        dataset.write_review_queue(remaining, out_dir=Path(labels_dir))

        changed = [
            name
            for name in schema.VIDEO_LABEL_NAMES
            if decisions[name] != int(pd.to_numeric(video_row.get(name, 0), errors="coerce") or 0)
        ]
        st.cache_data.clear()
        st.success(
            f"Saved {capture_id}."
            + (f" You changed: {', '.join(changed)}." if changed else " Auto labels confirmed.")
        )
        st.rerun()


main()
