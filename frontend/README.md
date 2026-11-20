# CaptureCheck — web interface

Upload a capture in a browser, watch it being checked, and read the score with
the evidence behind it.

React 18 + TypeScript + Vite. No UI framework and no state library: the whole
app is five components and one `fetch` wrapper, and the thing it is wrapped
around is a three-endpoint API.

## Running it

The front end is useless on its own — it needs the Python API for everything.

```bash
# once
npm install

# development: this on :5173, the API on :8000, /api proxied across
npm run dev
capturecheck serve        # in another terminal

# production: build, then one server on one port
npm run build
capturecheck serve        # serves dist/ and the API together
```

`vite.config.ts` proxies `/api` to `http://127.0.0.1:8000`. Point it elsewhere
with `CAPTURECHECK_API=http://host:port npm run dev`.

## How it works

```
Uploader  ──POST /api/captures──▶  job id
                                     │
App  ──GET /api/captures/{id}────────┘   every 0.9s until done
     │
     └─▶ ScoreCard · ProblemList · StatsPanel
```

Checking a capture is allowed 60 seconds, which is far too long to hold an HTTP
request open, so an upload returns a job and the page polls it. `followJob` in
`src/api.ts` does the polling and reports every update, which is what lets the
progress card name the step instead of showing an unexplained spinner.

The progress bar is deliberately indeterminate. The server says which step it
is on but not how far through it is, and a made-up percentage is a guess
dressed as information.

## What the browser does not decide

The score bands, the label names and their meanings, the accepted file types
and the upload ceiling all come from `/api/health` and `/api/labels`. None of
them are written in the TypeScript.

That is on purpose. `capturecheck/labeling/schema.py` is the one place label
names live, and a copy of them here would be a second place — silently wrong
the first time a threshold moves. The only duplicated numbers are the colour
cut-offs in `ScoreCard.tsx`, which decide which shade to draw and are checked
against the API in `tests/test_api.py`.

## Layout

```
src/
├── App.tsx               the page and its four states: idle, uploading, working, done
├── api.ts                fetch wrappers and the polling loop
├── types.ts              the shapes the API sends
├── styles.css            one stylesheet, light and dark
└── components/
    ├── Uploader.tsx      drag and drop, with the checks done before uploading
    ├── Progress.tsx      which step, and how long it has taken
    ├── ScoreCard.tsx     the score, its band, and how much the ensemble disagreed
    ├── ProblemList.tsx   what was found, what is suspected, what passed
    └── StatsPanel.tsx    the measurements, in plain English
```

## A note on what is shown

The page keeps two things apart on purpose:

- **What was found** — the rules fired, and there is evidence pointing at a
  frame or a timestamp. Measured, not predicted.
- **What the model suspects** — the measurements stayed inside their
  thresholds but the model disagrees. No evidence to point at.

A model hunch reading as a fact would be the easiest way to make this interface
misleading, so they never share a list.

The same goes for the score: `ScoreCard` shows how far apart the five ensemble
members were, and says so plainly when they disagree by more than 0.3.
