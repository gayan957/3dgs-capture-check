import { useCallback, useEffect, useState } from "react";
import { followJob, getHealth, getLabels, uploadCapture } from "./api";
import { ProblemList } from "./components/ProblemList";
import { Progress } from "./components/Progress";
import { ScoreCard } from "./components/ScoreCard";
import { StatsPanel } from "./components/StatsPanel";
import { ThemeToggle } from "./components/ThemeToggle";
import { Uploader } from "./components/Uploader";
import type { Health, Job, LabelDef } from "./types";

type Phase =
  | { name: "idle" }
  | { name: "uploading"; filename: string }
  | { name: "working"; job: Job }
  | { name: "done"; job: Job }
  | { name: "error"; message: string };

export function App() {
  const [health, setHealth] = useState<Health | null>(null);
  const [labels, setLabels] = useState<LabelDef[]>([]);
  const [boot, setBoot] = useState<string | null>(null);
  const [phase, setPhase] = useState<Phase>({ name: "idle" });

  useEffect(() => {
    Promise.all([getHealth(), getLabels()])
      .then(([h, l]) => {
        setHealth(h);
        setLabels(l);
      })
      .catch((e: Error) =>
        setBoot(
          `Cannot reach the CaptureCheck server (${e.message}). Start it with \`capturecheck serve\`.`,
        ),
      );
  }, []);

  const check = useCallback(async (file: File) => {
    setPhase({ name: "uploading", filename: file.name });
    try {
      const started = await uploadCapture(file);
      setPhase({ name: "working", job: started });
      const finished = await followJob(started.job_id, (job) =>
        setPhase({ name: "working", job }),
      );
      if (finished.status === "failed") {
        setPhase({
          name: "error",
          message: finished.error ?? "The capture could not be checked.",
        });
      } else {
        setPhase({ name: "done", job: finished });
      }
    } catch (e) {
      setPhase({ name: "error", message: (e as Error).message });
    }
  }, []);

  return (
    <div className="page">
      <header className="masthead">
        <div className="brand">
          <h1>CaptureCheck</h1>
          <p className="tagline">Will your footage make a good 3D Gaussian Splat?</p>
          <p className="lede">
            CaptureCheck evaluates video quality for 3D Gaussian Splatting by checking
            camera movement, scene coverage, and frame overlap. It provides a quality
            score and identifies what needs to be re-shot before training, saving time
            and effort.
          </p>
        </div>
        <div className="masthead-right">
          {/* Only worth saying when it is not true: a named run id means nothing
              to a visitor, but "no model" changes how to read the page. */}
          {health && !health.can_score && (
            <div className="server-state">
              <span className="dot warn" aria-hidden="true" />
              <span>no trained model: rules only</span>
            </div>
          )}
          <ThemeToggle />
        </div>
      </header>

      {boot && (
        <div className="card error" role="alert">
          <h2>Server not running</h2>
          <p>{boot}</p>
        </div>
      )}

      {health && (
        <>
          {phase.name === "idle" && (
            <>
              <Uploader
                accepts={health.accepts}
                maxUploadMb={health.max_upload_mb}
                disabled={false}
                onPick={check}
              />
              <WhatItLooksFor />
            </>
          )}

          {phase.name === "uploading" && (
            <section className="card progress-card">
              <div className="progress-head">
                <span className="filename">{phase.filename}</span>
              </div>
              <div className="bar">
                <div className="bar-fill" />
              </div>
              <p className="stage">uploading&hellip;</p>
            </section>
          )}

          {phase.name === "working" && (
            <Progress job={phase.job} budgetS={health.budget_s} />
          )}

          {phase.name === "error" && (
            <div className="card error" role="alert">
              <h2>Could not check that</h2>
              <p>{phase.message}</p>
              <button className="again" onClick={() => setPhase({ name: "idle" })}>
                Try another capture
              </button>
            </div>
          )}

          {phase.name === "done" && phase.job.result && (
            <>
              <div className="result-head">
                <h2 className="result-name">{phase.job.result.capture_id}</h2>
                <button className="again" onClick={() => setPhase({ name: "idle" })}>
                  Check another
                </button>
              </div>
              <ScoreCard result={phase.job.result} />
              <ProblemList problems={phase.job.result.problems} labels={labels} />
              <StatsPanel result={phase.job.result} />
              {phase.job.result.review && (
                <div className="card review-note">
                  {/* The reason names raw measurements and thresholds, which is
                      for whoever tunes them, not for a visitor reading a score. */}
                  <details>
                    <summary>Worth a human look</summary>
                    <p className="review-reason">{phase.job.result.review_reason}</p>
                  </details>
                </div>
              )}
            </>
          )}
        </>
      )}

      <footer className="foot">
        {health && (
          <p className="muted">
            CaptureCheck {health.version} &middot; <a href="/docs">API</a>
          </p>
        )}
      </footer>
    </div>
  );
}

/** Four short lines, so a first-time visitor knows what is being judged. */
function WhatItLooksFor() {
  return (
    <section className="card primer">
      <h2>What it checks</h2>
      <ul className="primer-list">
        <li>Did you walk around, or just turn on the spot?</li>
        <li>Did you film every side?</li>
        <li>Do neighbouring frames overlap enough?</li>
        <li>Are the frames sharp, lit evenly, and detailed enough to match?</li>
      </ul>
    </section>
  );
}
