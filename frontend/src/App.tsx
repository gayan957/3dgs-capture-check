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

  const busy = phase.name === "uploading" || phase.name === "working";

  return (
    <div className="page">
      <header className="masthead">
        <div className="brand">
          <h1>CaptureCheck</h1>
          <p className="tagline">
            Find out whether your footage will reconstruct in 3D, before you spend hours
            finding out the hard way.
          </p>
        </div>
        <div className="masthead-right">
          {health && (
            <div className="server-state">
              <span className={`dot${health.can_score ? " live" : " warn"}`} aria-hidden="true" />
              <span>
                {health.can_score ? (
                  <>
                    model <code>{health.model}</code>
                  </>
                ) : (
                  "no trained model: rules only"
                )}
              </span>
            </div>
          )}
          <ThemeToggle />
        </div>
      </header>

      {boot && (
        <div className="card error" role="alert">
          <h2>Server unavailable</h2>
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
              <h2>That did not work</h2>
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
                  <strong>Worth a human look.</strong> {phase.job.result.review_reason}
                </div>
              )}
            </>
          )}
        </>
      )}

      {busy && (
        <p className="busy-note">
          Leave this page open. The check runs on the server and the page updates itself.
        </p>
      )}

      <footer className="foot">
        <p>
          The score is predicted. The measurements and the evidence are not &mdash; they are
          read off your footage.
        </p>
        {health && (
          <p className="muted">
            CaptureCheck {health.version} &middot; Apache 2.0 &middot;{" "}
            <a href="/docs">API docs</a>
          </p>
        )}
      </footer>
    </div>
  );
}

/** A short primer, shown before anything is uploaded. */
function WhatItLooksFor() {
  return (
    <section className="card primer">
      <h2>What it looks for</h2>
      <p>
        3D reconstruction does not care whether your video looks nice. It cares whether the
        geometry can be solved &mdash; whether the camera moved enough for the software to
        work out where each frame was taken from.
      </p>
      <ul className="primer-list">
        <li>
          <strong>Did you walk, or turn on the spot?</strong> A sharp 4K pan from a tripod
          fails completely. A shaky phone video where you walked around works fine.
        </li>
        <li>
          <strong>Did you film every side?</strong> Whatever you never pointed the camera at
          becomes a hole, or invented geometry.
        </li>
        <li>
          <strong>Do neighbouring frames share detail?</strong> Move too fast and the
          software cannot link one frame to the next.
        </li>
        <li>
          <strong>Is there anything to grab onto?</strong> Blank walls, mirrors and glass
          all give the software nothing reliable to anchor to.
        </li>
      </ul>
    </section>
  );
}
