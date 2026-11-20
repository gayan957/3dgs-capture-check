import type { CaptureResult } from "../types";

/** Which visual band a score falls in. Mirrors capturecheck/presentation.py. */
function bandKey(score: number): string {
  if (score >= 0.75) return "good";
  if (score >= 0.55) return "fair";
  if (score >= 0.35) return "poor";
  return "bad";
}

interface Props {
  result: CaptureResult;
}

/** The headline: one number, what it means, and how sure the ensemble was. */
export function ScoreCard({ result }: Props) {
  if (result.score === null) {
    return (
      <section className="card score-card">
        <h2 className="score-none">No score</h2>
        <p className="muted">
          No model trained yet &mdash; the measurements below still apply.
        </p>
      </section>
    );
  }

  const score = result.score;
  const key = bandKey(score);
  const pct = Math.max(0, Math.min(1, score)) * 100;

  // How much the five ensemble members disagreed. A wide spread on a small
  // model is worth showing rather than hiding behind the average.
  const seeds = result.score_per_seed ?? [];
  const spread =
    seeds.length > 1 ? Math.max(...seeds) - Math.min(...seeds) : null;

  return (
    <section className={`card score-card band-${key}`}>
      <div className="score-main">
        <div className="score-number">
          <span className="score-value">{score.toFixed(2)}</span>
          <span className="score-band">{result.band}</span>
        </div>
        <div className="score-meter" aria-hidden="true">
          <div className="meter-track">
            <div className="meter-fill" style={{ width: `${pct}%` }} />
          </div>
          <div className="meter-ticks">
            <span>0</span>
            <span>unusable</span>
            <span>poor</span>
            <span>fair</span>
            <span>good</span>
            <span>1</span>
          </div>
        </div>
      </div>

      <dl className="score-facts">
        <div>
          <dt>Capture</dt>
          <dd>
            {result.n_frames} frames &middot; {result.input_type}
          </dd>
        </div>
        <div>
          <dt>Checked in</dt>
          <dd>{result.time_s === null ? "—" : `${result.time_s.toFixed(1)}s`}</dd>
        </div>
        <div>
          <dt>Model</dt>
          <dd>{result.model ?? "—"}</dd>
        </div>
        {spread !== null && (
          <div title={`How much the ${seeds.length} models disagreed`}>
            <dt>Agreement</dt>
            <dd className={spread > 0.3 ? "warn" : undefined}>
              {spread > 0.3 ? "mixed" : "close"}
            </dd>
          </div>
        )}
      </dl>
    </section>
  );
}
