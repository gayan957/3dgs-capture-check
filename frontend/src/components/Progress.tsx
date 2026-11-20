import type { Job } from "../types";

interface Props {
  job: Job;
  budgetS: number;
}

/**
 * What the server is doing, and for how long.
 *
 * The bar is deliberately indeterminate. The pipeline reports which step it is
 * on but not how far through it is, and inventing a percentage would be a
 * guess dressed up as information.
 */
export function Progress({ job, budgetS }: Props) {
  const over = job.elapsed_s > budgetS;
  return (
    <section className="card progress-card" aria-live="polite">
      <div className="progress-head">
        <span className="filename">{job.filename}</span>
        <span className={`elapsed${over ? " over" : ""}`}>
          {job.elapsed_s.toFixed(0)}s <span className="muted">/ {budgetS}s</span>
        </span>
      </div>
      <div className="bar" role="progressbar" aria-label={job.stage}>
        <div className="bar-fill" />
      </div>
      <p className="stage">{job.stage}&hellip;</p>
    </section>
  );
}
