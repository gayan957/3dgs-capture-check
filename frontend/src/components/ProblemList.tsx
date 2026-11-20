import type { LabelDef, Problem } from "../types";

interface Props {
  problems: Problem[];
  labels: LabelDef[];
}

/**
 * What is wrong, with the evidence.
 *
 * Found and suspected stay in separate lists: the rules can point at a frame,
 * the model cannot, and a hunch reading as a fact is the easiest way to make
 * this page misleading. What each problem means is on hover, not on screen --
 * the evidence line already says what happened.
 */
export function ProblemList({ problems, labels }: Props) {
  const meaning = new Map(labels.map((l) => [l.name, l.meaning]));
  const found = problems.filter((p) => p.flagged);
  const suspected = problems.filter((p) => p.suspected);
  const clear = problems.filter((p) => !p.flagged && !p.suspected);

  return (
    <section className="card">
      <h2>What was found</h2>

      {found.length === 0 ? (
        <p className="all-clear">Nothing wrong found.</p>
      ) : (
        <ul className="problems">
          {found.map((p) => (
            <li key={p.name} className="problem found" title={meaning.get(p.name)}>
              {p.evidence ? (
                // Collapsed by default: the name is the headline, the evidence
                // is the detail you open when you want to act on it.
                <details>
                  <summary>
                    <span className="problem-title">{p.title}</span>
                    {p.probability !== null && (
                      <span className="prob">{p.probability.toFixed(2)}</span>
                    )}
                  </summary>
                  <p className="evidence">{p.evidence}</p>
                </details>
              ) : (
                // Nothing to reveal, so no arrow promising there is.
                <div className="problem-head no-detail">
                  <span className="problem-title">{p.title}</span>
                  {p.probability !== null && (
                    <span className="prob">{p.probability.toFixed(2)}</span>
                  )}
                </div>
              )}
            </li>
          ))}
        </ul>
      )}

      {suspected.length > 0 && (
        <>
          <h3 className="subhead">Possibly also</h3>
          <ul className="problems">
            {suspected.map((p) => (
              <li key={p.name} className="problem suspected" title={meaning.get(p.name)}>
                <div className="problem-head">
                  <span className="problem-title">{p.title}</span>
                  <span className="prob">{p.probability?.toFixed(2)}</span>
                </div>
              </li>
            ))}
          </ul>
        </>
      )}

      {clear.length > 0 && (
        <details className="clear-list">
          <summary>{clear.length} checks passed</summary>
          <ul>
            {clear.map((p) => (
              <li key={p.name} title={meaning.get(p.name)}>
                <span className="tick" aria-hidden="true">
                  &#10003;
                </span>{" "}
                {p.title}
              </li>
            ))}
          </ul>
        </details>
      )}
    </section>
  );
}
