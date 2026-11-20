import type { LabelDef, Problem } from "../types";

interface Props {
  problems: Problem[];
  labels: LabelDef[];
}

/**
 * What is wrong, with the evidence.
 *
 * Three groups, because they carry different weight: the rules fired and can
 * point at a frame; the model suspects something the rules did not catch; or
 * nothing was found. Keeping them apart stops a model hunch reading as a fact.
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
        <p className="all-clear">No problems found by the rules.</p>
      ) : (
        <ul className="problems">
          {found.map((p) => (
            <li key={p.name} className="problem found">
              <div className="problem-head">
                <span className="problem-title">{p.title}</span>
                {p.probability !== null && (
                  <span className="prob" title="What the model thinks">
                    model {p.probability.toFixed(2)}
                  </span>
                )}
              </div>
              {p.evidence && <p className="evidence">{p.evidence}</p>}
              <p className="meaning">{meaning.get(p.name)}</p>
            </li>
          ))}
        </ul>
      )}

      {suspected.length > 0 && (
        <>
          <h3 className="subhead">The model also suspects</h3>
          <p className="subnote">
            The measurements stayed inside their thresholds, but the model has seen
            something like this before. No evidence to point at.
          </p>
          <ul className="problems">
            {suspected.map((p) => (
              <li key={p.name} className="problem suspected">
                <div className="problem-head">
                  <span className="problem-title">{p.title}</span>
                  <span className="prob">model {p.probability?.toFixed(2)}</span>
                </div>
                <p className="meaning">{meaning.get(p.name)}</p>
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
              <li key={p.name}>
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
