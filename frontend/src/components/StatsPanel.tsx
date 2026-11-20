import type { CaptureResult } from "../types";

type Kind = "percent" | "degrees" | "count" | "number";

interface Spec {
  key: string;
  label: string;
  help: string;
  kind: Kind;
  /** Higher is better, so a low value is the thing to worry about. */
  higherIsBetter: boolean;
}

/** The measurements that decide whether a capture reconstructs, in order. */
const HEADLINE: Spec[] = [
  {
    key: "walked_ratio",
    label: "Walked",
    help: "Share of frame pairs where the camera genuinely moved rather than turned on the spot. The single most important number here.",
    kind: "percent",
    higherIsBetter: true,
  },
  {
    key: "azimuth_span_deg",
    label: "Angle covered",
    help: "How far around the subject the camera travelled. 360 degrees means every side was filmed.",
    kind: "degrees",
    higherIsBetter: true,
  },
  {
    key: "overlap_mean",
    label: "Overlap",
    help: "How much detail neighbouring frames share on average. Too little and the frames cannot be linked.",
    kind: "percent",
    higherIsBetter: true,
  },
  {
    key: "blur_ratio",
    label: "Blurry frames",
    help: "Share of frames too soft to give reliable reference points.",
    kind: "percent",
    higherIsBetter: false,
  },
  {
    key: "diversity",
    label: "View variety",
    help: "How different the views are from one another. Low variety means you did not move around much.",
    kind: "number",
    higherIsBetter: true,
  },
  {
    key: "unique_views",
    label: "Distinct views",
    help: "Frames that are genuinely different from the others, after duplicates are discounted.",
    kind: "count",
    higherIsBetter: true,
  },
  {
    key: "coverage_empty_bins",
    label: "Unfilmed sectors",
    help: "Of 24 slices around the subject, how many were never filmed from.",
    kind: "count",
    higherIsBetter: false,
  },
  {
    key: "low_texture_ratio",
    label: "Blank frames",
    help: "Share of frames with too little texture to grab onto, such as plain walls.",
    kind: "percent",
    higherIsBetter: false,
  },
];

function format(value: number | null, kind: Kind): string {
  if (value === null || Number.isNaN(value)) return "unknown";
  switch (kind) {
    case "percent":
      return `${(value * 100).toFixed(0)}%`;
    case "degrees":
      return `${value.toFixed(0)}°`;
    case "count":
      return value.toFixed(0);
    default:
      return value.toFixed(2);
  }
}

interface Props {
  result: CaptureResult;
}

export function StatsPanel({ result }: Props) {
  const rest = Object.entries(result.stats).filter(
    ([k]) => !HEADLINE.some((s) => s.key === k),
  );

  return (
    <section className="card">
      <h2>The measurements</h2>
      <p className="subnote">
        These are measured, not predicted. They hold whatever the model says.
      </p>

      <div className="stat-grid">
        {HEADLINE.map((spec) => {
          const value = result.stats[spec.key] ?? null;
          const unknown = value === null || Number.isNaN(value);
          return (
            <div className="stat" key={spec.key} title={spec.help}>
              <div className="stat-label">{spec.label}</div>
              <div className={`stat-value${unknown ? " unknown" : ""}`}>
                {format(value, spec.kind)}
              </div>
              <div className="stat-help">{spec.help}</div>
            </div>
          );
        })}
      </div>

      {rest.length > 0 && (
        <details className="all-stats">
          <summary>Every measurement ({rest.length} more)</summary>
          <table>
            <tbody>
              {rest.map(([key, value]) => (
                <tr key={key}>
                  <th scope="row">{key.replace(/_/g, " ")}</th>
                  <td>
                    {value === null || Number.isNaN(value)
                      ? "unknown"
                      : Number(value).toFixed(3)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </details>
      )}
    </section>
  );
}
