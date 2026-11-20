import type { CaptureResult } from "../types";

type Kind = "percent" | "degrees" | "count" | "number";

interface Spec {
  key: string;
  label: string;
  /** Shown on hover, not on screen: the label carries the meaning. */
  help: string;
  kind: Kind;
}

/** The measurements that decide whether a capture reconstructs, in order. */
const HEADLINE: Spec[] = [
  {
    key: "walked_ratio",
    label: "Walked",
    help: "How often the camera really moved instead of turning on the spot. The most important number here.",
    kind: "percent",
  },
  {
    key: "azimuth_span_deg",
    label: "Angle covered",
    help: "How far around the subject you got. 360° means every side.",
    kind: "degrees",
  },
  {
    key: "overlap_mean",
    label: "Overlap",
    help: "How much neighbouring frames share. Too little and they cannot be linked.",
    kind: "percent",
  },
  {
    key: "blur_ratio",
    label: "Blurry frames",
    help: "Share of frames too soft to be useful.",
    kind: "percent",
  },
  {
    key: "diversity",
    label: "View variety",
    help: "How different the views are from one another.",
    kind: "number",
  },
  {
    key: "unique_views",
    label: "Distinct views",
    help: "Frames that are genuinely different, after duplicates are discounted.",
    kind: "count",
  },
  {
    key: "coverage_empty_bins",
    label: "Unfilmed sides",
    help: "Of 24 directions around the subject, how many you never filmed from.",
    kind: "count",
  },
  {
    key: "low_texture_ratio",
    label: "Blank frames",
    help: "Share of frames with too little detail, such as plain walls.",
    kind: "percent",
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
