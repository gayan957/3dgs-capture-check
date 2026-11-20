import type { Health, Job, LabelDef } from "./types";

/** The API lives behind /api, proxied in development and same-origin in production. */
const BASE = "/api";

async function unwrap<T>(response: Response): Promise<T> {
  if (response.ok) return (await response.json()) as T;
  // FastAPI puts the readable reason in `detail`; fall back to the status.
  let detail = `${response.status} ${response.statusText}`;
  try {
    const body = await response.json();
    if (body?.detail) detail = String(body.detail);
  } catch {
    /* not JSON: keep the status line */
  }
  throw new Error(detail);
}

export async function getHealth(): Promise<Health> {
  return unwrap<Health>(await fetch(`${BASE}/health`));
}

export async function getLabels(): Promise<LabelDef[]> {
  return unwrap<LabelDef[]>(await fetch(`${BASE}/labels`));
}

export async function uploadCapture(file: File, signal?: AbortSignal): Promise<Job> {
  const body = new FormData();
  body.append("file", file);
  return unwrap<Job>(await fetch(`${BASE}/captures`, { method: "POST", body, signal }));
}

export async function getJob(jobId: string): Promise<Job> {
  return unwrap<Job>(await fetch(`${BASE}/captures/${jobId}`));
}

/**
 * Poll a job until it finishes.
 *
 * Checking a capture is allowed a minute, so this is a long poll by design.
 * `onTick` is called with every update so the page can show the stage rather
 * than an unexplained spinner.
 */
export async function followJob(
  jobId: string,
  onTick: (job: Job) => void,
  intervalMs = 900,
): Promise<Job> {
  for (;;) {
    const job = await getJob(jobId);
    onTick(job);
    if (job.status === "done" || job.status === "failed") return job;
    await new Promise((resolve) => setTimeout(resolve, intervalMs));
  }
}
