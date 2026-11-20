// The shapes the API sends. Kept in step with capturecheck/api/app.py.

export interface Band {
  at_least: number;
  name: string;
}

export interface Health {
  ok: boolean;
  version: string;
  model: string | null;
  can_score: boolean;
  budget_s: number;
  max_upload_mb: number;
  accepts: string[];
  bands: Band[];
}

export interface Problem {
  name: string;
  title: string;
  flagged: boolean;
  probability: number | null;
  evidence: string | null;
  /** The rules found nothing, but the model is not convinced. */
  suspected: boolean;
}

export interface CaptureResult {
  capture_id: string;
  input_type: string;
  n_frames: number;
  score: number | null;
  band: string | null;
  score_per_seed: number[];
  model: string | null;
  backend: string | null;
  problems: Problem[];
  stats: Record<string, number | null>;
  review: boolean;
  review_reason: string;
  time_s: number | null;
  budget_s: number;
}

export type JobStatus = "queued" | "running" | "done" | "failed";

export interface Job {
  job_id: string;
  filename: string;
  status: JobStatus;
  stage: string;
  elapsed_s: number;
  queued_s: number;
  result: CaptureResult | null;
  error: string | null;
}

export interface LabelDef {
  name: string;
  title: string;
  meaning: string;
}
