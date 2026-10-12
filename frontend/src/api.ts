export const API = import.meta.env.VITE_API_BASE_URL || "";

export type Camera = {
  id: string;
  name: string;
  source_type: "webcam" | "rtsp" | "video";
  source_label: string;
  status: "disconnected" | "connecting" | "connected" | "reconnecting" | "degraded" | "error" | string;
  active: boolean;
  last_frame_at: string | null;
  last_error: string | null;
  inference_ms: number | null;
  fps: number | null;
  frame_number: number;
};

export type Incident = {
  event_id: string;
  timestamp: string;
  event_class: string;
  confidence: number;
  camera_name: string;
  severity: string;
  status: string;
  acknowledged: number;
  source: string;
};

export type Page<T> = { items: T[]; page: number; page_size: number; total: number; has_more: boolean };
export type SystemStatus = {
  process: string;
  model: "loading" | "ready" | "error" | string;
  model_error: string | null;
  uptime_seconds: number;
  active_sources: number;
  max_active_sources: number;
  storage: { database_available: boolean; evidence_available: boolean };
};
export type Analytics = {
  total_events: number;
  by_class: Record<string, number>;
  by_severity: Record<string, number>;
  acknowledged: number;
  resolved: number;
  days: number;
  timezone: string;
  timeseries: { date: string; counts: Record<string, number> }[];
  cameras: Camera[];
};
export type Settings = { confidence_threshold: number; confirmation_frames: number; reconnect_attempts: number; max_active_sources: number };
export type ImageAnalysis = { timestamp: string; speed_ms: Record<string, number>; alert_state: string; detections: { class_name: string; confidence: number; xyxy: number[] }[]; confirmed_events: { event_id: string; event_class: string; confidence: number; timestamp: string }[]; annotated_jpeg_base64: string };

export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API}${path}`, {
      ...init,
      headers: { ...(init?.body ? { "Content-Type": "application/json" } : {}), ...init?.headers },
    });
  } catch {
    throw new Error("FireSight API is unreachable. Start the local backend and retry.");
  }
  if (!response.ok) {
    const body = await response.json().catch(() => null) as { detail?: unknown } | null;
    const detail = typeof body?.detail === "string" ? body.detail : `Request failed (${response.status})`;
    throw new Error(detail);
  }
  if (response.status === 204) return undefined as T;
  return response.json() as Promise<T>;
}

export async function apiUpload<T>(path: string, file: File): Promise<T> {
  const response = await fetch(`${API}${path}`, {
    method: "POST",
    headers: { "Content-Type": file.type || "application/octet-stream", "X-File-Name": encodeURIComponent(file.name) },
    body: file,
  }).catch(() => { throw new Error("FireSight API is unreachable. Start the local backend and retry."); });
  if (!response.ok) {
    const body = await response.json().catch(() => null) as { detail?: unknown } | null;
    throw new Error(typeof body?.detail === "string" ? body.detail : `Upload failed (${response.status})`);
  }
  return response.json() as Promise<T>;
}

export const apiJson = (method: string, body?: unknown): RequestInit => ({
  method,
  ...(body === undefined ? {} : { body: JSON.stringify(body) }),
});

export const formatTime = (value: string | null | undefined) => {
  if (!value) return "—";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString();
};
