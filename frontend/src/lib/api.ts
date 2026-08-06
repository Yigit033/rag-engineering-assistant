/**
 * Typed API client.
 *
 * WHY THE TYPES ARE NOT HAND-WRITTEN:
 *   `api-schema.d.ts` is generated from the backend's OpenAPI document
 *   (`npm run generate:api`). If a field is renamed on the server, the
 *   frontend stops compiling — the mismatch is caught at build time
 *   instead of surfacing as `undefined` in production.
 *
 *   This is the concrete payoff of designing the API first. Hand-written
 *   interfaces drift silently; generated ones cannot.
 */

import type { components } from "./api-schema";

// ---------------------------------------------------------------------------
// Types re-exported from the generated schema — single source of truth.
// ---------------------------------------------------------------------------
export type AskResponse = components["schemas"]["AskResponse"];
export type Citation = components["schemas"]["CitationOut"];
export type Source = components["schemas"]["SourceOut"];
export type DocumentInfo = components["schemas"]["DocumentOut"];
export type DocumentList = components["schemas"]["DocumentListResponse"];
export type UploadResult = components["schemas"]["UploadResponse"];
export type DeleteResult = components["schemas"]["DeleteResponse"];
export type SystemHealth = components["schemas"]["HealthResponse"];
export type ApiErrorBody = components["schemas"]["ErrorResponse"];

export const API_BASE =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") ?? "http://127.0.0.1:8001";

/**
 * A failed API call.
 *
 * The backend returns a uniform error body (`{error, detail, request_id}`).
 * We keep the machine-readable `code` separate from the human-readable
 * `message` so the UI can branch on the code without parsing prose —
 * parsing error strings is how UIs break on the next copy edit.
 */
export class ApiError extends Error {
  constructor(
    readonly status: number,
    readonly code: string,
    message: string,
    readonly requestId?: string,
  ) {
    super(message);
    this.name = "ApiError";
  }

  /** The server is temporarily unable to serve — retrying may work. */
  get isTransient(): boolean {
    return this.status === 503 || this.status === 502;
  }

  /** A precondition is missing (e.g. empty index) — the user must act. */
  get isPrecondition(): boolean {
    return this.status === 409;
  }
}

async function toApiError(response: Response): Promise<ApiError> {
  const requestId = response.headers.get("x-request-id") ?? undefined;
  try {
    const body = (await response.json()) as Partial<ApiErrorBody> & {
      detail?: unknown;
    };
    // FastAPI validation errors use `detail` as an array of issues; our own
    // handlers use `{error, detail}`. Normalise both into one shape.
    const detail =
      typeof body.detail === "string"
        ? body.detail
        : Array.isArray(body.detail)
          ? "The request was rejected as invalid."
          : response.statusText;
    return new ApiError(
      response.status,
      body.error ?? `http_${response.status}`,
      detail,
      requestId,
    );
  } catch {
    return new ApiError(
      response.status,
      `http_${response.status}`,
      response.statusText || "Unexpected error",
      requestId,
    );
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}${path}`, init);
  } catch {
    // Network-level failure: the API is not reachable at all. This is a
    // different problem from "the API returned an error" and deserves a
    // different message — otherwise users chase the wrong cause.
    throw new ApiError(
      0,
      "network_unreachable",
      `Cannot reach the API at ${API_BASE}. Is the server running?`,
    );
  }
  if (!response.ok) throw await toApiError(response);
  return (await response.json()) as T;
}

// ---------------------------------------------------------------------------
export const api = {
  health: () => request<SystemHealth>("/ready"),

  ask: (question: string, topK?: number, signal?: AbortSignal) =>
    request<AskResponse>("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, top_k: topK ?? null }),
      signal,
    }),

  documents: () => request<DocumentList>("/documents"),

  upload: (file: File, options: { index: boolean; overwrite: boolean }) => {
    const form = new FormData();
    form.append("file", file);
    form.append("index", String(options.index));
    form.append("overwrite", String(options.overwrite));
    // No Content-Type header: the browser must set the multipart boundary.
    return request<UploadResult>("/documents", { method: "POST", body: form });
  },

  remove: (fileName: string) =>
    request<DeleteResult>(`/documents/${encodeURIComponent(fileName)}`, {
      method: "DELETE",
    }),
};
