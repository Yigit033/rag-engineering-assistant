/**
 * Server-Sent Events over POST.
 *
 * WHY NOT `EventSource`:
 *   The browser's built-in `EventSource` only issues GET requests and cannot
 *   send a body. Our question can be long and must go in a POST body, so we
 *   read the response stream manually with `fetch` + `ReadableStream`.
 *
 * WHY THE BUFFER LOOP MATTERS:
 *   A network chunk has nothing to do with an SSE event boundary. One read
 *   may deliver half an event, or three events at once. Parsing each chunk
 *   as if it were a complete event produces truncated or merged text —
 *   a bug that only appears under real network conditions and is nearly
 *   impossible to reproduce locally. We buffer and split on the blank line
 *   that terminates an SSE event.
 */

import { API_BASE, ApiError } from "./api";

const DONE_MARKER = "[DONE]";
const ERROR_PREFIX = "[HATA]";

export interface StreamHandlers {
  onToken: (text: string) => void;
  onDone?: () => void;
  onError?: (error: Error) => void;
}

/**
 * Stream an answer token by token.
 *
 * Returns an abort function. Cancelling matters: if the user navigates away
 * or asks a new question, the previous stream must stop — otherwise two
 * answers interleave into the same view.
 */
export function streamAnswer(
  question: string,
  handlers: StreamHandlers,
  topK?: number,
): () => void {
  const controller = new AbortController();

  void (async () => {
    try {
      const response = await fetch(`${API_BASE}/ask/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, top_k: topK ?? null }),
        signal: controller.signal,
      });

      if (!response.ok) {
        const body = (await response.json().catch(() => null)) as {
          error?: string;
          detail?: string;
        } | null;
        throw new ApiError(
          response.status,
          body?.error ?? `http_${response.status}`,
          body?.detail ?? response.statusText,
          response.headers.get("x-request-id") ?? undefined,
        );
      }
      if (!response.body) throw new Error("The response carried no stream.");

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;

        // `stream: true` keeps multi-byte characters intact across chunk
        // boundaries. Without it a Turkish "ğ" split across two chunks
        // decodes into replacement characters.
        buffer += decoder.decode(value, { stream: true });

        // An SSE event ends with a blank line. Everything after the last
        // blank line is incomplete and stays in the buffer.
        const events = buffer.split("\n\n");
        buffer = events.pop() ?? "";

        for (const event of events) {
          const line = event.split("\n").find((l) => l.startsWith("data: "));
          if (!line) continue;

          const payload = line.slice(6);
          if (payload === DONE_MARKER) {
            handlers.onDone?.();
            return;
          }
          if (payload.startsWith(ERROR_PREFIX)) {
            handlers.onError?.(new Error(payload.slice(ERROR_PREFIX.length).trim()));
            return;
          }
          // The server escapes newlines so they cannot break event framing;
          // restore them for display.
          handlers.onToken(payload.replace(/\\n/g, "\n"));
        }
      }
      handlers.onDone?.();
    } catch (error) {
      // An abort is a deliberate user action, not a failure. Reporting it
      // as an error would flash a false alarm every time someone retypes.
      if (error instanceof DOMException && error.name === "AbortError") return;
      handlers.onError?.(
        error instanceof Error ? error : new Error("Streaming failed."),
      );
    }
  })();

  return () => controller.abort();
}
