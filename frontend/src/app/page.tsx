"use client";

import { useQuery } from "@tanstack/react-query";
import { useCallback, useEffect, useRef, useState } from "react";
import Link from "next/link";

import { AnswerCard, SourceList } from "@/components/AnswerCard";
import { ApiError, api, type AskResponse } from "@/lib/api";
import { streamAnswer } from "@/lib/stream";

type Phase = "idle" | "streaming" | "resolving" | "done" | "error";

export default function AskPage() {
  const [question, setQuestion] = useState("");
  const [phase, setPhase] = useState<Phase>("idle");
  const [streamed, setStreamed] = useState("");
  const [answer, setAnswer] = useState<AskResponse | null>(null);
  const [error, setError] = useState<ApiError | Error | null>(null);
  const [focused, setFocused] = useState<number | null>(null);

  const abortRef = useRef<(() => void) | null>(null);

  const { data: docs } = useQuery({
    queryKey: ["documents"],
    queryFn: api.documents,
  });

  // Cancel any in-flight stream when the page unmounts — otherwise the
  // request keeps running and tries to update a component that is gone.
  useEffect(() => () => abortRef.current?.(), []);

  const reset = () => {
    abortRef.current?.();
    setStreamed("");
    setAnswer(null);
    setError(null);
    setFocused(null);
  };

  const submit = useCallback(
    (event: React.FormEvent) => {
      event.preventDefault();
      const trimmed = question.trim();
      if (trimmed.length < 3 || phase === "streaming") return;

      reset();
      setPhase("streaming");

      /*
       * TWO-PHASE ANSWERING — a deliberate trade-off.
       *
       * The stream gives immediate feedback (first token in ~1s instead of
       * a blank screen for 5s). But citations can only be verified once the
       * full text exists, and the streaming endpoint cannot return the
       * structured citation payload mid-flight.
       *
       * So: stream for perceived speed, then fetch the verified answer to
       * get citations, sources and grounding. The user sees text early AND
       * ends up with a checkable result — rather than one or the other.
       */
      abortRef.current = streamAnswer(trimmed, {
        onToken: (text) => setStreamed((prev) => prev + text),
        onDone: async () => {
          setPhase("resolving");
          try {
            setAnswer(await api.ask(trimmed));
            setPhase("done");
          } catch (err) {
            setError(err as Error);
            setPhase("error");
          }
        },
        onError: (err) => {
          setError(err);
          setPhase("error");
        },
      });
    },
    [question, phase],
  );

  const focusSource = (marker: number) => {
    setFocused(marker);
    document
      .getElementById(`source-${marker}`)
      ?.scrollIntoView({ behavior: "smooth", block: "center" });
  };

  const indexEmpty = docs && docs.searchable === 0;

  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-2xl font-semibold tracking-tight">
          Ask your documents
        </h1>
        <p className="mt-1 text-sm text-ink-muted">
          Every answer cites the page it came from. If the answer is not in
          your documents, the assistant says so.
        </p>
      </header>

      {/* An empty index is a precondition problem, not an error. Say what
          to do instead of letting the user discover it by failing. */}
      {indexEmpty && (
        <div className="rounded-lg border border-warn/40 bg-warn-soft p-4 text-sm">
          <p className="font-medium text-warn">No searchable documents yet</p>
          <p className="mt-1 text-ink-muted">
            {docs.needs_ocr > 0
              ? `${docs.needs_ocr} document(s) are scanned images and cannot be searched without OCR.`
              : "Upload a PDF with a text layer to get started."}{" "}
            <Link href="/documents" className="text-accent underline">
              Manage documents
            </Link>
          </p>
        </div>
      )}

      <form onSubmit={submit} className="space-y-3">
        <label htmlFor="question" className="sr-only">
          Your question
        </label>
        <textarea
          id="question"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => {
            // Enter submits, Shift+Enter adds a newline — the convention
            // users already expect from chat interfaces.
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              submit(e);
            }
          }}
          rows={3}
          maxLength={1000}
          placeholder="What does the document say about…?"
          className="w-full resize-y rounded-lg border border-border bg-surface p-3 text-ink outline-none focus:border-accent focus:ring-2 focus:ring-accent/20"
        />
        <div className="flex items-center gap-3">
          <button
            type="submit"
            disabled={question.trim().length < 3 || phase === "streaming"}
            className="rounded-lg bg-accent px-4 py-2 text-sm font-medium text-white transition-opacity disabled:opacity-40"
          >
            {phase === "streaming" ? "Answering…" : "Ask"}
          </button>
          {phase === "streaming" && (
            <button
              type="button"
              onClick={() => {
                abortRef.current?.();
                setPhase("idle");
              }}
              className="text-sm text-ink-muted underline"
            >
              Stop
            </button>
          )}
          <span className="ml-auto text-xs text-ink-muted">
            {question.length}/1000 · Enter to send
          </span>
        </div>
      </form>

      {/* Streaming text. `aria-live="polite"` lets screen readers announce
          the answer as it arrives without interrupting the user. */}
      {(phase === "streaming" || phase === "resolving") && (
        <section
          aria-live="polite"
          aria-busy
          className="rounded-xl border border-border bg-surface-muted p-5"
        >
          <div className="text-xs font-medium tracking-wide text-ink-muted uppercase">
            {phase === "streaming" ? "Answering" : "Verifying citations"}
          </div>
          <p
            className={`mt-2 whitespace-pre-wrap text-ink ${
              phase === "streaming" ? "caret" : ""
            }`}
          >
            {streamed || "…"}
          </p>
        </section>
      )}

      {phase === "error" && error && <ErrorPanel error={error} />}

      {phase === "done" && answer && (
        <>
          <AnswerCard answer={answer} onSourceFocus={focusSource} />
          <SourceList sources={answer.sources} highlighted={focused} />
        </>
      )}
    </div>
  );
}

/**
 * Errors are explained by cause, not by status code. "409" means nothing to
 * a user; "your index is empty, add a document" is actionable.
 */
function ErrorPanel({ error }: { error: ApiError | Error }) {
  const api = error instanceof ApiError ? error : null;

  const guidance = (() => {
    if (!api) return "Something went wrong. Please try again.";
    if (api.code === "network_unreachable")
      return "The API server is not responding. Start it and try again.";
    if (api.isPrecondition)
      return "There is nothing indexed yet — upload a document first.";
    if (api.isTransient)
      return "The language model is temporarily unavailable. Try again shortly.";
    return api.message;
  })();

  return (
    <section
      role="alert"
      className="rounded-xl border border-danger/40 bg-danger-soft p-5"
    >
      <div className="flex items-center gap-2 text-sm font-medium text-danger">
        <span aria-hidden>×</span>
        Could not answer
      </div>
      <p className="mt-2 text-ink">{guidance}</p>
      {api?.requestId && (
        <p className="mt-3 font-mono text-[11px] text-ink-muted">
          request {api.requestId}
        </p>
      )}
    </section>
  );
}
