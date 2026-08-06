"use client";

import type { AskResponse, Source } from "@/lib/api";

/**
 * Answer presentation.
 *
 * THE CENTRAL UX DECISION: an abstention is NOT an error.
 *
 *   When the assistant says "this is not in the documents", that is the
 *   system working exactly as designed — it declined to invent an answer.
 *   Rendering it in red next to real failures would teach users that
 *   honesty looks like malfunction, and they would start preferring a
 *   system that always answers. That is precisely the wrong incentive.
 *
 *   So abstention gets its own calm, distinct treatment, and the copy
 *   explains what to do next.
 */

interface Props {
  answer: AskResponse;
  onSourceFocus?: (marker: number) => void;
}

export function AnswerCard({ answer, onSourceFocus }: Props) {
  if (answer.abstained) {
    return (
      <section
        aria-label="Answer"
        className="rounded-xl border border-abstain/30 bg-abstain-soft p-5"
      >
        <div className="flex items-center gap-2 text-sm font-medium text-abstain">
          <span aria-hidden>◇</span>
          Not found in the documents
        </div>
        <p className="mt-2 text-ink">{answer.answer}</p>
        <p className="mt-3 text-sm text-ink-muted">
          The assistant only answers from indexed documents. It declined
          rather than guessing. Try rephrasing, or add a document that covers
          this topic.
        </p>
        <Meta answer={answer} />
      </section>
    );
  }

  const ungrounded = !answer.grounded;

  return (
    <section
      aria-label="Answer"
      className={`rounded-xl border p-5 ${
        ungrounded
          ? "border-warn/40 bg-warn-soft"
          : "border-grounded/30 bg-grounded-soft"
      }`}
    >
      <div
        className={`flex items-center gap-2 text-sm font-medium ${
          ungrounded ? "text-warn" : "text-grounded"
        }`}
      >
        <span aria-hidden>{ungrounded ? "!" : "✓"}</span>
        {ungrounded ? "Answer without citations" : "Grounded answer"}
      </div>

      <p className="mt-2 whitespace-pre-wrap text-ink">{answer.answer}</p>

      {ungrounded && (
        <p className="mt-3 text-sm text-warn">
          The model did not cite a source, so this answer cannot be verified
          against the documents. Treat it with caution.
        </p>
      )}

      {answer.citations.length > 0 && (
        <div className="mt-4">
          <h3 className="text-xs font-semibold tracking-wide text-ink-muted uppercase">
            Citations
          </h3>
          <ul className="mt-2 flex flex-wrap gap-2">
            {answer.citations.map((c) => (
              <li key={c.marker}>
                <button
                  type="button"
                  onClick={() => onSourceFocus?.(c.marker)}
                  className="rounded-full border border-border bg-surface px-3 py-1 text-xs transition-colors hover:border-accent hover:text-accent"
                >
                  <span className="font-mono">[{c.marker}]</span> {c.source}
                </button>
              </li>
            ))}
          </ul>
        </div>
      )}

      <Meta answer={answer} />
    </section>
  );
}

function Meta({ answer }: { answer: AskResponse }) {
  return (
    <dl className="mt-4 flex flex-wrap gap-x-4 gap-y-1 border-t border-border/60 pt-3 text-[11px] text-ink-muted">
      <div className="flex gap-1">
        <dt>model</dt>
        <dd className="font-mono">{answer.model}</dd>
      </div>
      <div className="flex gap-1">
        <dt>prompt</dt>
        <dd className="font-mono">{answer.prompt_version}</dd>
      </div>
      <div className="flex gap-1">
        <dt>latency</dt>
        <dd className="font-mono">{answer.latency_ms} ms</dd>
      </div>
      {answer.groundedness != null && (
        <div className="flex gap-1">
          <dt>groundedness</dt>
          <dd className="font-mono">
            {(answer.groundedness * 100).toFixed(0)}%
          </dd>
        </div>
      )}
    </dl>
  );
}

/**
 * The context that was actually sent to the model.
 *
 * `retriever_hits` is surfaced deliberately. It answers "did hybrid search
 * actually work?" — a value above 1 means both the dense and the keyword
 * retriever independently selected this chunk. In an earlier version of this
 * system that number was always 1 because fusion was silently broken, and
 * nothing in the UI would have revealed it.
 */
export function SourceList({
  sources,
  highlighted,
}: {
  sources: Source[];
  highlighted?: number | null;
}) {
  if (sources.length === 0) return null;

  return (
    <section aria-label="Context used" className="mt-6">
      <h2 className="text-xs font-semibold tracking-wide text-ink-muted uppercase">
        Context sent to the model ({sources.length})
      </h2>
      <ol className="mt-2 space-y-2">
        {sources.map((s) => {
          const isHighlighted = highlighted === s.marker;
          return (
            <li
              key={s.chunk_id}
              id={`source-${s.marker}`}
              className={`rounded-lg border p-3 transition-colors ${
                isHighlighted
                  ? "border-accent bg-accent-soft"
                  : "border-border bg-surface-muted"
              }`}
            >
              <div className="flex flex-wrap items-center gap-2 text-xs">
                <span className="font-mono text-accent">[{s.marker}]</span>
                <span className="font-medium">{s.source}</span>
                <span className="text-ink-muted">score {s.score.toFixed(3)}</span>
                {s.retriever_hits > 1 && (
                  <span
                    className="rounded bg-grounded-soft px-1.5 py-0.5 text-grounded"
                    title="Found independently by both dense and keyword search"
                  >
                    consensus ×{s.retriever_hits}
                  </span>
                )}
              </div>
              <p className="mt-1.5 line-clamp-3 text-sm text-ink-muted">
                {s.preview}
              </p>
            </li>
          );
        })}
      </ol>
    </section>
  );
}
