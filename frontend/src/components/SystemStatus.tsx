"use client";

import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";

/**
 * Readiness indicator.
 *
 * Surfaces what `/ready` reports rather than a generic "online" dot. If the
 * index is empty or the reranker could not load, the user should see that —
 * a silently degraded system is the hardest kind of regression to notice.
 */
export function SystemStatus() {
  const { data, isPending, isError } = useQuery({
    queryKey: ["health"],
    queryFn: api.health,
    refetchInterval: 30_000,
  });

  if (isPending) {
    return <span className="text-xs text-ink-muted">checking…</span>;
  }

  if (isError) {
    return (
      <span className="flex items-center gap-1.5 text-xs text-danger">
        <span aria-hidden className="size-2 rounded-full bg-danger" />
        API unreachable
      </span>
    );
  }

  const degraded = data.components.filter((c) => !c.ok);
  const inactive = data.components.filter((c) =>
    c.detail.startsWith("devre dışı"),
  );

  return (
    <details className="group relative">
      <summary className="flex cursor-pointer list-none items-center gap-1.5 text-xs text-ink-muted">
        <span
          aria-hidden
          className={`size-2 rounded-full ${data.ready ? "bg-grounded" : "bg-warn"}`}
        />
        {data.ready ? "Ready" : "Not ready"}
        {inactive.length > 0 && (
          <span className="text-warn">· degraded</span>
        )}
      </summary>
      <div className="absolute right-0 z-20 mt-2 w-72 rounded-lg border border-border bg-surface p-3 text-xs shadow-lg">
        <dl className="space-y-1.5">
          {data.components.map((c) => (
            <div key={c.name} className="flex gap-2">
              <dt className="w-28 shrink-0 text-ink-muted">{c.name}</dt>
              <dd className={c.ok ? "" : "text-warn"}>{c.detail}</dd>
            </div>
          ))}
        </dl>
        <hr className="my-2 border-border" />
        <dl className="space-y-1 text-ink-muted">
          <div className="flex gap-2">
            <dt className="w-28 shrink-0">strategy</dt>
            <dd className="font-mono text-[11px]">{data.retrieval_strategy}</dd>
          </div>
          <div className="flex gap-2">
            <dt className="w-28 shrink-0">embedder</dt>
            <dd className="font-mono text-[11px]">{data.embedder_model}</dd>
          </div>
          <div className="flex gap-2">
            <dt className="w-28 shrink-0">model</dt>
            <dd className="font-mono text-[11px]">{data.llm_model}</dd>
          </div>
        </dl>
        {degraded.length > 0 && (
          <p className="mt-2 rounded bg-warn-soft p-2 text-warn">
            {degraded.length} component(s) not ready.
          </p>
        )}
      </div>
    </details>
  );
}
