"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useRef, useState } from "react";

import { ApiError, api, type DocumentInfo } from "@/lib/api";

const MAX_UPLOAD_MB = 50;

export default function DocumentsPage() {
  const queryClient = useQueryClient();
  const [notice, setNotice] = useState<{
    tone: "ok" | "warn" | "error";
    text: string;
  } | null>(null);
  const [pendingDelete, setPendingDelete] = useState<DocumentInfo | null>(null);

  const { data, isPending, isError, error } = useQuery({
    queryKey: ["documents"],
    queryFn: api.documents,
  });

  /*
   * After a mutation we invalidate BOTH queries.
   *
   * `documents` is obvious. `health` matters too: the readiness endpoint
   * reports the index vector count, and uploading or deleting changes it.
   * Forgetting the second one leaves a stale badge in the header — the kind
   * of small inconsistency that makes users distrust everything else.
   */
  const refreshAll = () => {
    void queryClient.invalidateQueries({ queryKey: ["documents"] });
    void queryClient.invalidateQueries({ queryKey: ["health"] });
  };

  const upload = useMutation({
    mutationFn: (file: File) =>
      api.upload(file, { index: true, overwrite: false }),
    onSuccess: (result) => {
      refreshAll();
      if (result.needs_ocr) {
        setNotice({
          tone: "warn",
          text: `"${result.file_name}" uploaded, but it is a scanned image with no text layer. It cannot be searched until OCR is added.`,
        });
      } else if (result.warning) {
        setNotice({ tone: "warn", text: result.warning });
      } else {
        setNotice({
          tone: "ok",
          text: `"${result.file_name}" indexed — ${result.chunk_count} chunks.`,
        });
      }
    },
    onError: (err) => setNotice({ tone: "error", text: describe(err) }),
  });

  const remove = useMutation({
    mutationFn: (fileName: string) => api.remove(fileName),
    onSuccess: (result) => {
      refreshAll();
      setPendingDelete(null);
      setNotice({
        tone: "ok",
        text: `"${result.file_name}" deleted — ${result.removed_chunks} chunks removed from the index.`,
      });
    },
    onError: (err) => setNotice({ tone: "error", text: describe(err) }),
  });

  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-2xl font-semibold tracking-tight">Documents</h1>
        <p className="mt-1 text-sm text-ink-muted">
          Only documents with a text layer can be searched. Scanned PDFs are
          listed but flagged — they are never silently ignored.
        </p>
      </header>

      <UploadZone
        onFile={(file) => {
          setNotice(null);
          upload.mutate(file);
        }}
        busy={upload.isPending}
      />

      {notice && (
        <div
          role="status"
          className={`rounded-lg border p-3 text-sm ${
            notice.tone === "ok"
              ? "border-grounded/30 bg-grounded-soft text-grounded"
              : notice.tone === "warn"
                ? "border-warn/40 bg-warn-soft text-warn"
                : "border-danger/40 bg-danger-soft text-danger"
          }`}
        >
          {notice.text}
        </div>
      )}

      {isPending && <p className="text-sm text-ink-muted">Loading…</p>}

      {isError && (
        <p role="alert" className="text-sm text-danger">
          {describe(error)}
        </p>
      )}

      {data && data.total === 0 && (
        <div className="rounded-lg border border-dashed border-border p-8 text-center">
          <p className="font-medium">No documents yet</p>
          <p className="mt-1 text-sm text-ink-muted">
            Upload a PDF above to build your searchable index.
          </p>
        </div>
      )}

      {data && data.total > 0 && (
        <>
          <dl className="flex flex-wrap gap-x-6 gap-y-1 text-sm text-ink-muted">
            <Stat label="documents" value={data.total} />
            <Stat label="searchable" value={data.searchable} />
            {data.needs_ocr > 0 && (
              <Stat label="need OCR" value={data.needs_ocr} tone="warn" />
            )}
            <Stat label="index vectors" value={data.index_vectors} />
            <Stat label="disk" value={formatBytes(data.disk_usage_bytes)} />
          </dl>

          <div className="overflow-x-auto rounded-xl border border-border">
            <table className="w-full text-sm">
              <caption className="sr-only">Indexed documents</caption>
              <thead className="bg-surface-muted text-left text-xs text-ink-muted uppercase">
                <tr>
                  <th scope="col" className="px-4 py-2.5 font-medium">
                    Document
                  </th>
                  <th scope="col" className="px-4 py-2.5 font-medium">
                    Status
                  </th>
                  <th scope="col" className="px-4 py-2.5 text-right font-medium">
                    Pages
                  </th>
                  <th scope="col" className="px-4 py-2.5 text-right font-medium">
                    Chunks
                  </th>
                  <th scope="col" className="px-4 py-2.5 text-right font-medium">
                    Size
                  </th>
                  <th scope="col" className="px-4 py-2.5" />
                </tr>
              </thead>
              <tbody>
                {data.documents.map((doc) => (
                  <tr key={doc.file_name} className="border-t border-border">
                    <td className="px-4 py-3 font-medium">{doc.file_name}</td>
                    <td className="px-4 py-3">
                      <StatusBadge doc={doc} />
                    </td>
                    <td className="px-4 py-3 text-right tabular-nums">
                      {doc.page_count || "—"}
                    </td>
                    <td className="px-4 py-3 text-right tabular-nums">
                      {doc.chunk_count || "—"}
                    </td>
                    <td className="px-4 py-3 text-right tabular-nums text-ink-muted">
                      {formatBytes(doc.size_bytes)}
                    </td>
                    <td className="px-4 py-3 text-right">
                      <button
                        type="button"
                        onClick={() => setPendingDelete(doc)}
                        className="text-xs text-ink-muted underline hover:text-danger"
                      >
                        Delete
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}

      {/* Deletion removes chunks from the search index too, so it is not
          reversible from the UI. Confirming is not friction for its own
          sake — it names the actual consequence. */}
      {pendingDelete && (
        <ConfirmDialog
          doc={pendingDelete}
          busy={remove.isPending}
          onCancel={() => setPendingDelete(null)}
          onConfirm={() => remove.mutate(pendingDelete.file_name)}
        />
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
function UploadZone({
  onFile,
  busy,
}: {
  onFile: (file: File) => void;
  busy: boolean;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [localError, setLocalError] = useState<string | null>(null);

  /*
   * Client-side checks are a COURTESY, not a control.
   *
   * The server validates extension, size and file name independently and
   * cannot be bypassed. Checking here just saves the user a 50 MB upload
   * that is going to be rejected anyway.
   */
  const accept = (file: File | undefined) => {
    if (!file) return;
    setLocalError(null);
    if (!file.name.toLowerCase().endsWith(".pdf")) {
      setLocalError("Only PDF files are supported.");
      return;
    }
    if (file.size > MAX_UPLOAD_MB * 1024 * 1024) {
      setLocalError(`File is larger than ${MAX_UPLOAD_MB} MB.`);
      return;
    }
    if (file.size === 0) {
      setLocalError("That file is empty.");
      return;
    }
    onFile(file);
  };

  return (
    <div>
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          accept(e.dataTransfer.files[0]);
        }}
        className={`rounded-xl border-2 border-dashed p-8 text-center transition-colors ${
          dragging ? "border-accent bg-accent-soft" : "border-border"
        }`}
      >
        <p className="text-sm">
          {busy ? (
            <span className="text-ink-muted">Uploading and indexing…</span>
          ) : (
            <>
              Drop a PDF here, or{" "}
              <button
                type="button"
                onClick={() => inputRef.current?.click()}
                className="text-accent underline"
              >
                choose a file
              </button>
            </>
          )}
        </p>
        <p className="mt-1 text-xs text-ink-muted">
          PDF with a text layer · up to {MAX_UPLOAD_MB} MB
        </p>
        <input
          ref={inputRef}
          type="file"
          accept="application/pdf,.pdf"
          className="sr-only"
          onChange={(e) => {
            accept(e.target.files?.[0]);
            e.target.value = ""; // allow re-selecting the same file
          }}
        />
      </div>
      {localError && (
        <p role="alert" className="mt-2 text-sm text-danger">
          {localError}
        </p>
      )}
    </div>
  );
}

function StatusBadge({ doc }: { doc: DocumentInfo }) {
  if (doc.needs_ocr) {
    return (
      <span
        className="rounded bg-warn-soft px-2 py-0.5 text-xs text-warn"
        title="Scanned image — no text layer. Not searchable."
      >
        needs OCR
      </span>
    );
  }
  if (doc.is_searchable) {
    return (
      <span className="rounded bg-grounded-soft px-2 py-0.5 text-xs text-grounded">
        searchable
      </span>
    );
  }
  return (
    <span
      className="rounded bg-surface-muted px-2 py-0.5 text-xs text-ink-muted"
      title={doc.error ?? undefined}
    >
      {doc.status}
    </span>
  );
}

function ConfirmDialog({
  doc,
  busy,
  onCancel,
  onConfirm,
}: {
  doc: DocumentInfo;
  busy: boolean;
  onCancel: () => void;
  onConfirm: () => void;
}) {
  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="confirm-title"
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4"
      onClick={onCancel}
    >
      <div
        className="w-full max-w-md rounded-xl border border-border bg-surface p-5"
        onClick={(e) => e.stopPropagation()}
      >
        <h2 id="confirm-title" className="font-semibold">
          Delete “{doc.file_name}”?
        </h2>
        <p className="mt-2 text-sm text-ink-muted">
          This removes the file and its {doc.chunk_count} indexed chunk
          {doc.chunk_count === 1 ? "" : "s"}. Answers will no longer be able to
          cite it. This cannot be undone.
        </p>
        <div className="mt-5 flex justify-end gap-2">
          <button
            type="button"
            onClick={onCancel}
            className="rounded-lg border border-border px-3 py-1.5 text-sm"
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={onConfirm}
            disabled={busy}
            className="rounded-lg bg-danger px-3 py-1.5 text-sm text-white disabled:opacity-50"
          >
            {busy ? "Deleting…" : "Delete"}
          </button>
        </div>
      </div>
    </div>
  );
}

function Stat({
  label,
  value,
  tone,
}: {
  label: string;
  value: string | number;
  tone?: "warn";
}) {
  return (
    <div className="flex gap-1.5">
      <dt>{label}</dt>
      <dd
        className={`font-medium tabular-nums ${tone === "warn" ? "text-warn" : "text-ink"}`}
      >
        {value}
      </dd>
    </div>
  );
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function describe(error: unknown): string {
  if (error instanceof ApiError) {
    if (error.code === "network_unreachable") return error.message;
    if (error.code === "document_exists")
      return "A document with that name already exists. Delete it first, or rename the file.";
    if (error.code === "unsupported_file_type")
      return "Only PDF files are supported.";
    if (error.code === "file_too_large")
      return `That file is larger than the ${MAX_UPLOAD_MB} MB limit.`;
    if (error.code === "invalid_file_name")
      return "That file name is not allowed. Rename it and try again.";
    return error.message;
  }
  return "Something went wrong. Please try again.";
}
