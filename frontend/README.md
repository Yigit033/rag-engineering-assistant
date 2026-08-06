# RAG Assistant — Web UI

Next.js 16 · React 19 · Tailwind 4 · TanStack Query

## Run

The API must be running first:

```bash
# from the repository root
uvicorn rag_assistant.api.app:app --port 8001
```

```bash
npm install
npm run dev            # http://localhost:3000
```

Point the UI at a different API with `.env.local`:

```
NEXT_PUBLIC_API_URL=http://127.0.0.1:8001
```

## Types are generated, not written

```bash
npm run generate:api   # reads /openapi.json -> src/lib/api-schema.d.ts
```

Rename a field on the server and the frontend **stops compiling**. Hand-written
interfaces drift silently; generated ones cannot. Run this after any API change.

## Design decisions

| Decision | Why |
|---|---|
| **Abstention is not an error** | When the assistant says "not in the documents", that is the system working. Styling it like a failure would teach users that honesty looks like malfunction — and push them toward a system that always answers. |
| Stream first, then verify | The stream gives a first token in ~1s; the follow-up `/ask` call returns citations and sources. Users get speed *and* a checkable result. |
| `retriever_hits` shown as "consensus" | Reveals whether hybrid search actually worked. In an earlier version this number was always 1 because fusion was silently broken — nothing in a normal UI would have exposed that. |
| Errors explained by cause | "409" means nothing to a user. "Your index is empty, add a document" is actionable. |
| Client-side validation is a courtesy | The server validates independently and cannot be bypassed. The browser check only saves a doomed 50 MB upload. |
| Delete asks for confirmation | It removes chunks from the search index too — not reversible from the UI. The dialog names the actual consequence. |
| Scanned PDFs flagged, never hidden | A document that cannot be searched must look different from one that can, or users will assume it is included. |

## Accessibility

Skip link · `aria-live` on the streaming answer · labelled form controls ·
focus-visible styling · `prefers-reduced-motion` respected · semantic table
with caption.
