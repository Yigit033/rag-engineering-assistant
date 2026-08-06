"use client";

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useState } from "react";

/**
 * Query client provider.
 *
 * WHY TANSTACK QUERY AND NOT PLAIN `useEffect` + `useState`:
 *   The document list is shared by several views and must refresh after an
 *   upload or delete. Hand-rolling that means every mutation has to remember
 *   which views to refresh — and one will be forgotten, leaving a stale list
 *   that shows a document the user just deleted.
 *   Query invalidation makes the refresh a property of the data, not of the
 *   component that happened to trigger it.
 *
 * WHY THE CLIENT IS CREATED IN STATE:
 *   A module-level client would be shared across users during server-side
 *   rendering — one visitor's data could leak into another's cache.
 */
export function Providers({ children }: { children: React.ReactNode }) {
  const [client] = useState(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: {
            staleTime: 15_000,
            refetchOnWindowFocus: false,
            // Do not retry client errors (bad request, not found): the
            // request is wrong, repeating it wastes time and hides the cause.
            retry: (failureCount, error) => {
              const status = (error as { status?: number }).status ?? 0;
              if (status >= 400 && status < 500) return false;
              return failureCount < 2;
            },
          },
        },
      }),
  );

  return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
}
