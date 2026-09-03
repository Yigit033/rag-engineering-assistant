import type { Metadata } from "next";
import Link from "next/link";

import { SystemStatus } from "@/components/SystemStatus";
import { Providers } from "./providers";
import "./globals.css";

export const metadata: Metadata = {
  title: "RAG Assistant",
  description:
    "Grounded question answering over your own documents. Answers cite their sources — and the system declines to answer when the information is not there.",
};

const NAV = [
  { href: "/", label: "Ask" },
  { href: "/documents", label: "Documents" },
];

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body className="min-h-dvh" suppressHydrationWarning>
        <Providers>
          {/* Keyboard users should be able to skip the nav. Costs one
              element; without it every tab session starts with the nav. */}
          <a
            href="#main"
            className="sr-only focus:not-sr-only focus:absolute focus:top-2 focus:left-2 focus:z-50 focus:rounded focus:bg-accent focus:px-3 focus:py-2 focus:text-white"
          >
            Skip to content
          </a>

          <header className="border-b border-border bg-surface-muted">
            <div className="mx-auto flex max-w-5xl items-center gap-6 px-5 py-3">
              <Link href="/" className="font-semibold tracking-tight">
                RAG Assistant
              </Link>
              <nav className="flex gap-1 text-sm" aria-label="Main">
                {NAV.map((item) => (
                  <Link
                    key={item.href}
                    href={item.href}
                    className="rounded px-3 py-1.5 text-ink-muted transition-colors hover:bg-surface hover:text-ink"
                  >
                    {item.label}
                  </Link>
                ))}
              </nav>
              <div className="ml-auto">
                <SystemStatus />
              </div>
            </div>
          </header>

          <main id="main" className="mx-auto max-w-5xl px-5 py-8">
            {children}
          </main>

          <footer className="mx-auto max-w-5xl px-5 pb-10 text-xs text-ink-muted">
            Answers are grounded in the indexed documents only. When the
            information is not present, the assistant says so instead of
            guessing.
          </footer>
        </Providers>
      </body>
    </html>
  );
}
