import type { Metadata } from "next";
import type { ReactNode } from "react";
import "./globals.css";

// Root layout of the hosted build only (npm run build:share); the atlas uses layout.tsx.
export const metadata: Metadata = {
  title: "Phase 2 · Embedding-Geometrie",
  description: "Embedding-space geometry of supervised and semi-supervised metric learning (Phase 2, B6).",
  robots: { index: false, follow: false },
};

export default function ShareLayout({ children }: Readonly<{ children: ReactNode }>) {
  return (
    <html lang="de">
      <body>{children}</body>
    </html>
  );
}
