import type { Metadata } from "next";
import type { ReactNode } from "react";
import "./globals.css";

export const metadata: Metadata = {
  metadataBase: new URL("https://trial-atlas.local"),
  title: "Trial Atlas",
  description: "Compare metric-learning HPO runs, search trajectories, score distributions, and runtime trade-offs.",
  openGraph: {
    title: "Trial Atlas",
    description: "Compare runs. Find the signal.",
    type: "website",
    images: [
      {
        url: "/og.png",
        width: 1731,
        height: 909,
        alt: "Trial Atlas run comparison dashboard",
      },
    ],
  },
  twitter: {
    card: "summary_large_image",
    title: "Trial Atlas",
    description: "Compare runs. Find the signal.",
    images: ["/og.png"],
  },
};

export default function RootLayout({ children }: Readonly<{ children: ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
