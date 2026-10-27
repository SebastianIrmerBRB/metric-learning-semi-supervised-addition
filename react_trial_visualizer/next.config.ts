import { PHASE_DEVELOPMENT_SERVER } from "next/constants";
import type { NextConfig } from "next";

// The published site is a static export, but the config studio needs route
// handlers to read and write files on this machine. `next dev` keeps them;
// every other phase drops the `.ts` routes from the graph so the export still
// builds exactly as it did before.
//
// TRIAL_ATLAS_SHARE=1 (npm run build:share) builds the hosted Phase-2 site
// instead: only app/**/*.share.tsx, exported to share/.
export default function config(phase: string): NextConfig {
  const isDevServer = phase === PHASE_DEVELOPMENT_SERVER;
  const isShare = !isDevServer && process.env.TRIAL_ATLAS_SHARE === "1";

  return {
    output: isDevServer ? undefined : "export",
    pageExtensions: isDevServer ? ["ts", "tsx"] : isShare ? ["share.tsx"] : ["tsx"],
    ...(isShare ? { distDir: "share" } : {}),
    poweredByHeader: false,
    trailingSlash: true,
  };
}
