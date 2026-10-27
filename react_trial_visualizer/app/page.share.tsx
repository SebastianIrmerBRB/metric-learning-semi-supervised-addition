import { Phase2Explorer } from "@/components/Phase2Explorer";
import { loadAblationSnapshot, loadGeometrySnapshot } from "@/lib/phase2-server";

export const dynamic = "force-static";

/**
 * The hosted Phase-2 page (npm run build:share). SHARE_TABS=geometry (default) bundles only
 * the geometry snapshot; SHARE_TABS=all adds the ablations.
 */
function withoutLocalPaths<T extends { roots: string[] }>(snapshot: T | null): T | null {
  // Log roots are absolute paths on this machine; the page only needs them as run labels.
  return snapshot && { ...snapshot, roots: snapshot.roots.map((root) => root.split(/[\\/]/).filter(Boolean).at(-1) ?? root) };
}

export default function SharePage() {
  const withAblations = process.env.SHARE_TABS?.trim() === "all";
  return (
    <Phase2Explorer
      shared
      ablations={withAblations ? withoutLocalPaths(loadAblationSnapshot()) : null}
      geometry={withoutLocalPaths(loadGeometrySnapshot())}
    />
  );
}
