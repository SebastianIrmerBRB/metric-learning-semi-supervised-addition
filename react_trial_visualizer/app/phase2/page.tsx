import type { Metadata } from "next";
import { Phase2Explorer } from "@/components/Phase2Explorer";
import { loadAblationSnapshot, loadGeometrySnapshot } from "@/lib/phase2-server";

export const dynamic = "force-static";

export const metadata: Metadata = {
  title: "Phase 2 · Trial Atlas",
  description: "2x2 ablation, named mechanism controls and embedding-space geometry (B6).",
};

export default function Phase2Page() {
  return <Phase2Explorer ablations={loadAblationSnapshot()} geometry={loadGeometrySnapshot()} />;
}
