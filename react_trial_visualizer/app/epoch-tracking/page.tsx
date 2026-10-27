import type { Metadata } from "next";
import { EpochTrackingExplorer } from "@/components/EpochTrackingExplorer";
import { loadEpochTrackingSnapshot } from "@/lib/epoch-tracking-server";

export const dynamic = "force-static";

export const metadata: Metadata = {
  title: "Epoch Tracking · Trial Atlas",
  description:
    "Watch graph structure and embedding geometry unfold across the epochs of a fold, from the epoch -1 baseline onward.",
};

export default function EpochTrackingPage() {
  return <EpochTrackingExplorer snapshot={loadEpochTrackingSnapshot()} />;
}
