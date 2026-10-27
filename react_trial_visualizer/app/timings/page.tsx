import type { Metadata } from "next";
import { TimingsExplorer } from "@/components/TimingsExplorer";
import { loadTimingSnapshots } from "@/lib/timings-server";

export const dynamic = "force-static";

export const metadata: Metadata = {
  title: "Timings · Trial Atlas",
  description: "Clean per-epoch cost and estimated HPO cost per method, dataset, and loss.",
};

export default function TimingsPage() {
  return <TimingsExplorer snapshots={loadTimingSnapshots()} />;
}
