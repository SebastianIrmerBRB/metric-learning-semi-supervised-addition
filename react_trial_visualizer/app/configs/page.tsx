import type { Metadata } from "next";
import { ConfigOverview } from "@/components/configs/ConfigOverview";

export const dynamic = "force-static";

export const metadata: Metadata = {
  title: "Run configs · Trial Atlas",
  description: "What every HPO study, seeded series and Phase-2 replay ran with: protocol checks, comparisons, runtime and coverage.",
};

// The snapshot (public/study-configs.json, several MB) is fetched in the browser
// rather than inlined into the page.
export default function ConfigsPage() {
  return <ConfigOverview />;
}
