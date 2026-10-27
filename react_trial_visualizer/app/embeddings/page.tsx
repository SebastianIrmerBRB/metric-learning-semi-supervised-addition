import type { Metadata } from "next";
import { EmbeddingExplorer } from "@/components/EmbeddingExplorer";
import { loadEmbeddingViewIndex } from "@/lib/embedding-views-server";

export const dynamic = "force-static";

export const metadata: Metadata = {
  title: "Embeddings · Trial Atlas",
  description:
    "Explore one run's embedding projection in 2D or 3D, with labeled, unlabeled, validation and test points toggled independently.",
};

export default function EmbeddingsPage() {
  return <EmbeddingExplorer index={loadEmbeddingViewIndex()} />;
}
