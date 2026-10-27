import type { Metadata } from "next";
import { EmbeddingExplorer } from "@/components/EmbeddingExplorer";
import { loadEmbeddingViewIndex } from "@/lib/embedding-views-server";

export const dynamic = "force-static";

export const metadata: Metadata = {
  title: "Embeddings · Phase 2",
};

// Hosted build only: build-share-site.mjs points EMBEDDING_VIEWS_INDEX at the published subset.
export default function SharedEmbeddingsPage() {
  return <EmbeddingExplorer index={loadEmbeddingViewIndex()} shared />;
}
