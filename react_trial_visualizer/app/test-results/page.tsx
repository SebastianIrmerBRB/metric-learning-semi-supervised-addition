import type { Metadata } from "next";
import { TestResultsExplorer } from "@/components/TestResultsExplorer";
import { loadTestResultsFile } from "@/lib/test-results-server";

export const dynamic = "force-static";

export const metadata: Metadata = {
  title: "Test Results · Trial Atlas",
  description: "Compare final test-set performance across metric-learning runs and seeds.",
};

export default function TestResultsPage() {
  return <TestResultsExplorer initialFile={loadTestResultsFile()} />;
}
