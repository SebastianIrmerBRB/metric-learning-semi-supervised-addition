import fs from "node:fs";
import path from "node:path";
import type { TestResultFile } from "@/lib/test-results";

export function loadTestResultsFile(): TestResultFile | null {
  const configuredPath = process.env.TEST_RESULTS_CSV?.trim();
  const candidates = [
    configuredPath ? path.resolve(configuredPath) : null,
    path.resolve(process.cwd(), "..", "trials-files", "test-results.csv"),
    path.resolve(process.cwd(), "..", "test-results.csv"),
  ].filter((candidate): candidate is string => candidate != null);

  for (const candidate of candidates) {
    try {
      if (!fs.statSync(candidate).isFile()) continue;
      return {
        name: path.basename(candidate),
        text: fs.readFileSync(candidate, "utf8"),
      };
    } catch {
      // Try the next supported location. Both comparison surfaces allow upload.
    }
  }
  return null;
}
