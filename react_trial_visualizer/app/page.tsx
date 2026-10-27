import fs from "node:fs";

import path from "node:path";
import { TrialVisualizer } from "@/components/TrialVisualizer";
import { loadTestResultsFile } from "@/lib/test-results-server";
import type { SeedFile } from "@/lib/trials";

export const dynamic = "force-static";

const FALLBACK_CSV = `number,state,value,param:batch_sampler,param:lr,attr:best_valid_mean_average_precision_at_r,attr:last_epoch,attr:selected_epoch,duration_seconds
0,COMPLETE,0.512,16:4,0.0000028,0.512,420,361,742
1,COMPLETE,0.548,32:4,0.00004,0.548,68,34,122
2,COMPLETE,0.559,64:4,0.00019,0.559,31,7,67
3,COMPLETE,0.541,128:4,0.0017,0.541,28,4,58
4,COMPLETE,0.566,256:4,0.000053,0.566,45,22,104
5,COMPLETE,0.552,512:8,0.066,0.552,33,10,76
6,COMPLETE,0.571,128:8,0.00043,0.571,28,5,62
7,COMPLETE,0.547,64:8,0.022,0.547,28,8,69
8,COMPLETE,0.579,32:2,0.000019,0.579,95,70,217
9,COMPLETE,0.568,256:8,0.387,0.568,50,26,114`;

function loadSeedFiles(): SeedFile[] {
  const trialDirectories = [
    path.resolve(process.cwd(), "..", "trials-files", "scheduled"),
    path.resolve(process.cwd(), "..", "trials-files_ol", "cars"),
  ];

  for (const trialDirectory of trialDirectories) {
    try {
      const files = fs
        .readdirSync(trialDirectory)
        .filter((name) => name.toLowerCase().endsWith(".csv"))
        .sort()
        .map((name) => ({
          name,
          text: fs.readFileSync(path.join(trialDirectory, name), "utf8"),
        }));

      if (files.length > 0) return files;
    } catch {
      // Try the next repository data location.
    }
  }

  return [{ name: "example_metric_learning_trials.csv", text: FALLBACK_CSV }];
}

export default function Home() {
  return (
    <TrialVisualizer
      initialFiles={loadSeedFiles()}
      initialTestResultsFile={loadTestResultsFile()}
    />
  );
}
