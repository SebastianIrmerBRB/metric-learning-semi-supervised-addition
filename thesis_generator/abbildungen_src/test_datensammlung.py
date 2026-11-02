"""Behavior checks: result edits propagate, mismatched folds are rejected."""
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

import datensammlung as data
from erstelle_uebersicht import overview
from pruefe_daten import validate


class DataBundleTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="thesis-data-test-")
        folder = Path(self.tmp.name)
        for p in data.DATA.iterdir():
            if p.is_file():
                shutil.copyfile(p, folder / p.name)  # copies, symbolic links need extra rights on Windows
        self.patch = patch.object(data, "DATA", folder)
        self.patch.start()
        data.clear_cache()

    def tearDown(self):
        self.patch.stop()
        data.clear_cache()
        self.tmp.cleanup()

    def edit_result(self, with_fold):
        p = data.DATA / "results.csv"
        r = pd.read_csv(p)
        mask = (r.dataset_key == "cars") & (r.method == "iscen_arc") & (r.k == 4) & (r.seed == 2) & (r.scope == "mean")
        self.assertEqual(mask.sum(), 1)
        run_id = r.loc[mask, "run_id"].iloc[0]
        r.loc[mask, "mean_average_precision_at_r"] += 0.001
        r.to_csv(p, index=False)
        if with_fold:
            p = data.DATA / "folds.csv"
            f = pd.read_csv(p, low_memory=False)
            selected = f.index[f.run_id == run_id]
            f.loc[selected[0], "mean_average_precision_at_r"] += 0.001 * len(selected)
            f.to_csv(p, index=False)
        data.clear_cache()

    def paired_gain(self):
        r = data.read_csv("cars_ergebnisse/data/phase1_paired_differences.csv")
        return r[(r.method == "iscen_arc") & (r.k == 4) & (r.scope == "mean") &
                 (r.metric == "mean_average_precision_at_r")]["mean"].iloc[0]

    def test_result_and_fold_edit_updates_gain_and_overview(self):
        before = self.paired_gain()
        self.edit_result(with_fold=True)
        self.assertAlmostEqual(self.paired_gain() - before, 0.02, places=10)
        summary = overview()
        gain = summary[(summary.dataset == "cars") & (summary.method == "iscen") &
                       (summary.loss == "arc") & (summary.budget == 4)].paired_mean_gain_pp.iloc[0]
        self.assertAlmostEqual(gain, self.paired_gain(), places=10)
        result = validate()
        self.assertTrue(result["passed"], result["errors"])

    def test_result_edit_without_matching_folds_is_rejected(self):
        self.edit_result(with_fold=False)
        result = validate()
        self.assertFalse(result["passed"])
        self.assertTrue(any("Mean differs from fold mean" in e for e in result["errors"]))

    def test_original_import_check_rejects_edits(self):
        self.edit_result(with_fold=True)
        result = validate(verify_snapshot=True)
        self.assertFalse(result["passed"])
        self.assertIn("Changed since import: results.csv", result["errors"])

    def test_generated_view_is_written_without_other_outputs(self):
        with tempfile.TemporaryDirectory(prefix="thesis-output-test-") as out, patch.object(data, "OUTPUT", Path(out)):
            data.read_csv("geometrie/korrelation_weitere.csv")
            written = Path(out) / "generiert" / "geometrie" / "korrelation_weitere.csv"
            self.assertTrue(written.is_file())
            self.assertEqual(sorted(p.name for p in Path(out).iterdir()), ["generiert"])
            stamp = written.stat().st_mtime
            data.read_csv("geometrie/mittel_weitere.csv")  # newer than data and generator: not written again
            self.assertEqual(written.stat().st_mtime, stamp)


if __name__ == "__main__":
    unittest.main()
