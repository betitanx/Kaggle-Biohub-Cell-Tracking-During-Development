import json
import math
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class PublicSolutionManifestTests(unittest.TestCase):
    def test_manifest_is_a_reproducibility_queue_not_a_score_source(self):
        payload = json.loads((ROOT / "config" / "public_solutions.json").read_text(encoding="utf-8"))

        self.assertEqual(payload["competition"], "biohub-cell-tracking-during-development")
        self.assertGreaterEqual(len(payload["entries"]), 5)
        for entry in payload["entries"]:
            self.assertTrue(entry["url"].startswith("https://www.kaggle.com/code/"))
            self.assertTrue(entry["status"] != "reproduced")
            for key in ("observed_public_score", "observed_best_score"):
                value = entry.get(key)
                if value is not None:
                    self.assertTrue(math.isfinite(float(value)))

    def test_method_rich_public_lead_is_present_for_reimplementation(self):
        payload = json.loads((ROOT / "config" / "public_solutions.json").read_text(encoding="utf-8"))
        entry = next(item for item in payload["entries"] if item["id"] == "adaptive_threshold_ilp_postprocess")

        self.assertIn("adaptive threshold + subvoxel refinement improves detections", entry["hypotheses"])
        self.assertIn("ILP", " ".join(entry["evidence"]))


if __name__ == "__main__":
    unittest.main()
