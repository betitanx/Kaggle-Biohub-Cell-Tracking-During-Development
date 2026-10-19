import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from biohub_lab.evaluation import convert_csv_to_geff


class EvaluationAdapterTests(unittest.TestCase):
    def test_converter_receives_absolute_paths_when_cwd_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "baseline"
            (baseline / "scripts").mkdir(parents=True)
            (baseline / "scripts" / "csv_to_geffs.py").write_text("", encoding="utf-8")
            submission = root / "runs" / "candidate" / "validation_submission.csv"
            submission.parent.mkdir(parents=True)
            submission.write_text("dataset,t,z,y,x,parent_id\n", encoding="utf-8")
            output = root / "runs" / "candidate" / "pred_geffs"

            previous_cwd = Path.cwd()
            os.chdir(root)
            try:
                with patch("biohub_lab.evaluation.subprocess.run") as run:
                    run.return_value.returncode = 0
                    result = convert_csv_to_geff(
                        Path("runs/candidate/validation_submission.csv"),
                        Path("runs/candidate/pred_geffs"),
                        baseline,
                    )
            finally:
                os.chdir(previous_cwd)

        command = run.call_args.args[0]
        self.assertEqual(result, output.resolve())
        self.assertEqual(command[command.index("--csv") + 1], str(submission.resolve()))
        self.assertEqual(command[command.index("--out-dir") + 1], str(output.resolve()))
        self.assertEqual(run.call_args.kwargs["cwd"], str(baseline.resolve()))


if __name__ == "__main__":
    unittest.main()
