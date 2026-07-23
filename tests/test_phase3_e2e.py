"""Lightweight Phase 3 gate test — runs a reduced end-to-end validation."""

from __future__ import annotations

import os
import unittest
from pathlib import Path


class Phase3GateTest(unittest.TestCase):
    def test_phase3_validation_harness_passes(self) -> None:
        # Keep CI-friendly defaults; full 1M stress is run manually / in scripts.
        env = os.environ.copy()
        env["PHASE3_STRESS_CANDLES"] = "50000"
        env["PHASE3_STRESS_STREAM"] = "25000"
        env["PHASE3_TRAIN_BARS"] = "8000"
        env["PHASE3_ARTIFACT_DIR"] = "ai_artifacts/phase3_ci"
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "scripts/phase3_validation.py"],
            cwd=str(Path(__file__).resolve().parents[1]),
            env=env,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            result.returncode,
            0,
            msg=f"stdout:\n{result.stdout[-2000:]}\nstderr:\n{result.stderr[-2000:]}",
        )
        self.assertIn("'passed': True", result.stdout.replace("True", "True"))


if __name__ == "__main__":
    unittest.main()
