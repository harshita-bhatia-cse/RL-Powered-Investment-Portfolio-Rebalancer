
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from src.api.analytics_api import app


class PPOValidationAPITests(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)

    def _make_report_and_result(self):
        scenario_id = "validation-123456789"
        scenario_seed = 123456789

        ppo_metrics = {
            "scenario_id": scenario_id,
            "scenario_seed": scenario_seed,
            "steps": 1,
            "ending_nav": 101000.0,
            "episode_reward": 0.01,
            "gross_turnover_sum": 0.2,
            "one_way_turnover_sum": 0.1,
            "transaction_fees_sum": 10.0,
            "slippage_cost_sum": 5.0,
            "transaction_cost_sum": 15.0,
        }

        baselines = []

        for policy, nav in [
            ("equal_weight_rebalanced", 102000.0),
            ("buy_and_hold_after_initial_equal_target", 101500.0),
        ]:
            baselines.append({
                "policy": policy,
                "scenario_id": scenario_id,
                "scenario_seed": scenario_seed,
                "steps": 1,
                "ending_nav": nav,
                "episode_reward": 0.02,
                "gross_turnover_sum": 0.1,
                "one_way_turnover_sum": 0.05,
                "transaction_fees_sum": 5.0,
                "slippage_cost_sum": 2.5,
                "transaction_cost_sum": 7.5,
                "ending_weights": {
                    "SPY": 0.5,
                    "CASH": 0.5,
                },
            })

        report = {
            "source_type": "synthetic",
            "synthetic": True,
            "scenario_id": scenario_id,
            "scenario_seed": scenario_seed,
            "starting_nav": 100000.0,
            "training_validation_seeds_distinct": True,
            "validation_was_used_for_training_or_model_selection": False,
            "notes": "Synthetic software sanity checks only.",
            "ppo": ppo_metrics,
            "baselines": baselines,
        }

        step = SimpleNamespace(
            nav=101000.0,
            reward=0.01,
            gross_turnover=0.2,
            one_way_turnover=0.1,
            transaction_fees=10.0,
            slippage_cost=5.0,
            transaction_cost=15.0,
            weights={
                "SPY": 0.6,
                "CASH": 0.4,
            },
            terminated=True,
            truncated=False,
        )

        result = SimpleNamespace(
            scenario_id=scenario_id,
            seed=scenario_seed,
            steps=[step],
        )

        return report, result

    def test_ppo_validation_returns_reproducible_results(self):
        report, result = self._make_report_and_result()

        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_dir = Path(temp_dir)

            (artifact_dir / "trained_synthetic_ppo_seed42.zip").write_bytes(
                b"mock-model"
            )

            (artifact_dir / "validation_results.json").write_text(
                json.dumps(report),
                encoding="utf-8",
            )

            with patch.dict(
                os.environ,
                {"PPO_ARTIFACT_DIR": temp_dir},
            ):
                with patch(
                    "src.rl.ppo_agent.build_ppo_environments",
                    return_value=SimpleNamespace(
                        validation_env=object()
                    ),
                ), patch(
                    "src.rl.ppo_agent.load_ppo_model",
                    return_value=object(),
                ), patch(
                    "src.rl.ppo_agent.evaluate_ppo",
                    return_value=result,
                ):
                    response = self.client.get("/api/ppo/validation")

        self.assertEqual(response.status_code, 200)

        data = response.json()

        self.assertTrue(data["synthetic"])
        self.assertEqual(data["source_type"], "synthetic")
        self.assertEqual(data["scenario_id"], "validation-123456789")

        # Verify that the large seed is preserved as an exact string.
        self.assertEqual(data["scenario_seed"], "123456789")
        self.assertIsInstance(data["scenario_seed"], str)

        self.assertEqual(data["steps"], 1)
        self.assertAlmostEqual(
            data["ppo"]["ending_nav"],
            101000.0,
        )

        self.assertEqual(
            set(data["baselines"].keys()),
            {
                "equal_weight_rebalanced",
                "buy_and_hold_after_initial_equal_target",
            },
        )

    def test_missing_artifacts_return_503(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.dict(
                os.environ,
                {"PPO_ARTIFACT_DIR": temp_dir},
            ):
                response = self.client.get("/api/ppo/validation")

        self.assertEqual(response.status_code, 503)
        self.assertIn(
            "missing",
            response.json()["detail"].lower(),
        )

    def test_invalid_report_returns_503(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_dir = Path(temp_dir)

            (artifact_dir / "trained_synthetic_ppo_seed42.zip").write_bytes(
                b"mock-model"
            )

            (artifact_dir / "validation_results.json").write_text(
                json.dumps({
                    "source_type": "historical",
                    "synthetic": False,
                }),
                encoding="utf-8",
            )

            with patch.dict(
                os.environ,
                {"PPO_ARTIFACT_DIR": temp_dir},
            ):
                response = self.client.get("/api/ppo/validation")

        self.assertEqual(response.status_code, 503)
        self.assertIn(
            "synthetic",
            response.json()["detail"].lower(),
        )


if __name__ == "__main__":
    unittest.main()
