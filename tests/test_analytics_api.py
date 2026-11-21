
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
        ppo_metrics = {
            "scenario_id": result.scenario_id,
            "seed": result.seed,
            "steps": 1,
            "initial_portfolio_value": 100000.0,
            "final_portfolio_value": 101000.0,
            "cumulative_return": 0.01,
            "annualized_return": 0.01,
            "annualized_volatility": 0.02,
            "sharpe_ratio": 0.5,
            "sortino_ratio": 0.7,
            "maximum_drawdown": -0.01,
            "total_transaction_costs": 15.0,
            "portfolio_turnover": 0.1,
            "episode_reward": 0.01,
            "gross_turnover_sum": 0.2,
            "one_way_turnover_sum": 0.1,
            "transaction_fees_sum": 10.0,
            "slippage_cost_sum": 5.0,
            "transaction_cost_sum": 15.0,
            "ending_weights": {"SPY": 0.6, "CASH": 0.4},
            "terminated": True,
            "truncated": False,
        }
        benchmark_names = (
            "SPY buy-and-hold",
            "Equal-weight",
            "Monthly rebalancing",
            "Threshold rebalancing",
        )
        benchmarks = {
            name: {
                key: value
                for key, value in ppo_metrics.items()
                if key not in {
                    "scenario_id",
                    "seed",
                    "steps",
                    "episode_reward",
                    "gross_turnover_sum",
                    "one_way_turnover_sum",
                    "transaction_fees_sum",
                    "slippage_cost_sum",
                    "transaction_cost_sum",
                    "ending_weights",
                    "terminated",
                    "truncated",
                }
            }
            for name in benchmark_names
        }
        evaluation = {
            "evaluation_type": "synthetic-only",
            "period": {
                "warmup_sessions": 63,
                "period_sessions": 21,
                "evaluation_sessions": 21,
            },
            "cost_assumptions": {
                "initial_capital": 100000.0,
                "transaction_cost_rate": 0.001,
                "slippage_bps": 5.0,
            },
            "ppo": ppo_metrics,
            "benchmarks": benchmarks,
            "strategies": {"Synthetic PPO": ppo_metrics, **benchmarks},
        }

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
                    "src.rl.phase4_evaluation.evaluate_phase4",
                    return_value=evaluation,
                ):
                    response = self.client.get("/api/ppo/validation")

        self.assertEqual(response.status_code, 200, response.text)

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
        self.assertEqual(
            set(data["evaluation"]["strategies"].keys()),
            {"Synthetic PPO", *benchmark_names},
        )
        self.assertEqual(
            data["evaluation"]["cost_assumptions"]["initial_capital"],
            data["starting_nav"],
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
            "not found",
            response.json()["detail"].lower(),
        )

    def test_missing_validation_report_uses_fresh_shared_evaluation(self):
        metrics = {
            "scenario_id": "validation-no-report",
            "seed": 42,
            "steps": 1,
            "initial_portfolio_value": 100000.0,
            "final_portfolio_value": 101000.0,
            "episode_reward": 0.01,
            "gross_turnover_sum": 0.2,
            "one_way_turnover_sum": 0.1,
            "transaction_fees_sum": 10.0,
            "slippage_cost_sum": 5.0,
            "transaction_cost_sum": 15.0,
            "ending_weights": {"SPY": 0.6, "CASH": 0.4},
            "terminated": True,
            "truncated": False,
        }
        strategies = {"Synthetic PPO": metrics, "Equal-weight": {"final_portfolio_value": 99000.0}}
        evaluation = {
            "evaluation_type": "synthetic-only",
            "period": {"evaluation_sessions": 21},
            "cost_assumptions": {"initial_capital": 100000.0},
            "ppo": metrics,
            "strategies": strategies,
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            (Path(temp_dir) / "model.zip").write_bytes(b"mock-model")
            with patch.dict(os.environ, {"PPO_ARTIFACT_DIR": temp_dir}):
                with patch(
                    "src.rl.ppo_agent.build_ppo_environments",
                    return_value=SimpleNamespace(validation_env=object()),
                ), patch(
                    "src.rl.ppo_agent.load_ppo_model",
                    return_value=object(),
                ), patch(
                    "src.rl.phase4_evaluation.evaluate_phase4",
                    return_value=evaluation,
                ):
                    response = self.client.get("/api/ppo/validation")

        self.assertEqual(response.status_code, 200, response.text)
        data = response.json()
        self.assertEqual(data["baselines"], {})
        self.assertEqual(data["baseline_source"], "shared_evaluation_engine")
        self.assertEqual(data["evaluation"]["strategies"], strategies)

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
