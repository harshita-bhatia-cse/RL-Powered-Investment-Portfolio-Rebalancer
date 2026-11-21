import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import gymnasium as gym

from src.rl.phase4_evaluation import (
    calculate_portfolio_metrics,
    evaluate_phase4,
    save_phase4_diagnostics,
    save_phase4_csv,
    save_phase4_results,
    summarize_allocation_diagnostics,
)
from src.rl.phase4_training import (
    HistoricalDataAuthorization,
    HistoricalDataAuthorizationError,
    TrainingOrchestrator,
)
from src.rl.ppo_agent import EvaluationStep, create_ppo_agent
from src.rl.ppo_config import PPOConfig, TrainingOptions
from src.rl.synthetic_data import SyntheticMarketConfig
from src.rl.ppo_agent import build_ppo_environments
from src.rl.benchmarks import evaluate_synthetic_benchmarks
from src.rl.environment.portfolio_env import ASSET_UNIVERSE


def make_evaluation_step(
    *,
    action,
    requested_weights,
    applied_weights,
    nav,
    gross_turnover,
    transaction_fees,
    slippage_cost,
):
    return EvaluationStep(
        observation=np.zeros(43, dtype=np.float32),
        action=np.asarray(action, dtype=np.float32),
        reward=0.0,
        nav=nav,
        weights=dict(applied_weights),
        gross_turnover=gross_turnover,
        one_way_turnover=gross_turnover / 2.0,
        transaction_fees=transaction_fees,
        slippage_cost=slippage_cost,
        transaction_cost=transaction_fees + slippage_cost,
        terminated=False,
        truncated=False,
        requested_weights=dict(requested_weights),
        applied_weights=dict(applied_weights),
    )


class Phase4WorkflowTests(unittest.TestCase):
    def setUp(self):
        try:
            import stable_baselines3
            import torch
        except ImportError:
            stable_baselines3 = None
        self.sb3_available = stable_baselines3 is not None

    def test_portfolio_metrics_match_known_nav_series(self):
        metrics = calculate_portfolio_metrics(
            [110.0, 99.0, 108.0],
            initial_capital=100.0,
            total_transaction_costs=1.25,
            portfolio_turnover=0.5,
        )
        self.assertAlmostEqual(metrics["final_portfolio_value"], 108.0)
        self.assertAlmostEqual(metrics["cumulative_return"], 0.08)
        self.assertAlmostEqual(metrics["maximum_drawdown"], -0.10)
        self.assertEqual(metrics["total_transaction_costs"], 1.25)
        self.assertEqual(metrics["portfolio_turnover"], 0.5)
        daily_returns = np.array([0.1, -0.1, 108.0 / 99.0 - 1.0])
        daily_volatility = np.std(daily_returns, ddof=1)
        self.assertAlmostEqual(
            metrics["annualized_return"],
            (1.08 ** (252.0 / 3.0)) - 1.0,
        )
        self.assertAlmostEqual(
            metrics["annualized_volatility"],
            daily_volatility * np.sqrt(252.0),
        )
        self.assertAlmostEqual(
            metrics["sharpe_ratio"],
            np.mean(daily_returns) / daily_volatility * np.sqrt(252.0),
        )
        downside_deviation = np.sqrt(np.mean(np.minimum(daily_returns, 0.0) ** 2))
        self.assertAlmostEqual(
            metrics["sortino_ratio"],
            np.mean(daily_returns) * 252.0
            / (downside_deviation * np.sqrt(252.0)),
        )

    def test_all_synthetic_benchmarks_use_the_same_window_and_cost(self):
        returns = np.zeros((130, 6), dtype=np.float64)
        returns[:, 0] = np.tile([0.01, -0.005], 65)
        returns[:, 1] = np.tile([-0.004, 0.008], 65)
        returns[:, 2] = 0.001
        frame = evaluate_synthetic_benchmarks(
            returns,
            initial_capital=100_000.0,
            transaction_cost_rate=0.001,
            slippage_bps=5.0,
            period_sessions=21,
            warmup_sessions=63,
        )
        self.assertEqual(
            set(frame.index),
            {
                "SPY buy-and-hold",
                "Equal-weight",
                "Monthly rebalancing",
                "Threshold rebalancing",
            },
        )
        for row in frame.to_dict("index").values():
            self.assertTrue(np.isfinite(list(row.values())).all())
            self.assertGreater(row["final_portfolio_value"], 0.0)
            self.assertGreaterEqual(row["portfolio_turnover"], 0.0)
            self.assertGreaterEqual(row["total_transaction_costs"], 0.0)
        self.assertNotAlmostEqual(
            frame.loc["Equal-weight", "portfolio_turnover"],
            frame.loc["Monthly rebalancing", "portfolio_turnover"],
        )
        self.assertEqual(
            frame["initial_portfolio_value"].unique().tolist(),
            [100_000.0],
        )

    def test_allocation_diagnostics_summarize_applied_weights_and_costs(self):
        names = (*ASSET_UNIVERSE, "CASH")
        requested_one = dict(zip(names, [0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.3]))
        applied_one = dict(zip(names, [0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.3]))
        requested_two = dict(zip(names, [0.3, 0.0, 0.1, 0.0, 0.2, 0.1, 0.3]))
        applied_two = dict(zip(names, [0.3, 0.0, 0.1, 0.0, 0.2, 0.1, 0.3]))
        steps = [
            make_evaluation_step(
                action=[0.1] * 7,
                requested_weights=requested_one,
                applied_weights=applied_one,
                nav=101_000.0,
                gross_turnover=0.6,
                transaction_fees=10.0,
                slippage_cost=5.0,
            ),
            make_evaluation_step(
                action=[0.2] * 7,
                requested_weights=requested_two,
                applied_weights=applied_two,
                nav=102_000.0,
                gross_turnover=0.4,
                transaction_fees=20.0,
                slippage_cost=10.0,
            ),
        ]

        diagnostics = summarize_allocation_diagnostics(steps)
        summary = diagnostics["summary"]

        self.assertEqual(summary["decision_count"], 2)
        self.assertAlmostEqual(summary["average_cash_allocation"], 0.3)
        self.assertAlmostEqual(summary["minimum_cash_allocation"], 0.3)
        self.assertAlmostEqual(summary["maximum_cash_allocation"], 0.3)
        self.assertAlmostEqual(summary["average_risky_asset_allocation"], 0.7)
        self.assertAlmostEqual(summary["average_gross_turnover"], 0.5)
        self.assertAlmostEqual(summary["average_one_way_turnover"], 0.25)
        self.assertAlmostEqual(summary["total_transaction_fees"], 30.0)
        self.assertAlmostEqual(summary["total_slippage_costs"], 15.0)
        self.assertAlmostEqual(summary["total_transaction_costs"], 45.0)
        self.assertAlmostEqual(
            summary["total_transaction_costs"],
            summary["total_transaction_fees"] + summary["total_slippage_costs"],
        )
        self.assertAlmostEqual(
            summary["asset_allocation_distribution"]["SPY"]["average"], 0.25
        )
        np.testing.assert_allclose(
            diagnostics["rebalance_history"][0]["raw_action"],
            [0.1] * 7,
        )
        self.assertEqual(
            diagnostics["rebalance_history"][1]["requested_target_weights"],
            requested_two,
        )
        self.assertEqual(
            diagnostics["rebalance_history"][1]["applied_portfolio_weights"],
            applied_two,
        )
        self.assertEqual(
            diagnostics["rebalance_history"][1]["portfolio_nav"], 102_000.0
        )

    def test_allocation_diagnostics_reject_invalid_weights(self):
        names = (*ASSET_UNIVERSE, "CASH")
        invalid_weights = dict(zip(names, [0.4, 0.4, 0.1, 0.1, 0.1, 0.0, 0.1]))
        step = make_evaluation_step(
            action=[0.0] * 7,
            requested_weights=invalid_weights,
            applied_weights=invalid_weights,
            nav=100_000.0,
            gross_turnover=0.0,
            transaction_fees=0.0,
            slippage_cost=0.0,
        )
        with self.assertRaisesRegex(ValueError, "sum to one"):
            summarize_allocation_diagnostics([step])

    def test_diagnostics_save_does_not_replace_baseline_evaluation(self):
        diagnostics = {
            "summary": {"decision_count": 1},
            "rebalance_history": [{"decision_number": 1}],
        }
        evaluation = {
            "ppo": {
                "source_type": "synthetic",
                "synthetic": True,
                "scenario_id": "validation-example",
                "seed": 42,
                "allocation_diagnostics": diagnostics,
            }
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory)
            baseline_path = output / "evaluation.json"
            baseline_path.write_text('{"baseline":"keep"}', encoding="utf-8")

            diagnostics_path = save_phase4_diagnostics(output, evaluation)

            self.assertEqual(
                baseline_path.read_text(encoding="utf-8"),
                '{"baseline":"keep"}',
            )
            saved = json.loads(diagnostics_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["scenario_id"], "validation-example")
            self.assertEqual(saved["summary"], diagnostics["summary"])

    def test_historical_training_authorization_safeguard(self):
        class HistoricalEnv(gym.Env):
            metadata = {"synthetic": False, "source_type": "historical"}
            data_metadata = {"synthetic": False, "source_type": "historical"}

            @property
            def unwrapped(self):
                return self

        orchestrator = TrainingOrchestrator(
            config=PPOConfig(),
            seed=7,
            output_dir="unused",
        )
        with self.assertRaisesRegex(HistoricalDataAuthorizationError, "blocked"):
            orchestrator._guard_environment(HistoricalEnv())

        authorized = TrainingOrchestrator(
            config=PPOConfig(),
            seed=7,
            output_dir="unused",
            historical_authorization=HistoricalDataAuthorization(
                authorized=True,
                reference="LIC-TEST",
            ),
        )
        with self.assertRaisesRegex(HistoricalDataAuthorizationError, "not enabled"):
            authorized._guard_environment(HistoricalEnv())

    def test_synthetic_training_guard_accepts_synthetic_environment(self):
        bundle = build_ppo_environments(
            SyntheticMarketConfig(sessions=130), seed=11
        )
        orchestrator = TrainingOrchestrator(
            config=PPOConfig(),
            seed=11,
            output_dir="unused",
        )
        orchestrator._guard_environment(bundle.train_env)

    def test_phase4_evaluation_and_structured_output(self):
        if not self.sb3_available:
            raise unittest.SkipTest("Stable-Baselines3 and Torch are required")
        bundle = build_ppo_environments(
            SyntheticMarketConfig(sessions=130), seed=29
        )
        config = PPOConfig(
            n_steps=8,
            batch_size=4,
            net_arch=(16, 16),
            seed=29,
            device="cpu",
        )
        model = create_ppo_agent(bundle.train_env, config)
        evaluation = evaluate_phase4(
            model,
            bundle.test_env,
            initial_capital=50_000.0,
        )

        self.assertEqual(evaluation["evaluation_type"], "synthetic-only")
        self.assertEqual(
            set(evaluation["benchmarks"]),
            {
                "SPY buy-and-hold",
                "Equal-weight",
                "Monthly rebalancing",
                "Threshold rebalancing",
            },
        )
        self.assertTrue(evaluation["ppo"]["synthetic"])
        self.assertEqual(evaluation["cost_assumptions"]["initial_capital"], 50_000.0)
        self.assertEqual(evaluation["ppo"]["initial_portfolio_value"], 50_000.0)
        self.assertTrue(
            all(
                row["initial_portfolio_value"] == 50_000.0
                for row in evaluation["benchmarks"].values()
            )
        )
        self.assertEqual(len(evaluation["strategies"]), 5)
        ppo = evaluation["ppo"]
        diagnostics = ppo["allocation_diagnostics"]
        self.assertEqual(
            diagnostics["summary"]["decision_count"],
            ppo["steps"],
        )
        self.assertEqual(
            len(diagnostics["rebalance_history"]),
            ppo["steps"],
        )
        self.assertAlmostEqual(
            diagnostics["summary"]["total_transaction_costs"],
            ppo["total_transaction_costs"],
        )
        self.assertEqual(evaluation["strategies"]["Synthetic PPO"], ppo)
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory)
            experiment_path, evaluation_path = save_phase4_results(
                output,
                experiment={
                    "seed": 29,
                    "data_provenance": "synthetic",
                    "environment_type": "SyntheticPortfolioEnv",
                },
                evaluation=evaluation,
            )
            csv_path = save_phase4_csv(output, evaluation)
            self.assertTrue(experiment_path.is_file())
            self.assertTrue(evaluation_path.is_file())
            self.assertTrue(csv_path.is_file())
            loaded = json.loads(evaluation_path.read_text(encoding="utf-8"))
            self.assertEqual(loaded["evaluation_type"], "synthetic-only")
