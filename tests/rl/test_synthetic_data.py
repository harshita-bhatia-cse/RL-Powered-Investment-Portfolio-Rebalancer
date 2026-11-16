import unittest
from dataclasses import asdict, replace

import numpy as np

from src.rl.environment.synthetic_portfolio_env import SyntheticPortfolioEnv
from src.rl.synthetic_data import (
    GENERATED_SERIES,
    REGIME_NAMES,
    RegimeConfig,
    SyntheticMarketConfig,
    generate_synthetic_market,
    generate_synthetic_splits,
)


class SyntheticMarketDataTests(unittest.TestCase):
    def setUp(self):
        self.config = SyntheticMarketConfig(sessions=100)

    def test_dataset_shapes_order_metadata_and_environment_compatibility(self):
        dataset = generate_synthetic_market(self.config, seed=41)

        self.assertEqual(dataset.asset_returns.shape, (100, 6))
        self.assertEqual(dataset.benchmark_returns.shape, (100,))
        self.assertEqual(dataset.vix_levels.shape, (100,))
        self.assertEqual(dataset.asset_names, ("SPY", "QQQ", "IEF", "GLD", "XLU", "XLV"))
        self.assertEqual(dataset.series_names, GENERATED_SERIES)
        self.assertTrue(dataset.metadata["synthetic"])
        self.assertEqual(dataset.metadata["source_type"], "synthetic")
        self.assertEqual(dataset.metadata["seed"], 41)
        self.assertEqual(
            dataset.metadata["generation_config"], asdict(self.config)
        )

        env = SyntheticPortfolioEnv(**dataset.environment_kwargs())
        observation, info = env.reset(seed=1)
        self.assertEqual(observation.shape, (43,))
        self.assertTrue(env.observation_space.contains(observation))
        self.assertTrue(info["synthetic"])

    def test_all_generated_values_are_finite_and_valid_simple_returns(self):
        dataset = generate_synthetic_market(self.config, seed=42)

        self.assertTrue(np.isfinite(dataset.asset_returns).all())
        self.assertTrue(np.isfinite(dataset.benchmark_returns).all())
        self.assertTrue(np.isfinite(dataset.vix_levels).all())
        self.assertTrue(np.all(dataset.asset_returns > -1.0))
        self.assertTrue(np.all(dataset.benchmark_returns > -1.0))
        self.assertTrue(np.all(dataset.vix_levels > 0.0))
        self.assertFalse(dataset.asset_returns.flags.writeable)
        self.assertFalse(dataset.benchmark_returns.flags.writeable)
        self.assertFalse(dataset.vix_levels.flags.writeable)

    def test_same_configuration_and_seed_reproduce_paths_and_config_identity(self):
        first = generate_synthetic_market(self.config, seed=123)
        second = generate_synthetic_market(self.config, seed=123)

        np.testing.assert_array_equal(first.asset_returns, second.asset_returns)
        np.testing.assert_array_equal(first.benchmark_returns, second.benchmark_returns)
        np.testing.assert_array_equal(first.vix_levels, second.vix_levels)
        self.assertEqual(first.regime_labels, second.regime_labels)
        self.assertEqual(
            first.metadata["configuration_hash"],
            second.metadata["configuration_hash"],
        )
        self.assertEqual(first.seed, second.seed)

    def test_different_seeds_produce_different_scenarios(self):
        first = generate_synthetic_market(self.config, seed=123)
        second = generate_synthetic_market(self.config, seed=124)

        self.assertFalse(np.array_equal(first.asset_returns, second.asset_returns))
        self.assertNotEqual(first.regime_labels, second.regime_labels)

    def test_configured_regime_transitions_are_applied(self):
        transition = ((1.0, 0.0, 0.0),) * len(REGIME_NAMES)
        initial = (0.0, 1.0, 0.0)
        config = replace(
            self.config,
            transition_matrix=transition,
            initial_regime_probabilities=initial,
        )

        dataset = generate_synthetic_market(config, seed=9)

        self.assertEqual(dataset.regime_labels[0], "bear")
        self.assertEqual(set(dataset.regime_labels[1:]), {"bull"})

    def test_rejects_non_symmetric_non_unit_diagonal_and_non_psd_correlations(self):
        matrix = np.asarray(self.config.correlation_matrix).copy()
        matrix[0, 1] = 0.1
        asymmetric = replace(self.config, correlation_matrix=tuple(map(tuple, matrix)))
        with self.assertRaisesRegex(ValueError, "symmetric"):
            asymmetric.validate()

        matrix = np.asarray(self.config.correlation_matrix).copy()
        matrix[0, 0] = 0.9
        bad_diagonal = replace(self.config, correlation_matrix=tuple(map(tuple, matrix)))
        with self.assertRaisesRegex(ValueError, "unit diagonal"):
            bad_diagonal.validate()

        matrix = np.eye(len(GENERATED_SERIES))
        matrix[0, 1] = matrix[1, 0] = 1.1
        out_of_bounds = replace(
            self.config, correlation_matrix=tuple(map(tuple, matrix))
        )
        with self.assertRaisesRegex(ValueError, "between -1 and 1"):
            out_of_bounds.validate()

        matrix = np.eye(len(GENERATED_SERIES))
        matrix[0, 1] = matrix[1, 0] = 0.9
        matrix[0, 2] = matrix[2, 0] = 0.9
        matrix[1, 2] = matrix[2, 1] = -0.9
        non_psd = replace(self.config, correlation_matrix=tuple(map(tuple, matrix)))
        with self.assertRaisesRegex(ValueError, "positive semidefinite"):
            non_psd.validate()

    def test_rejects_invalid_regime_and_transition_parameters(self):
        bad_regime = replace(
            self.config.regimes[0],
            annualized_volatilities=(0.1,) * 6,
        )
        bad_config = replace(
            self.config,
            regimes=(bad_regime, *self.config.regimes[1:]),
        )
        with self.assertRaisesRegex(ValueError, "annualized_volatilities"):
            bad_config.validate()

        bad_transitions = replace(
            self.config,
            transition_matrix=((0.5, 0.5, 0.5), *self.config.transition_matrix[1:]),
        )
        with self.assertRaisesRegex(ValueError, "rows must sum to 1"):
            bad_transitions.validate()

    def test_split_scenarios_have_stable_distinct_seeds_and_no_shared_arrays(self):
        first = generate_synthetic_splits(self.config, seed=500)
        second = generate_synthetic_splits(self.config, seed=500)
        datasets = (first.train, first.validation, first.test)
        repeat_datasets = (second.train, second.validation, second.test)

        self.assertEqual(
            tuple(dataset.seed for dataset in datasets),
            tuple(dataset.seed for dataset in repeat_datasets),
        )
        self.assertEqual(len({dataset.seed for dataset in datasets}), 3)
        for dataset, repeated in zip(datasets, repeat_datasets):
            np.testing.assert_array_equal(dataset.asset_returns, repeated.asset_returns)
            self.assertEqual(dataset.metadata["scenario_id"], repeated.metadata["scenario_id"])
            self.assertEqual(
                dataset.metadata["configuration_hash"],
                first.train.metadata["configuration_hash"],
            )
        for left_index, left in enumerate(datasets):
            for right in datasets[left_index + 1 :]:
                self.assertFalse(np.shares_memory(left.asset_returns, right.asset_returns))
                self.assertFalse(np.array_equal(left.asset_returns, right.asset_returns))
                self.assertNotEqual(left.seed, right.seed)

    def test_split_session_override_preserves_other_configuration(self):
        splits = generate_synthetic_splits(self.config, seed=8, sessions=80)

        self.assertEqual(splits.train.asset_returns.shape, (80, 6))
        self.assertEqual(splits.validation.config.sessions, 80)
        self.assertEqual(splits.test.config.correlation_matrix, self.config.correlation_matrix)

    def test_dataset_rejects_invalid_metadata_and_seed(self):
        with self.assertRaisesRegex(ValueError, "between 0 and 2\\*\\*64"):
            generate_synthetic_market(self.config, seed=-1)
        with self.assertRaisesRegex(ValueError, "non-empty"):
            generate_synthetic_market(self.config, seed=1, scenario_id=" ")

        dataset = generate_synthetic_market(self.config, seed=1)
        with self.assertRaisesRegex(ValueError, "must identify synthetic"):
            replace(dataset, metadata={"source_type": "historical", "synthetic": False})

    def test_regime_specific_parameters_are_configurable(self):
        regimes = list(self.config.regimes)
        regimes[0] = replace(
            regimes[0],
            annualized_log_returns=(0.2,) * len(GENERATED_SERIES),
            annualized_volatilities=(0.01,) * len(GENERATED_SERIES),
        )
        config = replace(self.config, regimes=tuple(regimes))

        dataset = generate_synthetic_market(config, seed=2)

        self.assertEqual(dataset.config.regimes[0].annualized_log_returns, (0.2,) * 7)
        self.assertEqual(dataset.metadata["generation_config"]["regimes"][0]["name"], "bull")


if __name__ == "__main__":
    unittest.main()
