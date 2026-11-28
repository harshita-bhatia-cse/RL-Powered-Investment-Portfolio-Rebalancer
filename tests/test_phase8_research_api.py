import asyncio
import os
import unittest
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError

from src.api.analytics_api import (
    account_database_unavailable_response,
    app as application,
)
from src.api.phase8_research_api import router
from src.auth.dependencies import get_current_user
from src.auth.security import create_access_token
from src.database.session import get_db
from src.services.phase7_2_research_store import Phase72ResearchStoreError


_DIGEST = "a" * 64
_SEEDS = (101, 202, 303, 404, 505)
_CONFIGURATION_METRICS = {
    configuration: {
        "mean_validation_sharpe": 1.0,
        "mean_maximum_drawdown": -0.1,
        "mean_maximum_drawdown_magnitude": 0.1,
        "mean_turnover": 0.2,
        "seed_aggregates": [
            {
                "training_seed": seed,
                "mean_validation_sharpe": 1.0,
                "mean_maximum_drawdown": -0.1,
                "mean_maximum_drawdown_magnitude": 0.1,
                "mean_turnover": 0.2,
            }
            for seed in _SEEDS
        ],
    }
    for configuration in ("A", "B", "C")
}
_STATUS = {
    "format": "phase72-run-manifest-v2",
    "manifest_status": "test_complete",
    "protocol_id": "phase7_2_synthetic_ppo_comparison",
    "protocol_version": 1,
    "protocol_sha256": _DIGEST,
    "synthetic_only": True,
    "run_count": 15,
    "trained_run_count": 15,
    "validation_result_count": 300,
    "final_test_result_count": 100,
    "selected_configuration": "A",
    "final_test_scenario_count": 100,
    "integrity_verified": True,
}
_SELECTION = {
    "format": "phase72-selection-v1",
    "protocol_id": "phase7_2_synthetic_ppo_comparison",
    "protocol_version": 1,
    "protocol_sha256": _DIGEST,
    "selected_configuration": "A",
    "selection_sha256": _DIGEST,
    "selection_rule": {
        "primary": "highest mean validation Sharpe ratio",
        "aggregation": "mean over 20 roots per seed, then mean over all five seeds",
        "tie_breakers": [
            "lowest mean maximum drawdown magnitude",
            "lowest mean portfolio turnover",
            "lexicographically smallest configuration id",
        ],
        "drawdown_note": (
            "The evaluation metric is non-positive; minimize its absolute "
            "magnitude to prefer a smaller drawdown."
        ),
    },
    "selection_tolerance": 1e-12,
    "training_seeds_retained": list(_SEEDS),
    "configuration_metrics": _CONFIGURATION_METRICS,
    "tie_breaker_outcomes": {
        "primary_metric": {
            "best_value": 1.0,
            "candidates_within_tolerance": ["A"],
        },
        "drawdown_tie_breaker": {
            "best_magnitude": 0.1,
            "candidates_within_tolerance": ["A"],
        },
        "turnover_tie_breaker": {
            "best_value": 0.2,
            "candidates_within_tolerance": ["A"],
        },
        "configuration_id_tie_breaker": {
            "selected": "A",
            "remaining_candidates": ["A"],
        },
    },
}
_FINAL_TEST_OVERALL = {
    "synthetic_only": True,
    "selected_configuration": "A",
    "scenario_count": 100,
    "seed_count": 5,
    "overall_aggregate": {
        "mean_maximum_drawdown": -0.1,
        "mean_sharpe_ratio": 1.0,
        "mean_turnover": 0.2,
        "scenario_count": 100,
    },
}
_FINAL_TEST_SEEDS = [
    {
        "training_seed": seed,
        "scenario_count": 20,
        "mean_maximum_drawdown": -0.1,
        "mean_sharpe_ratio": 1.0,
        "mean_turnover": 0.2,
    }
    for seed in _SEEDS
]
_RESPONSES = {
    "read_status": _STATUS,
    "read_selection": _SELECTION,
    "read_validation_aggregates": {
        "synthetic_only": True,
        "configuration_metrics": _CONFIGURATION_METRICS,
    },
    "read_final_test_overall": _FINAL_TEST_OVERALL,
    "read_final_test_seed_aggregates": _FINAL_TEST_SEEDS,
}
_ENDPOINTS = (
    "/status",
    "/selection",
    "/validation",
    "/final-test",
    "/final-test/seeds",
    "/integrity",
)
_BASE_PATH = "/api/research/phase7-2"
_GENERIC_503_DETAIL = (
    "Stored synthetic Phase 7.2 research artifacts are unavailable "
    "or could not be verified."
)


class FakeResearchStore:
    def __init__(self, events=None, responses=None):
        self.events = events if events is not None else []
        self.responses = responses if responses is not None else _RESPONSES
        self.events.append("construct")

    def verify(self):
        self.events.append("verify")
        return deepcopy(_STATUS)

    def _read(self, method_name):
        self.events.append(method_name)
        return deepcopy(self.responses[method_name])

    def read_status(self):
        return self._read("read_status")

    def read_selection(self):
        return self._read("read_selection")

    def read_validation_aggregates(self):
        return self._read("read_validation_aggregates")

    def read_final_test_overall(self):
        return self._read("read_final_test_overall")

    def read_final_test_seed_aggregates(self):
        return self._read("read_final_test_seed_aggregates")


class Phase8ResearchAPITests(unittest.TestCase):
    def setUp(self):
        self.app = FastAPI()
        self.app.include_router(router)
        self.app.dependency_overrides[get_db] = lambda: None
        self.client = TestClient(self.app)

    def tearDown(self):
        self.client.close()
        self.app.dependency_overrides.clear()

    def authenticate(self):
        self.app.dependency_overrides[get_current_user] = (
            lambda: SimpleNamespace(id="test-user")
        )

    def assert_generic_503(self, response, sentinel):
        self.assertEqual(response.status_code, 503, response.text)
        self.assertEqual(response.json(), {"detail": _GENERIC_503_DETAIL})
        self.assertNotIn(sentinel, response.text)
        self.assertNotIn("C:\\", response.text)
        self.assertNotIn("/private/", response.text)

    def test_all_endpoints_require_authentication(self):
        with patch("src.api.phase8_research_api.Phase72ResearchStore") as factory:
            for endpoint in _ENDPOINTS:
                with self.subTest(endpoint=endpoint):
                    response = self.client.get(_BASE_PATH + endpoint)
                    self.assertEqual(response.status_code, 401, response.text)
            factory.assert_not_called()

    def test_all_endpoints_validate_success_responses_and_provenance(self):
        self.authenticate()
        instances = []

        def make_store():
            instance = FakeResearchStore()
            instances.append(instance)
            return instance

        expected_reads = {
            "/status": ("read_status", "protocol_id"),
            "/selection": ("read_selection", "selected_configuration"),
            "/validation": ("read_validation_aggregates", "configuration_metrics"),
            "/final-test": ("read_final_test_overall", "scenario_count"),
            "/final-test/seeds": (
                "read_final_test_seed_aggregates",
                "seed_aggregates",
            ),
            "/integrity": (None, "integrity_verified"),
        }
        with patch(
            "src.api.phase8_research_api.Phase72ResearchStore",
            side_effect=make_store,
        ) as factory:
            for endpoint in _ENDPOINTS:
                with self.subTest(endpoint=endpoint):
                    response = self.client.get(_BASE_PATH + endpoint)
                    self.assertEqual(response.status_code, 200, response.text)
                    body = response.json()
                    self.assertTrue(body["synthetic_only"])
                    self.assertEqual(
                        body["provenance"],
                        "stored synthetic Phase 7.2 experiment results",
                    )
                    read_name, expected_field = expected_reads[endpoint]
                    self.assertIn(expected_field, body)
                    if endpoint == "/final-test/seeds":
                        self.assertEqual(
                            body["seed_aggregates"],
                            _FINAL_TEST_SEEDS,
                        )
                    expected_events = ["construct", "verify"]
                    if read_name is not None:
                        expected_events.append(read_name)
                    self.assertEqual(instances[-1].events, expected_events)

        self.assertEqual(factory.call_count, len(_ENDPOINTS))
        self.assertEqual(len({id(instance) for instance in instances}), len(_ENDPOINTS))

    def test_store_construction_and_verification_failures_return_generic_503(self):
        self.authenticate()
        constructor_sentinel = "CONSTRUCTOR_FAILURE_SENTINEL"
        with patch(
            "src.api.phase8_research_api.Phase72ResearchStore",
            side_effect=Phase72ResearchStoreError(constructor_sentinel),
        ):
            response = self.client.get(_BASE_PATH + "/status")
        self.assert_generic_503(response, constructor_sentinel)

        verify_sentinel = "VERIFY_FAILURE_SENTINEL"

        class VerifyFailureStore:
            def verify(self):
                raise Phase72ResearchStoreError(verify_sentinel)

        with patch(
            "src.api.phase8_research_api.Phase72ResearchStore",
            return_value=VerifyFailureStore(),
        ):
            response = self.client.get(_BASE_PATH + "/integrity")
        self.assert_generic_503(response, verify_sentinel)

    def test_store_read_failures_return_generic_503(self):
        self.authenticate()
        sentinel = "READ_FAILURE_SENTINEL"

        class ReadFailureStore(FakeResearchStore):
            def read_status(self):
                raise Phase72ResearchStoreError(sentinel)

        with patch(
            "src.api.phase8_research_api.Phase72ResearchStore",
            return_value=ReadFailureStore(),
        ):
            response = self.client.get(_BASE_PATH + "/status")

        self.assert_generic_503(response, sentinel)

    def test_unicode_and_os_errors_return_generic_503(self):
        self.authenticate()
        cases = (
            (
                "unicode",
                UnicodeError("UNICODE_FAILURE_SENTINEL"),
                "UNICODE_FAILURE_SENTINEL",
            ),
            (
                "filesystem",
                OSError("C:\\private\\ARTIFACT_PATH_SENTINEL\\manifest.json"),
                "C:\\private\\ARTIFACT_PATH_SENTINEL\\manifest.json",
            ),
        )
        for label, error, sentinel in cases:
            with self.subTest(error=label):
                with patch(
                    "src.api.phase8_research_api.Phase72ResearchStore",
                    side_effect=error,
                ):
                    response = self.client.get(_BASE_PATH + "/status")
                self.assert_generic_503(response, sentinel)

    def test_response_validation_failure_returns_generic_503(self):
        self.authenticate()
        sentinel = "RESPONSE_VALIDATION_SENTINEL"
        invalid_status = deepcopy(_STATUS)
        invalid_status["protocol_version"] = sentinel

        class InvalidResponseStore(FakeResearchStore):
            def read_status(self):
                self.events.append("read_status")
                return deepcopy(invalid_status)

        with patch(
            "src.api.phase8_research_api.Phase72ResearchStore",
            return_value=InvalidResponseStore(),
        ):
            response = self.client.get(_BASE_PATH + "/status")

        self.assert_generic_503(response, sentinel)

    def test_repeated_requests_create_and_verify_fresh_stores(self):
        self.authenticate()
        instances = []

        def make_store():
            instance = FakeResearchStore()
            instances.append(instance)
            return instance

        with patch(
            "src.api.phase8_research_api.Phase72ResearchStore",
            side_effect=make_store,
        ) as factory:
            first = self.client.get(_BASE_PATH + "/selection")
            second = self.client.get(_BASE_PATH + "/selection")

        self.assertEqual(first.status_code, 200, first.text)
        self.assertEqual(second.status_code, 200, second.text)
        self.assertEqual(factory.call_count, 2)
        self.assertIsNot(instances[0], instances[1])
        self.assertEqual(
            [instance.events for instance in instances],
            [
                ["construct", "verify", "read_selection"],
                ["construct", "verify", "read_selection"],
            ],
        )


class DatabaseMiddlewareIntegrationTests(unittest.TestCase):
    JWT_SECRET = "phase8-middleware-test-secret-long-enough-123456"

    def setUp(self):
        self.previous_database_ready = getattr(
            application.state,
            "account_database_ready",
            False,
        )
        application.state.account_database_ready = True
        self.secret_patch = patch.dict(
            os.environ,
            {"AUTH_JWT_SECRET": self.JWT_SECRET},
        )
        self.secret_patch.start()
        self.token, _expires_in = create_access_token("phase8-test-user")
        self.client = TestClient(application, raise_server_exceptions=False)
        self.previous_overrides = application.dependency_overrides.copy()
        application.dependency_overrides[get_db] = lambda: _FakeAuthSession()

    def tearDown(self):
        self.client.close()
        application.dependency_overrides.clear()
        application.dependency_overrides.update(self.previous_overrides)
        application.state.account_database_ready = self.previous_database_ready
        self.secret_patch.stop()

    @property
    def auth_headers(self):
        return {"Authorization": f"Bearer {self.token}"}

    def assert_persistence_unavailable(self, response, *sentinels):
        self.assertEqual(response.status_code, 503, response.text)
        self.assertEqual(
            response.json(),
            {"detail": "Account persistence is unavailable"},
        )
        for sentinel in sentinels:
            self.assertNotIn(sentinel, response.text)

    def test_operational_database_errors_on_research_account_and_portfolio_paths(self):
        cases = (
            "/api/research/phase7-2/status",
            "/api/auth/me",
            "/api/portfolios",
        )
        for path in cases:
            with self.subTest(path=path):
                def unavailable_database():
                    raise OperationalError(
                        "SELECT credential_column FROM private_table",
                        {"password": "DB_CREDENTIAL_SENTINEL"},
                        RuntimeError("DB_EXCEPTION_SENTINEL C:\\private\\db.sqlite"),
                    )

                application.dependency_overrides[get_db] = unavailable_database
                response = self.client.get(path, headers=self.auth_headers)
                self.assert_persistence_unavailable(
                    response,
                    "SELECT",
                    "DB_CREDENTIAL_SENTINEL",
                    "DB_EXCEPTION_SENTINEL",
                    "C:\\private\\db.sqlite",
                )

    def test_operational_database_errors_on_unrelated_paths_are_not_reclassified(self):
        paths = (
            "/api/backtest/test-operational-error",
            "/api/test-operational-error",
        )
        added_paths = []
        try:
            for path in paths:
                def raise_database_error():
                    raise OperationalError(
                        "SELECT isolated_test",
                        {},
                        RuntimeError("UNRELATED_DB_ERROR_SENTINEL"),
                    )

                application.add_api_route(
                    path,
                    raise_database_error,
                    methods=["GET"],
                )
                added_paths.append(path)

            for path in paths:
                with self.subTest(path=path):
                    response = self.client.get(path)
                    self.assertEqual(response.status_code, 500)
                    self.assertNotIn(
                        "Account persistence is unavailable",
                        response.text,
                    )
        finally:
            application.router.routes[:] = [
                route
                for route in application.router.routes
                if getattr(route, "path", None) not in added_paths
            ]

    def test_research_route_when_account_database_is_not_ready(self):
        previous_ready = application.state.account_database_ready
        previous_overrides = application.dependency_overrides.copy()
        try:
            application.state.account_database_ready = False
            application.dependency_overrides.pop(get_db, None)
            with patch(
                "src.database.session.SessionLocal",
                side_effect=AssertionError("database session must not be opened"),
            ) as session_factory:
                response = self.client.get(
                    _BASE_PATH + "/status",
                    headers=self.auth_headers,
                )
            session_factory.assert_not_called()
        finally:
            application.dependency_overrides.clear()
            application.dependency_overrides.update(previous_overrides)
            application.state.account_database_ready = previous_ready

        self.assertEqual(response.status_code, 503, response.text)
        self.assertEqual(
            response.json(),
            {"detail": "Account persistence is unavailable"},
        )
        for sentinel in (
            "C:\\private\\",
            "SELECT",
            "AUTH_JWT_SECRET",
            "database session must not be opened",
        ):
            self.assertNotIn(sentinel, response.text)

    def test_actual_backtest_health_route_is_outside_account_database_path(self):
        fake_health = {
            "available": True,
            "dataset_validation": {"valid": True, "partitions": {}},
            "supported_strategies": [],
            "corporate_actions_verified": False,
            "limitation": "Stubbed health result; no data was loaded.",
        }
        with patch(
            "src.api.analytics_api.HistoricalBacktestService.health",
            return_value=fake_health,
        ):
            response = self.client.get("/api/backtest/health")

        self.assertEqual(response.status_code, 200, response.text)
        self.assertTrue(response.json()["available"])
        self.assertNotIn(
            "Account persistence is unavailable",
            response.text,
        )

        async def raise_operational_error(_request):
            raise OperationalError(
                "SELECT path_match_check",
                {},
                RuntimeError("PATH_MATCH_SENTINEL"),
            )

        def middleware_request(path):
            scope = {
                "type": "http",
                "asgi": {"version": "3.0"},
                "http_version": "1.1",
                "method": "GET",
                "scheme": "http",
                "path": path,
                "raw_path": path.encode("ascii"),
                "query_string": b"",
                "headers": [],
                "server": ("testserver", 80),
                "client": ("testclient", 123),
            }
            return Request(scope)

        async def invoke_middleware(path):
            return await account_database_unavailable_response(
                middleware_request(path),
                raise_operational_error,
            )

        with self.assertRaises(OperationalError):
            asyncio.run(invoke_middleware("/api/backtest/health"))

        research_response = asyncio.run(
            invoke_middleware("/api/research/phase7-2/status")
        )
        self.assertEqual(research_response.status_code, 503)
        self.assertEqual(
            research_response.body,
            b'{"detail":"Account persistence is unavailable"}',
        )

    def test_artifact_errors_keep_the_research_generic_response(self):
        application.dependency_overrides[get_db] = lambda: _FakeAuthSession()
        sentinel = "ARTIFACT_ERROR_SENTINEL"
        with patch(
            "src.api.phase8_research_api.Phase72ResearchStore",
            side_effect=Phase72ResearchStoreError(
                f"{sentinel} C:\\private\\phase72\\manifest.json"
            ),
        ):
            response = self.client.get(
                _BASE_PATH + "/status",
                headers=self.auth_headers,
            )

        self.assertEqual(response.status_code, 503, response.text)
        self.assertEqual(response.json(), {"detail": _GENERIC_503_DETAIL})
        self.assertNotIn(sentinel, response.text)
        self.assertNotIn("C:\\private\\phase72", response.text)
        self.assertNotIn("Account persistence is unavailable", response.text)

    def test_missing_and_invalid_credentials_remain_401(self):
        missing = self.client.get(_BASE_PATH + "/status")
        invalid = self.client.get(
            _BASE_PATH + "/status",
            headers={"Authorization": "Bearer not-a-valid-token"},
        )
        self.assertEqual(missing.status_code, 401, missing.text)
        self.assertEqual(invalid.status_code, 401, invalid.text)


class _FakeAuthSession:
    def get(self, _model, identity):
        return SimpleNamespace(id=identity)


if __name__ == "__main__":
    unittest.main()
