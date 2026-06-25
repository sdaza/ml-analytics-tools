# %%
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ModelManager depends on the optional [modeling] extra (mlflow, scikit-learn);
# skip the whole module when those deps aren't installed instead of erroring.
pytest.importorskip("sklearn")
pytest.importorskip("mlflow")

from sklearn.ensemble import RandomForestClassifier  # noqa: E402

import ml_analytics.model_manager as model_manager_module  # noqa: E402


@pytest.fixture
def mock_model_manager():
    with patch("ml_analytics.model_manager.ModelManager") as MockModelManager:
        mock_instance = MockModelManager.return_value
        mock_instance.log_model = MagicMock()
        mock_instance.set_model_alias = MagicMock()
        mock_instance.load_latest_model = MagicMock()
        mock_instance.loadl_model = MagicMock()
        yield mock_instance


def test_model_manager_operations(mock_model_manager):
    # Use the mocked instance directly
    model_manager = mock_model_manager

    # Mock input data
    input_data = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})
    predictions = np.array([0, 1, 0])

    # Train a sample model
    model = RandomForestClassifier()
    model.fit(input_data, predictions)

    # Test log_model
    model_manager.log_model(
        model=model,
        input_data=input_data,
        predictions=predictions,
        flavor="sklearn",
        description="A RandomForest model for classification",
        tags={"status": "testing"},
        register_model=False,
    )
    model_manager.log_model.assert_called_once()

    # Test set_model_alias
    model_manager.set_model_alias(alias="test", version=1)
    model_manager.set_model_alias.assert_called_once_with(alias="test", version=1)

    # Test get_latest_model
    model_manager.load_latest_model.return_value = model
    latest_model = model_manager.load_latest_model()
    assert latest_model.predict(input_data).tolist() == predictions.tolist()

    # Test load_model
    model_manager.load_model.return_value = model
    alias_model = model_manager.load_model(alias="test")
    assert alias_model.predict(input_data).tolist() == predictions.tolist()

    model_manager.load_model.return_value = model
    version_model = model_manager.load_model(version=1)
    assert version_model.predict(input_data).tolist() == predictions.tolist()


def _build_model_manager_with_patched_mlflow(monkeypatch, **kwargs):
    monkeypatch.delenv("MLFLOW_REGISTRY_URI", raising=False)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.delenv("MLFLOW_WORKSPACE", raising=False)

    patches = [
        patch("ml_analytics.model_manager.get_credential_value", side_effect=KeyError),
        patch("ml_analytics.model_manager.AuthServiceClient"),
        patch("ml_analytics.model_manager.MlflowClient"),
        patch("ml_analytics.model_manager.mlflow.set_tracking_uri"),
        patch("ml_analytics.model_manager.mlflow.set_registry_uri"),
        patch("ml_analytics.model_manager.ModelManager._setup_experiment"),
    ]

    started = [p.start() for p in patches]
    try:
        manager = model_manager_module.ModelManager(user="user@example.com", **kwargs)
        return manager, {
            "set_tracking_uri": started[3],
            "set_registry_uri": started[4],
        }
    finally:
        for p in reversed(patches):
            p.stop()


def test_model_manager_sets_databricks_uc_for_uc_model_name(monkeypatch):
    manager, mocks = _build_model_manager_with_patched_mlflow(
        monkeypatch,
        model_name="dev.ml_models.churn_model",
        tracking_uri="databricks",
    )

    assert manager.registry_uri == "databricks-uc"
    mocks["set_tracking_uri"].assert_called_once_with("databricks")
    mocks["set_registry_uri"].assert_called_once_with("databricks-uc")


def test_model_manager_does_not_auto_set_uc_for_workspace_model_name(monkeypatch):
    manager, mocks = _build_model_manager_with_patched_mlflow(
        monkeypatch,
        model_name="churn_model",
        tracking_uri="databricks",
    )

    assert manager.registry_uri is None
    mocks["set_registry_uri"].assert_not_called()


def test_model_manager_use_databricks_uc_forces_registry_uri(monkeypatch):
    manager, mocks = _build_model_manager_with_patched_mlflow(
        monkeypatch,
        model_name="churn_model",
        tracking_uri="https://adb-123.cloud.databricks.com",
        use_databricks_uc=True,
    )

    assert manager.registry_uri == "databricks-uc"
    mocks["set_registry_uri"].assert_called_once_with("databricks-uc")


def test_model_manager_explicit_registry_uri_wins(monkeypatch):
    manager, mocks = _build_model_manager_with_patched_mlflow(
        monkeypatch,
        model_name="dev.ml_models.churn_model",
        tracking_uri="databricks",
        registry_uri="databricks",
        use_databricks_uc=True,
    )

    assert manager.registry_uri == "databricks"
    mocks["set_registry_uri"].assert_called_once_with("databricks")


@pytest.mark.parametrize(
    ("model_name", "expected"),
    [
        ("dev.ml_models.churn_model", True),
        ("churn_model", False),
        ("dev..churn_model", False),
        (".dev.ml_models.churn_model", False),
        ("hive_metastore.default.churn_model", False),
    ],
)
def test_model_manager_detects_exact_unity_catalog_model_names(model_name, expected):
    assert model_manager_module.ModelManager._is_unity_catalog_model_name(model_name) is expected


# %%
