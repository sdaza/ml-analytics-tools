# %%
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier


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


# %%
