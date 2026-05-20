# %%

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import (
    auc as sk_auc,
)
from sklearn.metrics import (
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ml_analytics.model_tools import (
    get_balanced_accuracy,
    get_features,
    get_metrics,
    get_metrics_surv,
    get_performance,
    get_performance_surv,
    mcc_score,
    pr_auc_score,
    survival_mae,
)


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "cat_col1": ["a", "b", "a", "c", "b"],
            "cat_col2": pd.Categorical(["x", "y", "x", "z", "y"]),
            "num_col1": [1, 2, 3, 4, 5],
            "num_col2": [1.1, 2.2, 3.3, 4.4, 5.5],
            "target": [0, 1, 0, 1, 0],
        }
    )


@pytest.fixture
def classification_data():
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1, 0, 0, 0, 1, 1, 0])
    y_prob = np.array([0.1, 0.9, 0.6, 0.8, 0.2, 0.4, 0.3, 0.7, 0.85, 0.35])
    return y_true, y_pred, y_prob


@pytest.fixture
def survival_data():
    observed_time = np.array([10, 15, 20, 25, 30, 12, 18, 22, 28, 32])
    event_indicator = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1])  # 1 for event, 0 for censored
    predicted_time = np.array([12, 10, 22, 20, 35, 10, 20, 25, 26, 30])
    return observed_time, event_indicator, predicted_time


def test_get_features(sample_df):
    cat_feats, num_feats = get_features(sample_df, target_col="target")
    assert sorted(cat_feats) == sorted(["cat_col1", "cat_col2"])
    assert sorted(num_feats) == sorted(["num_col1", "num_col2"])  # target is int64, so it will be here

    cat_feats_only, num_feats_only = get_features(sample_df[["cat_col1", "cat_col2"]])
    assert sorted(cat_feats_only) == sorted(["cat_col1", "cat_col2"])
    assert num_feats_only == []

    cat_feats_only, num_feats_only = get_features(sample_df[["num_col1", "num_col2"]])
    assert cat_feats_only == []
    assert sorted(num_feats_only) == sorted(["num_col1", "num_col2"])


def test_get_balanced_accuracy(classification_data):
    y_true, y_pred, _ = classification_data
    # Manually calculate for y_true, y_pred
    # recall_0 = TP0 / (TP0 + FN0) = (y_true=0, y_pred=0).sum() / (y_true=0).sum()
    # recall_1 = TP1 / (TP1 + FN1) = (y_true=1, y_pred=1).sum() / (y_true=1).sum()
    # y_true = [0, 1, 0, 1, 0, 1, 0, 0, 1, 1] (5 zeros, 5 ones)
    # y_pred = [0, 1, 1, 1, 0, 0, 0, 1, 1, 0]
    # True Negatives (TN for class 0): (0,0) -> 3 times (idx 0, 4, 6)
    # False Positives (FP for class 0 / FN for class 1): (0,1) -> 2 times (idx 2, 7)
    # False Negatives (FN for class 0 / FP for class 1): (1,0) -> 2 times (idx 5, 9)
    # True Positives (TP for class 1): (1,1) -> 3 times (idx 1, 3, 8)

    # recall_pos_label_0 = TN / (TN + FP) where FP means (true=0, pred=1)
    recall_0 = recall_score(y_true, y_pred, pos_label=0)  # 3 / (3+2) = 3/5 = 0.6
    # recall_pos_label_1 = TP / (TP + FN) where FN means (true=1, pred=0)
    recall_1 = recall_score(y_true, y_pred, pos_label=1)  # 3 / (3+2) = 3/5 = 0.6
    expected_ba = (recall_0 + recall_1) / 2  # (0.6 + 0.6) / 2 = 0.6

    assert get_balanced_accuracy(y_true, y_pred) == pytest.approx(expected_ba)

    # Test case where one class has no true positives for its prediction
    y_true_mix = np.array([0, 0, 0, 1])
    y_pred_mix = np.array([0, 0, 0, 0])  # recall_0 = 1.0, recall_1 = 0.0
    recall_0_mix = recall_score(y_true_mix, y_pred_mix, pos_label=0)
    recall_1_mix = recall_score(y_true_mix, y_pred_mix, pos_label=1, zero_division=0)
    expected_ba_mix = (recall_0_mix + recall_1_mix) / 2
    assert get_balanced_accuracy(y_true_mix, y_pred_mix) == pytest.approx(expected_ba_mix)


def test_pr_auc_score(classification_data):
    y_true, _, y_prob = classification_data
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    expected_pr_auc = sk_auc(recall, precision)
    assert pr_auc_score(y_true, y_prob) == pytest.approx(expected_pr_auc)


def test_mcc_score(classification_data):
    y_true, y_pred, _ = classification_data
    # y_true = [0, 1, 0, 1, 0, 1, 0, 0, 1, 1]
    # y_pred = [0, 1, 1, 1, 0, 0, 0, 1, 1, 0]
    # TP = 3 (y_true=1, y_pred=1 at indices 1, 3, 8)
    # TN = 3 (y_true=0, y_pred=0 at indices 0, 4, 6)
    # FP = 2 (y_true=0, y_pred=1 at indices 2, 7)
    # FN = 2 (y_true=1, y_pred=0 at indices 5, 9)
    # MCC = (TP*TN - FP*FN) / sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    # MCC = (3*3 - 2*2) / sqrt((3+2)*(3+2)*(3+2)*(3+2))
    # MCC = (9 - 4) / sqrt(625) = 5 / 25 = 0.2
    expected_mcc = (3 * 3 - 2 * 2) / np.sqrt((3 + 2) * (3 + 2) * (3 + 2) * (3 + 2))
    assert mcc_score(y_true, y_pred) == pytest.approx(expected_mcc)

    # Test case where denominator is zero
    y_true_zero_denom = np.array([0, 0, 0, 0])
    y_pred_zero_denom = np.array([1, 1, 1, 1])  # TP=0, TN=0, FP=4, FN=0 -> (TP+FP)=4, (TP+FN)=0 -> denom=0
    assert mcc_score(y_true_zero_denom, y_pred_zero_denom) == 0.0

    y_true_all_correct = np.array([0, 1, 0, 1])
    y_pred_all_correct = np.array([0, 1, 0, 1])  # TP=2, TN=2, FP=0, FN=0 -> MCC = (4-0)/sqrt(2*2*2*2) = 4/4 = 1
    assert mcc_score(y_true_all_correct, y_pred_all_correct) == pytest.approx(1.0)

    y_true_all_incorrect = np.array([0, 1, 0, 1])
    y_pred_all_incorrect = np.array([1, 0, 1, 0])  # TP=0, TN=0, FP=2, FN=2 -> MCC = (0-4)/sqrt(2*2*2*2) = -4/4 = -1
    assert mcc_score(y_true_all_incorrect, y_pred_all_incorrect) == pytest.approx(-1.0)


def test_survival_mae(survival_data):
    observed_time, event_indicator, predicted_time = survival_data
    # errors = np.zeros_like(observed_time, dtype=float)
    # event_mask = (event_indicator == 1)
    # errors[event_mask] = np.abs(observed_time[event_mask] - predicted_time[event_mask])
    # censored_mask = (event_indicator == 0)
    # predicted_early_mask = (censored_mask) & (predicted_time < observed_time)
    # errors[predicted_early_mask] = observed_time[predicted_early_mask] - predicted_time[predicted_early_mask]
    # return np.mean(errors)

    # observed_time   = [10, 15, 20, 25, 30, 12, 18, 22, 28, 32]
    # event_indicator = [ 1,  0,  1,  0,  1,  1,  0,  1,  0,  1]
    # predicted_time  = [12, 10, 22, 20, 35, 10, 20, 25, 26, 30]
    # errors:
    # idx 0 (event): abs(10-12) = 2
    # idx 1 (censored, pred < obs): 15-10 = 5
    # idx 2 (event): abs(20-22) = 2
    # idx 3 (censored, pred < obs): 25-20 = 5
    # idx 4 (event): abs(30-35) = 5
    # idx 5 (event): abs(12-10) = 2
    # idx 6 (censored, pred > obs): 0 (no error as per formula)
    # idx 7 (event): abs(22-25) = 3
    # idx 8 (censored, pred < obs): 28-26 = 2
    # idx 9 (event): abs(32-30) = 2
    # total_error = 2+5+2+5+5+2+0+3+2+2 = 28
    # mean_error = 28 / 10 = 2.8
    expected_mae = 2.8
    assert survival_mae(observed_time, event_indicator, predicted_time) == pytest.approx(expected_mae)

    # Test with boolean event indicators
    event_indicator_bool = event_indicator == 1
    assert survival_mae(observed_time, event_indicator_bool, predicted_time) == pytest.approx(expected_mae)

    # Test case: all events, all perfectly predicted
    obs_perfect = np.array([10, 20])
    event_perfect = np.array([1, 1])
    pred_perfect = np.array([10, 20])
    assert survival_mae(obs_perfect, event_perfect, pred_perfect) == 0.0

    # Test case: all censored, all predicted early
    obs_cens_early = np.array([10, 20])
    event_cens_early = np.array([0, 0])
    pred_cens_early = np.array([5, 15])  # errors: 5, 5. Mean = 5
    assert survival_mae(obs_cens_early, event_cens_early, pred_cens_early) == 5.0

    # Test case: all censored, all predicted late (or same)
    obs_cens_late = np.array([10, 20])
    event_cens_late = np.array([0, 0])
    pred_cens_late = np.array([10, 25])  # errors: 0, 0. Mean = 0
    assert survival_mae(obs_cens_late, event_cens_late, pred_cens_late) == 0.0


def test_get_metrics(classification_data):
    y_true, y_pred, y_prob = classification_data
    metrics = get_metrics(y_true, y_pred, y_prob)

    expected_ba = get_balanced_accuracy(y_true, y_pred)
    expected_mcc = mcc_score(y_true, y_pred)
    expected_precision = precision_score(y_true, y_pred, zero_division=0)
    expected_recall = recall_score(y_true, y_pred, zero_division=0)
    expected_f1 = f1_score(y_true, y_pred, zero_division=0)
    expected_pr_auc = pr_auc_score(y_true, y_prob)
    expected_roc_auc = roc_auc_score(y_true, y_prob)

    assert metrics["balanced_accuracy"] == pytest.approx(expected_ba)
    assert metrics["mcc"] == pytest.approx(expected_mcc)
    assert metrics["precision"] == pytest.approx(expected_precision)
    assert metrics["recall"] == pytest.approx(expected_recall)
    assert metrics["f1"] == pytest.approx(expected_f1)
    assert metrics["pr_auc"] == pytest.approx(expected_pr_auc)
    assert metrics["roc_auc"] == pytest.approx(expected_roc_auc)

    # Test with prefix
    prefix = "test"
    metrics_prefixed = get_metrics(y_true, y_pred, y_prob, prefix=prefix)
    assert metrics_prefixed[f"{prefix}_balanced_accuracy"] == pytest.approx(expected_ba)
    assert metrics_prefixed[f"{prefix}_mcc"] == pytest.approx(expected_mcc)
    assert f"{prefix}_roc_auc" in metrics_prefixed


def test_get_metrics_surv(survival_data):
    observed_time, event_indicator, predicted_time = survival_data
    metrics = get_metrics_surv(observed_time, event_indicator, predicted_time)

    expected_mae = survival_mae(observed_time, event_indicator, predicted_time)
    # For concordance_index, lifelines is used directly, so we trust its calculation
    # but ensure the key is present and the value is float.
    from lifelines.utils import concordance_index as lifelines_c_index

    expected_c_index = lifelines_c_index(observed_time, predicted_time, event_indicator)

    assert metrics["mae"] == pytest.approx(expected_mae)
    assert "c_index" in metrics
    assert isinstance(metrics["c_index"], float)
    assert metrics["c_index"] == pytest.approx(expected_c_index)

    # Test with prefix
    prefix = "surv_test"
    metrics_prefixed = get_metrics_surv(observed_time, event_indicator, predicted_time, prefix=prefix)
    assert metrics_prefixed[f"{prefix}_mae"] == pytest.approx(expected_mae)
    assert f"{prefix}_c_index" in metrics_prefixed
    assert isinstance(metrics_prefixed[f"{prefix}_c_index"], float)


# More complex functions like get_performance and get_performance_surv
# would require more elaborate setup and mocking, especially for logging.
# Here's a very basic smoke test for get_performance.


def test_get_performance_smoke(classification_data):
    y_true, y_pred, y_prob = classification_data
    df = pd.DataFrame({"target": y_true, "prediction": y_pred, "probability": y_prob, "group": ["A"] * 5 + ["B"] * 5})

    # Test with grouping
    performance_df = get_performance(
        df, "target", "prediction", "probability", grouping_cols=["group"], test_size_minimum_per_group=3
    )
    assert performance_df is not None
    assert len(performance_df) == 2  # Two groups A and B
    assert "balanced_accuracy" in performance_df.columns
    assert "rank_correlation" in performance_df.columns  # check one of the specific columns

    # Test without grouping (fake group)
    performance_df_no_group = get_performance(df, "target", "prediction", "probability", test_size_minimum_per_group=3)
    assert performance_df_no_group is not None
    assert len(performance_df_no_group) == 1
    assert "fake_group" in performance_df_no_group.columns
    assert performance_df_no_group["n_observations"].iloc[0] == len(y_true)

    # Test with a group too small
    df_small_group = pd.DataFrame(
        {"target": [0, 1, 0], "prediction": [0, 1, 0], "probability": [0.1, 0.9, 0.2], "group": ["C"] * 3}
    )
    performance_small_group = get_performance(
        df_small_group, "target", "prediction", "probability", grouping_cols=["group"], test_size_minimum_per_group=5
    )
    assert performance_small_group is None


def test_get_performance_surv_smoke(survival_data):
    observed_time, event_indicator, predicted_time = survival_data
    df = pd.DataFrame(
        {"time": observed_time, "event": event_indicator, "prediction": predicted_time, "group": ["A"] * 5 + ["B"] * 5}
    )

    # Test with grouping
    performance_df = get_performance_surv(
        df, "time", "event", "prediction", grouping_cols=["group"], test_size_minimum_per_group=3
    )
    assert performance_df is not None
    assert len(performance_df) == 2
    assert "mae" in performance_df.columns
    assert "km_median_surv_months" in performance_df.columns

    # Test without grouping
    performance_df_no_group = get_performance_surv(df, "time", "event", "prediction", test_size_minimum_per_group=3)
    assert performance_df_no_group is not None
    assert len(performance_df_no_group) == 1
    assert "fake_group" in performance_df_no_group.columns

    # Test with a group too small
    df_small_group = pd.DataFrame(
        {"time": [10, 20, 30], "event": [1, 0, 1], "prediction": [12, 18, 32], "group": ["C"] * 3}
    )
    performance_small_group = get_performance_surv(
        df_small_group, "time", "event", "prediction", grouping_cols=["group"], test_size_minimum_per_group=5
    )
    assert performance_small_group is None
