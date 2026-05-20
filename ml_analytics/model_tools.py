"""
Set of utility functions for training/testing models
"""

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from catboost import CatBoostClassifier, CatBoostRegressor, EFeaturesSelectionAlgorithm, EShapCalcType, Pool
from lifelines import KaplanMeierFitter
from lifelines.utils import concordance_index
from sklearn.metrics import (
    auc,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, train_test_split

from ml_analytics.utils import get_logger, log_and_raise_error

logger = get_logger("modeling-tools")


def prepare_catboost_data(
    df: pd.DataFrame,
    cat_features: list[str],
    feature_list: list[str] | None = None,
) -> pd.DataFrame:
    """
    Prepare a DataFrame for CatBoost training or inference.

    Runs ``infer_objects`` to downcast object dtypes, then converts every
    categorical column to ``str`` so CatBoost never receives Python ``None``
    or ``np.nan`` values in categorical columns (which would raise an error).
    Numeric columns are left untouched.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame (will be copied internally).
    cat_features : list[str]
        Columns to treat as categorical.
    feature_list : list[str], optional
        Subset of columns to check for missing values (for logging only).
        If None, all columns are checked.

    Returns
    -------
    pd.DataFrame
        Pre-processed copy of ``df``.
    """
    df = df.copy()
    df = df.infer_objects(copy=False)

    check_cols = feature_list if feature_list is not None else df.columns.tolist()
    missing = df[check_cols].isnull().sum()
    n_missing = missing[missing > 0].shape[0]
    logger.info(f"Features with missing values: {n_missing}")

    for col in cat_features:
        if col in df.columns:
            df[col] = df[col].astype(str)

    return df


def make_catboost_pool(
    df: pd.DataFrame,
    feature_list: list[str],
    cat_features: list[str],
    label=None,
    **pool_kwargs,
) -> Pool:
    """
    Build a CatBoost ``Pool`` with automatic preprocessing.

    Calls :func:`prepare_catboost_data` before constructing the pool, so you
    never have to remember to handle missing values or cast categorical columns
    manually.

    Parameters
    ----------
    df : pd.DataFrame
        Source DataFrame (will be copied internally by ``prepare_catboost_data``).
    feature_list : list[str]
        Feature columns to include in the pool.
    cat_features : list[str]
        Columns to treat as categorical.
    label : array-like or str, optional
        Target values, or the name of the target column in ``df``.
        If a string, it is extracted from ``df`` before subsetting features.
    **pool_kwargs :
        Any additional keyword arguments forwarded to ``catboost.Pool``
        (e.g. ``group_id``, ``weight``).

    Returns
    -------
    catboost.Pool
    """
    if isinstance(label, str):
        label = df[label]

    df = prepare_catboost_data(df, cat_features=cat_features, feature_list=feature_list)

    present_cat = [c for c in cat_features if c in feature_list]
    cat_indices = [feature_list.index(c) for c in present_cat]

    return Pool(
        data=df[feature_list],
        label=label,
        feature_names=feature_list,
        cat_features=cat_indices,
        **pool_kwargs,
    )


def get_features(df, target_col=None):
    """
    Get categorical and numerical features from a DataFrame.
    Args:
        df (pd.DataFrame): DataFrame to extract features from.
        target_col (str, optional): Name of the target column. Defaults to None.
    Returns:
        tuple: List of categorical features and list of numerical features.
    """

    if target_col is not None:
        df = df.drop(columns=[target_col], errors="ignore").copy()

    categorical_features = df.select_dtypes(include=["object", "category"]).columns.tolist()
    numerical_features = df.select_dtypes(include=["int32", "int64", "float64", "float32"]).columns.tolist()
    return categorical_features, numerical_features


def get_balanced_accuracy(y_true, y_pred):
    acc = (recall_score(y_true, y_pred, pos_label=0) + recall_score(y_true, y_pred, pos_label=1)) / 2
    return acc


def pr_auc_score(y_true, y_pred_proba):
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    return auc(recall, precision)


def brier_score(y_true, y_pred_proba):
    """
    Calculate the Brier score for binary classification.

    The Brier score is a proper scoring rule that measures the accuracy of probabilistic predictions.
    Lower scores are better, with 0 being perfect and 0.25 being the worst possible score for a binary classifier.

    Args:
        y_true (array-like): True binary labels (0 or 1).
        y_pred_proba (array-like): Predicted probabilities for the positive class.

    Returns:
        float: Brier score (lower is better).
    """
    return brier_score_loss(y_true, y_pred_proba)


def expected_calibration_error(y_true, y_pred_proba, n_bins=10):
    """
    Calculate the Expected Calibration Error (ECE) for binary classification.

    ECE measures the difference between predicted probabilities and actual outcomes
    across different confidence bins. It indicates how well-calibrated the model's
    probability estimates are.

    Args:
        y_true (array-like): True binary labels (0 or 1).
        y_pred_proba (array-like): Predicted probabilities for the positive class.
        n_bins (int): Number of bins to use for calibration curve. Default is 10.

    Returns:
        float: Expected Calibration Error (lower is better, 0 is perfect calibration).
    """
    try:
        # Calculate bin boundaries
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        # Calculate ECE
        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers, strict=True):
            # Find samples in this bin
            in_bin = (y_pred_proba > bin_lower) & (y_pred_proba <= bin_upper)
            prop_in_bin = in_bin.mean()

            if prop_in_bin > 0:
                # Calculate accuracy and confidence for this bin
                accuracy_in_bin = y_true[in_bin].mean() if in_bin.sum() > 0 else 0
                avg_confidence_in_bin = y_pred_proba[in_bin].mean() if in_bin.sum() > 0 else 0

                # Add weighted calibration error for this bin
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece

    except Exception as e:
        logger.warning(f"Error calculating ECE: {e}. Returning NaN.")
        return np.nan


def mcc_score(y_true, y_pred):
    """
    Calculate Matthews correlation coefficient (MCC) for binary classification.

    Args:
        y_true (array-like): True binary labels.
        y_pred (array-like): Predicted binary labels.

    Returns:
        float: Matthews correlation coefficient.
    """
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    tn = ((y_true == 0) & (y_pred == 0)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()

    numerator = (tp * tn) - (fp * fn)
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    if denominator == 0:
        return 0.0

    return numerator / denominator


def survival_mae(observed_time, event_indicator, predicted_time):
    """
    Calculates a Mean Absolute Error (MAE) suitable for survival data,
    considering censored observations.
    Ensures inputs are treated as NumPy arrays to avoid Pandas indexing issues.

    Args:
        observed_time_pd (pd.Series): Series of observed times.
        event_indicator_pd (pd.Series): Series indicating if the event was observed.
        predicted_time_pd (pd.Series): Series of predicted event times.

    Returns:
        float: The mean absolute error.
    """

    errors = np.zeros_like(observed_time, dtype=float)

    event_mask = (event_indicator == 1) | (event_indicator is True)
    errors[event_mask] = np.abs(observed_time[event_mask] - predicted_time[event_mask])
    censored_mask = (event_indicator == 0) | (event_indicator is False)
    predicted_early_mask = (censored_mask) & (predicted_time < observed_time)
    errors[predicted_early_mask] = observed_time[predicted_early_mask] - predicted_time[predicted_early_mask]

    return np.mean(errors)


def get_metrics(y_obs, y_pred, y_prob, prefix=None):
    metrics = {}

    if prefix is None:
        prefix = ""
    else:
        prefix = prefix + "_"

    metrics[prefix + "balanced_accuracy"] = get_balanced_accuracy(y_obs, y_pred)
    metrics[prefix + "mcc"] = mcc_score(y_obs, y_pred)
    metrics[prefix + "precision"] = precision_score(y_obs, y_pred, zero_division=0)
    metrics[prefix + "recall"] = recall_score(y_obs, y_pred, zero_division=0)
    metrics[prefix + "f1"] = f1_score(y_obs, y_pred, zero_division=0)
    metrics[prefix + "pr_auc"] = pr_auc_score(y_obs, y_prob)
    metrics[prefix + "roc_auc"] = roc_auc_score(y_obs, y_prob)
    metrics[prefix + "brier_score"] = brier_score(y_obs, y_prob)
    metrics[prefix + "ece"] = expected_calibration_error(y_obs, y_prob)
    return metrics


def get_performance(data, target_col, pred_col, prob_col, grouping_cols=None, test_size_minimum_per_group=50):
    """
    Calculates performance metrics for each combination of grouping columns.

    Args:
        data (pd.DataFrame): DataFrame containing target, predictions, probabilities, and grouping columns.
        target_col (str): Name of the target column.
        pred_col (str): Name of the prediction column (0 or 1).
        prob_col (str): Name of the probability column (probability of class 1).
        grouping_cols (list, optional): List of column names to group by.
                                        Defaults to ['product', 'event_type', 'country_code'].
        test_size_minimum_per_group (int, optional): Minimum number of observations required in each group to calculate metrics.
                                        Defaults to 50.

    Returns:
        pd.DataFrame: DataFrame with grouping columns, number of observations, and performance metrics for each group.
    """  # noqa: E501

    data = data.copy()
    if grouping_cols is None:
        data["fake_group"] = 1
        grouping_cols = ["fake_group"]
    else:
        grouping_cols = [col for col in grouping_cols if col in data.columns]
        if not grouping_cols:
            logger.error("No valid grouping columns found!")

    results = []
    for name, group in data.groupby(grouping_cols):
        if not isinstance(name, tuple):
            name = (name,)
        n_observations = len(group)

        # Skip group if too small early on
        if n_observations == 0:
            logger.warning(f"Skipping empty group {name}.")
            continue

        if n_observations > test_size_minimum_per_group:
            y_true = group[target_col].values.ravel()
            y_pred = group[pred_col].values.ravel()
            y_prob = group[prob_col].values.ravel()

            tp = ((y_true == 1) & (y_pred == 1)).sum() / n_observations
            tn = ((y_true == 0) & (y_pred == 0)).sum() / n_observations
            fp = ((y_true == 0) & (y_pred == 1)).sum() / n_observations
            fn = ((y_true == 1) & (y_pred == 0)).sum() / n_observations

            # Calculate rank correlation
            rank_correlation = np.nan  # Default to NaN
            binning_agg = pd.DataFrame(columns=["bin", "target_rate"])

            try:
                bins_labels = pd.qcut(y_prob, q=10, labels=False, duplicates="drop")
            except ValueError:
                try:
                    bins_labels = pd.cut(y_prob, bins=10, labels=False, include_lowest=True)
                except ValueError:
                    logger.warning(f"Could not compute bins for group {name}. Setting rank_correlation to NaN.")
                    bins_labels = None

            if bins_labels is not None:
                try:
                    binning_df = pd.DataFrame({"bin": bins_labels, "target_rate": y_true})
                    binning_agg = binning_df.groupby("bin")["target_rate"].mean().reset_index()
                    if len(binning_agg) > 4:
                        rank_correlation = binning_agg[["bin", "target_rate"]].corr(method="spearman").iloc[0, 1]
                        target_bin_correlation = binning_agg[["bin", "target_rate"]].corr(method="pearson").iloc[0, 1]
                    else:
                        logger.warning(
                            f"Not enough bins with data to calculate correlation for group {name}. Setting rank_correlation to NaN."  # noqa: E501
                        )  # noqa: E501
                        rank_correlation = np.nan
                except Exception as e:
                    logger.error(f"Error during binning aggregation or correlation for group {name}: {e}")

            if len(np.unique(y_true)) > 1:
                metrics = get_metrics(y_true, y_pred, y_prob)

            binning_agg = binning_agg.sort_values(by="bin", ascending=False)
            group_result = dict(zip(grouping_cols, name, strict=False))
            group_result["n_observations"] = n_observations
            group_result["target_rate"] = np.mean(y_true)
            group_result["score_average"] = np.mean(y_prob)
            group_result["score_std"] = np.std(y_prob)
            group_result["rank_correlation"] = rank_correlation
            group_result["target_bin_correlation"] = target_bin_correlation
            group_result["number_deciles"] = len(binning_agg)
            group_result["tp_perc"] = tp * 100
            group_result["tn_perc"] = tn * 100
            group_result["fp_perc"] = fp * 100
            group_result["fn_perc"] = fn * 100
            group_result.update(metrics)
            group_result["first_decile_target_rate"] = binning_agg.iloc[0, 1] if len(binning_agg) > 0 else np.nan
            group_result["second_decile_target_rate"] = binning_agg.iloc[1, 1] if len(binning_agg) > 1 else np.nan
            group_result["third_decile_target_rate"] = binning_agg.iloc[2, 1] if len(binning_agg) > 2 else np.nan
            results.append(group_result)
        else:
            logger.warning(f"Group {name} has too few observations ({n_observations}). Skipping.")
            continue

    if not results:
        logger.warning("No groups found or processed. Returning empty DataFrame.")
        return None
    else:
        performance_df = pd.DataFrame(results)

        def create_improvement_column(df, new_column, baseline, new_value):
            df[new_column] = np.where(
                (df[baseline].notna()) & (df[baseline] != 0) & (df[new_value].notna()),
                (df[new_value] - df[baseline]) / df[baseline],
                np.nan,
            )
            return df

        performance_df = create_improvement_column(performance_df, "pr_auc_improvement", "target_rate", "pr_auc")
        performance_df = create_improvement_column(
            performance_df, "first_decile_improvement", "target_rate", "first_decile_target_rate"
        )  # noqa: E501
        performance_df = create_improvement_column(
            performance_df, "second_decile_improvement", "target_rate", "second_decile_target_rate"
        )  # noqa: E501

        performance_df = performance_df.round(3)
        return performance_df


def get_metrics_surv(observed_time, event_indicator, predicted_time, prefix=None):
    """
    Calculate performance metrics for survival analysis.

    Args:
        observed_time (array-like): Observed times.
        event_indicator (array-like): Event indicators (1 if event occurred, 0 if censored).
        predicted_time (array-like): Predicted times.

    Returns:
        dict: Dictionary of performance metrics.
    """
    metrics = {}

    if prefix is None:
        prefix = ""
    else:
        prefix = prefix + "_"

    metrics[prefix + "mae"] = survival_mae(observed_time, event_indicator, predicted_time)
    metrics[prefix + "c_index"] = concordance_index(observed_time, predicted_time, event_indicator)
    return metrics


def get_performance_surv(data, time_col, event_col, pred_col, grouping_cols=None, test_size_minimum_per_group=50):
    """
    Calculates performance metrics for each combination of grouping columns.
    Args:
        data (pd.DataFrame): DataFrame containing target, predictions, probabilities, and grouping columns.
        time_col (str): Name of the time column.
        event_col (str): Name of the event column.
        pred_col (str): Name of the prediction column (predicted survival time).
        grouping_cols (list, optional): List of column names to group by.
                                        Defaults to ['product', 'event_type', 'country_code'].
    Returns:
        pd.DataFrame: DataFrame with grouping columns, number of observations, and performance metrics for each group.
    """

    kmf = KaplanMeierFitter()

    data = data.copy()
    if grouping_cols is None:
        data["fake_group"] = 1
        grouping_cols = ["fake_group"]
    else:
        grouping_cols = [col for col in grouping_cols if col in data.columns]
        if not grouping_cols:
            logger.error("No valid grouping columns found!")

    results = []
    for name, group in data.groupby(grouping_cols):
        if not isinstance(name, tuple):
            name = (name,)
        n_observations = len(group)
        # Skip group if too small early on
        if n_observations == 0:
            logger.warning(f"Skipping empty group {name}.")
            continue

        if n_observations > test_size_minimum_per_group:
            kmf.fit(durations=group[time_col], event_observed=group[event_col])
            print()
            metrics = get_metrics_surv(group[time_col], group[event_col], group[pred_col])
            group_result = dict(zip(grouping_cols, name, strict=False))
            group_result["n_observations"] = n_observations
            group_result["avg_months"] = np.mean(group[time_col])
            group_result["median_months"] = np.median(group[time_col])
            group_result["km_median_surv_months"] = kmf.median_survival_time_
            group_result["event_rate"] = np.mean(group[event_col])
            group_result["avg_predicted_months"] = np.mean(group[pred_col])
            group_result["median_predicted_months"] = np.median(group[pred_col])
            group_result.update(metrics)
            results.append(group_result)
        else:
            logger.warning(f"Group {name} has too few observations ({n_observations}). Skipping.")
            continue

    if not results:
        logger.warning("No groups found or processed. Returning empty DataFrame.")
        return None
    else:
        performance_df = pd.DataFrame(results)
        performance_df = performance_df.round(3)
        return performance_df


def time_split(data, date_column, test_ratio=0.3, max_date=None):
    """
    Splits the data into training and testing sets based on a date column.

    Parameters:
        data (pd.DataFrame): The input dataframe to be split.
        date_column (str): The name of the date column to use for splitting.
        test_ratio (float): The ratio of the data to be used for testing. Default is 0.3.
        max_date (str): The maximum date to consider for the splittIf None, all data is used. Default is None.

    Returns:
        pd.DataFrame, pd.DataFrame: The training and testing dataframes.
    """

    if test_ratio <= 0 or test_ratio >= 1:
        log_and_raise_error(logger=logger, message="test_ratio must be between 0 and 1!")

    if max_date is not None:
        data = data[data[date_column] < max_date].copy()

    data[date_column] = pd.to_datetime(data[date_column]).dt.date

    data = data.sort_values(by=date_column)
    cutoff_date = data[date_column].quantile(1 - test_ratio)

    # Split the data into training and testing sets
    train_data = data[data[date_column] < cutoff_date]
    test_data = data[data[date_column] >= cutoff_date]

    return train_data, test_data


def catboost_feature_selection(
    train_df: pd.DataFrame,
    feature_list: list[str],
    target_column: str | list[str],  # Modified to handle both single and multiple targets
    cat_features: list[str],
    id_col: str = "id_col",
    num_features_to_select: int = None,
    define_best_num_features: bool = False,
    model: CatBoostClassifier | CatBoostRegressor = None,
    model_task: str = "classification",
    algorithm: EFeaturesSelectionAlgorithm = None,
    shap_calc_type: EShapCalcType = EShapCalcType.Regular,
    force_to_include: list[str] = None,
    steps: int = 15,
    iterations: int = 100,
    random_seed: int = 42,
    logging_level: str = "Verbose",
    min_count_binary: int = 30,
    sample_ratio: float = None,
) -> list[str]:
    """
    Perform feature selection using CatBoost's built-in feature selection with GroupKFold validation.
    Now supports both classification and regression tasks, including survival analysis.

    Parameters:
    - train_df: pd.DataFrame. The training DataFrame.
    - feature_list: list[str]. List of features to consider for selection.
    - target_column: str or list[str]. The target column name(s). For survival: ['y_lower', 'y_upper']
    - cat_features: list[str]. List of categorical feature names for CatBoost.
    - id_col: str. Column name for grouping (default: 'id_col').
    - model: CatBoostClassifier or CatBoostRegressor instance. If None, creates appropriate model based on target.
    - model_task: str. Type of model task: 'classification', 'regression', or 'survival'.
    - algorithm: EFeaturesSelectionAlgorithm. If None, defaults to RecursiveByShapValues.
    - force_to_include: list[str]. Features to always include in final set.
    - steps: int. Number of steps for feature selection.
    - iterations: int. Number of iterations for the CatBoost model.
    - random_seed: int. Random seed for reproducibility.
    - logging_level: str. Logging level for CatBoost.
    - min_count_binary: int. Minimum count for binary features to be included.
    - sample_ratio: float. If provided (e.g., 0.3), sample this ratio of unique IDs using stratified sampling by target for faster iteration.

    Returns:
    - list[str]. List of selected features.
    """  # noqa: E501
    feature_list = feature_list.copy()
    train_df = prepare_catboost_data(train_df, cat_features=cat_features, feature_list=feature_list)

    if algorithm is None:
        algorithm = EFeaturesSelectionAlgorithm.RecursiveByShapValues

    # Sample data by unique IDs if sample_ratio is specified
    if sample_ratio is not None:
        if sample_ratio <= 0 or sample_ratio >= 1:
            log_and_raise_error(logger, "sample_ratio must be between 0 and 1!")
        # Get unique IDs and their corresponding target values for stratification
        if isinstance(target_column, list):
            # For survival models, aggregate first target column
            unique_ids_df = train_df.groupby(id_col, as_index=False)[target_column[0]].first()
            if model_task == "survival":
                # Also need second target column for event indicator
                unique_ids_df[target_column[1]] = train_df.groupby(id_col)[target_column[1]].first().values
                stratify_col = (unique_ids_df[target_column[1]] != -1).astype(int)
            else:
                stratify_col = unique_ids_df[target_column[0]]
        else:
            unique_ids_df = train_df.groupby(id_col, as_index=False)[target_column].first()
            stratify_col = unique_ids_df[target_column]

        # Sample unique IDs with stratification
        sampled_ids, _ = train_test_split(
            unique_ids_df[id_col], train_size=sample_ratio, stratify=stratify_col, random_state=random_seed
        )

        # Filter dataframe to keep only sampled IDs
        original_size = len(train_df)
        train_df = train_df[train_df[id_col].isin(sampled_ids)].copy()
        logger.info(f"Sampled {len(sampled_ids)} unique IDs ({sample_ratio:.1%}) from {len(unique_ids_df)} total IDs")
        logger.info(
            f"Reduced dataset from {original_size:,} to {len(train_df):,} rows ({len(train_df) / original_size:.1%}) for faster feature selection iteration"  # noqa: E501
        )

    # Determine if this is a survival/regression task
    if model_task is None:
        log_and_raise_error(logger, "model_task must be specified as 'classification', 'regression', or 'survival'!")
    if model_task not in ["classification", "regression", "survival"]:
        log_and_raise_error(logger, "model_task must be one of 'classification', 'regression', or 'survival'!")

    if model is None:
        if model_task == "regression":
            model = CatBoostRegressor(iterations=iterations, loss_function="RMSE", verbose=0, random_seed=random_seed)
        elif model_task == "survival":
            # Survival analysis with AFT loss
            model = CatBoostRegressor(
                iterations=iterations, loss_function="SurvivalAft:dist=Normal", verbose=0, random_seed=random_seed
            )
        else:
            model = CatBoostClassifier(
                iterations=iterations, loss_function="Logloss", verbose=0, random_seed=random_seed
            )

    if num_features_to_select is not None and define_best_num_features:
        num_features_to_select = None
    if num_features_to_select is None:
        define_best_num_features = True

    # Remove features with only one value
    for col in feature_list[:]:
        if train_df[col].dropna().nunique() == 1:
            feature_list.remove(col)

    # Remove binary features with insufficient counts
    for col in feature_list[:]:
        if train_df[col].dropna().nunique() == 2:
            unique_vals = set(train_df[col].dropna().unique())
            if unique_vals == {0, 1} or unique_vals == {1, 0} or unique_vals == {0.0, 1.0}:
                if train_df[col].value_counts().get(1, 0) < min_count_binary:
                    feature_list.remove(col)

    # Get correlation between numberic future, and print the features with more than +-0.80 correlation
    num_features = [f for f in feature_list if f not in cat_features]
    corr_matrix = train_df[num_features].corr()
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > 0.8:
                high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
    if high_corr_pairs:
        logger.info(f"Found {len(high_corr_pairs)} highly correlated feature pairs (|correlation| > 0.8)")
    else:
        logger.info("No highly correlated feature pairs found (|correlation| > 0.8).")

    if force_to_include is None:
        force_to_include = []

    # Only exclude forced features from selection - let algorithm decide on categorical features
    selection_feature_list = [f for f in feature_list if f not in force_to_include]
    # Include categorical features from both selection list and forced features
    all_features_for_pool = selection_feature_list + force_to_include
    ncat_features = [c for c in cat_features if c in all_features_for_pool]

    logger.info(
        f"Feature selection: {len(selection_feature_list)} selectable, {len(force_to_include)} forced, {len(ncat_features)} categorical"  # noqa: E501
    )

    if model_task in ["regression", "survival"]:
        gkf = GroupKFold(n_splits=3)
        groups = train_df[id_col]
        if model_task == "survival":
            # For survival, use event indicator for stratification approximation
            stratify_col = (train_df[target_column[1]] != -1).astype(int)
            splits = list(StratifiedGroupKFold(n_splits=3).split(train_df, stratify_col, groups=groups))
        else:
            splits = list(gkf.split(train_df, groups=groups))
    else:
        sgkf = StratifiedGroupKFold(n_splits=3)
        groups = train_df[id_col]
        splits = list(sgkf.split(train_df, train_df[target_column], groups=groups))

    train_idx = np.concatenate([splits[0][0], splits[1][0]])
    val_idx = splits[2][1]

    train_fold = train_df.iloc[train_idx]
    val_fold = train_df.iloc[val_idx]

    train_pool = make_catboost_pool(train_fold, all_features_for_pool, cat_features, label=train_fold[target_column])
    eval_pool = make_catboost_pool(val_fold, all_features_for_pool, cat_features, label=val_fold[target_column])

    # Perform feature selection
    selected_features_algo = []
    try:
        if define_best_num_features:
            fsummary = model.select_features(
                train_pool,
                eval_set=eval_pool,
                features_for_select=selection_feature_list,
                num_features_to_select=1,
                steps=steps,
                algorithm=algorithm,
                shap_calc_type=shap_calc_type,
                train_final_model=False,
                logging_level=logging_level,
                plot=False,
            )

            loss_features = pd.DataFrame(
                {
                    "Features_Removed": fsummary["loss_graph"]["removed_features_count"],
                    "Loss": fsummary["loss_graph"]["loss_values"],
                }
            )

            best_idx = loss_features["Loss"].idxmin()
            best_features_kept = len(selection_feature_list) - loss_features.loc[best_idx, "Features_Removed"]
            best_loss_value = loss_features.loc[best_idx, "Loss"]

            logger.info(
                f"Optimal features: {best_features_kept} selected + {len(force_to_include)} forced = {best_features_kept + len(force_to_include)} total (loss: {best_loss_value:.6f})"  # noqa: E501
            )

            # plot of losses vs number of features removed
            plt.figure(figsize=(8, 5))
            plt.plot(
                loss_features["Features_Removed"],
                loss_features["Loss"],
                marker="o",
                linestyle="-",
                color="b",
            )
            plt.axvline(
                x=loss_features.loc[best_idx, "Features_Removed"],
                color="r",
                linestyle="--",
                label="Best number of features",
            )
            plt.title("Feature Selection Loss Graph")
            plt.xlabel("Number of Features Removed")
            plt.ylabel("Loss")
            plt.legend()
            plt.grid()
            plt.show()

            summary = model.select_features(
                train_pool,
                eval_set=eval_pool,
                features_for_select=selection_feature_list,
                num_features_to_select=best_features_kept,
                steps=steps,
                algorithm=algorithm,
                shap_calc_type=shap_calc_type,
                train_final_model=False,
                logging_level=logging_level,
                plot=False,
            )

        else:
            summary = model.select_features(
                train_pool,
                eval_set=eval_pool,
                features_for_select=selection_feature_list,
                num_features_to_select=num_features_to_select,
                steps=steps,
                algorithm=algorithm,
                shap_calc_type=shap_calc_type,
                train_final_model=False,
                logging_level=logging_level,
                plot=False,
            )
        selected_features_algo = summary["selected_features_names"]
        logger.info(f"Selected {len(selected_features_algo)} features by algorithm")

    except Exception as e:
        logger.error(f"Error during feature selection: {e}")
        raise

    # Combine selected features with forced features
    final_features = selected_features_algo + force_to_include

    logger.info(
        f"Final feature set: {len(final_features)} features ({len(selected_features_algo)} selected, {len(force_to_include)} forced)"  # noqa: E501
    )

    return final_features


def plot_score_bins(
    df,
    prob_col,
    target_col,
    bins=10,
    show_plot=True,
    figsize=(10, 6),
    title_placeholder=None,
    xlabel=None,
    ylabel=None,
    save_path=None,
):
    """
    Add 'propensity' and 'bins' columns to the input DataFrame, group the data,
    and plot the average target for each bin using a bar plot.

    Parameters:
        df (pd.DataFrame): DataFrame containing the target column.
        prob_col: Name of the column containing predicted probabilities.
        target_col (str): Name of the target column to aggregate.
        bins (int): Number of quantile bins for the 'propensity' column. Default is 10.
        show_plot (bool): Whether to display the plot immediately. Default is True.as_integer_ratio
        figsize (tuple): Size of the plot. Default is (10, 6).
        title_placeholder (str): Title for the plot. If None, a default title is generated.
        save_path (str): Path to save the plot. If None, the plot is not saved.

    Returns:
        pd.DataFrame: Aggregated DataFrame with bin means for target and propensity.
    """

    if title_placeholder is None:
        title_placeholder = ""
    else:
        title_placeholder = f" - {title_placeholder}"

    df = df.copy()
    df.loc[:, "bins"] = pd.qcut(df[prob_col], q=bins, labels=False, duplicates="drop") + 1
    tab = df.groupby("bins").agg({target_col: "mean", prob_col: "mean"}).reset_index()

    figure, ax = plt.subplots(figsize=figsize)

    ax = sns.barplot(x=tab["bins"], y=tab[target_col])
    if xlabel is not None:
        plt.xlabel(xlabel)
    else:
        plt.xlabel("Score bin")

    if ylabel is not None:
        plt.ylabel(f"Average {ylabel}")
    else:
        ylabel = f"Average {target_col.replace('_', ' ').title()}"
        plt.ylabel(ylabel)

    for index, row in tab.iterrows():
        bar_patch = None
        for patch in ax.patches:
            if abs(patch.get_height() - row[target_col]) < 1e-6:
                if abs(patch.get_x() + patch.get_width() / 2 - index) < 0.5:
                    bar_patch = patch
                    break
        if bar_patch:
            x_pos = bar_patch.get_x() + bar_patch.get_width() / 2.0
            y_pos = bar_patch.get_height()
            ax.text(x_pos, y_pos, f"{row[target_col]:.2f}", ha="center", va="bottom", color="black", fontweight="bold")
        else:
            ax.text(
                index,
                row[target_col],
                f"{row[target_col]:.2f}",
                ha="center",
                va="bottom",
                color="black",
                fontweight="bold",
            )

    plt.axhline(y=df[target_col].mean(), color="r", linestyle="--", label="Overall target rate")

    # compute improvement (last bing versus average rate)
    improvement = (tab[target_col].iloc[-1] - df[target_col].mean()) / df[target_col].mean()

    plt.title(
        f"{ylabel} by score bin ({bins} bins) {title_placeholder}\n"
        f"Improvement (highest bin vs average) : {improvement:.2%}"
    )

    if save_path is not None:
        plt.savefig(save_path)

    if show_plot:
        plt.show()
    else:
        plt.close()


def shap_plot(pipeline, data, features, output_path="shap_summary.png", show_plot=True):
    """
    Create and save a SHAP summary plot using a scikit-learn pipeline.
    If a whole column is missing in test data that was present in training, it fills that column with zeros.

    Parameters:
        pipeline: scikit-learn Pipeline containing a 'preprocessor' and a 'classifier'
        test_set: DataFrame for test data (used for computing SHAP values)
        features: list of feature column names
        output_path: file path where the plot will be saved

    Returns:
        None. The plot is saved to output_path.
    """

    data_transformed = pipeline.named_steps["preprocessor"].transform(data[features])
    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    # feature_names = [name.replace('cat__', '').replace('num__', '') for name in feature_names]

    df = pd.DataFrame(data_transformed, columns=feature_names)

    model = pipeline.named_steps["classifier"]
    explainer = shap.Explainer(model)
    shap_values = explainer.shap_values(df)

    plt.figure()
    shap.summary_plot(shap_values, df, show=show_plot)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_trend(
    data,
    x_col,
    y_col,
    hue_col,
    xlabel="",
    ylabel=None,
    title=None,
    figsize=(12, 6),
    rotation=65,
    vline_date=None,
    show_grid=False,
    show_plot=True,
    save_path=None,
):
    """
    Generates a line plot for time series data, grouped by a hue column.

    Args:
        data (pd.DataFrame): DataFrame containing the data to plot.
        x_col (str): Name of the column for the x-axis (should be datetime-like).
        y_col (str): Name of the column for the y-axis.
        hue_col (str): Name of the column to group and color lines by.
        title (str): Title for the plot.
        xlabel (str, optional): Label for the x-axis. Defaults to ''.
        ylabel (str, optional): Label for the y-axis. Defaults to y_col name.
        figsize (tuple, optional): Figure size. Defaults to (12, 6).
        rotation (int, optional): Rotation angle for x-axis labels. Defaults to 65.
        vline_date (str, optional): Date string (e.g., 'YYYY-MM-DD') to draw a
                                     vertical line. Defaults to None.
        show_grid (bool, optional): Whether to display the background grid. Defaults to False.
        show_plot (bool, optional): Whether to display the plot using plt.show(). Defaults to True.
        save_path (str, optional): Path to save the figure. If None, figure is not saved. Defaults to None.
    """
    if ylabel is None:
        ylabel = y_col

    data = data.copy()
    data[x_col] = data[x_col].dt.to_timestamp()

    plt.figure(figsize=figsize)
    # sns.set_style("whitegrid" if show_grid else "white") # Set style based on show_grid

    ax = sns.lineplot(data=data, x=x_col, y=y_col, hue=hue_col, marker="o")

    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    # Add vertical line if specified
    if vline_date:
        try:
            vline_dt = pd.to_datetime(vline_date)
            plt.axvline(vline_dt, color="red", linestyle="--")
        except ValueError:
            print(f"Warning: Could not parse vline_date '{vline_date}'. Skipping vertical line.")

    plt.grid(show_grid)
    plt.xticks(rotation=rotation, ha="right")
    plt.legend(title=None)
    if title:
        plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()  # Adjust layout

    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()  # Close the plot if not showing to free memory
