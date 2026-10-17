"""Additional analyses that strengthen the Ghana household-waste manuscript.

The script reproduces the original data preparation, completes all baseline
models on the 12 outer folds, quantifies audit-budget utility, evaluates
urban/rural performance and calibration, estimates calibration after holding
out each region, and checks sensitivity to nearly constant survey weights.
"""
import argparse
import json
import time
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


warnings.filterwarnings("ignore", category=FutureWarning)
SEEDS = [2026, 3031, 4049]
NUM = ["usual_members", "h07a", "h07b", "h11b"]
CAT = [
    "region", "urbrur", "h01", "h02", "h03", "h04", "h05", "h06", "h08a", "h08b",
    "h09a", "h09b", "h10", "h11a", "h11c", "h12", "h13a1", "h13a2", "h13b1",
    "h13b2", "h13c1", "h13c2", "h13d1", "h13d2", "h13e1", "h13e2", "h13f1",
    "h13f2", "h13g1", "h13g2", "h13h1", "h13h2", "h13i1", "h13i2", "h13j1",
    "h13j2", "h13k1", "h13k2", "h13l1", "h13l2", "h13m1", "h13m2", "h13n1",
    "h13n2", "h13o1", "h13o2", "h13p1", "h13p2", "h13q1", "h13q2", "h13r1",
    "h13r2", "h13s1", "h13s2",
]
FEATURES = NUM + CAT
COLLECTION = {
    "Compaction truck", "Other vehicles", "Tricycle", "Central container",
    "Push carts/Walk-in attendant/Bicycle/Wheelbarrow",
}


def load_data(zip_path, csv_name):
    use = set(["s02", "restype", "distcode", "weight"] + FEATURES)
    with zipfile.ZipFile(zip_path) as archive:
        with archive.open(csv_name) as handle:
            data = pd.read_csv(handle, usecols=lambda c: c in use, dtype="string", low_memory=False)
    data = data[
        data.restype.eq("Occupied housing unit")
        & data.s02.notna()
        & data.s02.str.strip().ne("")
        & data.distcode.notna()
    ].copy()
    data.s02 = data.s02.str.strip()
    data.weight = pd.to_numeric(data.weight, errors="coerce")
    data = data[data.weight.gt(0)].copy()
    for column in NUM:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    for column in CAT:
        data[column] = data[column].fillna("<Missing>").astype(str)
    y = data.s02.isin(COLLECTION).astype(np.int8).to_numpy()
    return (
        data,
        data[FEATURES],
        y,
        data.weight.to_numpy(float),
        data.distcode.astype(str).to_numpy(),
        data.region.astype(str).to_numpy(),
    )


ORDINAL_PREP = ColumnTransformer(
    [
        ("num", SimpleImputer(strategy="median", add_indicator=True), NUM),
        (
            "cat",
            Pipeline(
                [
                    ("impute", SimpleImputer(strategy="most_frequent")),
                    (
                        "enc",
                        OrdinalEncoder(
                            handle_unknown="use_encoded_value",
                            unknown_value=-1,
                            encoded_missing_value=-1,
                            dtype=np.float32,
                        ),
                    ),
                ]
            ),
            CAT,
        ),
    ],
    remainder="drop",
)

ONEHOT_PREP = ColumnTransformer(
    [
        (
            "num",
            Pipeline(
                [
                    ("imp", SimpleImputer(strategy="median", add_indicator=True)),
                    ("sc", StandardScaler(with_mean=False)),
                ]
            ),
            NUM,
        ),
        (
            "cat",
            Pipeline(
                [
                    ("imp", SimpleImputer(strategy="most_frequent")),
                    ("oh", OneHotEncoder(handle_unknown="ignore", min_frequency=30, dtype=np.float32)),
                ]
            ),
            CAT,
        ),
    ],
    sparse_threshold=1.0,
)


def make_hgb(max_iter=80):
    return Pipeline(
        [
            ("prep", clone(ORDINAL_PREP)),
            (
                "model",
                HistGradientBoostingClassifier(
                    max_iter=max_iter,
                    learning_rate=0.07,
                    max_leaf_nodes=31,
                    min_samples_leaf=100,
                    l2_regularization=1.0,
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=10,
                    random_state=2026,
                ),
            ),
        ]
    )


def metric_row(y, prob, weight=None, threshold=0.5):
    pred = (np.asarray(prob) >= threshold).astype(np.int8)
    return {
        "accuracy": accuracy_score(y, pred, sample_weight=weight),
        "balanced_accuracy": balanced_accuracy_score(y, pred, sample_weight=weight),
        "precision": precision_score(y, pred, sample_weight=weight, zero_division=0),
        "recall": recall_score(y, pred, sample_weight=weight, zero_division=0),
        "f1": f1_score(y, pred, sample_weight=weight, zero_division=0),
        "roc_auc": roc_auc_score(y, prob, sample_weight=weight),
        "average_precision": average_precision_score(y, prob, sample_weight=weight),
        "brier": brier_score_loss(y, prob, sample_weight=weight),
        "mcc": matthews_corrcoef(y, pred, sample_weight=weight),
        "log_loss": log_loss(y, np.c_[1 - np.asarray(prob), prob], sample_weight=weight, labels=[0, 1]),
        "threshold": threshold,
    }


def calibration_stats(y, prob, weight):
    eps = 1e-6
    p = np.clip(np.asarray(prob, float), eps, 1 - eps)
    logits = np.log(p / (1 - p)).reshape(-1, 1)
    model = LogisticRegression(C=1e6, max_iter=1000).fit(logits, y, sample_weight=weight)
    edges = np.linspace(0, 1, 11)
    bin_id = np.clip(np.digitize(prob, edges) - 1, 0, 9)
    ece = 0.0
    total = np.sum(weight)
    for b in range(10):
        mask = bin_id == b
        if not np.any(mask):
            continue
        share = np.sum(weight[mask]) / total
        observed = np.average(np.asarray(y)[mask], weights=weight[mask])
        predicted = np.average(np.asarray(prob)[mask], weights=weight[mask])
        ece += share * abs(observed - predicted)
    return {
        "calibration_intercept": float(model.intercept_[0]),
        "calibration_slope": float(model.coef_[0, 0]),
        "ece_10bin": float(ece),
    }


def complete_baselines(X, y, weight, groups, existing):
    rows = existing.copy().to_dict("records")
    present = set(zip(existing["repeat"], existing["fold"], existing["model"]))
    for repeat, seed in enumerate(SEEDS, 1):
        cv = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed)
        for fold, (train, test) in enumerate(cv.split(X, y, groups), 1):
            rng = np.random.default_rng(seed + fold)
            fit = np.sort(rng.choice(train, size=min(200000, len(train)), replace=False))
            specifications = []
            prevalence = np.average(y[fit], weights=weight[fit])
            specifications.append(("Weighted prevalence baseline", np.repeat(prevalence, len(test))))
            if (repeat, fold, "Regularized logistic SGD") not in present:
                logistic = Pipeline(
                    [
                        ("prep", clone(ONEHOT_PREP)),
                        (
                            "model",
                            SGDClassifier(
                                loss="log_loss", alpha=1e-5, max_iter=1000, tol=1e-4,
                                average=True, early_stopping=True, validation_fraction=0.1,
                                n_iter_no_change=10, random_state=seed + fold,
                            ),
                        ),
                    ]
                )
                logistic.fit(X.iloc[fit], y[fit], model__sample_weight=weight[fit])
                specifications.append(("Regularized logistic SGD", logistic.predict_proba(X.iloc[test])[:, 1]))
            if (repeat, fold, "Random forest") not in present:
                forest = Pipeline(
                    [
                        ("prep", clone(ORDINAL_PREP)),
                        (
                            "model",
                            RandomForestClassifier(
                                n_estimators=250, max_features="sqrt", min_samples_leaf=30,
                                n_jobs=4, random_state=seed + fold, class_weight=None,
                            ),
                        ),
                    ]
                )
                forest.fit(X.iloc[fit], y[fit], model__sample_weight=weight[fit])
                specifications.append(("Random forest", forest.predict_proba(X.iloc[test])[:, 1]))
            for model_name, prob in specifications:
                if (repeat, fold, model_name) in present:
                    continue
                rows.append(
                    {
                        "repeat": repeat,
                        "fold": fold,
                        "seed": seed,
                        "model": model_name,
                        "n_fit": len(fit),
                        "n_test": len(test),
                        "n_test_districts": pd.Series(groups[test]).nunique(),
                        **metric_row(y[test], prob, weight[test]),
                    }
                )
            print(json.dumps({"completed_baselines": [repeat, fold]}), flush=True)
    return pd.DataFrame(rows).sort_values(["repeat", "fold", "model"]).reset_index(drop=True)


def model_summary(metrics):
    cols = ["roc_auc", "average_precision", "f1", "balanced_accuracy", "brier", "mcc"]
    summary = metrics.groupby("model")[cols].agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    return summary.reset_index()


def paired_differences(metrics):
    value_cols = ["roc_auc", "average_precision", "f1", "balanced_accuracy", "brier", "mcc"]
    hgb = metrics[metrics.model.eq("Histogram gradient boosting")].set_index(["repeat", "fold"])
    rng = np.random.default_rng(8173)
    rows = []
    for comparator in ["Regularized logistic SGD", "Random forest", "Weighted prevalence baseline"]:
        comp = metrics[metrics.model.eq(comparator)].set_index(["repeat", "fold"])
        for metric in value_cols:
            diff = (hgb[metric] - comp[metric]).dropna().to_numpy(float)
            boot = np.array([np.mean(rng.choice(diff, size=len(diff), replace=True)) for _ in range(5000)])
            rows.append(
                {
                    "comparison": f"HGB minus {comparator}",
                    "metric": metric,
                    "n_paired_folds": len(diff),
                    "mean_difference": np.mean(diff),
                    "ci_low": np.quantile(boot, 0.025),
                    "ci_high": np.quantile(boot, 0.975),
                }
            )
    return pd.DataFrame(rows)


def audit_budget(oof):
    table = oof.copy()
    table["gap"] = 1 - table.y.astype(int)
    table["gap_score"] = 1 - table.prob.astype(float)
    table = table.sort_values("gap_score", ascending=False).reset_index(drop=True)
    total_weight = table.weight.sum()
    total_gap_weight = np.sum(table.weight * table.gap)
    gap_prevalence = total_gap_weight / total_weight
    rows = []
    cumulative_weight = table.weight.cumsum().to_numpy()
    for budget in [0.05, 0.10, 0.20, 0.30]:
        stop = int(np.searchsorted(cumulative_weight, budget * total_weight, side="right")) + 1
        selected = table.iloc[:stop]
        selected_weight = selected.weight.sum()
        gap_weight = np.sum(selected.weight * selected.gap)
        precision = gap_weight / selected_weight
        recall = gap_weight / total_gap_weight
        rows.append(
            {
                "inspection_budget": budget,
                "n_selected": len(selected),
                "weighted_selected_share": selected_weight / total_weight,
                "gap_prevalence": gap_prevalence,
                "precision_gap": precision,
                "recall_gap": recall,
                "lift_over_random": precision / gap_prevalence,
                "false_positive_workload": 1 - precision,
            }
        )
    return pd.DataFrame(rows)


def subgroup_results(oof, residence):
    joined = oof.copy()
    joined["residence"] = residence[joined.row_id.to_numpy(int)]
    rows = []
    for value, part in joined.groupby("residence"):
        yy = part.y.to_numpy(int)
        pp = part.prob.to_numpy(float)
        ww = part.weight.to_numpy(float)
        rows.append(
            {
                "subgroup": value,
                "n": len(part),
                "prevalence": np.average(yy, weights=ww),
                **metric_row(yy, pp, ww),
                **calibration_stats(yy, pp, ww),
            }
        )
    return pd.DataFrame(rows)


def regional_transfer_calibration(X, y, weight, regions):
    rows = []
    for index, region in enumerate(sorted(pd.unique(regions))):
        test = np.flatnonzero(regions == region)
        train = np.flatnonzero(regions != region)
        rng = np.random.default_rng(9000 + index)
        fit = np.sort(rng.choice(train, size=min(100000, len(train)), replace=False))
        model = make_hgb(70)
        model.fit(X.iloc[fit], y[fit], model__sample_weight=weight[fit])
        prob = model.predict_proba(X.iloc[test])[:, 1]
        rows.append(
            {
                "held_out_region": region,
                "n_test": len(test),
                "prevalence": np.average(y[test], weights=weight[test]),
                **metric_row(y[test], prob, weight[test]),
                **calibration_stats(y[test], prob, weight[test]),
                "mean_predicted": np.average(prob, weights=weight[test]),
                "observed_to_predicted_ratio": np.average(y[test], weights=weight[test])
                / np.average(prob, weights=weight[test]),
            }
        )
        print(json.dumps({"completed_region": region}), flush=True)
    return pd.DataFrame(rows)


def weighting_sensitivity(oof, X, y, weight, groups):
    evaluation_rows = []
    for weighting, sw in [("survey-weighted evaluation", oof.weight.to_numpy(float)), ("unweighted evaluation", None)]:
        evaluation_rows.append(
            {
                "analysis": weighting,
                "n": len(oof),
                **metric_row(oof.y.to_numpy(int), oof.prob.to_numpy(float), sw),
            }
        )
    fit_rows = []
    oof_lookup = oof.set_index("row_id")
    seed = SEEDS[0]
    cv = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed)
    for fold, (train, test) in enumerate(cv.split(X, y, groups), 1):
        rng = np.random.default_rng(seed + fold)
        fit = np.sort(rng.choice(train, size=min(200000, len(train)), replace=False))
        model = make_hgb(80)
        model.fit(X.iloc[fit], y[fit])
        prob = model.predict_proba(X.iloc[test])[:, 1]
        weighted_prob = oof_lookup.loc[test, "prob"].to_numpy(float)
        fit_rows.append(
            {
                "fold": fold,
                "n_fit": len(fit),
                "n_test": len(test),
                **metric_row(y[test], prob, weight[test]),
                "probability_correlation_with_weighted_model": np.corrcoef(prob, weighted_prob)[0, 1],
                "mean_absolute_probability_difference": np.mean(np.abs(prob - weighted_prob)),
            }
        )
        print(json.dumps({"completed_unweighted_hgb": fold}), flush=True)
    return pd.DataFrame(evaluation_rows), pd.DataFrame(fit_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-zip", required=True, type=Path)
    parser.add_argument("--csv-name", default="housingandsanitation_10%_20221011d.csv")
    parser.add_argument("--oof-path", required=True, type=Path)
    parser.add_argument("--existing-metrics", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    data, X, y, weight, groups, regions = load_data(args.data_zip, args.csv_name)
    oof = pd.read_csv(args.oof_path)
    existing = pd.read_csv(args.existing_metrics)

    extended = complete_baselines(X, y, weight, groups, existing)
    extended.to_csv(args.output_dir / "extended_model_metrics.csv", index=False)
    model_summary(extended).to_csv(args.output_dir / "extended_model_summary.csv", index=False)
    paired_differences(extended).to_csv(args.output_dir / "paired_model_differences.csv", index=False)
    audit_budget(oof).to_csv(args.output_dir / "inspection_budget.csv", index=False)
    subgroup_results(oof, data.urbrur.astype(str).to_numpy()).to_csv(
        args.output_dir / "urban_rural_performance.csv", index=False
    )
    regional_transfer_calibration(X, y, weight, regions).to_csv(
        args.output_dir / "regional_transfer_calibration.csv", index=False
    )
    evaluation, fit = weighting_sensitivity(oof, X, y, weight, groups)
    evaluation.to_csv(args.output_dir / "weighting_evaluation_sensitivity.csv", index=False)
    fit.to_csv(args.output_dir / "unweighted_hgb_sensitivity.csv", index=False)
    print(json.dumps({"complete_seconds": round(time.time() - start, 1)}), flush=True)


if __name__ == "__main__":
    main()
