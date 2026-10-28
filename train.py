"""End-to-end churn model: compare models with CV, tune the best, pick a cost-based
threshold, evaluate on the held-out test set, and explain the model with SHAP.

Usage:  python train.py
Writes: reports/metrics.json, reports/figures/*.png, models/churn_model.joblib
"""
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay, classification_report,
                             confusion_matrix, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold, cross_val_predict,
                                     cross_val_score, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data_cleaning import load_clean
from src.features import CATEGORICAL_COLS, NUMERIC_COLS, add_features

ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "reports" / "figures"
MODEL_PATH = ROOT / "models" / "churn_model.joblib"
SEED = 42
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

# Business assumption: losing a churner costs 5x a retention offer.
# (Theoretical optimum for calibrated probabilities is FP/(FP+FN) = 1/6 ≈ 0.17.)
COST_FN, COST_FP = 5, 1
THRESHOLDS = np.round(np.arange(0.10, 0.701, 0.05), 2)

MODELS = {
    "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "RandomForest": RandomForestClassifier(class_weight="balanced", random_state=SEED, n_jobs=-1),
    "GradientBoosting": GradientBoostingClassifier(random_state=SEED),
}

PARAM_GRIDS = {
    "LogisticRegression": {"model__C": [0.01, 0.1, 1, 10]},
    "RandomForest": {
        "model__n_estimators": [200, 400],
        "model__max_depth": [6, 10, None],
        "model__min_samples_split": [2, 10],
    },
    "GradientBoosting": {
        "model__n_estimators": [100, 200],
        "model__learning_rate": [0.05, 0.1],
        "model__max_depth": [2, 3],
    },
}


def load_data():
    df = add_features(load_clean())
    X = df[NUMERIC_COLS + CATEGORICAL_COLS]
    y = df["Churn"]
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)


def build_pipeline(model):
    # Inside a Pipeline the scaler/encoder are fit on training folds only -> no leakage.
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_COLS),
    ])
    return Pipeline([("prep", preprocessor), ("model", model)])


def compare_models(X_train, y_train):
    rows = []
    for name, model in MODELS.items():
        scores = cross_val_score(build_pipeline(model), X_train, y_train,
                                 cv=CV, scoring="roc_auc", n_jobs=-1)
        rows.append({"model": name, "cv_roc_auc_mean": scores.mean(), "cv_roc_auc_std": scores.std()})
        print(f"{name:<20} ROC-AUC {scores.mean():.4f} ± {scores.std():.4f}")
    return pd.DataFrame(rows).sort_values("cv_roc_auc_mean", ascending=False)


def tune(name, X_train, y_train):
    search = GridSearchCV(build_pipeline(MODELS[name]), PARAM_GRIDS[name],
                          cv=CV, scoring="roc_auc", n_jobs=-1)
    search.fit(X_train, y_train)
    print(f"Best {name} params: {search.best_params_}  CV ROC-AUC {search.best_score_:.4f}")
    return search


def business_cost(y_true, proba, threshold):
    tn, fp, fn, tp = confusion_matrix(y_true, (proba >= threshold).astype(int)).ravel()
    return COST_FN * fn + COST_FP * fp


def pick_threshold(pipe, X_train, y_train):
    """Choose the threshold on out-of-fold training predictions so the test set stays untouched."""
    oof = cross_val_predict(clone(pipe), X_train, y_train, cv=CV, method="predict_proba", n_jobs=-1)[:, 1]
    costs = pd.DataFrame({"threshold": THRESHOLDS,
                          "cost": [business_cost(y_train, oof, t) for t in THRESHOLDS]})
    best = float(costs.loc[costs["cost"].idxmin(), "threshold"])

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(costs["threshold"], costs["cost"], marker="o")
    ax.axvline(best, color="tab:red", linestyle="--", label=f"chosen = {best:.2f}")
    ax.axvline(0.5, color="grey", linestyle=":", label="default = 0.50")
    ax.set(xlabel="Decision threshold", ylabel=f"Cost ({COST_FN}·FN + {COST_FP}·FP)",
           title="Business cost vs threshold (out-of-fold, train)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "threshold_cost.png", dpi=150)
    plt.close(fig)
    return best, costs


def evaluate(pipe, X_test, y_test, threshold):
    proba = pipe.predict_proba(X_test)[:, 1]
    results = {"test_roc_auc": roc_auc_score(y_test, proba)}
    for label, t in [("default_0.50", 0.5), ("tuned", threshold)]:
        pred = (proba >= t).astype(int)
        report = classification_report(y_test, pred, output_dict=True)
        results[label] = {
            "threshold": t,
            "churn_precision": report["1"]["precision"],
            "churn_recall": report["1"]["recall"],
            "churn_f1": report["1"]["f1-score"],
            "accuracy": report["accuracy"],
            "business_cost": int(business_cost(y_test, proba, t)),
            "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        }
        print(f"\nThreshold {t:.2f}\n{classification_report(y_test, pred)}")

    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(y_test, proba, ax=ax, name="Tuned model")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_title("ROC curve (test set)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "roc_curve.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y_test, (proba >= threshold).astype(int), ax=ax,
                                            display_labels=["Stay", "Churn"], cmap="Blues")
    ax.set_title(f"Confusion matrix (threshold = {threshold:.2f})")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    return results


def explain(pipe, X_train, X_test, top_n=10):
    prep, model = pipe.named_steps["prep"], pipe.named_steps["model"]
    names = [n.split("__", 1)[1] for n in prep.get_feature_names_out()]
    X_test_t = pd.DataFrame(prep.transform(X_test), columns=names)

    if isinstance(model, LogisticRegression):
        explainer = shap.LinearExplainer(model, prep.transform(X_train))
    else:
        explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(X_test_t)
    if isinstance(values, list):  # older shap: one array per class
        values = values[1]
    if values.ndim == 3:  # newer shap on RF: (samples, features, classes)
        values = values[:, :, 1]

    shap.summary_plot(values, X_test_t, max_display=15, show=False)
    plt.title("SHAP summary (test set)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Sum one-hot columns back into their source feature (e.g. all Contract_* -> Contract).
    source = [next((c for c in CATEGORICAL_COLS if n.startswith(c + "_")), n) for n in names]
    grouped = pd.DataFrame(values, columns=names).T.groupby(source).sum().T
    importance = grouped.abs().mean().sort_values(ascending=False)
    return importance.head(top_n)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    X_train, X_test, y_train, y_test = load_data()
    print(f"Train {X_train.shape}, test {X_test.shape}, churn rate {y_train.mean():.3f}\n")

    comparison = compare_models(X_train, y_train)
    # Tune every candidate (grids are small) and keep the best tuned CV score.
    searches = {name: tune(name, X_train, y_train) for name in MODELS}
    comparison["tuned_cv_roc_auc"] = comparison["model"].map({n: s.best_score_ for n, s in searches.items()})
    best_name = max(searches, key=lambda n: searches[n].best_score_)
    search = searches[best_name]
    pipe = search.best_estimator_
    print(f"Selected model: {best_name}")

    threshold, costs = pick_threshold(pipe, X_train, y_train)
    print(f"Cost-optimal threshold: {threshold:.2f}")
    results = evaluate(pipe, X_test, y_test, threshold)
    drivers = explain(pipe, X_train, X_test)
    print(f"\nTest ROC-AUC: {results['test_roc_auc']:.4f}\nTop churn drivers (mean |SHAP|):\n{drivers}")

    metrics = {
        "cv_comparison": comparison.round(4).to_dict(orient="records"),
        "best_model": best_name,
        "best_params": search.best_params_,
        "best_cv_roc_auc": round(search.best_score_, 4),
        "threshold_costs_train_oof": costs.to_dict(orient="records"),
        **results,
        "top_drivers_mean_abs_shap": drivers.round(4).to_dict(),
    }
    (ROOT / "reports" / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipe, "threshold": threshold}, MODEL_PATH)
    print(f"\nSaved metrics to reports/metrics.json and model to {MODEL_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
