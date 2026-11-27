"""
Classical models on the 32x32 datasets, as a baseline for the radial encoding.

They use the bands of the best radial encoding run in `results.json`, and the same
cross-validation folds and selection rule (highest mean cross-validation accuracy
on the training set) as `run_experiments.py`.
Bands are standardised with the statistics of the data each model is fitted on.

Run from the repository root with:

    uv run python experiments/radial_encoding/run_classical_baseline.py
"""

import json

import numpy as np
from run_experiments import OUTPUT, load_datasets, spatial_folds
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

CLASSICAL_OUTPUT = OUTPUT.parent / "classical_results.json"
CS = [0.1, 1.0, 10.0, 100.0, 1000.0]

MODELS = {
    "svm_linear": (SVC(kernel="linear"), {"C": CS}),
    "svm_rbf": (
        SVC(kernel="rbf"),
        {"C": CS, "gamma": ["scale", 0.001, 0.01, 0.1, 1.0, 10.0]},
    ),
    "svm_poly": (
        SVC(kernel="poly"),
        {"C": CS[:-1], "degree": [2, 3], "gamma": ["scale", 0.1, 1.0]},
    ),
    "logistic_regression": (
        LogisticRegression(max_iter=10000),
        {"C": [0.01, *CS]},
    ),
    "knn": (KNeighborsClassifier(), {"n_neighbors": [1, 3, 5, 11, 21, 51, 101]}),
    "random_forest": (
        RandomForestClassifier(n_estimators=300, random_state=0),
        {"max_depth": [None, 3, 5, 10], "min_samples_leaf": [1, 5, 20]},
    ),
}


def main() -> None:
    """Search every model and save all runs."""
    data = load_datasets()
    folds = spatial_folds(data["train"])
    y_train = data["train"]["label"].to_numpy()
    y_test = data["test"]["label"].to_numpy()

    best_quantum = json.loads(OUTPUT.read_text())["selected"]["overall"]
    features = best_quantum["features"]
    x_train = data["train"][features].to_numpy()
    x_test = data["test"][features].to_numpy()

    runs = []
    for name, (model, grid) in MODELS.items():
        search = GridSearchCV(
            Pipeline([("scaler", StandardScaler()), ("model", model)]),
            {f"model__{key}": values for key, values in grid.items()},
            cv=folds,
            refit=False,
            n_jobs=-1,
        )
        search.fit(x_train, y_train)

        # Refit every setting on all the training data to report all of them
        results = search.cv_results_
        for i, params in enumerate(results["params"]):
            fitted = Pipeline([("scaler", StandardScaler()), ("model", model)])
            fitted.set_params(**params).fit(x_train, y_train)
            predicted = fitted.predict(x_test)
            runs.append(
                {
                    "features": features,
                    "model": name,
                    "params": {k.removeprefix("model__"): v for k, v in params.items()},
                    "results": {
                        "cv_accuracy_mean": float(results["mean_test_score"][i]),
                        "cv_accuracy_std": float(results["std_test_score"][i]),
                        "train_accuracy": float(fitted.score(x_train, y_train)),
                        "test_accuracy": float(np.mean(predicted == y_test)),
                        "test_balanced_accuracy": float(
                            balanced_accuracy_score(y_test, predicted)
                        ),
                        "predicted_labels": predicted.tolist(),
                    },
                }
            )
        print(name, "done", flush=True)

    def best(candidates: list) -> dict:
        return max(candidates, key=lambda r: r["results"]["cv_accuracy_mean"])

    output = {
        "selected": {
            "overall": best(runs),
            "per_model": [best([r for r in runs if r["model"] == m]) for m in MODELS],
        },
        "runs": runs,
    }
    CLASSICAL_OUTPUT.write_text(json.dumps(output) + "\n")

    print(f"Saved {len(runs)} runs to {CLASSICAL_OUTPUT}")
    for run in [output["selected"]["overall"], *output["selected"]["per_model"]]:
        summary = {k: v for k, v in run["results"].items() if k != "predicted_labels"}
        print(run["model"], run["features"], run["params"])
        print("  ", {k: round(v, 3) for k, v in summary.items()})


if __name__ == "__main__":
    main()
