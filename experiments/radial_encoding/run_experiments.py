"""
Hyperparameter experiments for the radial encoding on the 32x32 datasets.

Every encoding, simulation, kernel and model step is run by `qkernel.Pipeline`.

- Stage 1 varies the features, encoding, Hamiltonian, simulation and SVM `C`,
  with a fixed kernel.
- Stage 2 varies the kernel and `C` for the best simulations of stage 1.

Configurations are ranked by cross-validation accuracy on the training set only.
Encodings which are not embeddable (train or test) are recorded and not simulated.

Run from the repository root with:

    uv run python experiments/radial_encoding/run_experiments.py
"""

import copy
import itertools
import json
import os
import tempfile
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold

from qkernel.encodings import RadialEncoding
from qkernel.model import SVM
from qkernel.pipeline import HAMILTONIANS_DICT, MODELS_DICT, SIMULATORS_DICT, Pipeline
from qkernel.utils import check_embeddability

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent / "results.json"
DATASETS = {"train": "datasets/train_32.csv", "test": "datasets/test_32.csv"}

CONSTRAINTS = {
    "max_atoms": 80,
    "max_distance_from_origin": 38.0,
    "min_distance_between_atoms": 5.0,
}

# Stage 1
FEATURE_SETS = [
    ["b02", "b03", "b04"],
    ["b03", "b04", "b08"],
    ["b02", "b03", "b04", "b08"],
]
SHIFTS = [0.0, 0.5, 1.0, 2.0]
SCALINGS = [3.0, 5.4, 8.0, 12.0, 18.0]
OMEGAS = [np.pi, 2 * np.pi, 4 * np.pi]
DELTAS = [0.0, np.pi]
DURATIONS = [0.33, 0.66, 1.32]
C6 = 865723.02
STAGE_1_KERNEL = {
    "excitations": True,
    "distance_fn": "exp",
    "distance_kwargs": {"mu": 1.0},
}

# Stage 2
N_BEST_SIMULATIONS = 4  # per feature set
KERNELS = [
    {"excitations": e, "distance_fn": f, "distance_kwargs": {"mu": mu}}
    for e, f, mu in itertools.product(
        [True, False], ["exp", "exp_js"], [0.25, 1.0, 4.0, 16.0]
    )
]

CS = [1.0, 10.0, 100.0, 1000.0]

# Cross-validation folds are made of whole spatial blocks, because neighbouring
# pixels are strongly correlated.
N_FOLDS = 4
BLOCK_SIDE = 8

N_WORKERS = max(1, (os.cpu_count() or 2) - 2)


def load_datasets() -> dict:
    """Load the train and test dataframes with lowercase column names."""
    return {
        name: pd.read_csv(ROOT / path).rename(columns=str.lower)
        for name, path in DATASETS.items()
    }


def spatial_folds(df: pd.DataFrame) -> list:
    """Stratified cross-validation folds which keep spatial blocks together."""
    lat_rank = df["latitude"].rank(method="dense").astype(int) - 1
    lon_rank = df["longitude"].rank(method="dense").astype(int) - 1
    blocks = (lat_rank // BLOCK_SIDE) * 1000 + lon_rank // BLOCK_SIDE
    splitter = StratifiedGroupKFold(n_splits=N_FOLDS)
    return list(splitter.split(df, df["label"], groups=blocks))


def make_pipeline(config: dict, data: dict) -> Pipeline:
    """Create a pipeline for a configuration with `features` and `args`."""
    features = config["features"]
    return Pipeline(
        RadialEncoding,
        data["train"][features].to_numpy(),
        data["test"][features].to_numpy(),
        data["train"]["label"].to_numpy(),
        data["test"]["label"].to_numpy(),
        copy.deepcopy(config["args"]),
    )


def check_encoding(features: list, kwargs: dict, data: dict) -> dict:
    """Encode both datasets and check the embeddability conditions."""
    pipeline = make_pipeline(
        {"features": features, "args": {"encoding": {"kwargs": kwargs}}}, data
    )
    pipeline.compute_encoding()

    report = {"features": features, "encoding_kwargs": kwargs, "embeddable": True}
    for name in ["train", "test"]:
        graphs = getattr(pipeline, f"x_{name}")
        failed = check_embeddability(graphs, **CONSTRAINTS)
        report[name] = {
            "n_graphs": len(graphs),
            "n_atoms": graphs.shape[1],
            "max_distance_from_origin": float(np.linalg.norm(graphs, axis=2).max()),
            "min_distance_between_atoms": float(min(pdist(g).min() for g in graphs)),
            "n_graphs_failing": len(failed),
            "violated_constraints": sorted({v for f in failed.values() for v in f}),
            "feature_max_used_to_normalise": float(
                data[name][features].to_numpy().max()
            ),
        }
        report["embeddable"] &= not failed
    return report


def evaluate_models(pipeline: Pipeline, config: dict, folds: list) -> list:
    """Cross-validate, train and test every `C` on the current Gram matrices."""
    runs = []
    for c in CS:
        t1 = time.time()
        model_args = {"type": "svc", "kwargs": {"C": c}}
        pipeline.args["model"] = model_args
        pipeline.model_cls = MODELS_DICT[model_args["type"]]

        cv_scores = []
        for train_idx, val_idx in folds:
            model = SVM(**model_args["kwargs"])
            model.train(
                pipeline.gram_train[np.ix_(train_idx, train_idx)],
                pipeline.y_train[train_idx],
            )
            cv_scores.append(
                model.get_accuracy(
                    pipeline.gram_train[np.ix_(val_idx, train_idx)],
                    pipeline.y_train[val_idx],
                )
            )

        pipeline.run_model()
        args = copy.deepcopy(pipeline.args)
        runs.append(
            {
                "features": config["features"],
                "args": args,
                "results": {
                    "cv_accuracy_mean": float(np.mean(cv_scores)),
                    "cv_accuracy_std": float(np.std(cv_scores)),
                    "cv_accuracy_folds": [float(s) for s in cv_scores],
                    "train_accuracy": float(pipeline.train_accuracy),
                    "test_accuracy": float(pipeline.test_accuracy),
                    "test_balanced_accuracy": float(
                        balanced_accuracy_score(
                            pipeline.y_test, pipeline.predicted_labels
                        )
                    ),
                    "test_confusion_matrix": confusion_matrix(
                        pipeline.y_test, pipeline.predicted_labels, labels=[1, 2, 3]
                    ).tolist(),
                    "predicted_labels": pipeline.predicted_labels.tolist(),
                    "model_time": time.time() - t1,
                },
            }
        )
    return runs


def run_simulation_config(task: tuple) -> tuple:
    """
    Simulate one configuration once and evaluate it with several kernels.

    Returns the list of runs and the list of failures.
    """
    config, kernels, stage = task
    warnings.filterwarnings("ignore", message="divide by zero")
    data = load_datasets()
    folds = spatial_folds(data["train"])
    pipeline = make_pipeline(config, data)

    t1 = time.time()
    pipeline.compute_encoding()
    pipeline.hamiltonian_cls = HAMILTONIANS_DICT[
        pipeline.args["hamiltonian"]["backend"]
    ]
    pipeline.compute_hamiltonian()
    pipeline.simulator_cls = SIMULATORS_DICT[pipeline.args["simulator"]["backend"]]
    pipeline.run_simulation()
    simulation_time = time.time() - t1

    runs, failures = [], []
    for kernel_kwargs in kernels:
        pipeline.args["kernel"] = {"kwargs": kernel_kwargs}
        try:
            t2 = time.time()
            pipeline.compute_kernel()
            kernel_time = time.time() - t2
            for run in evaluate_models(pipeline, config, folds):
                run["stage"] = stage
                run["results"]["simulation_time"] = simulation_time
                run["results"]["kernel_time"] = kernel_time
                runs.append(run)
        except ValueError as error:
            failures.append(
                {
                    "stage": stage,
                    "features": config["features"],
                    "args": copy.deepcopy(pipeline.args),
                    "error": str(error),
                }
            )
    return runs, failures


def run_stage(tasks: list, name: str) -> tuple:
    """Run the tasks of a stage in parallel, printing progress."""
    results = [None] * len(tasks)
    t1 = time.time()
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {
            pool.submit(run_simulation_config, t): i for i, t in enumerate(tasks)
        }
        for done, future in enumerate(as_completed(futures), start=1):
            results[futures[future]] = future.result()
            if done % max(1, len(tasks) // 20) == 0 or done == len(tasks):
                print(
                    f"{name}: {done}/{len(tasks)} simulations, {time.time() - t1:.0f}s",
                    flush=True,
                )
    runs = [run for r, _ in results for run in r]
    failures = [failure for _, f in results for failure in f]
    return runs, failures


def best_run(runs: list) -> dict:
    """Run with the highest cross-validation accuracy (first one if tied)."""
    return max(runs, key=lambda r: r["results"]["cv_accuracy_mean"])


def verify_with_full_pipeline(run: dict, data: dict) -> dict:
    """Repeat a run from scratch with `Pipeline.run` and compare the results."""
    with tempfile.TemporaryDirectory() as path:
        config = copy.deepcopy(run)
        config["args"]["results"] = {"path": path, "name": "verification"}
        pipeline = make_pipeline(config, data)
        pipeline.run()
    return {
        "train_accuracy": float(pipeline.train_accuracy),
        "test_accuracy": float(pipeline.test_accuracy),
        "identical_predictions": pipeline.predicted_labels.tolist()
        == run["results"]["predicted_labels"],
    }


def save(output: dict) -> None:
    """Write the results as JSON, with one list item per line."""
    parts = []
    for key, value in output.items():
        if isinstance(value, list) and value:
            items = ",\n".join(f"  {json.dumps(item)}" for item in value)
            parts.append(f" {json.dumps(key)}: [\n{items}\n ]")
        else:
            parts.append(f" {json.dumps(key)}: {json.dumps(value)}")
    OUTPUT.write_text("{\n" + ",\n".join(parts) + "\n}\n")


def main() -> None:
    """Run both stages and save everything to `results.json`."""
    warnings.filterwarnings("ignore", message="divide by zero")
    t1 = time.time()
    data = load_datasets()

    encodings = [
        check_encoding(features, {"shift": shift, "scaling": scaling}, data)
        for features, shift, scaling in itertools.product(
            FEATURE_SETS, SHIFTS, SCALINGS
        )
    ]
    embeddable = [e for e in encodings if e["embeddable"]]
    print(f"{len(embeddable)}/{len(encodings)} encodings are embeddable", flush=True)

    stage_1_tasks = []
    for encoding, omega, delta, duration in itertools.product(
        embeddable, OMEGAS, DELTAS, DURATIONS
    ):
        config = {
            "features": encoding["features"],
            "args": {
                "encoding": {"kwargs": encoding["encoding_kwargs"]},
                "hamiltonian": {
                    "backend": "qutip",
                    "kwargs": {"omega": omega, "delta": delta, "c6": C6},
                },
                "simulator": {"backend": "qutip", "kwargs": {"duration": duration}},
                "kernel": {"kwargs": STAGE_1_KERNEL},
                "model": {"type": "svc", "kwargs": {"C": CS[0]}},
            },
        }
        stage_1_tasks.append((config, [STAGE_1_KERNEL], 1))

    runs, failures = run_stage(stage_1_tasks, "Stage 1")

    stage_2_tasks = []
    for features in FEATURE_SETS:
        ranked = sorted(
            (r for r in runs if r["features"] == features),
            key=lambda r: r["results"]["cv_accuracy_mean"],
            reverse=True,
        )
        chosen = []
        for run in ranked:
            simulation = {
                k: run["args"][k] for k in ["encoding", "hamiltonian", "simulator"]
            }
            if simulation not in chosen:
                chosen.append(simulation)
            if len(chosen) == N_BEST_SIMULATIONS:
                break
        for simulation in chosen:
            config = {
                "features": features,
                "args": {**simulation, "kernel": {}, "model": {}},
            }
            stage_2_tasks.append((config, KERNELS, 2))

    stage_2_runs, stage_2_failures = run_stage(stage_2_tasks, "Stage 2")
    runs += stage_2_runs
    failures += stage_2_failures

    selected = best_run(runs)
    output = {
        "metadata": {
            "datasets": DATASETS,
            "encoding": "RadialEncoding",
            "embeddability_constraints": CONSTRAINTS,
            "constraint_codes": {
                "0": "too many atoms",
                "1": "atom too far from the origin",
                "2": "atoms too close to each other",
            },
            "search_space": {
                "stage_1": {
                    "features": FEATURE_SETS,
                    "shift": SHIFTS,
                    "scaling": SCALINGS,
                    "omega": OMEGAS,
                    "delta": DELTAS,
                    "duration": DURATIONS,
                    "c6": C6,
                    "kernel": STAGE_1_KERNEL,
                    "C": CS,
                },
                "stage_2": {
                    "best_simulations_per_feature_set": N_BEST_SIMULATIONS,
                    "kernels": KERNELS,
                    "C": CS,
                },
            },
            "selection": (
                f"highest mean accuracy in {N_FOLDS}-fold cross-validation on the "
                f"training set, with folds made of {BLOCK_SIDE}x{BLOCK_SIDE} pixel "
                "spatial blocks"
            ),
            "n_runs": len(runs),
            "n_failures": len(failures),
            "total_time": None,
        },
        "selected": {
            "overall": selected,
            "per_feature_set": [
                best_run([r for r in runs if r["features"] == f]) for f in FEATURE_SETS
            ],
            "full_pipeline_verification": verify_with_full_pipeline(selected, data),
        },
        "encodings": encodings,
        "failures": failures,
        "runs": runs,
    }
    output["metadata"]["total_time"] = time.time() - t1
    save(output)
    print(f"Saved {len(runs)} runs to {OUTPUT}", flush=True)
    summary = {k: v for k, v in selected["results"].items() if "predicted" not in k}
    print(json.dumps({"features": selected["features"], "args": selected["args"]}))
    print(json.dumps(summary))
    print(output["selected"]["full_pipeline_verification"])


if __name__ == "__main__":
    main()
