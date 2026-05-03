"""Aggregate saved metrics into CSV tables for the final notebook."""

import argparse
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"
DEFAULT_TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"

MODEL_TABLES = {
    "baseline": "baseline_summary.csv",
    "concept": "concept_predictor_summary.csv",
    "cbm_independent": "independent_cbm_summary.csv",
    "cbm_joint": "joint_cbm_summary.csv",
    "hybrid": "hybrid_cbm_summary.csv",
}

MODEL_LABELS = {
    "baseline": "Baseline",
    "concept": "Concept predictor",
    "cbm_independent": "Independent CBM",
    "cbm_joint": "Joint CBM",
    "hybrid": "Hybrid CBM",
}

SUMMARY_METRIC_KEYS = [
    "accuracy",
    "auroc_ovr",
    "macro_concept_accuracy",
    "macro_concept_f1",
]

CONCEPT_DETAIL_KEYS = [
    ("concept_accuracy_per_concept", "concept_accuracy"),
    ("concept_f1_per_concept", "concept_f1"),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create CSV summary tables from saved experiment metrics.",
    )
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=DEFAULT_METRICS_DIR,
        help="Directory containing saved metric JSON/CSV files.",
    )
    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=DEFAULT_TABLES_DIR,
        help="Directory where summary CSV tables will be saved.",
    )
    return parser.parse_args()


def warn(message):
    print(f"WARNING: {message}")


def read_json(path):
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as exc:
        warn(f"Could not parse JSON file {path.name}: {exc}")
    except OSError as exc:
        warn(f"Could not read JSON file {path.name}: {exc}")

    return None


def read_csv(path):
    try:
        return pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError) as exc:
        warn(f"Could not read CSV file {path.name}: {exc}")

    return None


def write_table(rows, output_path):
    if not rows:
        warn(f"No rows available for {output_path.name}; table was not written.")
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Wrote {output_path}")
    return True


def write_dataframe_table(dataframe, output_path):
    if dataframe is None or dataframe.empty:
        warn(f"No rows available for {output_path.name}; table was not written.")
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False)
    print(f"Wrote {output_path}")
    return True


def get_best_val_loss(history):
    if not isinstance(history, dict):
        return None

    if "best_val_loss" in history:
        return history["best_val_loss"]

    best_losses = []
    for value in history.values():
        if isinstance(value, dict) and "best_val_loss" in value:
            best_losses.append(value["best_val_loss"])

    if not best_losses:
        return None

    return min(best_losses)


def get_epochs_trained(history):
    if not isinstance(history, dict):
        return None

    if "epochs_trained" in history:
        return history["epochs_trained"]

    epochs = []
    for value in history.values():
        if isinstance(value, dict) and "epochs_trained" in value:
            epochs.append(value["epochs_trained"])

    if not epochs:
        return None

    return max(epochs)


def make_model_summary_row(path, data):
    args = data.get("args", {})
    test_metrics = data.get("test_metrics", {})
    history = data.get("history", {})
    model = data.get("model")

    row = {
        "source_file": path.name,
        "run_name": data.get("run_name", path.stem),
        "model": model,
        "model_label": MODEL_LABELS.get(model, model),
        "seed": args.get("seed"),
        "epochs": args.get("epochs"),
        "batch_size": args.get("batch_size"),
        "lr": args.get("lr"),
        "lambda_concept": args.get("lambda_concept"),
        "side_dropout": args.get("side_dropout"),
        "early_stopping_patience": data.get("early_stopping_patience"),
        "epochs_trained": get_epochs_trained(history),
        "best_val_loss": get_best_val_loss(history),
        "checkpoint_path": data.get("checkpoint_path"),
    }

    for metric_key in SUMMARY_METRIC_KEYS:
        if metric_key in test_metrics:
            row[metric_key] = test_metrics[metric_key]

    return row


def make_concept_detail_rows(path, data):
    test_metrics = data.get("test_metrics", {})
    concept_names = test_metrics.get("concept_names", [])
    if not concept_names:
        return []

    detail_values = {}
    for source_key, output_key in CONCEPT_DETAIL_KEYS:
        values = test_metrics.get(source_key)
        if values is not None:
            detail_values[output_key] = values

    if not detail_values:
        return []

    model = data.get("model")
    args = data.get("args", {})
    run_name = data.get("run_name", path.stem)
    rows = []

    for concept_index, concept_name in enumerate(concept_names):
        row = {
            "source_file": path.name,
            "run_name": run_name,
            "model": model,
            "model_label": MODEL_LABELS.get(model, model),
            "seed": args.get("seed"),
            "concept_index": concept_index,
            "concept_name": concept_name,
        }

        for output_key, values in detail_values.items():
            if concept_index < len(values):
                row[output_key] = values[concept_index]
            else:
                warn(
                    f"{path.name} has no {output_key} value for concept "
                    f"{concept_index} ({concept_name})."
                )

        rows.append(row)

    return rows


def collect_model_metrics(metrics_dir):
    rows_by_model = {model: [] for model in MODEL_TABLES}
    concept_detail_rows = []

    for path in sorted(metrics_dir.glob("*.json")):
        if path.stem.endswith("_interventions"):
            continue

        data = read_json(path)
        if not isinstance(data, dict):
            continue

        model = data.get("model")
        test_metrics = data.get("test_metrics")
        if model not in MODEL_TABLES or not isinstance(test_metrics, dict):
            warn(f"Skipping {path.name}: not a recognized experiment metrics file.")
            continue

        rows_by_model[model].append(make_model_summary_row(path, data))
        concept_detail_rows.extend(make_concept_detail_rows(path, data))

    return rows_by_model, concept_detail_rows


def collect_dropout_sweeps(metrics_dir):
    frames = []
    for path in sorted(metrics_dir.glob("hybrid_dropout_sweep*.csv")):
        dataframe = read_csv(path)
        if dataframe is None:
            continue

        dataframe.insert(0, "source_file", path.name)
        dataframe.insert(1, "sweep_run", path.stem)
        frames.append(dataframe)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True, sort=False)


def infer_model_from_intervention_name(stem):
    if stem.startswith("cbm_independent"):
        return "cbm_independent"
    if stem.startswith("cbm_joint") or stem.startswith("smoke_joint"):
        return "cbm_joint"
    if stem.startswith("hybrid") or stem.startswith("smoke_hybrid"):
        return "hybrid"
    return None


def collect_intervention_rankings(metrics_dir):
    rows = []
    json_stems = set()

    for path in sorted(metrics_dir.glob("*_interventions.json")):
        data = read_json(path)
        if not isinstance(data, dict):
            continue

        results = data.get("results")
        if not isinstance(results, list):
            warn(f"Skipping {path.name}: missing intervention results list.")
            continue

        json_stems.add(path.stem)
        model = data.get("model") or infer_model_from_intervention_name(path.stem)
        checkpoint_args = data.get("checkpoint_args", {})

        for result in results:
            if not isinstance(result, dict):
                warn(f"Skipping malformed intervention row in {path.name}.")
                continue

            row = {
                "source_file": path.name,
                "intervention_run": path.stem,
                "model": model,
                "model_label": MODEL_LABELS.get(model, model),
                "checkpoint_path": data.get("checkpoint_path"),
                "checkpoint_seed": checkpoint_args.get("seed"),
                "checkpoint_epochs": checkpoint_args.get("epochs"),
                "checkpoint_batch_size": checkpoint_args.get("batch_size"),
                "checkpoint_lr": checkpoint_args.get("lr"),
                "lambda_concept": checkpoint_args.get("lambda_concept"),
                "side_dropout": checkpoint_args.get("side_dropout"),
                "evaluation_seed": data.get("seed"),
                "evaluation_batch_size": data.get("batch_size"),
                "threshold": data.get("threshold"),
                "ranking_metric": data.get("ranking_metric"),
            }
            row.update(result)
            rows.append(row)

    for path in sorted(metrics_dir.glob("*_interventions.csv")):
        if path.stem in json_stems:
            continue

        dataframe = read_csv(path)
        if dataframe is None:
            continue

        model = infer_model_from_intervention_name(path.stem)
        dataframe.insert(0, "source_file", path.name)
        dataframe.insert(1, "intervention_run", path.stem)
        dataframe.insert(2, "model", model)
        dataframe.insert(3, "model_label", MODEL_LABELS.get(model, model))
        rows.extend(dataframe.to_dict(orient="records"))

    return rows


def main():
    args = parse_args()
    metrics_dir = args.metrics_dir
    tables_dir = args.tables_dir

    if not metrics_dir.exists():
        warn(f"Metrics directory does not exist: {metrics_dir}")
        return

    rows_by_model, concept_detail_rows = collect_model_metrics(metrics_dir)

    all_model_rows = []
    for model, output_name in MODEL_TABLES.items():
        rows = rows_by_model[model]
        all_model_rows.extend(rows)
        if not rows:
            warn(f"No {MODEL_LABELS[model]} metrics found in {metrics_dir}.")
        write_table(rows, tables_dir / output_name)

    write_table(all_model_rows, tables_dir / "model_comparison_summary.csv")
    write_table(concept_detail_rows, tables_dir / "concept_metric_details.csv")

    dropout_sweeps = collect_dropout_sweeps(metrics_dir)
    if dropout_sweeps.empty:
        warn(f"No dropout sweep CSV files found in {metrics_dir}.")
    write_dataframe_table(dropout_sweeps, tables_dir / "dropout_sweep_summary.csv")

    intervention_rows = collect_intervention_rankings(metrics_dir)
    if not intervention_rows:
        warn(f"No intervention ranking files found in {metrics_dir}.")
    write_table(intervention_rows, tables_dir / "intervention_ranking_summary.csv")


if __name__ == "__main__":
    main()
