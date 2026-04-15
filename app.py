from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, render_template, request, send_file

from main import PipelineResult, build_config, run_pipeline


PROJECT_ROOT = Path(__file__).resolve().parent
app = Flask(__name__, template_folder="templates", static_folder="static")


def _load_json(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_plot_key(path: str) -> str:
    return Path(path).name.replace('.png', '.html')


def summarize_result(result: PipelineResult) -> dict[str, object]:
    baseline_rf = float(result.accuracy_comparison["rf_baseline"])
    defended_rf = float(result.accuracy_comparison["rf_defended"])
    reduction = baseline_rf - defended_rf
    return {
        "backend": result.backend,
        "config": result.config,
        "dataset_path": result.dataset_path,
        "total_rows": result.total_rows,
        "baseline_rows": result.baseline_rows,
        "defended_rows": result.defended_rows,
        "metrics": result.metrics,
        "accuracy_comparison": result.accuracy_comparison,
        "plots": {key: f"/plots/{_safe_plot_key(value)}" for key, value in result.plots.items()},
        "story": {
            "leakage_strength": "High" if baseline_rf >= 0.9 else "Moderate",
            "defense_effectiveness": "Visible reduction" if reduction >= 0.05 else "Limited reduction",
            "rf_reduction": round(reduction * 100, 2),
            "torch_reduction": round(
                (float(result.accuracy_comparison["torch_baseline"]) - float(result.accuracy_comparison["torch_defended"])) * 100,
                2,
            ),
            "suspicious_accuracy": baseline_rf >= 0.98,
        },
    }


def load_existing_summary() -> dict[str, object] | None:
    config = build_config()
    baseline = _load_json(config.metrics_dir / "metrics_baseline.json")
    defended = _load_json(config.metrics_dir / "metrics_defended.json")
    comparison = _load_json(config.metrics_dir / "accuracy_comparison.json")
    dataset_path = config.dataset_path
    if baseline is None or defended is None or comparison is None or not dataset_path.exists():
        return None

    dataframe = pd.read_csv(dataset_path)
    plots = {
        "accuracy_comparison": str(config.plots_dir / "accuracy_comparison.html"),
        "feature_importance_baseline": str(config.plots_dir / "feature_importance_baseline.html"),
        "feature_importance_defended": str(config.plots_dir / "feature_importance_defended.html"),
        "confusion_matrix_rf_baseline": str(config.plots_dir / "confusion_matrix_rf_baseline.html"),
        "confusion_matrix_rf_defended": str(config.plots_dir / "confusion_matrix_rf_defended.html"),
        "confusion_matrix_torch_baseline": str(config.plots_dir / "confusion_matrix_torch_baseline.html"),
        "confusion_matrix_torch_defended": str(config.plots_dir / "confusion_matrix_torch_defended.html"),
    }
    pipeline_result = PipelineResult(
        backend=str(config.backend_preference),
        config=config.as_dict(),
        dataset_path=str(dataset_path),
        total_rows=int(len(dataframe)),
        baseline_rows=int((dataframe["defense"] == 0).sum()),
        defended_rows=int((dataframe["defense"] == 1).sum()),
        metrics={"baseline": baseline, "defended": defended},
        accuracy_comparison={key: float(value) for key, value in comparison.items()},
        plots=plots,
    )
    return summarize_result(pipeline_result)


@app.get("/")
def index():
    return render_template("index.html", initial_state=load_existing_summary())


@app.get("/api/results")
def get_results():
    state = load_existing_summary()
    if state is None:
        return jsonify({"available": False})
    return jsonify({"available": True, "result": state})


@app.post("/api/run")
def run_simulation():
    payload = request.get_json(silent=True) or {}
    overrides = {
        "samples_per_operation": int(payload.get("samples_per_operation", 50)),
        "torch_epochs": int(payload.get("torch_epochs", 10)),
        "backend_preference": str(payload.get("backend_preference", "mock")),
        "defense_noise_scale": float(payload.get("defense_noise_scale", 0.08)),
    }
    try:
        result = run_pipeline(overrides=overrides)
    except Exception as exc:  # pragma: no cover - UI safety
        return jsonify({"ok": False, "error": str(exc)}), 500
    return jsonify({"ok": True, "result": summarize_result(result)})


@app.get("/plots/<path:filename>")
def serve_plot(filename: str):
    target = (PROJECT_ROOT / "artifacts" / "plots" / filename).resolve()
    plots_root = (PROJECT_ROOT / "artifacts" / "plots").resolve()
    if plots_root not in target.parents and target.parent != plots_root:
        return jsonify({"error": "invalid plot path"}), 400
    if not target.exists():
        return jsonify({"error": "plot not found"}), 404
    return send_file(target)


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
