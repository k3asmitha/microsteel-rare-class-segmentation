"""
run_experiment.py
====================
THE entrypoint for one experiment run: train -> load best checkpoint ->
evaluate on the test split via the FROZEN evaluator -> save results.

This is what the user actually runs for each of the 8 Type II experiments
(and later, for any other architecture/configuration/type combination).
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.common import load_config
from src.training.train import train_one_experiment
from src.inference.predict import load_trained_predict_fn
from src.eval.evaluator import evaluate_dataset, save_evaluation_results


def _test_stems_for_type(config, true_type):
    import csv
    out_root = config["paths"]["output_dir"]
    with open(os.path.join(out_root, "manifest.csv"), encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [r["stem"] for r in rows if r["true_type"] == true_type and r["split"] == "test"]


def run_one_experiment(config: dict, architecture: str, configuration: str,
                        true_type: str, run_id: str, smoke_test: bool = False,
                        verbose: bool = True) -> dict:
    out_root = config["paths"]["output_dir"]
    run_dir = os.path.join(out_root, "training_runs", run_id)

    if smoke_test:
        config = json.loads(json.dumps(config))  # deep copy, don't mutate caller's config
        config["training"]["epochs"] = config["training"].get("smoke_test_epochs", 2)
        config["training"]["early_stopping_patience"] = 999
        config["training"]["deeplab_encoder_weights"] = None  # avoid needing network for a quick smoke test
        config["training"]["max_batches_per_epoch"] = config["training"].get("smoke_test_max_batches", 3)

    if verbose:
        print("=" * 70)
        print(f"RUN: {run_id}  ({architecture} / {configuration} / Type-{true_type})"
              f"{'  [SMOKE TEST]' if smoke_test else ''}")
        print("=" * 70)

    t0 = time.time()
    train_result = train_one_experiment(config, architecture, configuration, true_type,
                                         run_dir, verbose=verbose)
    train_time = time.time() - t0
    if verbose:
        print(f"Training done in {train_time/60:.1f} min. "
              f"Best val_loss={train_result['best_val_loss']:.4f} "
              f"at epoch {train_result['n_epochs_run']}.")

    predict_fn, ckpt_meta = load_trained_predict_fn(train_result["checkpoint_path"], config)

    if verbose:
        print(f"Evaluating on test split (Type {true_type})...")
    t1 = time.time()
    test_stems = _test_stems_for_type(config, true_type)
    if smoke_test:
        test_stems = test_stems[:1]  # full-image sliding-window reconstruction is
                                       # itself slow on CPU (every patch needs a
                                       # forward pass) -- 1 image is enough to
                                       # mechanically prove reconstruct->metrics
                                       # ->aggregate wiring works with a real
                                       # (if undertrained) model
    eval_result = evaluate_dataset(config, predict_fn, split="test",
                                    image_stems=test_stems,
                                    architecture=architecture, configuration=configuration)
    eval_time = time.time() - t1

    if verbose:
        print(eval_result["summary_table"])
        print(f"Evaluation done in {eval_time:.1f}s.")

    written_dirs = save_evaluation_results(eval_result, out_root)

    summary = {
        "run_id": run_id,
        "architecture": architecture,
        "configuration": configuration,
        "true_type": true_type,
        "smoke_test": smoke_test,
        "train_time_seconds": round(train_time, 1),
        "eval_time_seconds": round(eval_time, 1),
        "checkpoint_path": train_result["checkpoint_path"],
        "checkpoint_metadata": ckpt_meta,
        "n_epochs_run": train_result["n_epochs_run"],
        "best_val_loss": train_result["best_val_loss"],
        "n_test_images": eval_result["n_images"],
        "results_written_to": written_dirs,
    }

    summary_path = os.path.join(run_dir, "run_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    summary["summary_path"] = summary_path

    if verbose:
        print(f"\nRun summary written to {summary_path}")
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", required=True, choices=["DeepLabV3Plus", "AttentionUNet"])
    parser.add_argument("--configuration", required=True, choices=["baseline", "patch", "loss", "both"])
    parser.add_argument("--true-type", required=True, choices=["I", "II", "III"])
    parser.add_argument("--run-id", required=True, help="e.g. II-DL-B")
    parser.add_argument("--smoke-test", action="store_true",
                         help="2 epochs, random-init encoder (no network needed), for a quick mechanical check")
    args = parser.parse_args()

    cfg = load_config("configs/config.yaml")
    result = run_one_experiment(cfg, args.architecture, args.configuration,
                                 args.true_type, args.run_id, smoke_test=args.smoke_test)
    print("\n" + json.dumps({k: v for k, v in result.items() if k != "checkpoint_metadata"}, indent=2, default=str))
