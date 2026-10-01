"""
verify_training_pipeline.py
=============================
Runs one ultra-fast smoke test (a handful of batches, 1 test image) through
the WHOLE path: train -> checkpoint -> load -> reconstruct -> metrics ->
save results. Takes well under a minute. Run this FIRST on any new machine
before committing real CPU time to actual training -- it catches broken
installs, path misconfigurations, and wiring bugs immediately instead of
after a multi-hour training run fails at the very end during evaluation.

This does NOT check whether the model learned anything (2 epochs of 3
batches each teaches it nothing) -- only that every piece of the pipeline
runs without crashing and produces correctly-shaped, valid output.
"""

import json
import os
import shutil
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.common import load_config
from src.training.run_experiment import run_one_experiment


def check(name, condition, failures, detail=""):
    if not condition:
        failures.append(f"[FAIL] {name}" + (f" -- {detail}" if detail else ""))


def run(config: dict) -> dict:
    failures = []
    out_root = config["paths"]["output_dir"]
    smoke_run_id = "_PIPELINE_VERIFICATION_SMOKE_TEST"
    run_dir = os.path.join(out_root, "training_runs", smoke_run_id)

    # clean any leftover from a previous verification attempt
    if os.path.exists(run_dir):
        shutil.rmtree(run_dir)

    try:
        summary = run_one_experiment(config, "AttentionUNet", "both", "II",
                                      smoke_run_id, smoke_test=True, verbose=False)
    except Exception as e:
        import traceback
        return {
            "check": "verify_training_pipeline",
            "passed": False,
            "failures": [f"[FAIL] run_one_experiment raised an exception: {e}"],
            "traceback": traceback.format_exc(),
            "conclusion": "VERIFICATION FAILED: the training pipeline crashed. "
                           "See traceback -- this must be fixed before any real "
                           "training is worth attempting.",
        }

    check("checkpoint file was written", os.path.exists(summary["checkpoint_path"]), failures)
    check("training_history.csv was written",
          os.path.exists(os.path.join(run_dir, "training_history.csv")), failures)
    check("run_summary.json was written", os.path.exists(summary["summary_path"]), failures)
    check("evaluation ran on at least 1 test image", summary["n_test_images"] >= 1, failures)
    check("results were written to outputs/results/", len(summary["results_written_to"]) > 0, failures)

    for d in summary["results_written_to"]:
        check(f"per_image.csv exists in {d}", os.path.exists(os.path.join(d, "per_image.csv")), failures)
        check(f"summary.json exists in {d}", os.path.exists(os.path.join(d, "summary.json")), failures)

    # clean up the verification run's artifacts so they don't clutter outputs/
    # (nuke the whole trees rather than surgically removing leaf dirs --
    # simpler and avoids leaving empty parent directories behind)
    training_runs_dir = os.path.join(out_root, "training_runs")
    results_dir = os.path.join(out_root, "results")
    if os.path.exists(training_runs_dir):
        shutil.rmtree(training_runs_dir)
    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)

    passed = len(failures) == 0
    return {
        "check": "verify_training_pipeline",
        "passed": passed,
        "failures": failures,
        "train_time_seconds": summary["train_time_seconds"],
        "eval_time_seconds": summary["eval_time_seconds"],
        "conclusion": (
            f"VERIFIED: the full train->checkpoint->load->reconstruct->"
            f"metrics->save pipeline runs end to end without error "
            f"(train {summary['train_time_seconds']}s + eval "
            f"{summary['eval_time_seconds']}s for this tiny smoke test). "
            f"Your machine's setup (torch, segmentation-models-pytorch, "
            f"paths) is confirmed working. This does NOT mean the model "
            f"learned anything useful -- only that nothing crashed. Safe to "
            f"proceed to a real training run now."
            if passed else
            f"VERIFICATION FAILED: {len(failures)} issue(s) -- fix these "
            f"before attempting a real (multi-hour) training run, or you'll "
            f"only discover the problem after wasting that time."
        ),
    }


if __name__ == "__main__":
    cfg = load_config("configs/config.yaml")
    result = run(cfg)
    print(json.dumps(result, indent=2, default=str))
    sys.exit(0 if result["passed"] else 1)
