"""
train.py
==========
The common training loop, shared across all 4 configs x 2 architectures.
CPU-first: no CUDA calls anywhere, no assumption a GPU exists. Designed to
be run with very few epochs first (see configs/config.yaml's
training.smoke_test_epochs) to mechanically verify the whole path works
before committing real CPU time to a full run.

CONFIGURATION -> (sampling, loss) mapping, per the 2x2 ablation design:
    baseline -> uniform sampling, plain CE
    patch    -> pixel-weighted TiN oversampling, plain CE
    loss     -> uniform sampling, class-weighted CE
    both     -> pixel-weighted TiN oversampling, class-weighted CE
"""

import csv
import json
import os
import sys
import time

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.common import load_config
from src.models.build_model import build_model
from src.training.dataset_torch import MicroSteelTorchDataset, load_rows
from src.training.sampler_torch import build_weighted_sampler
from src.training.losses import build_loss


def get_class_names_in_id_order(codebook_path: str) -> list:
    """Class names MUST be ordered by integer class ID (0,1,2,...), since
    this order defines which weight tensor index applies to which class in
    nn.CrossEntropyLoss(weight=...). Reading from class_codebook.json's
    class_id_to_name (not from config.yaml's per-type membership lists,
    which aren't guaranteed ID-ordered) is the correct source."""
    with open(codebook_path, encoding="utf-8") as f:
        codebook = json.load(f)
    id_to_name = {int(k): v for k, v in codebook["class_id_to_name"].items()}
    return [id_to_name[i] for i in sorted(id_to_name.keys())]


def train_one_experiment(config: dict, architecture: str, configuration: str,
                          true_type: str, output_dir: str, verbose: bool = True) -> dict:
    """Trains one (architecture, configuration, true_type) combination end
    to end, returns a dict with the best checkpoint path and training
    history. Does NOT run full-image evaluation -- that happens separately
    via src/inference/predict.py using the frozen evaluator, since running
    sliding-window reconstruction every epoch would waste CPU time that's
    better spent on more training epochs."""
    torch.manual_seed(config["training"].get("seed", 0))

    out_root = config["paths"]["output_dir"]
    data_root = config["paths"]["data_root"]
    patch_size = config["patching"]["patch_size"]
    manifest_rows = load_rows(os.path.join(out_root, "manifest.csv"))
    patch_rows = load_rows(os.path.join(out_root, "patch_index.csv"))
    class_names = get_class_names_in_id_order(os.path.join(out_root, "class_codebook.json"))
    num_classes = len(class_names)

    train_cfg = config["training"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- datasets ---
    augment = train_cfg.get("augment", True)
    train_ds = MicroSteelTorchDataset(data_root, manifest_rows, patch_rows,
                                       true_type, "train", augment=augment,
                                       seed=train_cfg.get("seed", 0))
    val_ds = MicroSteelTorchDataset(data_root, manifest_rows, patch_rows,
                                     true_type, "val", augment=False)

    uses_oversampling = configuration in ("patch", "both")
    batch_size = train_cfg["batch_size"]

    if uses_oversampling:
        sampler = build_weighted_sampler(
            train_ds.patch_rows,
            config["sampler"]["target_tin_pixel_multiplier"],
            config["sampler"]["max_tin_pixel_fraction"],
            patch_size,
            config["sampler"]["weight_cap_percentile"],
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                                   drop_last=True, num_workers=0)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                   drop_last=True, num_workers=0)

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             drop_last=False, num_workers=0)

    # --- model, loss, optimizer, scheduler ---
    model, _ = build_model(architecture, num_classes, config)
    model.to(device)

    # Weights come from THIS true_type's train pixels only (and only the
    # classes present in that type) -- not pooled across all types.
    criterion = build_loss(configuration, manifest_rows, class_names,
                            config["loss"]["scheme"], true_type,
                            config["expected_type_class_membership"],
                            effective_number_beta=config["loss"].get("effective_number_beta"),
                            split="train", device=str(device))

    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=train_cfg.get("lr_patience", 3)
    )

    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, "best_checkpoint.pt")
    history_path = os.path.join(output_dir, "training_history.csv")

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    patience = train_cfg.get("early_stopping_patience", 10)
    max_epochs = train_cfg.get("epochs", 30)
    history = []

    if verbose:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"[{architecture}/{configuration}/Type-{true_type}] "
              f"{n_params:,} params, {len(train_ds)} train patches "
              f"({'oversampled' if uses_oversampling else 'uniform'}), "
              f"{len(val_ds)} val patches, device={device}")

    for epoch in range(1, max_epochs + 1):
        epoch_start = time.time()
        max_batches = train_cfg.get("max_batches_per_epoch")  # smoke-test-only limiter

        model.train()
        train_loss_sum, train_batches = 0.0, 0
        for batch_idx, (images, labels) in enumerate(train_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()
            train_batches += 1
        train_loss = train_loss_sum / max(train_batches, 1)

        model.eval()
        val_loss_sum, val_batches = 0.0, 0
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(val_loader):
                if max_batches is not None and batch_idx >= max_batches:
                    break
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                loss = criterion(logits, labels)
                val_loss_sum += loss.item()
                val_batches += 1
        val_loss = val_loss_sum / max(val_batches, 1)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - epoch_start

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "architecture": architecture,
                "configuration": configuration,
                "true_type": true_type,
                "epoch": epoch,
                "val_loss": val_loss,
                "num_classes": num_classes,
            }, checkpoint_path)
        else:
            epochs_without_improvement += 1

        history.append({"epoch": epoch, "train_loss": round(train_loss, 6),
                         "val_loss": round(val_loss, 6), "lr": current_lr,
                         "epoch_seconds": round(epoch_time, 1),
                         "improved": improved})

        if verbose:
            marker = " *" if improved else ""
            print(f"  epoch {epoch:3d}/{max_epochs}: train_loss={train_loss:.4f} "
                  f"val_loss={val_loss:.4f} lr={current_lr:.2e} "
                  f"({epoch_time:.1f}s){marker}")

        if epochs_without_improvement >= patience:
            if verbose:
                print(f"  early stopping at epoch {epoch} "
                      f"(no val_loss improvement for {patience} epochs)")
            break

    with open(history_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss",
                                                 "lr", "epoch_seconds", "improved"])
        writer.writeheader()
        writer.writerows(history)

    return {
        "architecture": architecture,
        "configuration": configuration,
        "true_type": true_type,
        "checkpoint_path": checkpoint_path,
        "history_path": history_path,
        "best_val_loss": best_val_loss,
        "n_epochs_run": len(history),
        "class_names": class_names,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", required=True, choices=["DeepLabV3Plus", "AttentionUNet"])
    parser.add_argument("--configuration", required=True, choices=["baseline", "patch", "loss", "both"])
    parser.add_argument("--true-type", required=True, choices=["I", "II", "III"])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--smoke-test", action="store_true",
                         help="Override epochs to a tiny number for a quick mechanical check")
    args = parser.parse_args()

    cfg = load_config("configs/config.yaml")
    if args.smoke_test:
        cfg["training"]["epochs"] = cfg["training"].get("smoke_test_epochs", 2)
        cfg["training"]["early_stopping_patience"] = 999  # don't early-stop during a smoke test

    result = train_one_experiment(cfg, args.architecture, args.configuration,
                                   args.true_type, args.output_dir)
    print(json.dumps({k: v for k, v in result.items() if k != "class_names"}, indent=2))
