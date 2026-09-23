"""Resource-bounded supervised warm-start for the ORCS-style SONIC adapter.

The input archive must contain ``base_action`` produced by the exact frozen SONIC build and
``target_action`` from an accepted teacher/rollout. This command intentionally refuses to invent
a base action from joint references; PPO fine-tuning or privileged distillation starts only after
the action-aligned archive exists.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .sonic_adapter import SonicAdapterConfig, SonicConditionAdapter, stage2_augmentation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=20260923)
    args = parser.parse_args()
    with np.load(args.dataset, allow_pickle=False) as archive:
        required = {"base_action", "target_action", "state", "history", "manifold",
                    "corridor", "sdf", "command", "primitive", "split"}
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError(
                f"adapter dataset is not action-aligned; missing {missing}. "
                "Generate frozen-SONIC base_action and accepted teacher target_action first."
            )
        values = {key: np.asarray(archive[key]) for key in required}
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    tensors = {key: torch.as_tensor(value, device=device) for key, value in values.items()
               if key not in {"primitive", "split"}}
    primitive_count = int(np.max(values["primitive"])) + 1
    one_hot = torch.nn.functional.one_hot(
        torch.as_tensor(values["primitive"], device=device).long(), primitive_count).float()
    augmentation = stage2_augmentation(
        tensors["state"].float(), tensors["history"].float(), tensors["manifold"].float(),
        tensors["corridor"].float(), tensors["sdf"].float(), tensors["command"].float(), one_hot)
    base = tensors["base_action"].float()
    target = tensors["target_action"].float()
    model = SonicConditionAdapter(SonicAdapterConfig(
        condition_dim=int(augmentation.shape[1]), rank=args.rank, alpha=args.alpha)).to(device)
    parity = model.zero_init_parity(base[:min(64, len(base))], augmentation[:min(64, len(base))])
    if parity != 0.0:
        raise RuntimeError(f"zero-init adapter changed frozen SONIC by {parity}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    train_indices = np.flatnonzero(values["split"] == 0)
    val_indices = np.flatnonzero(values["split"] == 1)
    test_indices = np.flatnonzero(values["split"] == 2)
    generator = np.random.default_rng(args.seed)
    eval_indices = val_indices if len(val_indices) else train_indices

    def mse(index: np.ndarray) -> float:
        with torch.no_grad():
            tensor_index = torch.as_tensor(index, device=device)
            return float(torch.mean((model(base[tensor_index], augmentation[tensor_index]) -
                                     target[tensor_index]) ** 2).item())

    base_mse = {
        "train": float(np.mean((values["base_action"][train_indices] -
                                 values["target_action"][train_indices]) ** 2)),
        "validation": float(np.mean((values["base_action"][val_indices] -
                                      values["target_action"][val_indices]) ** 2))
        if len(val_indices) else None,
        "test": float(np.mean((values["base_action"][test_indices] -
                                values["target_action"][test_indices]) ** 2))
        if len(test_indices) else None,
    }
    history = []
    best_validation = float("inf")
    best_epoch = 0
    best_state = None
    for epoch in range(args.epochs):
        generator.shuffle(train_indices)
        model.train()
        losses = []
        for begin in range(0, len(train_indices), args.batch_size):
            index = torch.as_tensor(train_indices[begin:begin + args.batch_size], device=device)
            prediction = model(base[index], augmentation[index])
            loss = torch.mean((prediction - target[index]) ** 2)
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); losses.append(float(loss.item()))
        model.eval()
        validation = mse(eval_indices)
        history.append({"epoch": epoch + 1, "train_mse": float(np.mean(losses)),
                        "validation_mse": validation})
        if validation < best_validation:
            best_validation = validation
            best_epoch = epoch + 1
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    fitted_mse = {
        "train": mse(train_indices),
        "validation": mse(val_indices) if len(val_indices) else None,
        "test": mse(test_indices) if len(test_indices) else None,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    torch.save(model.export_state(), args.out / "sonic_condition_adapter.pt")
    report = {
        "dataset": str(args.dataset), "zero_init_base_parity_max_abs": parity,
        "train_count": int(len(train_indices)), "validation_count": int(len(val_indices)),
        "test_count": int(len(test_indices)),
        "trainable_parameters": model.trainable_parameter_count(), "history": history,
        "seed": args.seed, "best_epoch": best_epoch,
        "base_mse": base_mse, "fitted_mse": fitted_mse,
        "boundary": (
            "supervised warm-start only; privileged PPO and the full MuJoCo acceptance matrix "
            "are required before replacing the zero adapter"
        ),
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({**report, "history": history[-3:]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
