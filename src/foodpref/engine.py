from __future__ import annotations

import contextlib
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader

from .utils import compute_binary_classification_metrics


def create_grad_scaler(device: torch.device) -> torch.amp.GradScaler:
    return torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))


def _autocast_context(device: torch.device) -> contextlib.AbstractContextManager:
    if device.type != "cuda":
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.float16)


def run_epoch(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
) -> Dict[str, Any]:
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_examples = 0
    all_targets = []
    all_preds = []
    all_probs_high = []

    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True).float()
        batch_size = int(y.shape[0])

        if training:
            optimizer.zero_grad(set_to_none=True)

        with _autocast_context(device):
            if not hasattr(model, "forward_raw"):
                raise AttributeError(
                    "FMRIConvNet must expose forward_raw() so training can use BCEWithLogitsLoss safely under autocast."
                )
            logits_high = model.forward_raw(x)
            loss = criterion(logits_high, y)

        if training:
            if scaler is not None and device.type == "cuda":
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        probs_high = torch.sigmoid(logits_high.detach().float()).view(-1)
        preds = (probs_high >= 0.5).long()

        total_loss += float(loss.detach().item()) * batch_size
        total_examples += batch_size
        all_targets.append(y.detach().long().cpu())
        all_preds.append(preds.cpu())
        all_probs_high.append(probs_high.cpu())

    if total_examples <= 0:
        raise ValueError("epoch produced zero examples.")

    targets = torch.cat(all_targets, dim=0)
    preds = torch.cat(all_preds, dim=0)
    probs_high = torch.cat(all_probs_high, dim=0)
    metrics = compute_binary_classification_metrics(targets, preds)
    metrics["loss"] = float(total_loss / total_examples)
    metrics["mean_prob_high"] = float(probs_high.mean().item())
    return metrics


def train_one_epoch(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.amp.GradScaler],
    device: torch.device,
) -> Dict[str, Any]:
    return run_epoch(
        model=model,
        loader=loader,
        criterion=criterion,
        device=device,
        optimizer=optimizer,
        scaler=scaler,
    )


def evaluate_one_epoch(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> Dict[str, Any]:
    with torch.no_grad():
        return run_epoch(
            model=model,
            loader=loader,
            criterion=criterion,
            device=device,
        )
