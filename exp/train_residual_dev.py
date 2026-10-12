
"""DLinear Residual DEV: 3 folds x 3 draws, no Lockbox."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from exp.residual_model import DLinearResidual
from exp.timeseries_data import build_training_data
from exp.splits import universe_draws, fold_masks
from exp.train_dlinear import (
    SEED, SPLIT, FOLDS, EMBARGO_DAYS,
    split_calibration, prepare_arrays, set_seed,
)
from src.data import score, label_of
from src.model import KRule

ALPHAS = [-2, -1, -0.5, -0.25, 0, 0.25, 0.5, 1, 2]
OUTPUT = Path("runs/residual_dev")


def evaluate_one(draw, fold):
    print(
        f"\n=== Draw {draw['draw']} / Fold {fold['name']} ===",
        flush=True,
    )

    train_panel = pd.concat(
        [build_training_data(u) for u in draw["train_universes"]],
        ignore_index=True,
    )
    test_panel = build_training_data(draw["test_universe"])

    tr, te = fold_masks(
        train_panel, test_panel, fold, EMBARGO_DAYS
    )

    train = train_panel.loc[tr].dropna(
        subset=["gap_pct", "ret_pct", "label"]
    ).copy()
    test = test_panel.loc[te].dropna(
        subset=["gap_pct", "ret_pct", "label"]
    ).copy()

    if train.empty or test.empty:
        raise ValueError("Train/Test data is empty")

    fit, calib = split_calibration(train)

    assert not set(draw["unseen"]) & set(train["symbol"])
    assert fit["target"].max() < (
        calib["target"].min() - pd.Timedelta(days=EMBARGO_DAYS)
    )
    assert train["target"].max() < (
        pd.Timestamp(fold["start"])
        - pd.Timedelta(days=EMBARGO_DAYS)
    )
    assert (
        (test["target"] >= pd.Timestamp(fold["start"]))
        & (test["target"] < pd.Timestamp(fold["end"]))
    ).all()

    print(
        f"Fit={len(fit)} Calib={len(calib)} Test={len(test)}",
        flush=True,
    )

    xf, xc, xv, _, _, _ = prepare_arrays(
        fit, calib, test, gap_only=False
    )

    target = (
        fit["ret_pct"].to_numpy(dtype=np.float32)
        - fit["gap_pct"].to_numpy(dtype=np.float32)
    )

    y_fit = torch.from_numpy(
        np.clip(target, -15, 15).copy()
    )

    seed = SEED + 100 * draw["draw"] + ord(fold["name"])
    set_seed(seed)

    model = DLinearResidual()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(15):
        model.train()
        total = 0.0

        for idx in torch.randperm(len(xf)).split(128):
            pred = model(xf[idx])
            loss = F.smooth_l1_loss(pred, y_fit[idx])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total += loss.item() * len(idx)

        if epoch in (0, 14):
            print(
                f"Epoch {epoch+1:02d}/15 "
                f"Loss={total/len(xf):.4f}",
                flush=True,
            )

    model.eval()

    with torch.no_grad():
        r_cal = model(xc).numpy()
        r_test = model(xv).numpy()

    gap_cal = calib["gap_pct"].to_numpy(float)
    gap_test = test["gap_pct"].to_numpy(float)

    y_cal = calib["label"].to_numpy(int)
    y_test = test["label"].to_numpy(int)

    objective = lambda y, p: score(y, p)["score"]

    # Baseline and candidate use the SAME calibration set
    b2 = KRule()
    b2.fit(gap_cal, y_cal, objective)
    b2_pred = b2.predict(gap_test)

    # alpha=0 reproduces the calibrated B2
    best_alpha = 0
    best_k = float(b2.k)
    best_cal_score = objective(
        y_cal, b2.predict(gap_cal)
    )

    for alpha in ALPHAS:
        if alpha == 0:
            continue

        cal_signal = gap_cal + alpha * r_cal

        rule = KRule()
        rule.fit(cal_signal, y_cal, objective)

        candidate_cal_score = objective(
            y_cal, rule.predict(cal_signal)
        )

        if candidate_cal_score > best_cal_score + 1e-12:
            best_alpha = alpha
            best_k = float(rule.k)
            best_cal_score = candidate_cal_score

    test_signal = gap_test + best_alpha * r_test

    residual_pred = label_of(
        best_k * test_signal
    ).to_numpy(dtype=int)

    b2_metrics = score(y_test, b2_pred)
    residual_metrics = score(y_test, residual_pred)

    delta = residual_metrics["score"] - b2_metrics["score"]

    print(
        f"B2={b2_metrics['score']:.4f} "
        f"Residual={residual_metrics['score']:.4f} "
        f"Delta={delta:+.4f} "
        f"alpha={best_alpha} k={best_k:.2f}",
        flush=True,
    )

    metrics = []

    for name, m, k in [
        ("B2", b2_metrics, b2.k),
        ("Residual", residual_metrics, best_k),
    ]:
        metrics.append({
            "draw": draw["draw"],
            "fold": fold["name"],
            "model": name,
            "alpha": 0 if name == "B2" else best_alpha,
            "k": k,
            **m,
        })

    predictions = test[
        ["symbol", "target", "ret_pct", "label"]
    ].copy()
    predictions["draw"] = draw["draw"]
    predictions["fold"] = fold["name"]
    predictions["is_unseen"] = predictions["symbol"].isin(
        draw["unseen"]
    )
    predictions["pred_B2"] = b2_pred
    predictions["pred_Residual"] = residual_pred

    return metrics, predictions


def main():
    set_seed(SEED)

    symbols = sorted(
        pd.read_parquet(
            "dataset/daily.parquet",
            columns=["symbol"],
        )["symbol"].unique()
    )

    draws = universe_draws(symbols, SPLIT, SEED)

    all_metrics = []
    all_predictions = []

    OUTPUT.mkdir(parents=True, exist_ok=True)

    for draw in draws:
        for fold in FOLDS:
            metrics, predictions = evaluate_one(draw, fold)
            all_metrics.extend(metrics)
            all_predictions.append(predictions)

            # Save progress after each run
            pd.DataFrame(all_metrics).to_csv(
                OUTPUT / "metrics.csv", index=False
            )
            pd.concat(
                all_predictions, ignore_index=True
            ).to_parquet(
                OUTPUT / "predictions.parquet",
                index=False,
            )

    result = pd.DataFrame(all_metrics)
    pred = pd.concat(all_predictions, ignore_index=True)

    print("\n=== Mean DEV Metrics ===")
    print(
        result.groupby("model")[
            ["score", "accuracy", "direction",
             "big_recall", "big_prec"]
        ].mean().round(4).to_string()
    )

    wide = result.pivot(
        index=["draw", "fold"],
        columns="model",
        values="score",
    )

    delta = wide["Residual"] - wide["B2"]

    print("\n=== Paired Delta by Draw/Fold ===")
    print(delta.round(4).to_string())
    print("Mean Delta:", round(float(delta.mean()), 4))
    print("Positive runs:", int((delta > 0).sum()), "/ 9")

    print("\n=== Alpha choices ===")
    print(
        result[result["model"] == "Residual"][
            ["draw", "fold", "alpha", "k"]
        ].to_string(index=False)
    )

    print("\n=== Seen / Unseen Scores ===")
    for group, subset in pred.groupby("is_unseen"):
        name = "Unseen" if group else "Seen"
        b = score(subset["label"], subset["pred_B2"])
        r = score(subset["label"], subset["pred_Residual"])
        print(
            f"{name}: B2={b['score']:.4f} "
            f"Residual={r['score']:.4f} "
            f"Delta={r['score']-b['score']:+.4f}"
        )

    print("\nSaved to:", OUTPUT)
    print("Residual DEV experiment completed")


if __name__ == "__main__":
    main()
