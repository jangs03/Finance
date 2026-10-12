
"""DLinear DEV experiment: B2 vs Gap-only vs DLinear + Gap."""

from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from exp.dlinear_model import DLinearClassifier
from exp.splits import universe_draws, fold_masks
from exp.timeseries_data import build_training_data
from src.data import score
from src.model import KRule, CLASS_CENTERS

SEED = 20261005
EPOCHS = 15
BATCH_SIZE = 128
LR = 1e-3
EMBARGO_DAYS = 14

SPLIT = {
    "n_draws": 3,
    "n_unseen": 10,
    "n_seen": 10,
    "universe_size": 20,
}

FOLDS = [
    {"name": "A", "start": "2025-07-01", "end": "2025-11-01"},
    {"name": "B", "start": "2025-11-01", "end": "2026-03-01"},
    {"name": "C", "start": "2026-03-01", "end": "2026-06-01"},
]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def split_calibration(train):
    """마지막 20% 날짜를 KRule calibration 용도로 사용."""
    dates = np.sort(train["target"].unique())

    if len(dates) < 30:
        raise ValueError("Calibration에 사용할 날짜가 부족함")

    cut_idx = max(1, int(len(dates) * 0.8))
    calib_start = pd.Timestamp(dates[cut_idx])

    fit = train[
        train["target"] < calib_start - pd.Timedelta(days=EMBARGO_DAYS)
    ].copy()

    calib = train[
        train["target"] >= calib_start
    ].copy()

    if fit.empty or calib.empty:
        raise ValueError("Fit 또는 Calibration 데이터가 비어 있음")

    return fit, calib


def prepare_arrays(fit, calib, test, gap_only=False):
    """정규화 통계는 Fit 데이터에서만 계산."""

    x_fit = np.stack(fit["sequence"].to_numpy()).astype(np.float32)
    x_cal = np.stack(calib["sequence"].to_numpy()).astype(np.float32)
    x_test = np.stack(test["sequence"].to_numpy()).astype(np.float32)

    g_fit = fit[["gap_pct"]].to_numpy(dtype=np.float32)
    g_cal = calib[["gap_pct"]].to_numpy(dtype=np.float32)
    g_test = test[["gap_pct"]].to_numpy(dtype=np.float32)

    mean = x_fit.mean(axis=(0, 1), keepdims=True)
    std = np.maximum(
        x_fit.std(axis=(0, 1), keepdims=True), 1e-6
    )

    x_fit = (x_fit - mean) / std
    x_cal = (x_cal - mean) / std
    x_test = (x_test - mean) / std

    if gap_only:
        x_fit = np.zeros_like(x_fit)
        x_cal = np.zeros_like(x_cal)
        x_test = np.zeros_like(x_test)

    gap_mean = g_fit.mean(axis=0, keepdims=True)
    gap_std = np.maximum(
        g_fit.std(axis=0, keepdims=True), 1e-6
    )

    g_fit = (g_fit - gap_mean) / gap_std
    g_cal = (g_cal - gap_mean) / gap_std
    g_test = (g_test - gap_mean) / gap_std

    arrays = [x_fit, x_cal, x_test, g_fit, g_cal, g_test]

    if not all(np.isfinite(a).all() for a in arrays):
        raise ValueError("NaN 또는 Inf 입력 발견")

    return [
        torch.from_numpy(np.array(a, copy=True)).float()
        for a in arrays
    ]


def train_model(fit, calib, test, gap_only, seed):
    """모델 학습 → Calibration KRule → DEV Test 예측."""

    set_seed(seed)

    xf, xc, xv, gf, gc, gv = prepare_arrays(
        fit, calib, test, gap_only=gap_only
    )

    yf = torch.from_numpy(
        fit["label"].to_numpy(dtype=np.int64, copy=True)
    )
    yc = calib["label"].to_numpy(dtype=np.int64, copy=True)

    model = DLinearClassifier(
        seq_len=20,
        n_features=5,
        gap_dim=1,
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR
    )

    for epoch in range(EPOCHS):
        model.train()

        indices = torch.randperm(len(xf))
        total_loss = 0.0

        for idx in indices.split(BATCH_SIZE):
            logits = model(xf[idx], gf[idx])
            loss = F.cross_entropy(logits, yf[idx])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(idx)

        if epoch in (0, EPOCHS - 1):
            print(
                f"    Epoch {epoch + 1:02d}/{EPOCHS} "
                f"Loss={total_loss / len(xf):.4f}",
                flush=True,
            )

    model.eval()

    with torch.no_grad():
        cal_prob = torch.softmax(
            model(xc, gc), dim=1
        ).numpy()

        test_prob = torch.softmax(
            model(xv, gv), dim=1
        ).numpy()

    cal_signal = cal_prob @ CLASS_CENTERS
    test_signal = test_prob @ CLASS_CENTERS

    # 독립 Calibration 기간에서만 KRule 학습
    rule = KRule()
    rule.fit(
        cal_signal,
        yc,
        lambda y, p: score(y, p)["score"],
    )

    pred = rule.predict(test_signal)

    return pred, rule.k


def run_one(draw, fold):
    """하나의 Draw × Fold 평가."""

    train_parts = [
        build_training_data(u)
        for u in draw["train_universes"]
    ]

    train_panel = pd.concat(train_parts, ignore_index=True)
    test_panel = build_training_data(draw["test_universe"])

    tr_mask, te_mask = fold_masks(
        train_panel,
        test_panel,
        fold,
        embargo_days=EMBARGO_DAYS,
    )

    train = train_panel.loc[tr_mask].copy()
    test = test_panel.loc[te_mask].copy()

    train = train.dropna(subset=["gap_pct", "label"])
    test = test.dropna(subset=["gap_pct", "label"])

    if train.empty or test.empty:
        raise ValueError("Train 또는 Test 데이터가 비어 있음")

    assert not set(draw["unseen"]) & set(train["symbol"])

    fit, calib = split_calibration(train)

    assert fit["target"].max() < (
        calib["target"].min() - pd.Timedelta(days=EMBARGO_DAYS)
    )

    print(
        f"  Fit={len(fit)} "
        f"Calibration={len(calib)} "
        f"Test={len(test)}",
        flush=True,
    )

    y_test = test["label"].to_numpy(dtype=int)

    # B2: 과거 학습 데이터의 gap에서 최적 k 결정
    b2 = KRule()
    b2.fit(
        train["gap_pct"].to_numpy(dtype=float),
        train["label"].to_numpy(dtype=int),
        lambda y, p: score(y, p)["score"],
    )

    b2_pred = b2.predict(
        test["gap_pct"].to_numpy(dtype=float)
    )

    seed = SEED + draw["draw"] * 100 + ord(fold["name"])

    print("  Training Gap-only NN...", flush=True)
    gap_pred, gap_k = train_model(
        fit, calib, test, gap_only=True, seed=seed
    )

    print("  Training DLinear + Gap...", flush=True)
    dlinear_pred, dlinear_k = train_model(
        fit, calib, test, gap_only=False, seed=seed
    )

    predictions = {
        "B2": b2_pred,
        "Gap-only": gap_pred,
        "DLinear+Gap": dlinear_pred,
    }

    k_values = {
        "B2": b2.k,
        "Gap-only": gap_k,
        "DLinear+Gap": dlinear_k,
    }

    rows = []

    for name, pred in predictions.items():
        m = score(y_test, pred)

        rows.append({
            "draw": draw["draw"],
            "fold": fold["name"],
            "model": name,
            "k": k_values[name],
            **m,
        })

        print(
            f"  {name:12s} "
            f"Score={m['score']:.4f} "
            f"Accuracy={m['accuracy']:.4f} "
            f"BigRecall={m['big_recall']:.4f}",
            flush=True,
        )

    # 개별 예측도 보존하여 나중에 동일 행 paired 평가에 사용
    pred_table = test[
        ["symbol", "target", "label", "ret_pct"]
    ].copy()

    pred_table["draw"] = draw["draw"]
    pred_table["fold"] = fold["name"]

    for name, pred in predictions.items():
        pred_table["pred_" + name] = pred

    return rows, pred_table


def main():
    set_seed(SEED)

    symbols = sorted(
        pd.read_parquet(
            "dataset/daily.parquet",
            columns=["symbol"],
        )["symbol"].unique()
    )

    draws = universe_draws(symbols, SPLIT, SEED)

    all_results = []
    all_predictions = []

    for draw in draws:
        for fold in FOLDS:
            print(
                f"\n=== Draw {draw['draw']} / "
                f"Fold {fold['name']} ===",
                flush=True,
            )

            rows, pred_table = run_one(draw, fold)
            all_results.extend(rows)
            all_predictions.append(pred_table)

    result = pd.DataFrame(all_results)
    predictions = pd.concat(
        all_predictions, ignore_index=True
    )

    output = Path("runs/dlinear_dev")
    output.mkdir(parents=True, exist_ok=True)

    result.to_csv(output / "metrics.csv", index=False)
    predictions.to_parquet(
        output / "predictions.parquet", index=False
    )

    print("\n=== Mean Score across 9 Draw-Fold runs ===")

    summary = result.groupby("model")[
        ["score", "accuracy", "big_recall"]
    ].mean()

    print(summary.round(4).to_string())

    wide = result.pivot(
        index=["draw", "fold"],
        columns="model",
        values="score",
    )

    delta = wide["DLinear+Gap"] - wide["B2"]
    ablation = wide["DLinear+Gap"] - wide["Gap-only"]

    print("\n=== Paired Fold-Draw Differences ===")
    print("DLinear - B2:")
    print(delta.round(4).to_string())
    print("Mean Delta:", round(float(delta.mean()), 4))

    print("\nDLinear - Gap-only:")
    print(ablation.round(4).to_string())
    print("Mean Ablation Delta:", round(float(ablation.mean()), 4))

    print("\nSaved to:", output)
    print("DEV experiment completed")


if __name__ == "__main__":
    main()
