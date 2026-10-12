
"""EDA-selected features: DEV Fold A x Draw 0 smoke test."""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from exp.dlinear_model_eda import DLinearEDAClassifier
from exp.timeseries_data_eda import (
    build_eda_training_data,
    prepare_eda_features,
)
from exp.splits import universe_draws, fold_masks
from exp.train_dlinear import (
    SEED, SPLIT, FOLDS, EMBARGO_DAYS,
    split_calibration, prepare_arrays, set_seed,
)
from src.data import score
from src.model import KRule, CLASS_CENTERS


class EDAOnlyClassifier(nn.Module):
    def __init__(self, n_features=11):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 32),
            nn.ReLU(),
            nn.Linear(32, 5),
        )

    def forward(self, features):
        return self.net(features)


def fit_nn(model, x, eda, y, use_sequence=False):
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-3
    )

    y = torch.from_numpy(
        y.to_numpy(dtype=np.int64, copy=True)
    )

    for epoch in range(15):
        model.train()
        total_loss = 0.0

        for idx in torch.randperm(len(y)).split(128):
            if use_sequence:
                logits = model(x[idx], eda[idx])
            else:
                logits = model(eda[idx])

            loss = F.cross_entropy(logits, y[idx])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(idx)

        if epoch in (0, 14):
            print(
                f"  Epoch {epoch+1:02d}/15 "
                f"Loss={total_loss / len(y):.4f}",
                flush=True,
            )


def predict_signal(model, sequence, eda, use_sequence):
    model.eval()

    with torch.no_grad():
        if use_sequence:
            logits = model(sequence, eda)
        else:
            logits = model(eda)

        probabilities = torch.softmax(logits, dim=1)

    return probabilities.numpy() @ CLASS_CENTERS


def main():
    set_seed(SEED)

    symbols = sorted(
        pd.read_parquet(
            "dataset/daily.parquet",
            columns=["symbol"],
        )["symbol"].unique()
    )

    draw = universe_draws(symbols, SPLIT, SEED)[0]
    fold = FOLDS[0]

    print("[1/4] Preparing EDA feature data", flush=True)

    train_panel = pd.concat(
        [
            build_eda_training_data(u)
            for u in draw["train_universes"]
        ],
        ignore_index=True,
    )

    test_panel = build_eda_training_data(
        draw["test_universe"]
    )

    tr, te = fold_masks(
        train_panel, test_panel, fold, EMBARGO_DAYS
    )

    train = train_panel.loc[tr].dropna(
        subset=["gap_pct", "resid_gap", "label"]
    ).copy()

    test = test_panel.loc[te].dropna(
        subset=["gap_pct", "resid_gap", "label"]
    ).copy()

    if train.empty or test.empty:
        raise ValueError("Train 또는 Test 데이터가 비어 있음")

    assert not set(train["symbol"]) & set(draw["unseen"])
    assert train["target"].max() < (
        pd.Timestamp(fold["start"])
        - pd.Timedelta(days=EMBARGO_DAYS)
    )
    assert (
        (test["target"] >= pd.Timestamp(fold["start"]))
        & (test["target"] < pd.Timestamp(fold["end"]))
    ).all()

    fit, calib = split_calibration(train)

    seq_fit, seq_cal, seq_test, _, _, _ = prepare_arrays(
        fit, calib, test, gap_only=False
    )

    eda_fit, eda_cal, eda_test, columns = (
        prepare_eda_features(fit, calib, test)
    )

    eda_fit = torch.from_numpy(eda_fit.copy())
    eda_cal = torch.from_numpy(eda_cal.copy())
    eda_test = torch.from_numpy(eda_test.copy())

    y_cal = calib["label"].to_numpy(dtype=int)
    y_test = test["label"].to_numpy(dtype=int)

    objective = lambda y, p: score(y, p)["score"]

    print(
        f"Fit={len(fit)} Calib={len(calib)} "
        f"Test={len(test)} Features={len(columns)}",
        flush=True,
    )

    # B2: 동일 Calibration 및 Test 행 사용
    b2 = KRule()
    b2.fit(
        calib["gap_pct"].to_numpy(dtype=float),
        y_cal,
        objective,
    )
    b2_pred = b2.predict(
        test["gap_pct"].to_numpy(dtype=float)
    )

    print("[2/4] Training EDA-only NN", flush=True)

    set_seed(SEED)
    eda_model = EDAOnlyClassifier(len(columns))

    fit_nn(
        eda_model,
        seq_fit,
        eda_fit,
        fit["label"],
        use_sequence=False,
    )

    print("[3/4] Training DLinear + EDA", flush=True)

    set_seed(SEED)
    dlinear = DLinearEDAClassifier(
        seq_len=20,
        n_features=5,
        n_eda_features=len(columns),
    )

    fit_nn(
        dlinear,
        seq_fit,
        eda_fit,
        fit["label"],
        use_sequence=True,
    )

    print("[4/4] Calibration and DEV evaluation")

    results = {"B2": score(y_test, b2_pred)}
    best_k = {"B2": b2.k}

    for name, model, with_seq in [
        ("EDA-only", eda_model, False),
        ("DLinear+EDA", dlinear, True),
    ]:
        cal_signal = predict_signal(
            model, seq_cal, eda_cal, with_seq
        )
        test_signal = predict_signal(
            model, seq_test, eda_test, with_seq
        )

        rule = KRule()
        rule.fit(cal_signal, y_cal, objective)

        prediction = rule.predict(test_signal)

        results[name] = score(y_test, prediction)
        best_k[name] = rule.k

    print("\n=== Fold A / Draw 0 Results ===")
    for name, metrics in results.items():
        print(
            f"{name:12s} "
            f"Score={metrics['score']:.4f} "
            f"Accuracy={metrics['accuracy']:.4f} "
            f"BigRecall={metrics['big_recall']:.4f} "
            f"k={best_k[name]:.2f}"
        )

    print("\n=== Paired Score Differences ===")
    print(
        "EDA-only - B2:",
        round(
            results["EDA-only"]["score"]
            - results["B2"]["score"], 4
        ),
    )
    print(
        "DLinear+EDA - B2:",
        round(
            results["DLinear+EDA"]["score"]
            - results["B2"]["score"], 4
        ),
    )
    print(
        "DLinear+EDA - EDA-only:",
        round(
            results["DLinear+EDA"]["score"]
            - results["EDA-only"]["score"], 4
        ),
    )

    print("EDA Smoke Test completed")


if __name__ == "__main__":
    main()
