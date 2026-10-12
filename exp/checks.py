"""E0 재현성 · G0 실행 가능성 점검.

기능  : e0()  연구용 패널(전체 표 벡터 계산)과 제출 경로(day API 하루 계산)의 feature·정답이 같은지 대조
        g0()  제출 경로의 하루 예측 시간과 LLM 사용 조건을 점검
구성  : e0() · g0() · _save()
역할  : 실험 결과를 믿을 수 있는지(E0), 제출 환경에서 돌아가는지(G0)를 실험 전에 확인
통과 기준 (E0): label 불일치 0, feature 최대 절대오차 < 1e-9, NaN 위치 일치, 문자열 열(titles_r1) 일치,
                대상일 pre 봉에 09:00 봉 없음
"""

import json
import os
import time

import numpy as np
import pandas as pd

from src.data import Dataset
from src.model import DEPLOYABLE, TEXT_COLS, Model, day_features, day_tables, groups_of, resolve_features

from . import config as C
from .lockbox import LOCKBOX_START
from .panel import calendar, universe_panel
from .splits import universe_draws

ALL_GROUPS = ["core", "scale", "cross", "event", "analyst", "news", "regime", "seq", "premkt", "har", "txt_recent", "txt_body", "txt_lead",
              "txt_count", "txt_tone_gdelt", "txt_tone_lm", "txt_novelty", "txt_event", "txt_meta", "txt_text"]


def _save(name, obj):
    d = C.RUNS / "checks"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{name}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return path


# -----------------------------------------------------------------------------
# 기능  : E0 대조
# input : n_days  표본 대상일 수,  seed,  groups  비교할 feature 묶음
# output: (통과 여부, 결과 dict)  결과는 runs/checks/e0_*.json 에도 저장
# 방법  : draw 0 의 test universe(20종목)로 universe_panel 을 만들고,
#         같은 20종목만 보이는 Dataset 의 day 마다 day_features 를 계산해 열별로 비교
# -----------------------------------------------------------------------------
def e0(n_days=8, seed=20261005, groups=ALL_GROUPS):
    cfg = C.DEFAULTS["split"]
    uni = universe_draws(sorted(pd.read_parquet(C.ROOT / "dataset" / "daily.parquet", columns=["symbol"])["symbol"].unique()),
                         cfg, seed)[0]["test_universe"]
    panel = universe_panel(uni, groups)
    cols = sorted((set(resolve_features(groups)) & DEPLOYABLE) - TEXT_COLS)
    tcols = sorted(set(resolve_features(groups)) & TEXT_COLS)
    cal = calendar()
    cal = cal[cal["target"] < LOCKBOX_START]          # 점검도 DEV 날짜만 (정답 비교가 있어서)
    rng = np.random.default_rng(C.derive_seed(seed, "e0"))
    picks = sorted(rng.choice(cal["date"].iloc[60:-1].to_numpy(), n_days, replace=False))
    ds = Dataset(symbols=uni)
    rows, ok = [], True
    for d in picks:
        day = ds.day(pd.Timestamp(d))
        x = day_features(day, groups_of(cols)).set_index("symbol")
        p = panel[panel["target"] == pd.Timestamp(day.target)].set_index("symbol").reindex(x.index)
        a, b = x[cols].to_numpy(float), p[cols].to_numpy(float)
        nan_mismatch = int((np.isnan(a) != np.isnan(b)).sum())
        diff = np.nanmax(np.abs(np.where(np.isnan(a) | np.isnan(b), 0, a - b)))
        worst = cols[int(np.nanargmax(np.nanmax(np.abs(np.nan_to_num(a - b)), axis=0)))] if diff > 0 else None
        text_mis = int(sum((x[c].fillna("") != p[c].fillna("")).sum() for c in tcols))
        y = day.y.set_index("symbol")["label"].astype(int)
        lab_mis = int((p["label"].astype(int) != y.reindex(p.index)).sum())
        tables, _ = day_tables(day, {"core"})
        pr = tables["price"]
        pre_t = pr[(pr["session"] == "pre") & (pd.to_datetime(pr["datetime"]).dt.normalize() == pd.Timestamp(day.target))]
        bar9 = int((pd.to_datetime(pre_t["datetime"]).dt.hour >= 9).sum())
        good = lab_mis == 0 and diff < 1e-9 and nan_mismatch == 0 and bar9 == 0 and len(x) == len(uni) and text_mis == 0
        ok &= good
        rows.append({"date": str(day.date.date()), "target": str(day.target.date()), "n": len(x),
                     "label_mismatch": lab_mis, "max_abs_diff": float(diff), "worst_col": worst,
                     "nan_mismatch": nan_mismatch, "text_mismatch": text_mis, "pre_bars_at_or_after_09": bar9,
                     "pass": good})
    res = {"pass": bool(ok), "universe": uni, "columns": cols, "days": rows}
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"[E0] {'통과' if ok else '실패'} → {_save('e0', res)}")
    return ok, res


# -----------------------------------------------------------------------------
# 기능  : G0 점검
#         1) 하루 예측 시간: 가격만 / +실적·애널리스트 / +뉴스 세 구성으로 측정
#         2) LLM 사용 조건: anthropic 패키지·인증 변수 유무 (채점 환경 허용 여부는 사람이 확인해야 함)
# input : n_days, seed
# output: 결과 dict (runs/checks/g0_*.json 에도 저장)
# -----------------------------------------------------------------------------
def g0(n_days=5, seed=20261005):
    from src.model import Pipeline
    cal = calendar()
    cal = cal[cal["target"] < LOCKBOX_START]
    rng = np.random.default_rng(C.derive_seed(seed, "g0"))
    picks = sorted(rng.choice(cal["date"].iloc[60:-1].to_numpy(), n_days, replace=False))
    ds = Dataset()
    variants = {"price": ["core", "scale", "cross"], "price+event": ["core", "scale", "cross", "event", "analyst"],
                "price+event+news": ["core", "scale", "cross", "event", "analyst", "news"]}
    timing = {}
    for name, feats in variants.items():
        m = Model(Pipeline({"name": name, "signal": {"type": "column", "features": feats}, "decision": {"type": "identity"}}))
        m.pipeline.signal.features = resolve_features(feats)      # 읽는 표의 범위만 늘려 시간 측정
        m.pipeline.decision.b = np.array([-2.5, -0.6, 0.6, 2.5])
        secs = []
        for d in picks:
            t = time.time()
            m.predict(ds.day(pd.Timestamp(d)))
            secs.append(time.time() - t)
        timing[name] = {"mean_s": float(np.mean(secs)), "max_s": float(np.max(secs))}
    try:
        import anthropic  # noqa: F401
        sdk = True
    except ImportError:
        sdk = False
    res = {"predict_seconds_per_day": timing,
           "llm": {"anthropic_sdk_installed": sdk,
                   "auth_env_present": any(os.environ.get(k) for k in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")),
                   "grading_env_allows_api": "미확인 — 교수·조교 확인 필요. 확인 전에는 LLM feature 를 제출 모델에 쓰지 않음"}}
    print(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"[G0] → {_save('g0', res)}")
    return res
