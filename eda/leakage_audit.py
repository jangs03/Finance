"""누수(look-ahead) 실측 감사.

기능  : 코드가 아니라 데이터·결과로 "09:30 이후 정보가 feature 에 섞였는가"를 검정함
구성  : A  장 시작 후 움직임(rest) 검정 — 모든 feature 와 rest 의 상관, 미래 feature(양성 대조군)와 비교
        B  가격 시각 — 갭이 시가 갭과 일치하고 장중 수익률은 예측하지 않는가
        C  실적 시각 — 반영 대상일의 갭이 다음 날 갭보다 큰가 (known_at 이 실제보다 이르면 반대로 나옴)
        D  애널리스트 시각 — 같은 검정
        E  뉴스 시각 — 장 마감 후 실적 발표 대비 "실적 결과" 기사의 도착 시각 분포
        F  Reddit 시각 — created_et 가 created_utc 의 ET 변환과 같은가
        G  분할·학습 — 학습 행이 평가 시작 14일 전에 끝나는가, feature 목록에 정답 계열 열이 없는가
역할  : 결과를 runs/audit/ 에 저장하고 요약을 출력. DEV 행만 씀 (lockbox 미사용)
실행  : python eda/leakage_audit.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from exp import config as C                                    # noqa: E402
from exp.lockbox import LOCKBOX_START                          # noqa: E402
from exp.panel import RESEARCH_GROUPS, calendar, load_tables, universe_panel   # noqa: E402
from exp.splits import fold_masks, partition, universe_draws   # noqa: E402
from src.model import DEPLOYABLE, TEXT_COLS, map_to_target     # noqa: E402

OUT = C.RUNS / "audit"
GROUPS = ["core", "scale", "cross", "event", "analyst", "news", "regime", "txt_count", "txt_tone_gdelt",
          "txt_tone_lm", "txt_novelty", "txt_event", "txt_meta", "txt_text"]
FORBIDDEN = {"label", "ret_pct", "rest", "ret_t", "close_t", "M", "D", "up"}


# -----------------------------------------------------------------------------
# 기능  : DEV 패널 (전체 종목을 20종목 묶음으로 나눠 교차 feature 계산, 연구용 reddit 포함)
# -----------------------------------------------------------------------------
def dev_panel():
    syms = sorted(load_tables()["daily"]["symbol"].unique())
    unis = partition(syms, 20, np.random.default_rng(C.derive_seed(20261005, "analysis")))
    x = pd.concat([universe_panel(u, GROUPS, ["reddit"]) for u in unis], ignore_index=True)
    return x[x["target"] < LOCKBOX_START].sort_values(["symbol", "target"]).reset_index(drop=True)


# -----------------------------------------------------------------------------
# A. rest 검정
# 수식  : rest = log(1 + 당일 수익률) − log(1 + 갭)  ≈ 09:30 이후 움직임 (cutoff 이후라 알 수 없는 값)
#         각 feature 의 Spearman ρ(feature, rest).  양성 대조군 = 같은 종목의 "다음 날" feature (미래 정보)
#         날짜 단위 block 재표본으로 ρ 의 표준오차 → |ρ| / SE > 3 이면 표시
# 해석  : 정상 feature 는 rest 를 거의 예측 못함 (|ρ| 작음). 누수가 있으면 미래 대조군처럼 커짐
# -----------------------------------------------------------------------------
def test_rest(x, n_boot=200):
    cols = sorted((DEPLOYABLE - TEXT_COLS) | set(RESEARCH_GROUPS["reddit"]))
    cols = [c for c in cols if c in x.columns and x[c].nunique(dropna=True) > 2]
    r = x["rest"].rank()
    dates = x["target"].to_numpy()
    ud = np.unique(dates)
    idx = [np.flatnonzero(dates == d) for d in ud]
    rng = np.random.default_rng(0)
    boots = [np.concatenate([idx[i] for i in rng.integers(0, len(ud), len(ud))]) for _ in range(n_boot)]
    rows = []
    for c in cols:
        f = x[c]
        fut = x.groupby("symbol")[c].shift(-1)             # 양성 대조군: 내일 값 (미래)
        ok = f.notna() & x["rest"].notna()
        rho = f[ok].rank().corr(r[ok])
        fr = f.rank().to_numpy()
        rr = r.to_numpy()
        bs = []
        for b in boots:
            m = ok.to_numpy()[b]
            if m.sum() > 100:
                bs.append(pd.Series(fr[b][m]).corr(pd.Series(rr[b][m])))
        se = float(np.nanstd(bs))
        okf = fut.notna() & x["rest"].notna()
        rho_f = fut[okf].rank().corr(r[okf])
        rows.append({"feature": c, "rho_rest": rho, "se": se, "z": rho / se if se else np.nan,
                     "rho_future_control": rho_f, "n": int(ok.sum())})
    return pd.DataFrame(rows).sort_values("z", key=lambda s: s.abs(), ascending=False)


# -----------------------------------------------------------------------------
# B. 가격 시각
# 수식  : open_gap = 시가 / 전일 종가 − 1,  intraday = 종가 / 시가 − 1
#         정상: ρ(gap, open_gap) 높음,  ρ(gap, intraday) ≈ 0 또는 음(되돌림)
# -----------------------------------------------------------------------------
def test_price(x):
    d = load_tables()["daily"][["symbol", "date_et", "open", "close", "prev_close"]].copy()
    d["target"] = pd.to_datetime(d["date_et"]).dt.normalize()
    d["open_gap"] = (d["open"] / d["prev_close"] - 1) * 100
    d["intraday"] = (d["close"] / d["open"] - 1) * 100
    m = x.merge(d[["symbol", "target", "open_gap", "intraday"]], on=["symbol", "target"])
    m = m[m["gap_missing"] == 0]
    return {"rho_gap_open_gap": float(m["gap_pct"].corr(m["open_gap"], method="spearman")),
            "rho_gap_intraday": float(m["gap_pct"].corr(m["intraday"], method="spearman")),
            "mean_abs(gap - open_gap)": float((m["gap_pct"] - m["open_gap"]).abs().mean()),
            "fallback_share": float(x["gap_fallback"].mean()), "n": len(m)}


# -----------------------------------------------------------------------------
# C·D. 이벤트 시각 (실적·애널리스트)
# 수식  : 이벤트를 반영 대상일 T 로 붙였을 때 |gap_T|, |gap_T+1|, |ret 기준일| 의 평균
#         known_at 이 실제보다 이르면(누수) 가격 반응이 T 가 아니라 T+1 에 나타남 → |gap_T+1| > |gap_T|
# -----------------------------------------------------------------------------
def test_event(x, name):
    e = load_tables()[name].copy()
    cal = calendar()
    e["target"] = map_to_target(e["known_at"], cal)
    e = e.dropna(subset=["target"])
    e = e[e["target"] < LOCKBOX_START]
    g = x[["symbol", "target", "gap_pct"]].copy()
    g["gap_next"] = g.groupby("symbol")["gap_pct"].shift(-1)
    d = load_tables()["daily"][["symbol", "date_et", "ret"]].copy()
    d["date"] = pd.to_datetime(d["date_et"]).dt.normalize()
    g = g.merge(cal, on="target").merge(d[["symbol", "date", "ret"]], on=["symbol", "date"], how="left")
    m = e.merge(g, on=["symbol", "target"]).drop_duplicates(["symbol", "target"])
    hour = pd.to_datetime(m["known_at"]).dt.hour
    grp = np.where(hour >= 16, "장 마감 후(≥16시)", np.where(hour < 9, "개장 전(<9시)", "장중"))
    out = m.assign(slot=grp).groupby("slot").agg(n=("gap_pct", "size"), abs_gap_T=("gap_pct", lambda s: s.abs().mean()),
                                                 abs_gap_T1=("gap_next", lambda s: s.abs().mean()),
                                                 abs_ret_base=("ret", lambda s: (s * 100).abs().mean()))
    base = x["gap_pct"].abs().mean()
    return out.round(3), float(base)


# -----------------------------------------------------------------------------
# E. 뉴스 시각: 장 마감 후(16:00) 실적 발표 기준, 같은 종목의 "실적 결과" 제목 기사가 언제 도착하나
#    발표 전(−6h ~ 0)에 결과 기사가 몰리면 뉴스 시각이 실제보다 이르다는 뜻 → 누수 위험
# -----------------------------------------------------------------------------
def test_news_timing():
    from exp.text import articles
    a = articles()
    a = a[(a["rel"] >= 1) & a["title"].str.contains(r"(?i)\b(?:beats?|miss(?:es|ed)?|results|earnings|eps)\b")]
    e = load_tables()["earnings"]
    e = e[pd.to_datetime(e["known_at"]).dt.hour == 16][["symbol", "known_at"]].rename(columns={"known_at": "ek"})
    m = a.merge(e, on="symbol")
    lag = (pd.to_datetime(m["known_at"]) - pd.to_datetime(m["ek"])).dt.total_seconds() / 3600
    m = m.assign(lag=lag)[(lag > -24) & (lag < 24)]
    bins = [-24, -6, -1, 0, 1, 3, 6, 24]
    h = pd.cut(m["lag"], bins).value_counts(normalize=True).sort_index()
    return {str(k): round(float(v), 3) for k, v in h.items()}, int(len(m))


# -----------------------------------------------------------------------------
# F. Reddit 시각: created_et == created_utc 를 America/New_York 로 변환한 값인가 (표본)
# -----------------------------------------------------------------------------
def test_reddit():
    import pyarrow.parquet as pq
    t = pq.ParquetFile(ROOT / "dataset" / "reddit" / "options.comments.parquet").read_row_group(0, columns=["created_utc", "created_et"]).to_pandas()
    conv = pd.to_datetime(t["created_utc"], unit="s", utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None)
    diff = (conv - pd.to_datetime(t["created_et"])).dt.total_seconds().abs()
    return {"max_abs_diff_sec": float(diff.max()), "n": len(t)}


# -----------------------------------------------------------------------------
# G. 분할·학습 구조 점검 (실험과 같은 분할 함수)
# -----------------------------------------------------------------------------
def test_splits():
    sp = C.DEFAULTS["split"]
    syms = sorted(load_tables()["daily"]["symbol"].unique())
    dr = universe_draws(syms, sp, 20261005)[0]
    tr = pd.concat([universe_panel(u, ["core"]) for u in dr["train_universes"]])
    te = universe_panel(dr["test_universe"], ["core"])
    res = []
    for f in sp["folds"]:
        a, b = fold_masks(tr, te, f, sp["embargo_days"])
        gap_days = (te[b]["target"].min() - tr[a]["target"].max()).days
        res.append({"fold": f["name"], "train_max": str(tr[a]["target"].max().date()),
                    "test_min": str(te[b]["target"].min().date()), "gap_days": gap_days,
                    "unseen_in_train": int(tr[a]["symbol"].isin(dr["unseen"]).sum()),
                    "lockbox_rows": int((te[b]["target"] >= LOCKBOX_START).sum() + (tr[a]["target"] >= LOCKBOX_START).sum())})
    feats = set().union(*[set(v) for v in RESEARCH_GROUPS.values()]) | DEPLOYABLE
    return res, sorted(feats & FORBIDDEN)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    x = dev_panel()
    report = {}
    A = test_rest(x)
    A.to_csv(OUT / "A_rest_test.csv", index=False)
    print("A. rest 검정 (상위 12, |z| 순)\n", A.head(12).round(4).to_string(index=False))
    report["A_flagged_|z|>3"] = A.loc[A["z"].abs() > 3, ["feature", "rho_rest", "z", "rho_future_control"]].round(4).to_dict("records")
    report["B_price"] = test_price(x)
    print("\nB. 가격", report["B_price"])
    for name in ("earnings", "analyst"):
        t, base = test_event(x, name)
        t.to_csv(OUT / f"C_{name}_timing.csv")
        report[f"C_{name}"] = {"table": t.reset_index().to_dict("records"), "abs_gap_all_days": round(base, 3)}
        print(f"\nC. {name} 시각 (전체 평균 |gap| {base:.3f})\n", t.to_string())
    report["E_news_vs_earnings_lag_hours"], n = test_news_timing()
    print(f"\nE. 16시 실적 발표 대비 결과 기사 도착 시각 분포 (n={n})", report["E_news_vs_earnings_lag_hours"])
    report["F_reddit"] = test_reddit()
    print("\nF. Reddit", report["F_reddit"])
    report["G_splits"], report["G_forbidden_in_features"] = test_splits()
    print("\nG. 분할", report["G_splits"], "\n   feature 목록의 정답 계열 열:", report["G_forbidden_in_features"])
    (OUT / "audit_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\n→ {OUT}")


if __name__ == "__main__":
    main()
