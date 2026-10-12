"""실험 실행.

기능  : config 의 모든 후보를 (fold × universe draw) 마다 학습·예측하고, 지표·paired 비교·판정을 저장
구성  : run() · export() · 내부 도우미 (_unit_frames, _compare, _report)
역할  : 단계(E1~E6)가 달라도 같은 함수로 돌림. 단계 차이는 config 의 후보 목록뿐임

저장 (runs/<name>_<config hash>_<code hash>/)
    config.json        해석된 config
    env.json           python·패키지·git·코드·데이터 지문
    universes.json     draw 별 unseen / seen / train universe
    predictions.parquet  fold, draw, candidate, symbol, target, seen, label, pred
    params.jsonl       후보·fold·draw 별로 고른 경계·k·regime 값
    metrics.csv        후보·fold·draw·scope(all/seen/unseen) 별 지표
    compare.csv        기준선 대비 ΔScore, CI, 판정
    report.md          결과 요약표
    DONE               완료 표시 (있으면 같은 config·코드로 다시 돌리지 않고 결과를 읽음)
"""

import json
import pickle
import time

import numpy as np
import pandas as pd

from src.model import ARTIFACT, Pipeline, load_model

from . import config as C
from . import lockbox as LB
from .metrics import evaluate
from .panel import RESEARCH_GROUPS, load_tables, universe_panel
from .splits import fold_masks, partition, universe_draws
from .stats import decide, holm, paired_bootstrap
from contextlib import nullcontext
from exp.news_fast import skip_text_block


def _symbols():
    return sorted(load_tables()["daily"]["symbol"].unique())


def _extra(cfg):
    return {k: RESEARCH_GROUPS[k] for k in cfg["data"]["research"]}


# -----------------------------------------------------------------------------
# 기능  : 실험 실행
# input : cfg  config.load() 결과,  force_lockbox  lockbox 재사용 허용,  rerun  DONE 무시
# lockbox: use_lockbox=true 면 freeze 로 고정한 계획과 같아야 하고 1회만 실행됨 (exp/lockbox.py)
# output: 결과 폴더 경로
# -----------------------------------------------------------------------------
def run(cfg, force_lockbox=False, rerun=False):
    env = C.environment()
    out = C.RUNS / f"{cfg['name']}_{cfg['hash']}_{env['code']}"
    use_lb = bool(cfg["split"]["use_lockbox"])
    if (out / "DONE").exists() and not rerun and not use_lb:
        print(f"[run] 같은 config·코드의 결과가 있음 → {out}")
        return out
    out.mkdir(parents=True, exist_ok=True)
    sp, seed = cfg["split"], int(cfg["seed"])
    folds = sp["folds"]
    lb_entry = None
    if use_lb:
        lb_entry = LB.open_run(cfg, force_lockbox)
        folds = [{"name": "LOCKBOX", "start": sp["lockbox_start"], "end": "2100-01-01"}]

    draws = universe_draws(_symbols(), sp, seed)
    (out / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    (out / "env.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    (out / "universes.json").write_text(json.dumps(draws, indent=2), encoding="utf-8")

    groups, research, extra = cfg["data"]["groups"], cfg["data"]["research"], _extra(cfg)
    preds, params, t0 = [], [], time.time()
    for dr in draws:
        # 경량 뉴스 처리는 e5_news_abn DEV 실험에만 적용
        fast_news = (
            cfg["name"] == "e5_news_abn"
            and not use_lb
            and "news" in groups
        )

        with (skip_text_block() if fast_news else nullcontext()):
            train_p = pd.concat(
                [universe_panel(u, groups, research) for u in dr["train_universes"]],
                ignore_index=True,
            )
            test_p = universe_panel(
                dr["test_universe"],
                groups,
                research,
                allow_lockbox=use_lb,
            )

        test_p["seen"] = test_p["symbol"].isin(dr["seen_test"])


        for fold in folds:
            trm, tem = fold_masks(train_p, test_p, fold, sp["embargo_days"])
            tr, te = train_p[trm], test_p[tem].reset_index(drop=True)
            for cand in cfg["candidates"]:
                spec = {**cand, "seed": C.derive_seed(seed, "model", cand["name"], dr["draw"], fold["name"]) % 2**31}
                pipe = Pipeline(spec, extra).fit(tr)
                preds.append(pd.DataFrame({
                    "fold": fold["name"], "draw": dr["draw"], "candidate": cand["name"],
                    "symbol": te["symbol"], "target": te["target"], "seen": te["seen"],
                    "label": te["label"].astype(int), "pred": pipe.predict(te)}))
                params.append({"fold": fold["name"], "draw": dr["draw"], "candidate": cand["name"], **pipe.params()})
            print(f"[run] draw {dr['draw']} fold {fold['name']}: 학습 {len(tr):,} / 평가 {len(te):,} "
                  f"({time.time() - t0:.0f}s)", flush=True)

    P = pd.concat(preds, ignore_index=True)
    P.to_parquet(out / "predictions.parquet", index=False)
    with open(out / "params.jsonl", "w", encoding="utf-8") as f:
        for r in params:
            f.write(json.dumps(r, ensure_ascii=False, default=float) + "\n")
    M = _metrics(P)
    M.to_csv(out / "metrics.csv", index=False)
    cmp_ = _compare(P, cfg) if cfg["eval"]["baseline"] else pd.DataFrame()
    cmp_.to_csv(out / "compare.csv", index=False)
    report = _report(cfg, env, M, cmp_)
    if use_lb:
        report += "\n" + LB.caveat([c["name"] for c in cfg["candidates"]])
    (out / "report.md").write_text(report, encoding="utf-8")
    if use_lb:
        LB.close_run(lb_entry, out)
    (out / "DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")
    print(f"[run] 완료 → {out}")
    return out


# -----------------------------------------------------------------------------
# 기능  : 후보·fold·draw·scope 별 지표
# output: DataFrame[candidate, fold, draw, scope, <metrics>]
# -----------------------------------------------------------------------------
def _metrics(P):
    rows = []
    for (c, f, d), g in P.groupby(["candidate", "fold", "draw"], sort=False):
        for scope, m in (("all", slice(None)), ("seen", g["seen"].to_numpy()), ("unseen", ~g["seen"].to_numpy())):
            h = g[m]
            rows.append({"candidate": c, "fold": f, "draw": d, "scope": scope, **evaluate(h["label"], h["pred"])})
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# 기능  : 기준선 대비 paired 비교와 판정
# output: DataFrame[candidate, delta, ci_low, ci_high, se, fold_<이름>, unseen_delta,
#                   flip_diff, extreme_ratio, decision, reasons]
# -----------------------------------------------------------------------------
def _compare(P, cfg):
    base_name, ev = cfg["eval"]["baseline"], cfg["eval"]
    base = P[P["candidate"] == base_name].reset_index(drop=True)
    rows = []
    for cand in [c["name"] for c in cfg["candidates"] if c["name"] != base_name]:
        cd = P[P["candidate"] == cand].reset_index(drop=True)
        units, unseen, flips, ext = [], [], [], []
        for (f, d), gb in base.groupby(["fold", "draw"], sort=False):
            gc = cd[(cd["fold"] == f) & (cd["draw"] == d)]
            assert (gb["symbol"].to_numpy() == gc["symbol"].to_numpy()).all(), "후보 간 평가 행이 다름"
            units.append({"key": (f, d), "dates": gb["target"].to_numpy(), "y": gb["label"].to_numpy(),
                          "base": gb["pred"].to_numpy(), "cand": gc["pred"].to_numpy()})
            u = ~gb["seen"].to_numpy()
            mb, mc = evaluate(gb["label"][u], gb["pred"][u]), evaluate(gc["label"][u], gc["pred"][u])
            unseen.append(mc["score"] - mb["score"])
            ab, ac = evaluate(gb["label"], gb["pred"]), evaluate(gc["label"], gc["pred"])
            flips.append(ac["flip_index"] - ab["flip_index"])
            ext.append(ac["pred_extreme"] / ab["pred_extreme"] if ab["pred_extreme"] else np.nan)
        bt = paired_bootstrap(units, ev["bootstrap"], ev["block_days"], cfg["seed"])
        fold_delta = pd.Series(bt["per_unit"]).groupby(level=0).mean().to_dict()
        rows.append({"candidate": cand, "delta": bt["delta"], "ci_low": bt["ci_low"], "ci_high": bt["ci_high"],
                     "se": bt["se"], "p": bt["p"], "fold_delta": fold_delta, "unseen_delta": float(np.nanmean(unseen)),
                     "flip_diff": float(np.nanmean(flips)), "extreme_ratio": float(np.nanmean(ext))})
    adj = holm([r["p"] for r in rows], cfg["adopt"].get("family_size")) if rows else []
    out = []
    for r, ph in zip(rows, adj):
        r["p_holm"] = float(ph)
        r["decision"], why = decide(r, cfg["adopt"])
        r["reasons"] = "; ".join(why)
        out.append({**{k: v for k, v in r.items() if k != "fold_delta"},
                    **{f"fold_{k}": v for k, v in r["fold_delta"].items()}})
    return pd.DataFrame(out)


# -----------------------------------------------------------------------------
# 기능  : DataFrame → markdown 표 (외부 패키지 없이)
# -----------------------------------------------------------------------------
def _md(df):
    cols = [df.index.name or ""] + [str(c) for c in df.columns]
    fmt = lambda v: f"{v:.4f}" if isinstance(v, float) else str(v)
    rows = [[str(i)] + [fmt(v) for v in r] for i, r in zip(df.index, df.to_numpy())]
    return "\n".join(["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] + ["| " + " | ".join(r) + " |" for r in rows])


# -----------------------------------------------------------------------------
# 기능  : 결과 요약 markdown
# -----------------------------------------------------------------------------
def _report(cfg, env, M, cmp_):
    a = M[M["scope"] == "all"].groupby("candidate", sort=False)[
        ["score", "accuracy", "direction", "magnitude_acc", "big_recall", "big_prec", "flip_index",
         "pred_extreme", "true_extreme"]].mean()
    un = M[M["scope"] == "unseen"].groupby("candidate", sort=False)["score"].mean().rename("unseen_score")
    lines = [f"# {cfg['name']} ({cfg['stage']})", "",
             f"- config `{cfg['path']}` hash `{cfg['hash']}` · code `{env['code']}` · data `{env['data']}` · git `{env['git_commit']}`"
             + (" (dirty)" if env["git_dirty"] else ""),
             f"- baseline `{cfg['eval']['baseline']}` · bootstrap {cfg['eval']['bootstrap']} · block {cfg['eval']['block_days']}일", "",
             "## 후보별 지표 (fold × draw 평균)", "", _md(a.join(un).round(4)), ""]
    if len(cmp_):
        show = cmp_.drop(columns=["reasons"]).set_index("candidate").round(4)
        lines += ["## 기준선 대비 paired ΔScore", "", _md(show), "", "## 판정 근거", ""]
        lines += [f"- **{r.candidate}** {r.decision}: {r.reasons}" for r in cmp_.itertuples()]
    return "\n".join(lines) + "\n"


# -----------------------------------------------------------------------------
# 기능  : 제출용 학습. 후보 하나를 train_until 이전 전체(모든 종목)로 학습해 artifacts/ 에 저장
#         학습 행의 교차 feature 는 전체 종목을 20종목 안팎 묶음으로 나눠 계산 (채점 universe 와 같은 크기)
# input : cfg  ([export] train_until, candidate 이름은 인자로),  name  후보 이름
# output: artifact 경로.  artifacts/pipeline.pkl (학습 결과) + pipeline.json (사람이 읽는 요약)
# 검사  : 연구용 feature 를 쓰면 저장하지 않음 (채점 환경에서 만들 수 없음)
#         lockbox 평가 전에는 lockbox 기간을 학습에 넣지 않음 (allow_lockbox=True 면 허용하고 노출로 기록)
# -----------------------------------------------------------------------------
def export(cfg, name, allow_lockbox=False):
    cand = next((c for c in cfg["candidates"] if c["name"] == name), None)
    if cand is None:
        raise SystemExit(f"후보 '{name}' 없음: {[c['name'] for c in cfg['candidates']]}")
    until = pd.Timestamp(cfg.get("export", {}).get("train_until", str(LB.LOCKBOX_START.date())))
    LB.allow_training(until, allow_lockbox, f"export {cfg['name']}/{name}")
    use_lb = until > LB.LOCKBOX_START
    seed = int(cfg["seed"])
    unis = partition(_symbols(), cfg["split"]["universe_size"], np.random.default_rng(C.derive_seed(seed, "export")))
    X = pd.concat([universe_panel(u, cfg["data"]["groups"], allow_lockbox=use_lb) for u in unis], ignore_index=True)
    X = X[X["target"] < until]
    pipe = Pipeline({**cand, "seed": C.derive_seed(seed, "model", name, "export") % 2**31}).fit(X)
    if not pipe.deployable:
        raise SystemExit(f"{name}: 연구용 feature 를 써서 제출할 수 없음 {sorted(set(pipe.features))}")
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    with open(ARTIFACT, "wb") as f:
        pickle.dump(pipe.state(), f)
    meta = {"candidate": name, "spec": pipe.spec, "params": pipe.params(), "train_rows": len(X),
            "train_until": str(until.date()), "config": cfg["path"], "config_hash": cfg["hash"],
            "universes": unis, "env": C.environment()}
    ARTIFACT.with_suffix(".json").write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    m = load_model()
    print(f"[export] {ARTIFACT} 저장. 복원 확인: {type(m.pipeline.signal).__name__} / {m.pipeline.params()['decision']}")
    return ARTIFACT
