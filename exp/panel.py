"""연구용 패널.

기능  : universe(종목 목록) 하나에 대해 (종목 × 대상일) feature + 정답 표를 만듦
구성  : load_tables() · calendar() · labels() · universe_panel() · RESEARCH_GROUPS
역할  : feature 는 src.model.build_features 를 그대로 호출함 → 제출 모델과 같은 계산
        기본으로 lockbox 행을 빼고 돌려줌 (exp.lockbox.dev_only) → 실수로 lockbox 를 볼 길을 없앰
        종목 교차 feature(gap_rank, mkt_gap, reg_*) 는 넘긴 universe 안에서만 계산되므로
        universe 를 20종목으로 주면 채점 환경과 같은 분포가 됨
        결과는 (universe, feature 묶음, 코드·데이터 지문) 으로 키를 만든 parquet 캐시에 저장
"""

import functools
import hashlib
import json
from tkinter.font import names

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.data import START, label_of
from src.model import GROUP_TABLES, NS, build_features

from .config import CACHE, ROOT, fingerprint
from .lockbox import dev_only

DATA = ROOT / "dataset"
SUBREDDITS = ["Daytrading", "StockMarket", "Superstonk", "ValueInvesting",
              "investing", "options", "stocks", "wallstreetbets"]

# 제출 환경에서 만들 수 없는 연구용 feature 묶음. Pipeline(extra_groups=...) 로 넘겨 씀
RESEARCH_GROUPS = {
    "reddit": ["mentions_abn"] + [f"act_{s}_abn" for s in SUBREDDITS],
    "news_llm_anon": ["llm_anon_sign", "llm_anon_severity", "llm_anon_unexpected", "llm_anon_n"],
    "news_llm_raw": ["llm_raw_sign", "llm_raw_severity", "llm_raw_unexpected", "llm_raw_n"],
    "reddit_llm": ["llm_reddit_bull", "llm_reddit_hype", "llm_reddit_n"],
    "finbert": ["fb_net_r1", "fb_neg_share_r1", "fb_n_r1"],                 # Colab FinBERT 결과 (exp/text.py)
    "finbert_emb": [f"fb_pc{i}" for i in range(32)],                         # FinBERT 임베딩 PCA 32성분
}
LLM_FILES = {"news_llm_anon": "llm_features_anon.parquet", "news_llm_raw": "llm_features_raw.parquet",
             "reddit_llm": "llm_reddit_features.parquet"}


# -----------------------------------------------------------------------------
# 기능  : 원천 표 전체를 한 번 읽어 메모리에 둠 (같은 프로세스 안에서 재사용)
# input : with_news  뉴스 표까지 읽을지 (약 800MB, 필요할 때만)
# output: {"daily", "price", "earnings", "analyst", ("news")}
# -----------------------------------------------------------------------------
@functools.lru_cache(maxsize=2)
def load_tables(with_news=False):
    t = {"daily": pd.read_parquet(DATA / "daily.parquet"),
         "price": pd.read_parquet(DATA / "price.parquet", columns=["symbol", "datetime", "close", "session", "known_at"]),
         "earnings": pd.read_parquet(DATA / "earnings.parquet"),
         "analyst": pd.read_parquet(DATA / "analyst.parquet")}
    if with_news:
        t["news"] = pq.read_table(DATA / "news.parquet", columns=["known_at", "symbols", "source", "title", "tone"]).to_pandas()
    return t


# -----------------------------------------------------------------------------
# 기능  : 기준일·대상일 쌍. 대상일 기준으로 START 이후만
# output: DataFrame[date, target]
# -----------------------------------------------------------------------------
def calendar():
    d = np.sort(load_tables()["daily"]["date_et"].astype(NS).dt.normalize().unique())
    cal = pd.DataFrame({"date": d[:-1], "target": d[1:]})
    return cal[cal["date"] >= START].reset_index(drop=True)


# -----------------------------------------------------------------------------
# 기능  : 대상일 정답
# 수식  : ret_pct = ret × 100,  label = label_of(ret_pct)
# output: DataFrame[symbol, target, ret_pct, label]
# -----------------------------------------------------------------------------
def labels():
    d = load_tables()["daily"].dropna(subset=["ret"])
    out = pd.DataFrame({"symbol": d["symbol"], "target": d["date_et"].astype(NS).dt.normalize(),
                        "ret_pct": d["ret"] * 100})
    out["label"] = label_of(out["ret_pct"]).astype(int)
    return out


# -----------------------------------------------------------------------------
# 기능  : 연구용 feature (Reddit, LLM) 를 붙임. 제출 모델에서는 만들 수 없는 값임
# input : x  패널,  names  RESEARCH_GROUPS 의 이름 목록
# output: 열이 붙은 패널 (원천 캐시가 없으면 NaN 열과 경고)
# -----------------------------------------------------------------------------
def _attach_research(x, names):
    key = ["symbol", "target"]
    if "reddit" in names:
        import sys
        sys.path.insert(0, str(ROOT / "eda"))
        import eda_utils as E                         # EDA 에서 만든 Reddit 집계를 재사용
        cal = E.trading_calendar(load_tables()["daily"])
        m, a = E.load_reddit(cal, sorted(load_tables()["daily"]["symbol"].unique()))
        tot = m.groupby(key)["mentions"].sum().reset_index()
        x = x.merge(E.abnormal(tot, "mentions", cal["target"].to_numpy())[key + ["mentions_abn"]], on=key, how="left")
        act = E.abnormal(a.rename(columns={"sub": "symbol"}), "items", cal["target"].to_numpy())
        act = act.pivot(index="target", columns="symbol", values="items_abn")
        act.columns = [f"act_{c}_abn" for c in act.columns]
        x = x.merge(act.reset_index(), on="target", how="left")
    if {"finbert", "finbert_emb"} & set(names):
        from .text import finbert_features
        include_embeddings = "finbert_emb" in names
        fb = finbert_features(
            sorted(x["symbol"].unique()),
            include_embeddings=include_embeddings
        )
    if len(fb.columns) > 2:
        x = x.merge(fb, on=key, how="left")
    for name, fname in LLM_FILES.items():
        if name in names:
            path = CACHE / fname
            if path.exists():
                x = x.merge(pd.read_parquet(path), on=key, how="left")
            else:
                print(f"[panel] {path.name} 없음. python -m exp llm 을 먼저 실행해야 함 (NaN 으로 채움)")
    for c in [c for n in names for c in RESEARCH_GROUPS[n]]:
        if c not in x.columns:
            x[c] = np.nan
    return x


def _research_stamp(name):
    """연구용 묶음의 원천 파일 수정 시각 (없으면 None). reddit 은 EDA 캐시라 코드 지문으로 충분함"""
    from .text import FINBERT_SCORES
    f = {"finbert": FINBERT_SCORES, "finbert_emb": FINBERT_SCORES}.get(name) or (CACHE / LLM_FILES[name] if name in LLM_FILES else None)
    return int(f.stat().st_mtime) if f is not None and f.exists() else None


# -----------------------------------------------------------------------------
# 기능  : universe 하나의 연구용 패널
# input : universe  종목 목록 (예: 20종목)
#         groups    feature 묶음 이름 (뉴스 표를 읽을지 결정하는 데 씀)
#         research  RESEARCH_GROUPS 이름 목록
#         allow_lockbox  False(기본) 면 lockbox 행(대상일 ≥ LOCKBOX_START)을 빼고 돌려줌
#                        True 는 lockbox 평가(runner, 고정된 계획)와 lockbox 평가 이후 export 에서만 씀
# output: DataFrame[symbol, date, target, <features>, ret_pct, label, rest]
#         rest = log(1 + ret) − log(1 + gap)   (갭 이후 움직임, 분석용. feature 로 쓰지 않음)
# 캐시  : exp/cache/panel_<key>.parquet,  key = sha256(universe, 뉴스 여부, research, 코드·데이터 지문)
# -----------------------------------------------------------------------------
def universe_panel(universe, groups=(), research=(), allow_lockbox=False):
    universe = sorted(universe)
    with_news = any("news" in GROUP_TABLES.get(g, ()) for g in groups)
    fp = fingerprint()
    stamps = [(n, _research_stamp(n)) for n in sorted(research)]     # 연구 결과 파일이 바뀌면 캐시도 새로
    key = hashlib.sha256(json.dumps([universe, with_news, stamps, fp], sort_keys=True).encode()).hexdigest()[:16]
    path = CACHE / f"panel_{key}.parquet"
    if path.exists():
        return dev_only(pd.read_parquet(path), allow_lockbox)
    tables = load_tables(with_news)
    x = build_features(tables, calendar(), universe)
    x = x.merge(labels(), on=["symbol", "target"], how="inner")
    x["rest"] = np.log1p(x["ret_pct"] / 100) - np.log1p(x["gap_pct"] / 100)
    x = _attach_research(x, research)
    CACHE.mkdir(parents=True, exist_ok=True)
    x.to_parquet(path, index=False)
    return dev_only(x, allow_lockbox)
