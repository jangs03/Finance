"""제출 파일이자 실험 공용 코어.

기능  : 대상일 09:30 전 정보로 feature 를 만들고, 신호(signal) → 결정 규칙(decision)
        → regime 보정(regime) 순서로 label 0~4 를 냄
구성  : 1. 상수·registry   2. feature (build_features / day_tables / day_features)
        3. 점수            4. 신호 head   5. 결정 규칙   6. regime 보정
        7. Pipeline        8. Model / load_model
역할  : 채점(predict)과 실험(exp/)이 이 파일의 같은 함수를 씀 → 학습·검증·제출의 feature 가 구조적으로 같음
        채점자는 이 파일 외의 src/ 를 원본으로 덮어쓰므로, 제출에 필요한 코드는 전부 여기 둠

실험할 때 바꾸는 곳은 registry 이름과 spec(dict) 뿐임.
    FEATURE_GROUPS   feature 묶음 이름 → 열 이름
    SIGNALS          신호 이름 → 클래스
    DECISIONS        결정 규칙 이름 → 클래스
    REGIMES          regime 보정 이름 → 클래스
spec 예:
    {"name": "B2", "signal": {"type": "gap"}, "decision": {"type": "k"}}
"""

import pickle
import sys
import warnings

import numpy as np
import pandas as pd

from .data import CUTOFF, FLAT, WEIGHT
from .paths import ARTIFACTS

# =============================================================================
# 1. 상수 · registry
# =============================================================================

NS = "datetime64[ns]"
HIST_DAYS = 100          # predict 때 읽는 일봉 길이. 가장 긴 rolling(60) + 여유
NEWS_HIST_DAYS = 30      # predict 때 읽는 뉴스 길이. NEWS_WINDOW + 여유
NEWS_WINDOW = 20         # 뉴스 평소 수준 = 직전 20개 대상일 창의 평균
BIG_CUT = 2.5            # 급등락 경계(%)
VOL_FLOOR = 0.5          # zgap 분모의 하한(%). 변동성 0 근처에서 폭발 방지
ZGAP_CLIP = 10.0
BETA_SHRINK = 0.5        # beta = 0.5 × rolling beta + 0.5 (1 쪽으로 수축)
SURP_CLIP = 50.0         # surprise_pct 절단(%)
TONE_TAIL = 5.0          # tone 꼬리 기준
MIN_LOO = 3              # 시장 갭을 계산할 최소 다른 종목 수
BIG_GAP = 1.0            # regime: 큰 갭 종목 기준(%)
RET_CLIP = 15.0          # 회귀 목표 절단(%)
DEFAULT_K = 5.25         # artifact 가 없을 때만 쓰는 갭 규칙 배율. EDA 값이라 lockbox 가 노출된 값임 (실험·제출은 학습으로 k 를 고름)
SEQ_LEN = 20             # 시계열 feature(seq) 길이: 기준일 포함 직전 20 거래일
CLASS_CENTERS = np.array([-4.0, -1.5, 0.0, 1.5, 4.0])   # 5-class 확률 → 기대 수익률(%) 신호

ARTIFACT = ARTIFACTS / "pipeline.pkl"
ARTIFACT_FORMAT = 1

# feature 묶음. 실험 config 에서는 이 이름으로 부름
FEATURE_GROUPS = {
    "core":    ["gap_pct", "gap_rank"],
    "scale":   ["vol5", "vol20", "atr14", "big_rate60", "zgap"],
    "cross":   ["mkt_gap", "resid_gap", "beta60"],
    "event":   ["earn", "surp_sign", "surp_abs", "surp_pct_w"],
    "analyst": ["an_n", "an_up", "an_down", "an_tgt_chg", "an_gap_agree"],
    "news":    ["news_n", "news_abn", "tone_mean", "tone_min", "tone_max", "tone_neg_tail", "news_missing"],
    "regime":  ["reg_med_vol20", "reg_med_absgap", "reg_gap_disp", "reg_mkt_gap", "reg_big_ratio"],
    # 시계열: ret_l{i} = (i−1) 거래일 전 일간 수익률(%), rng_l{i} = 같은 날 (고가−저가)/종가(%). l1 = 기준일
    "seq":     [f"ret_l{i}" for i in range(1, SEQ_LEN + 1)] + [f"rng_l{i}" for i in range(1, SEQ_LEN + 1)],
    # 프리마켓 경로: 대상일 pre 봉(known_at < 09:30) 종가의 범위, 갭을 시간외(post)·프리마켓(pre) 두 구간으로 분해
    "premkt":  ["pre_rng", "post_ret", "pre_ret"],
    # HAR 식 크기 변수: |갭|, 과거 |수익률| 의 1·5·22일 평균 (Corsi 2009 의 일·주·월 성분)
    "har":     ["abs_gap", "absr1", "absr5", "absr22"],
}
# 묶음마다 필요한 원천 표. day_tables() 가 필요한 표만 읽게 함
GROUP_TABLES = {
    "core": {"daily", "price"}, "scale": {"daily", "price"}, "cross": {"daily", "price"},
    "regime": {"daily", "price"}, "seq": {"daily", "price"}, "premkt": {"daily", "price"}, "har": {"daily", "price"}, "event": {"earnings"}, "analyst": {"analyst"}, "news": {"news"},
}
DEPLOYABLE = {c for cols in FEATURE_GROUPS.values() for c in cols}
KEY_COLS = ["symbol", "date", "target"]


# -----------------------------------------------------------------------------
# 기능  : feature 목록(묶음 이름 또는 열 이름 섞어서)을 열 이름 목록으로 풂
# input : items  ["core", "vol20", ...]
#         extra  FEATURE_GROUPS 에 없는 연구용 묶음 {이름: [열]} (exp 에서 넘김)
# output: 중복 없는 열 이름 목록 (입력 순서 유지)
# -----------------------------------------------------------------------------
def resolve_features(items, extra=None):
    groups = {**FEATURE_GROUPS, **(extra or {})}
    out = []
    for it in items or []:
        out += groups.get(it, [it])
    return list(dict.fromkeys(out))


# -----------------------------------------------------------------------------
# 기능  : 열 이름 목록을 계산하는 데 필요한 feature 묶음 이름 집합
# input : cols  열 이름 목록
# output: {"core", ...}  (core 는 신호 대체값 계산에 늘 필요해서 항상 포함)
# -----------------------------------------------------------------------------
def groups_of(cols):
    need = {"core"}
    for g, members in FEATURE_GROUPS.items():
        if set(cols) & set(members):
            need.add(g)
    return need


# =============================================================================
# 2. feature
# =============================================================================

# -----------------------------------------------------------------------------
# 기능  : 날짜별로 자기 자신을 뺀 나머지 종목의 중앙값 (leave-one-out median)
# 수식  : mkt_i,t = median_{j≠i} x_j,t   (다른 종목 유효값이 MIN_LOO 개 미만이면 NaN)
# input : values  값 Series,  groups  같은 index 의 날짜 Series
# output: values 와 같은 index 의 Series
# -----------------------------------------------------------------------------
def loo_median(values, groups):
    def _loo(a):
        m = np.tile(a, (len(a), 1))
        np.fill_diagonal(m, np.nan)
        cnt = (~np.isnan(m)).sum(axis=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(m, axis=1) if len(a) else m.sum(axis=1)
        med[cnt < MIN_LOO] = np.nan
        return med

    return values.groupby(groups).transform(lambda s: _loo(s.to_numpy(float)))


# -----------------------------------------------------------------------------
# 기능  : 이벤트 시각을 그 정보를 처음 쓸 수 있는 대상일로 붙임
# 수식  : 대상일 T 의 창 = [기준일 09:30, T 09:30)   ← known_at < cutoff 규칙
# input : ts       이벤트 시각 Series (known_at)
#         targets  DataFrame[date, target]  (기준일, 대상일)
# output: 대상일 Series (창 밖이면 NaT)
# -----------------------------------------------------------------------------
def map_to_target(ts, targets):
    t = targets.sort_values("target")
    lo = (t["date"] + CUTOFF).to_numpy(NS)
    hi = (t["target"] + CUTOFF).to_numpy(NS)
    tgt = t["target"].to_numpy(NS)
    v = pd.Series(ts).to_numpy(NS)
    idx = np.searchsorted(hi, v, side="right")
    ok = idx < len(hi)
    ok[ok] &= v[ok] >= lo[idx[ok]]
    out = np.full(len(v), np.datetime64("NaT"), dtype=NS)
    out[ok] = tgt[idx[ok]]
    return pd.Series(out, index=pd.Series(ts).index)


# -----------------------------------------------------------------------------
# 기능  : 일봉에서 기준일 16:00 까지 알 수 있는 종목별 특성 (모든 rolling 은 과거만)
# 수식  : ret_pct = ret × 100
#         vol_n   = std(ret_pct, 최근 n일)
#         atr14   = mean(TR, 14일),  TR = max(H−L, |H−C₋₁|, |L−C₋₁|) / C × 100
#         big_rate60 = mean(1[ret_pct < −2.5 or ≥ 2.5], 60일)
#         beta60  = 0.5 × cov(r, m)/var(m) (60일) + 0.5,  m = 자기 제외 중앙 수익률
#         ret_l{i} = ret_pct(기준일 − (i−1) 거래일),  rng_l{i} = (H−L)/C × 100 (같은 날),  i = 1..SEQ_LEN
#         absr1 = |ret_pct| (기준일),  absr5 / absr22 = mean(|ret_pct|, 최근 5 / 22일)
# input : daily  일봉 (symbol, date_et, high, low, close, prev_close, ret)
# output: DataFrame[symbol, date, close_d0, vol5, vol20, atr14, big_rate60, beta60, ret_l*, rng_l*, absr*]
# -----------------------------------------------------------------------------
def _daily_block(daily):
    d = daily.assign(date=daily["date_et"].astype(NS).dt.normalize())
    d = d.sort_values(["symbol", "date"]).reset_index(drop=True)
    sym = d["symbol"]

    def roll(s, n, how):
        return s.groupby(sym).transform(lambda x: getattr(x.rolling(n, min_periods=max(2, n // 2)), how)())

    r = d["ret"] * 100
    big = ((r < -BIG_CUT) | (r >= BIG_CUT)).astype(float).where(r.notna())
    tr = pd.concat([d["high"] - d["low"], (d["high"] - d["prev_close"]).abs(),
                    (d["low"] - d["prev_close"]).abs()], axis=1).max(axis=1) / d["close"] * 100
    m = loo_median(r, d["date"])
    ex, em, exm, emm = roll(r, 60, "mean"), roll(m, 60, "mean"), roll(r * m, 60, "mean"), roll(m * m, 60, "mean")
    beta = (exm - ex * em) / (emm - em ** 2)
    rng = (d["high"] - d["low"]) / d["close"] * 100
    seq = {f"{name}_l{i}": v.groupby(sym).shift(i - 1)
           for name, v in (("ret", r), ("rng", rng)) for i in range(1, SEQ_LEN + 1)}
    return pd.DataFrame({
        "symbol": sym, "date": d["date"], "close_d0": d["close"],
        "vol5": roll(r, 5, "std"), "vol20": roll(r, 20, "std"), "atr14": roll(tr, 14, "mean"),
        "big_rate60": roll(big, 60, "mean"),
        "beta60": (BETA_SHRINK * beta + (1 - BETA_SHRINK)).fillna(1.0),
        "absr1": r.abs(), "absr5": roll(r.abs(), 5, "mean"), "absr22": roll(r.abs(), 22, "mean"),
        **seq,
    })


# -----------------------------------------------------------------------------
# 기능  : 시간봉에서 하루 단위 시간외 가격
#         pre_last  그날 pre 봉 중 known_at < 그날 09:30 인 마지막 종가 (= 08:00 봉, 09:00 봉 제외)
#         post_last 그날 post 봉의 마지막 종가
# input : price  시간봉 (symbol, datetime, close, session, known_at)
# output: (pre_last Series, post_last Series)  index = (symbol, day)
# -----------------------------------------------------------------------------
def _extended_hours(price):
    p = price[price["session"] != "regular"].sort_values("datetime")
    day = p["datetime"].astype(NS).dt.normalize()
    pre = (p["session"] == "pre") & (p["known_at"].astype(NS) < day + CUTOFF)
    post = p["session"] == "post"
    last = lambda m: p[m].groupby([p.loc[m, "symbol"], day[m]])["close"].last()
    return last(pre), last(post)


# -----------------------------------------------------------------------------
# 기능  : 대상일 프리마켓 경로의 범위 (_extended_hours 와 같은 봉: pre, known_at < 그날 09:30)
# 수식  : pre_rng = (max close / min close − 1) × 100   (봉 1개뿐이면 0)
# input : price  시간봉
# output: Series index = (symbol, day)
# -----------------------------------------------------------------------------
def _premarket_range(price):
    p = price[price["session"] == "pre"]
    day = p["datetime"].astype(NS).dt.normalize()
    p = p[p["known_at"].astype(NS) < day + CUTOFF]
    g = p.groupby([p["symbol"], day[p.index]])["close"]
    return (g.max() / g.min() - 1) * 100


# -----------------------------------------------------------------------------
# 기능  : 실적 · 애널리스트 이벤트를 (종목, 대상일)로 집계
# 수식  : earn = 1[창 안에 실적 발표],  surp_sign = sign(실제 − 예상),  surp_abs = |실제 − 예상|
#         surp_pct_w = clip(surprise_pct, ±50)
#         an_tgt_chg = mean(clip(목표가/이전 목표가 − 1, ±0.5))
# input : e / a  실적·애널리스트 표,  targets  DataFrame[date, target]
# output: DataFrame[symbol, target, <feature>]
# -----------------------------------------------------------------------------
def _earnings_block(e, targets):
    e = e.assign(target=map_to_target(e["known_at"], targets)).dropna(subset=["target"]).sort_values("known_at")
    diff = e["eps_reported"] - e["eps_estimate"]
    e = e.assign(earn=1.0, surp_sign=np.sign(diff), surp_abs=diff.abs(),
                 surp_pct_w=e["surprise_pct"].clip(-SURP_CLIP, SURP_CLIP))
    return e.groupby(["symbol", "target"])[FEATURE_GROUPS["event"]].last().reset_index()


def _analyst_block(a, targets):
    a = a.assign(target=map_to_target(a["known_at"], targets)).dropna(subset=["target"])
    a = a.assign(an_n=1.0, an_up=(a["action"] == "up").astype(float),
                 an_down=(a["action"] == "down").astype(float),
                 an_tgt_chg=(a["target_current"] / a["target_prior"] - 1).clip(-0.5, 0.5))
    return (a.groupby(["symbol", "target"])
            .agg(an_n=("an_n", "sum"), an_up=("an_up", "sum"), an_down=("an_down", "sum"),
                 an_tgt_chg=("an_tgt_chg", "mean")).reset_index())


# -----------------------------------------------------------------------------
# 기능  : 뉴스를 (종목, 대상일)로 집계. 정규화한 제목 해시로 같은 창 안의 중복을 제거
# 수식  : news_n   = 중복 제거 후 기사 수 (없으면 0)
#         news_abn = log(1 + news_n) − log(1 + 직전 20개 창의 news_n 평균)
#         news_missing = 1[최근 21개 창의 기사 합 = 0]  (커버리지 없음 표시)
# input : n  뉴스 (known_at, symbols, title, tone),  targets,  universe
# output: DataFrame[symbol, target, news_*, tone_*]
# -----------------------------------------------------------------------------
def _news_block(n, targets, universe):
    n = n.assign(symbol=n["symbols"].fillna("").str.split(",")).explode("symbol")
    n = n[n["symbol"].isin(set(universe))]
    n = n.assign(target=map_to_target(n["known_at"], targets)).dropna(subset=["target"])
    norm = n["title"].fillna("").str.lower().str.replace(r"[^a-z0-9]+", " ", regex=True).str.strip()
    n = n.assign(h=pd.util.hash_pandas_object(norm, index=False).to_numpy())
    u = n.drop_duplicates(["symbol", "target", "h"])
    key = [u["symbol"], u["target"]]
    agg = u.groupby(key)["tone"].agg(news_n="size", tone_mean="mean", tone_min="min", tone_max="max")
    agg["tone_neg_tail"] = (u["tone"] < -TONE_TAIL).groupby(key).sum()

    grid = pd.MultiIndex.from_product([sorted(universe), np.sort(targets["target"].unique())],
                                      names=["symbol", "target"])
    out = agg.reindex(grid)
    out["news_n"] = out["news_n"].fillna(0.0)
    out["tone_neg_tail"] = out["tone_neg_tail"].fillna(0.0)
    g = out["news_n"].groupby(level="symbol")
    base = g.transform(lambda s: s.shift().rolling(NEWS_WINDOW, min_periods=NEWS_WINDOW).mean())
    out["news_abn"] = np.log1p(out["news_n"]) - np.log1p(base)
    out["news_missing"] = (g.transform(lambda s: s.rolling(NEWS_WINDOW + 1, min_periods=NEWS_WINDOW + 1).sum()) == 0).astype(float)
    return out.reset_index()


# =============================================================================
# 2-T. 텍스트 (뉴스 제목) — docs/text_pipeline.md
#      수집 → 정제(템플릿·근접중복·최초보도) → 기업 연결(R0/R1/R2) → 토큰화 → 표현 → 종목-대상일 집계
#      학습(전체 표)과 predict(최근 NEWS_HIST_DAYS 거래일)가 같은 값을 내도록, 모든 과거 참조는
#      "직전 몇 개 대상일 창" 단위로만 함 (FIRST_LOOKBACK, NEWS_WINDOW)
# =============================================================================

TEXT_RESOURCES = ARTIFACTS / "text_resources.json"
FIRST_LOOKBACK = 5        # 최초 보도: 직전 5개 창에 같은 클러스터가 없으면 새 이야기
RELEVANCE = ("r0", "r1", "r2")
NEGATORS = {"no", "not", "never", "without", "neither", "nor", "none", "cannot", "nobody", "nothing"}
STOPWORDS = {"a", "an", "the", "of", "in", "on", "at", "to", "for", "by", "with", "from", "and", "or", "as",
             "is", "are", "was", "were", "be", "been", "its", "it", "this", "that", "these", "those", "into",
             "over", "after", "before", "about", "s", "inc", "corp", "co", "ltd", "plc", "llc", "nyse", "nasdaq"}
# 기업 이름과 겹치는 일반 단어 티커는 cashtag($) 나 이름으로만 인정 (새 종목도 2글자 이하는 같은 규칙)
COMMON_TICKERS = {"NOW", "COST", "WELL", "GILD", "COP", "BA", "DE", "GE", "KO", "MS", "PM", "CB", "V", "ALL", "ARE"}
# 자동 생성 기사: 기관 보유(13F)·내부자 매매·공매도 잔고·종목 목록 기사. 새 정보가 아님 (w4-2 p17 boilerplate)
TEMPLATE_RX = (
    r"(?:shares (?:sold|bought|purchased|acquired) by|shares (?:in|of) .{0,80}(?:acquired|sold|bought|purchased) by"
    r"|(?:acquires?|buys?|sells?|purchases?|trims?|cuts?|lowers?|raises?|boosts?|increases?|decreases?|reduces?|grows?|lifts?)"
    r" (?:\$[\d.,]+ (?:million|billion) |new |[\d,]+ shares (?:of|in) |its )?(?:stake|position|holdings?|shares)\b"
    r"|(?:stake|position|holdings?) (?:in|raised|lowered|boosted|trimmed|cut|increased|decreased|reduced|lifted)\b"
    r"|has \$[\d.,]+ (?:million|billion|thousand)|invests? \$[\d.,]+ (?:million|billion|thousand) in|stock position"
    r"|takes position in|(?:sells?|buys?) [\d,]+ shares|sells? \$[\d.,]+ in stock|insider (?:selling|buying|sells|buys)"
    r"|short interest|stocks to (?:add|watch|buy)|watchlist)")
# 사건 유형 (w5-1 p22 named event). 소문자 제목에 적용
EVENT_RX = {
    "earn":  r"\b(?:earnings|eps|quarterly (?:results|profit|revenue)|q[1-4] (?:results|earnings|revenue|sales)|beats?|miss(?:es|ed)?|profit (?:rises|falls|jumps|drops))\b",
    "guid":  r"\b(?:guidance|outlook|forecasts?|raises? (?:its )?(?:full-year|annual|outlook|forecast)|cuts? (?:its )?(?:outlook|forecast))\b",
    "anal":  r"\b(?:upgraded?s?|downgraded?s?|price target|initiates? coverage|reiterates?|overweight|underweight|outperform|underperform)\b",
    "legal": r"\b(?:lawsuit|sued?s?|probe|investigation|settles?|settlement|fined?|antitrust|subpoena|recalls?|indicted|charges filed)\b",
    "mna":   r"\b(?:acquisition|acquires?|to acquire|merger|merges?|buyout|takeover|deal to buy|spin-?off|divest\w*)\b",
    "prod":  r"\b(?:launch\w*|unveil\w*|fda|approv\w+|clinical trial|patent|rollout|release[sd]? new)\b",
    "mgmt":  r"\b(?:ceo|cfo|chief executive|resign\w*|steps? down|appoint\w*|succeed\w*|layoffs?|job cuts)\b",
}

FEATURE_GROUPS.update({
    "txt_count": [f"n_clu_{r}" for r in RELEVANCE] + [f"n_first_{r}" for r in RELEVANCE]
                 + [f"n_clu_{r}_abn" for r in RELEVANCE] + [f"n_first_{r}_abn" for r in RELEVANCE] + ["n_src_r1"],
    "txt_tone_gdelt": ["tone_mean_r1", "tone_neg_share_r1", "tone_sd_r1"],
    "txt_tone_lm": ["lm_net_r1", "lm_neg_share_r1", "lm_unc_r1"],
    "txt_novelty": ["novelty_r1"],
    "txt_event": [f"ev_{e}_r1" for e in EVENT_RX] + ["ev_any_r1"],
    "txt_meta": ["n_scored_r1", "tmpl_share_r0", "alias_missing"],
    "txt_text": ["titles_r1"],
    # 개장 직전 뉴스 (리서치 3순위): ovn = [기준일 16:00, 대상일 09:30),  late = [대상일 06:30, 09:30)
    # 뉴스 본문 (리서치 4순위): R1 대표 기사의 본문 첫 BODY_CHARS 글자(lead). 본문은 기사의 약 7.5% 에만 있음
    "txt_body": ["n_body_r1", "body_share_r1", "body_mention_r1", "ev_any_body_r1"],
    "txt_lead": ["lead_r1", "doc_r1"],
    "txt_recent": ["n_clu_r1_ovn", "n_first_r1_ovn", "n_first_r1_ovn_abn", "n_clu_r1_late", "n_first_r1_late",
                   "ovn_share_r1", "tone_mean_r1_ovn"],
})
for _g in ("txt_count", "txt_tone_gdelt", "txt_tone_lm", "txt_novelty", "txt_event", "txt_meta", "txt_text",
           "txt_recent", "txt_body", "txt_lead"):
    GROUP_TABLES[_g] = {"news"}
TEXT_COLS = {"titles_r1", "lead_r1", "doc_r1"}  # 문자열 열 (숫자 비교·결측 채우기에서 제외)
BODY_CHARS = 600                                 # 본문은 첫 600 글자(lead)만 씀 (기사 앞부분에 핵심, 메모리 절약)
DEPLOYABLE = {c for cols in FEATURE_GROUPS.values() for c in cols}


# -----------------------------------------------------------------------------
# 기능  : 텍스트 자원 읽기 (artifacts/text_resources.json, `python -m exp text resources` 로 만듦)
# output: {"aliases": {티커: [이름]}, "lm": {"positive", "negative", "uncertainty": [단어]}, "version"} 또는 빈 dict
#         파일이 없으면 alias·LM 사전 없이 티커 매칭만 함 (LM feature 는 NaN)
# -----------------------------------------------------------------------------
_RES_CACHE = {}


def text_resources():
    key = TEXT_RESOURCES.stat().st_mtime if TEXT_RESOURCES.exists() else None
    if key not in _RES_CACHE:
        import json
        _RES_CACHE.clear()
        _RES_CACHE[key] = json.loads(TEXT_RESOURCES.read_text(encoding="utf-8")) if key else {}
    return _RES_CACHE[key]


# -----------------------------------------------------------------------------
# 기능  : 제목 정규화
#         clean_title  HTML 엔티티 복원, 뒤에 붙은 " - 출처" / " | 출처" (40자 이하) 제거
#         cluster_key  근접 중복 키 = 불용어·숫자 뺀 소문자 단어 집합의 해시
#                      (어순·구두점·출처 꼬리표만 다른 재게시 기사를 같은 이야기로 묶음, w4-2 p17)
# -----------------------------------------------------------------------------
def clean_title(s):
    import html
    s = s.fillna("").astype(str).map(html.unescape)
    return s.str.replace(r"\s+[-|–—]\s+[^-|–—]{1,40}$", "", regex=True).str.strip()


def cluster_key(titles):
    toks = titles.str.lower().str.findall(r"[a-z][a-z'&-]+")
    norm = toks.map(lambda ts: " ".join(sorted({t for t in ts if t not in STOPWORDS})))
    return pd.util.hash_pandas_object(norm, index=False).to_numpy()


# -----------------------------------------------------------------------------
# 기능  : 기업 연결 정규식
#         symbol_pattern(s)  그 종목을 가리키는 표현: $TICKER, 대문자 TICKER(일반 단어 티커 제외), alias 이름
#         _entity_finder     universe·alias 전체 회사 표현 → 제목에 나온 회사 집합 (R2 판정)
# -----------------------------------------------------------------------------
def symbol_pattern(sym, aliases):
    import re
    parts = [rf"\${re.escape(sym)}\b"]
    if sym not in COMMON_TICKERS and len(sym) > 2:
        parts.append(rf"\b{re.escape(sym)}\b")
    parts += [rf"(?i:(?<!\w){re.escape(a)}(?:'s)?(?!\w))" for a in sorted(aliases.get(sym, []), key=len, reverse=True)]
    return re.compile("|".join(parts))


def _entity_finder(symbols, aliases):
    pats = {s: symbol_pattern(s, aliases) for s in sorted(set(symbols) | set(aliases))}
    return lambda title: {s for s, p in pats.items() if p.search(title)}


# -----------------------------------------------------------------------------
# 기능  : LM 금융 사전 점수 (w5-1 p6). 부정어 뒤 3단어 안의 긍정 단어는 부정으로 셈
# 수식  : lm_net = (n_pos − n_neg) / n_words,  lm_unc = n_uncertainty / n_words
# input : titles (Series),  lm  {"positive", "negative", "uncertainty"}
# output: DataFrame[lm_net, lm_unc]  (사전이 없으면 NaN)
# -----------------------------------------------------------------------------
def lm_scores(titles, lm):
    import re
    if not lm:
        return pd.DataFrame({"lm_net": np.nan, "lm_unc": np.nan}, index=titles.index)
    pos, neg, unc = set(lm["positive"]), set(lm["negative"]), set(lm["uncertainty"])
    rx = re.compile(r"[A-Za-z]+")

    def one(t):
        w = [x.upper() for x in rx.findall(t)]
        if not w:
            return (np.nan, np.nan)
        p = n = u = 0
        for i, x in enumerate(w):
            negated = any(v.lower() in NEGATORS for v in w[max(0, i - 3):i])
            if x in pos:
                n, p = (n + 1, p) if negated else (n, p + 1)
            elif x in neg:
                n += 1
            if x in unc:
                u += 1
        return ((p - n) / len(w), u / len(w))

    v = np.array([one(t) for t in titles], dtype=float).reshape(-1, 2)
    return pd.DataFrame({"lm_net": v[:, 0], "lm_unc": v[:, 1]}, index=titles.index)


# -----------------------------------------------------------------------------
# 기능  : 기사 단위 텍스트 표 (정제 + 기업 연결). 집계·연구용 feature(FinBERT·LLM)가 같은 표를 씀
# input : news  (known_at, symbols, source, title, tone),  targets,  universe,  res (text_resources)
# output: DataFrame[symbol, target, widx, known_at, source, title, tone, title_id, ckey,
#                   is_template, rel (0/1/2), n_entities]
#         title_id = 정규화 제목 해시 (FinBERT·LLM 결과를 붙이는 키),  widx = 대상일 창 번호
#         rel: 0 태그만, 1 제목에 그 회사 언급, 2 언급된 회사가 그 회사 하나뿐
# -----------------------------------------------------------------------------
ARTICLE_COLS = ["symbol", "target", "widx", "known_at", "source", "title", "tone", "title_id", "ckey",
                "is_template", "rel", "n_entities", "lead"]


def text_articles(news, targets, universe, res=None):
    res = res if res is not None else text_resources()
    aliases = res.get("aliases", {})
    n = news.assign(symbol=news["symbols"].fillna("").str.split(",")).explode("symbol")
    n = n[n["symbol"].isin(set(universe))]
    n = n.assign(target=map_to_target(n["known_at"], targets)).dropna(subset=["target"])
    if n.empty:
        return pd.DataFrame(columns=ARTICLE_COLS)
    n = n.reset_index(drop=True)
    if "source" not in n.columns:
        n["source"] = ""
    lead = n["text"] if "text" in n.columns else pd.Series("", index=n.index)
    n["lead"] = lead.fillna("").astype(str).str.slice(0, BODY_CHARS).str.replace(r"\s+", " ", regex=True).str.strip()
    n["title"] = clean_title(n["title"])
    norm = n["title"].str.lower().str.replace(r"[^a-z0-9]+", " ", regex=True).str.strip()
    n["title_id"] = pd.util.hash_pandas_object(norm, index=False).astype(str).to_numpy()
    n["ckey"] = cluster_key(n["title"])
    n["is_template"] = n["title"].str.contains(TEMPLATE_RX, case=False, regex=True).astype(int)
    tgt = np.sort(targets["target"].astype(NS).unique())
    n["widx"] = np.searchsorted(tgt, n["target"].to_numpy(NS))
    rel = np.zeros(len(n), int)
    for s, idx in n.groupby("symbol").indices.items():
        rel[idx] = n["title"].iloc[idx].str.contains(symbol_pattern(s, aliases), regex=True).to_numpy().astype(int)
    find = _entity_finder(universe, aliases)
    ent = np.zeros(len(n), int)
    r1 = np.flatnonzero(rel == 1)
    ent[r1] = [len(find(t)) for t in n["title"].to_numpy()[r1]]
    rel[r1[ent[r1] <= 1]] = 2
    n["rel"], n["n_entities"] = rel, ent
    return n[ARTICLE_COLS].sort_values(["symbol", "known_at", "title_id"], kind="stable").reset_index(drop=True)


# -----------------------------------------------------------------------------
# 기능  : 신규성 (w5-1 p22~23). 오늘 대표 제목이 같은 종목의 직전 NEWS_WINDOW 창 제목과 얼마나 다른가
# 수식  : novelty = max_{오늘 제목 i} (1 − max_{과거 제목 j} cos(v_i, v_j)),  v = 해싱 TF 벡터(불용어 제외, l2)
#         과거 제목이 없으면 1. 상태가 없는 해싱 벡터라 학습·predict 가 같은 값을 냄
# input : reps  대표 기사 (symbol, widx, title), index 0..n-1
# output: Series index (symbol, widx) → novelty
# -----------------------------------------------------------------------------
def _novelty(reps):
    from sklearn.feature_extraction.text import HashingVectorizer
    if reps.empty:
        return pd.Series(dtype=float)
    hv = HashingVectorizer(n_features=2 ** 18, alternate_sign=False, norm="l2", lowercase=True,
                           token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z'&-]+\b", stop_words=sorted(STOPWORDS))
    V = hv.transform(reps["title"])
    out = {}
    for s, idx in reps.groupby("symbol", sort=False).indices.items():
        w = reps["widx"].to_numpy()[idx]
        for t in np.unique(w):
            today, past = idx[w == t], idx[(w >= t - NEWS_WINDOW) & (w < t)]
            if len(past) == 0:
                out[(s, t)] = 1.0
                continue
            sim = (V[today] @ V[past].T).toarray().max(axis=1)
            out[(s, t)] = float((1 - sim).max())
    return pd.Series(out, dtype=float)


# -----------------------------------------------------------------------------
# 기능  : 텍스트 feature 를 (종목, 대상일)로 집계
# 수식  : n_clu_r   = 관련도 ≥ r 인 비템플릿 기사의 클러스터 수 (r0 태그, r1 제목 언급, r2 그 회사만 언급)
#         n_first_r = 그중 직전 FIRST_LOOKBACK 창에 없던 클러스터 수 (새 이야기)
#         *_abn     = log(1 + x) − log(1 + 직전 NEWS_WINDOW 창 평균)
#         R1 대표 기사(클러스터별 첫 기사) 기준으로
#           tone_mean / tone_neg_share(tone<0 비율) / tone_sd(2개 이상)   ← GDELT tone, 규칙 2개 + 분산
#           lm_net / lm_neg_share(lm_net<0 비율) / lm_unc                  ← LM 금융 사전
#           novelty, ev_*(사건 유형 포함 여부), n_scored(대표 기사 수), titles_r1(대표 제목 " || " 연결)
#         txt_body: n_body_r1 = 본문이 있는 R1 클러스터 수,  body_share_r1 = n_body_r1 / n_clu_r1
#           body_mention_r1 = 본문 lead 안 그 회사 언급 횟수 평균 (본문 있는 대표 기사만),  ev_any_body_r1 = lead 에 사건 정규식
#           lead_r1 = 대표 기사 lead " || " 연결,  doc_r1 = titles_r1 + " || " + lead_r1  (문자열 열)
#         txt_recent: 대표 기사(클러스터 첫 기사) 시각이 ovn = [기준일 16:00, 대상일 09:30) / late = [대상일 06:30, 09:30)
#           n_clu_r1_{ovn,late}, n_first_r1_{ovn,late}, n_first_r1_ovn_abn, ovn_share_r1 = n_clu_r1_ovn / n_clu_r1,
#           tone_mean_r1_ovn (ovn 대표 기사 tone 평균)
#         기사가 없으면 count 는 0, 톤·신규성·사건은 NaN (문서 없음 ≠ 중립, w5-1 p19)
# -----------------------------------------------------------------------------
def _text_block(news, targets, universe):
    res = text_resources()
    a = text_articles(news, targets, universe, res)
    tgt = np.sort(targets["target"].astype(NS).unique())
    grid = pd.MultiIndex.from_product([sorted(universe), range(len(tgt))], names=["symbol", "widx"])
    out = pd.DataFrame(index=grid)
    nt = a[a["is_template"] == 0]
    clus = nt.groupby(["symbol", "widx", "ckey"]).agg(rel=("rel", "max")).reset_index()
    if len(clus):
        prev = clus[["symbol", "ckey", "widx"]].rename(columns={"widx": "w_prev"})
        pair = clus.merge(prev, on=["symbol", "ckey"])
        seen = pair[(pair["w_prev"] < pair["widx"]) & (pair["w_prev"] >= pair["widx"] - FIRST_LOOKBACK)]
        old = set(zip(seen["symbol"], seen["widx"], seen["ckey"]))
        clus["is_first"] = [k not in old for k in zip(clus["symbol"], clus["widx"], clus["ckey"])]
    else:
        clus["is_first"] = pd.Series(dtype=bool)
    for i, r in enumerate(RELEVANCE):
        c = clus[clus["rel"] >= i]
        out[f"n_clu_{r}"] = c.groupby(["symbol", "widx"]).size().reindex(grid).fillna(0.0)
        out[f"n_first_{r}"] = c[c["is_first"]].groupby(["symbol", "widx"]).size().reindex(grid).fillna(0.0)
    g = out.groupby(level="symbol")
    for r in RELEVANCE:
        for kind in ("clu", "first"):
            col = f"n_{kind}_{r}"
            base = g[col].transform(lambda s: s.shift().rolling(NEWS_WINDOW, min_periods=NEWS_WINDOW).mean())
            out[f"{col}_abn"] = np.log1p(out[col]) - np.log1p(base)
    alln = a.groupby(["symbol", "widx"]).agg(n=("is_template", "size"), t=("is_template", "sum"))
    out["tmpl_share_r0"] = (alln["t"] / alln["n"]).reindex(grid)
    r1 = nt[nt["rel"] >= 1]
    out["n_src_r1"] = r1.groupby(["symbol", "widx"])["source"].nunique().reindex(grid).fillna(0.0)

    reps = r1.groupby(["symbol", "widx", "ckey"], sort=False).agg(
        title=("title", "first"), known_at=("known_at", "first"), tone=("tone", "mean"),
        lead=("lead", "max")).reset_index()                    # max = 클러스터 안 비어 있지 않은 lead 하나 (문자열 순 최대)
    reps = reps.sort_values(["symbol", "widx", "known_at", "ckey"], kind="stable").reset_index(drop=True)
    reps = pd.concat([reps, lm_scores(reps["title"], res.get("lm"))], axis=1)
    low = reps["title"].str.lower()
    for e, rx in EVENT_RX.items():
        reps[f"ev_{e}"] = low.str.contains(rx, regex=True).astype(float)
    k = [reps["symbol"], reps["widx"]]
    agg = pd.DataFrame({
        "tone_mean_r1": reps.groupby(k)["tone"].mean(),
        "tone_neg_share_r1": (reps["tone"] < 0).astype(float).groupby(k).mean(),
        "tone_sd_r1": reps.groupby(k)["tone"].std(),
        "lm_net_r1": reps.groupby(k)["lm_net"].mean(),
        "lm_neg_share_r1": (reps["lm_net"] < 0).astype(float).where(reps["lm_net"].notna()).groupby(k).mean(),
        "lm_unc_r1": reps.groupby(k)["lm_unc"].mean(),
        "n_scored_r1": reps.groupby(k).size().astype(float),
        "titles_r1": reps.groupby(k)["title"].agg(" || ".join),
        **{f"ev_{e}_r1": reps.groupby(k)[f"ev_{e}"].max() for e in EVENT_RX},
    })
    agg.index.names = ["symbol", "widx"]
    out = out.join(agg)
    out["novelty_r1"] = _novelty(reps).reindex(grid)

    # 본문 lead
    has = reps["lead"].str.len().to_numpy() > 0
    ment = np.full(len(reps), np.nan)
    evb = np.full(len(reps), np.nan)
    for s, idx in reps.groupby("symbol").indices.items():
        idx = idx[has[idx]]
        if len(idx):
            ld = reps["lead"].iloc[idx]
            ment[idx] = ld.str.count(symbol_pattern(s, res.get("aliases", {}))).to_numpy(float)
            low_l = ld.str.lower()
            evb[idx] = np.max([low_l.str.contains(rx, regex=True).to_numpy(float) for rx in EVENT_RX.values()], axis=0)
    hb = reps[has]
    kb = [hb["symbol"], hb["widx"]]
    out["n_body_r1"] = hb.groupby(kb).size().reindex(grid).fillna(0.0)
    out["body_share_r1"] = out["n_body_r1"] / out["n_clu_r1"].where(out["n_clu_r1"] > 0)
    out["body_mention_r1"] = pd.Series(ment[has], index=hb.index).groupby(kb).mean().reindex(grid)
    out["ev_any_body_r1"] = pd.Series(evb[has], index=hb.index).groupby(kb).max().reindex(grid)
    out["lead_r1"] = hb.groupby(kb)["lead"].agg(" || ".join).reindex(grid)

    # 개장 직전 뉴스: 대표 기사 시각으로 창 안의 하위 구간을 나눔
    base_day = pd.Series(targets["date"].astype(NS).to_numpy(), index=targets["target"].astype(NS).to_numpy())
    base_day = base_day[~base_day.index.duplicated()]
    t_of = tgt[reps["widx"].to_numpy()] if len(reps) else np.array([], dtype=NS)
    ka = reps["known_at"].astype(NS).to_numpy()
    ovn = ka >= base_day.reindex(t_of).to_numpy(NS) + np.timedelta64(16, "h")
    late = ka >= t_of + np.timedelta64(390, "m")
    first = clus.set_index(["symbol", "widx", "ckey"])["is_first"]
    is_first = first.reindex(pd.MultiIndex.from_arrays([reps["symbol"], reps["widx"], reps["ckey"]])).fillna(False).to_numpy(bool)
    for name, m in (("ovn", ovn), ("late", late)):
        out[f"n_clu_r1_{name}"] = reps[m].groupby([reps["symbol"][m], reps["widx"][m]]).size().reindex(grid).fillna(0.0)
        out[f"n_first_r1_{name}"] = reps[m & is_first].groupby([reps["symbol"][m & is_first], reps["widx"][m & is_first]]).size().reindex(grid).fillna(0.0)
    b = out.groupby(level="symbol")["n_first_r1_ovn"].transform(lambda v: v.shift().rolling(NEWS_WINDOW, min_periods=NEWS_WINDOW).mean())
    out["n_first_r1_ovn_abn"] = np.log1p(out["n_first_r1_ovn"]) - np.log1p(b)
    out["ovn_share_r1"] = out["n_clu_r1_ovn"] / out["n_clu_r1"].where(out["n_clu_r1"] > 0)
    out["tone_mean_r1_ovn"] = reps[ovn].groupby([reps["symbol"][ovn], reps["widx"][ovn]])["tone"].mean().reindex(grid)
    out["n_scored_r1"] = out["n_scored_r1"].fillna(0.0)
    out["ev_any_r1"] = out[[f"ev_{e}_r1" for e in EVENT_RX]].max(axis=1)
    out["alias_missing"] = (~out.index.get_level_values("symbol").isin(list(res.get("aliases", {})))).astype(float)
    out = out.reset_index()
    out["target"] = tgt[out["widx"].to_numpy()]
    return out.drop(columns="widx")


# -----------------------------------------------------------------------------
# 기능  : 원천 표들로 (종목 × 대상일) feature 표를 만듦. 학습(전체 표)과 predict(하루치 표)가 이 함수를 같이 씀
# input : tables    {"daily", "price", "earnings"?, "analyst"?, "news"?}  (known_at < cutoff 로 이미 잘린 표)
#         targets   DataFrame[date, target]  기준일·대상일 쌍 (연속 거래일)
#         universe  그날 예측 대상 종목 목록. 종목 교차 feature 는 이 안에서만 계산
# output: DataFrame[symbol, date, target, close_d0, gap_fallback, gap_missing, <FEATURE_GROUPS 열>]
#         뉴스 표가 있으면 txt_* 묶음(_text_block)도 계산. titles_r1 은 문자열 열
# 수식  : gap_pct   = (대상일 08:00 pre 봉 종가 / 기준일 종가 − 1) × 100
#                     pre 봉이 없으면 기준일 post 마지막 종가로 대체(gap_fallback=1), 그것도 없으면 NaN(gap_missing=1)
#         gap_rank  = 그날 universe 안 gap_pct 순위 (0~1)
#         zgap      = clip(gap_pct / max(vol20, 0.5), ±10)
#         mkt_gap   = 자기 제외 universe gap_pct 중앙값,  resid_gap = gap_pct − beta60 × mkt_gap
#         reg_*     = 그날 universe 의 median vol20, median |gap|, std gap, median gap, mean 1[|gap| ≥ 1%]
#         an_gap_agree = sign(an_up − an_down) × sign(gap_pct)
# -----------------------------------------------------------------------------
def build_features(tables, targets, universe):
    uni = set(universe)
    targets = targets.assign(date=targets["date"].astype(NS), target=targets["target"].astype(NS))
    daily = tables["daily"]
    blk = _daily_block(daily[daily["symbol"].isin(uni)])
    x = targets[["date", "target"]].merge(blk, on="date")

    pre, post = _extended_hours(tables["price"][tables["price"]["symbol"].isin(uni)])
    pre_v = pd.Series(pre.reindex(pd.MultiIndex.from_arrays([x["symbol"], x["target"]])).to_numpy(), index=x.index)
    post_v = pd.Series(post.reindex(pd.MultiIndex.from_arrays([x["symbol"], x["date"]])).to_numpy(), index=x.index)
    px = pre_v.fillna(post_v)
    x["gap_fallback"] = (pre_v.isna() & post_v.notna()).astype(float)
    x["gap_missing"] = px.isna().astype(float)
    x["gap_pct"] = (px / x["close_d0"] - 1) * 100
    # 프리마켓 경로 (premkt): post_ret + pre_ret ≈ gap_pct (post 봉이 없으면 post_ret = NaN, pre_ret = gap)
    rng = _premarket_range(tables["price"][tables["price"]["symbol"].isin(uni)])
    x["pre_rng"] = rng.reindex(pd.MultiIndex.from_arrays([x["symbol"], x["target"]])).to_numpy()
    x["post_ret"] = (post_v / x["close_d0"] - 1) * 100
    x["pre_ret"] = (pre_v / post_v.fillna(x["close_d0"]) - 1) * 100
    x["abs_gap"] = x["gap_pct"].abs()

    x["gap_rank"] = x.groupby("target")["gap_pct"].rank(pct=True)
    x["zgap"] = (x["gap_pct"] / x["vol20"].clip(lower=VOL_FLOOR)).clip(-ZGAP_CLIP, ZGAP_CLIP)
    x["mkt_gap"] = loo_median(x["gap_pct"], x["target"])
    x["resid_gap"] = x["gap_pct"] - x["beta60"] * x["mkt_gap"]

    a = x.assign(_abs=x["gap_pct"].abs(), _big=(x["gap_pct"].abs() >= BIG_GAP).astype(float).where(x["gap_pct"].notna()))
    reg = a.groupby("target").agg(reg_med_vol20=("vol20", "median"), reg_med_absgap=("_abs", "median"),
                                  reg_gap_disp=("gap_pct", "std"), reg_mkt_gap=("gap_pct", "median"),
                                  reg_big_ratio=("_big", "mean"))
    x = x.merge(reg.reset_index(), on="target", how="left")

    key = ["symbol", "target"]
    ev, an = tables.get("earnings"), tables.get("analyst")
    if ev is not None and len(ev):
        x = x.merge(_earnings_block(ev[ev["symbol"].isin(uni)], targets), on=key, how="left")
    if an is not None and len(an):
        x = x.merge(_analyst_block(an[an["symbol"].isin(uni)], targets), on=key, how="left")
    if tables.get("news") is not None:
        x = x.merge(_news_block(tables["news"], targets, universe), on=key, how="left")
        x = x.merge(_text_block(tables["news"], targets, universe), on=key, how="left")
    for c in DEPLOYABLE - set(x.columns):
        x[c] = "" if c in TEXT_COLS else np.nan
    for c in TEXT_COLS - {"doc_r1"}:
        x[c] = x[c].fillna("")
    x["doc_r1"] = (x["titles_r1"] + " || " + x["lead_r1"]).where(x["lead_r1"].str.len() > 0, x["titles_r1"])
    for c in ("earn", "an_n", "an_up", "an_down"):
        x[c] = x[c].fillna(0.0)
    x["an_gap_agree"] = (np.sign(x["an_up"] - x["an_down"]) * np.sign(x["gap_pct"])).fillna(0.0)

    cols = KEY_COLS + ["close_d0", "gap_fallback", "gap_missing"] + [c for g in FEATURE_GROUPS.values() for c in g]
    return x[cols].sort_values(["target", "symbol"]).reset_index(drop=True)


# -----------------------------------------------------------------------------
# 기능  : Day 하나에서 build_features 에 넣을 원천 표를 day API 로 꺼냄 (cutoff 는 API 가 보장)
# input : day     src.data.Day
#         groups  필요한 feature 묶음 이름 집합 (필요한 표만 읽어 추론 시간을 줄임)
# output: (tables dict, targets DataFrame)
# -----------------------------------------------------------------------------
def day_tables(day, groups):
    need = set().union(*(GROUP_TABLES[g] for g in groups)) | {"daily", "price"}
    need_body = bool({"txt_body", "txt_lead"} & set(groups))       # 본문은 무거워 필요할 때만 읽음
    t = {"daily": day.daily(days=HIST_DAYS, columns=["symbol", "date_et", "high", "low", "close", "prev_close", "ret"]),
         "price": day.price(since=day.date, columns=["symbol", "datetime", "close", "session"])}
    since = day.date + CUTOFF
    if "earnings" in need:
        t["earnings"] = day.earnings(since=since)
    if "analyst" in need:
        t["analyst"] = day.analyst(since=since)
    if "news" in need:
        cols = ["symbols", "source", "title", "tone"] + (["text"] if need_body else [])
        t["news"] = day.news(days=NEWS_HIST_DAYS, columns=cols)
    dates = sorted(pd.to_datetime(t["daily"]["date_et"]).dt.normalize().unique())
    dates = [d for d in dates if d <= day.date]
    targets = pd.DataFrame({"date": dates, "target": dates[1:] + [day.target]})
    return t, targets


# -----------------------------------------------------------------------------
# 기능  : Day 하나의 feature 표 (그날 day.symbols 를 universe 로 씀)
# input : day, groups
# output: build_features 결과 중 대상일 = day.target 인 행
# -----------------------------------------------------------------------------
def day_features(day, groups):
    tables, targets = day_tables(day, groups)
    x = build_features(tables, targets, day.symbols)
    return x[x["target"] == pd.Timestamp(day.target)].reset_index(drop=True)


# =============================================================================
# 3. 점수
# =============================================================================

# -----------------------------------------------------------------------------
# 기능  : 과제 score 의 빠른 버전 (src.data.score 와 같은 값, 결정 규칙 탐색용)
# 수식  : score = 1 − Σ w·O / Σ w·E,  E_ij = n_i · m_j / N,  w_ij = (i−j)² (i−2)²
# input : y, p  정답·예측 label 배열 (0~4 정수)
# output: float (분모가 0 이면 NaN)
# -----------------------------------------------------------------------------
def fast_score(y, p):
    O = np.bincount(np.asarray(y, int) * 5 + np.asarray(p, int), minlength=25).reshape(5, 5).astype(float)
    n = O.sum()
    if n == 0:
        return float("nan")
    den = (WEIGHT * np.outer(O.sum(1), O.sum(0)) / n).sum()
    return float(1 - (WEIGHT * O).sum() / den) if den > 0 else float("nan")


# -----------------------------------------------------------------------------
# 기능  : 결정 규칙·regime 을 고를 때의 목적 함수 (Domain-robust 선택 원칙)
# 수식  : pooled   score(tune 전체)
#         mean     mean_d score_d          (d = tune 구간을 시간순으로 n_chunks 개로 나눈 조각)
#         mean_sd  mean_d score_d − λ · sd_d score_d
#         min      min_d score_d
# input : spec  {"criterion", "lam", "n_chunks"},  dates  tune 행의 대상일
# output: 호출 가능 객체  obj(y, p) → float (클수록 좋음)
# -----------------------------------------------------------------------------
class Objective:
    def __init__(self, spec, dates):
        spec = spec or {}
        self.criterion = spec.get("criterion", "pooled")
        self.lam = float(spec.get("lam", 1.0))
        n_chunks = int(spec.get("n_chunks", 3))
        u = np.sort(pd.unique(pd.Series(dates).to_numpy(NS)))
        edges = u[np.linspace(0, len(u), n_chunks + 1).astype(int)[1:-1]] if len(u) else []
        self.chunk = np.searchsorted(np.asarray(edges, dtype=NS), pd.Series(dates).to_numpy(NS), side="right")

    def __call__(self, y, p):
        if self.criterion == "pooled":
            return fast_score(y, p)
        s = np.array([fast_score(y[self.chunk == c], p[self.chunk == c]) for c in np.unique(self.chunk)])
        if self.criterion == "mean":
            return float(np.nanmean(s))
        if self.criterion == "mean_sd":
            return float(np.nanmean(s) - self.lam * np.nanstd(s))
        if self.criterion == "min":
            return float(np.nanmin(s))
        raise ValueError(f"criterion 은 pooled / mean / mean_sd / min 중 하나: {self.criterion}")


# =============================================================================
# 4. 신호 head
#    공통 인터페이스  fit(X, y_ret, y_label, seed) → self,  predict(X) → 신호
#    learnable=False 인 신호는 학습이 없어 tune 구간 전체로 결정 규칙만 고름
# =============================================================================

def _hgb_params(params, seed):
    p = dict(max_iter=150, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40,
             l2_regularization=1.0, early_stopping=False, random_state=seed)
    p.update(params or {})
    return p


# -----------------------------------------------------------------------------
# 기능  : 학습 없는 신호. 열 하나를 그대로 신호로 씀 (기본 gap_pct)
# output: s = X[col]  (%, NaN 이면 결정 규칙에서 보합)
# -----------------------------------------------------------------------------
class ColumnSignal:
    learnable = False
    kind = "scalar"

    def __init__(self, features=None, params=None):
        self.col = (params or {}).get("col", "gap_pct")
        self.features = [self.col]

    def fit(self, X, y_ret, y_label, seed=0):
        return self

    def predict(self, X):
        return X[self.col].to_numpy(float)

    def state(self):
        return {}

    def load(self, st):
        return self


# -----------------------------------------------------------------------------
# 기능  : 당일 수익률 회귀 (B3)
# input : X[features],  목표 y = clip(ret_pct, ±15)
# loss  : absolute_error (중앙값 회귀, 급등락 이상치에 강건)
# output: s = ŷ (%)
# -----------------------------------------------------------------------------
class RegressSignal:
    learnable = True
    kind = "scalar"

    def __init__(self, features, params=None):
        self.features, self.params, self.model = list(features), dict(params or {}), None

    def fit(self, X, y_ret, y_label, seed=0):
        from sklearn.ensemble import HistGradientBoostingRegressor
        self.model = HistGradientBoostingRegressor(loss="absolute_error", **_hgb_params(self.params, seed))
        self.model.fit(X[self.features], np.clip(y_ret, -RET_CLIP, RET_CLIP))
        return self

    def predict(self, X):
        return self.model.predict(X[self.features])

    def state(self):
        return {"model": self.model}

    def load(self, st):
        self.model = st["model"]
        return self


# -----------------------------------------------------------------------------
# 기능  : 5-class 직접 분류 (B4 / D0)
# input : X[features],  목표 label 0~4
# loss  : multinomial log loss
# output: s = Σ_c P(c) · center_c   (center = −4, −1.5, 0, 1.5, 4 %) → 결정 규칙이 score 기준으로 경계를 고름
# -----------------------------------------------------------------------------
class Classify5Signal:
    learnable = True
    kind = "scalar"

    def __init__(self, features, params=None):
        self.features, self.params, self.model = list(features), dict(params or {}), None

    def fit(self, X, y_ret, y_label, seed=0):
        from sklearn.ensemble import HistGradientBoostingClassifier
        self.model = HistGradientBoostingClassifier(**_hgb_params(self.params, seed))
        self.model.fit(X[self.features], y_label)
        return self

    def proba(self, X):
        P = np.zeros((len(X), 5))
        P[:, self.model.classes_.astype(int)] = self.model.predict_proba(X[self.features])
        return P

    def predict(self, X):
        return self.proba(X) @ CLASS_CENTERS

    def state(self):
        return {"model": self.model}

    def load(self, st):
        self.model = st["model"]
        return self


# -----------------------------------------------------------------------------
# 기능  : Direction + Magnitude 분해 (D1 / D2)
# 정의  : Move    = 1[|r| ≥ 0.6%]                 (전체 표본으로 학습)
#         Extreme = 1[|r| ≥ 2.5%]                 (Move 표본만으로 학습)
#         Up      = 1[r > 0]                      (Move 표본만으로 학습, 보합에는 방향을 강제하지 않음)
#         magnitude="ordinal"    Move head + Extreme head (이진 2개)  → D2
#         magnitude="multiclass" flat/normal/extreme 3-class 하나     → D1
# loss  : 각 head 의 log loss
# output: (n, 3) 배열 [P(move), P(extreme | move), P(up | move)]
# -----------------------------------------------------------------------------
class DecompSignal:
    learnable = True
    kind = "decomp"

    def __init__(self, features, params=None):
        params = dict(params or {})
        self.magnitude = params.pop("magnitude", "ordinal")
        if self.magnitude not in ("ordinal", "multiclass"):
            raise ValueError(f"magnitude 는 ordinal / multiclass 중 하나: {self.magnitude}")
        self.features, self.params, self.heads = list(features), params, {}

    def fit(self, X, y_ret, y_label, seed=0):
        from sklearn.ensemble import HistGradientBoostingClassifier
        mk = lambda: HistGradientBoostingClassifier(**_hgb_params(self.params, seed))
        y = np.asarray(y_label, int)
        move = y != FLAT
        ext = np.isin(y, (0, 4))
        Xf = X[self.features]
        if self.magnitude == "ordinal":
            self.heads["move"] = mk().fit(Xf, move.astype(int))
            self.heads["ext"] = mk().fit(Xf[move], ext[move].astype(int))
        else:
            self.heads["mag3"] = mk().fit(Xf, np.where(ext, 2, np.where(move, 1, 0)))
        self.heads["up"] = mk().fit(Xf[move], (np.asarray(y_ret)[move] > 0).astype(int))
        return self

    @staticmethod
    def _p(model, X, cls):
        P = model.predict_proba(X)
        idx = list(model.classes_).index(cls) if cls in model.classes_ else None
        return P[:, idx] if idx is not None else np.zeros(len(X))

    def predict(self, X):
        Xf = X[self.features]
        if self.magnitude == "ordinal":
            p_move, p_ext = self._p(self.heads["move"], Xf, 1), self._p(self.heads["ext"], Xf, 1)
        else:
            p0, p1, p2 = (self._p(self.heads["mag3"], Xf, c) for c in (0, 1, 2))
            p_move = 1 - p0
            p_ext = np.divide(p2, p1 + p2, out=np.zeros(len(Xf)), where=(p1 + p2) > 0)
        return np.column_stack([p_move, p_ext, self._p(self.heads["up"], Xf, 1)])

    def state(self):
        return {"heads": self.heads, "magnitude": self.magnitude}

    def load(self, st):
        self.heads, self.magnitude = st["heads"], st["magnitude"]
        return self


def _xgb_params(params, seed):
    p = dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
             min_child_weight=20, reg_lambda=1.0, tree_method="hist", n_jobs=4, random_state=seed)
    p.update(params or {})
    return p


# -----------------------------------------------------------------------------
# 기능  : XGBoost 회귀 신호 (M1). RegressSignal 의 HGB 를 XGBoost 로 바꾼 것
# input : X[features] (NaN 은 XGBoost 가 결측 방향을 학습),  목표 y = clip(ret_pct, ±15)
# loss  : reg:absoluteerror  L = |y − ŷ|  (중앙값 회귀)
# output: s = ŷ (%)
# -----------------------------------------------------------------------------
class XGBRegSignal(RegressSignal):
    def fit(self, X, y_ret, y_label, seed=0):
        from xgboost import XGBRegressor
        self.model = XGBRegressor(objective="reg:absoluteerror", **_xgb_params(self.params, seed))
        self.model.fit(X[self.features], np.clip(y_ret, -RET_CLIP, RET_CLIP))
        return self


# -----------------------------------------------------------------------------
# 기능  : XGBoost 5-class 분류 신호 (M1). Classify5Signal 의 HGB 를 XGBoost 로 바꾼 것
# input : X[features],  목표 label 0~4 (학습 구간에 없는 class 는 확률 0)
# loss  : multi:softprob  L = −log P(y)
# output: s = Σ_c P(c) · center_c
# -----------------------------------------------------------------------------
class XGBCls5Signal(Classify5Signal):
    def fit(self, X, y_ret, y_label, seed=0):
        from xgboost import XGBClassifier
        y = np.asarray(y_label, int)
        self.classes = np.unique(y)
        self.model = XGBClassifier(objective="multi:softprob", **_xgb_params(self.params, seed))
        self.model.fit(X[self.features], np.searchsorted(self.classes, y))
        return self

    def proba(self, X):
        P = np.zeros((len(X), 5))
        P[:, self.classes] = self.model.predict_proba(X[self.features])
        return P

    def state(self):
        return {"model": self.model, "classes": self.classes}

    def load(self, st):
        self.model, self.classes = st["model"], np.asarray(st["classes"], int)
        return self


# -----------------------------------------------------------------------------
# 기능  : LSTM 회귀 신호 (M1). 직전 SEQ_LEN 거래일의 (수익률, 일중 범위) 시퀀스 + 정적 feature
# input : 시퀀스 x_t = [ret_l{L+1−t}, rng_l{L+1−t}],  t = 1..L (과거 → 기준일 순서)
#         정적 z = features 중 seq 가 아닌 열 (예: gap_pct, gap_rank)
#         표준화 (x − μ)/σ 는 fit 구간 값으로 계산해 저장, 결측은 표준화 뒤 0
# 구조  : h_L = LSTM(x_1..x_L)  (hidden H)
#         ŷ   = W2 · ReLU(W1 · [h_L, z] + b1) + b2
# loss  : Huber(δ = 1) on y = clip(ret_pct, ±15)
# 학습  : Adam(lr, weight_decay), 고정 epoch, 배치 섞기는 seed 고정 generator
#         torch 는 학습에만 쓰고 predict 는 저장한 가중치로 numpy 순전파 (채점 환경에 torch 가 없어도 동작)
# output: s = ŷ (%)
# -----------------------------------------------------------------------------
class LSTMSignal:
    learnable = True
    kind = "scalar"
    DEFAULTS = dict(hidden=32, dense=32, epochs=15, batch=512, lr=1e-3, weight_decay=1e-4, huber=1.0, threads=4)

    def __init__(self, features, params=None):
        self.features = list(features)
        self.params = {**self.DEFAULTS, **(params or {})}
        seq = set(FEATURE_GROUPS["seq"])
        self.static = [c for c in self.features if c not in seq]
        self.use_seq = seq <= set(self.features)
        self.w = None

    def _arrays(self, X):
        n = len(X)
        if self.use_seq:
            idx = range(SEQ_LEN, 0, -1)                 # 과거 → 기준일
            S = np.stack([X[[f"ret_l{i}" for i in idx]].to_numpy(float),
                          X[[f"rng_l{i}" for i in idx]].to_numpy(float)], axis=2)
        else:
            S = np.zeros((n, 1, 2))
        Z = X[self.static].to_numpy(float) if self.static else np.zeros((n, 0))
        return S, Z

    def _norm(self, S, Z):
        return np.nan_to_num((S - self.mu_s) / self.sd_s), np.nan_to_num((Z - self.mu_z) / self.sd_z)

    def fit(self, X, y_ret, y_label, seed=0):
        import torch
        from torch import nn
        P = self.params
        S, Z = self._arrays(X)
        self.mu_s, self.sd_s = np.nanmean(S, axis=(0, 1)), np.nanstd(S, axis=(0, 1)) + 1e-6
        self.mu_z, self.sd_z = np.nanmean(Z, axis=0), np.nanstd(Z, axis=0) + 1e-6
        S, Z = self._norm(S, Z)
        y = np.clip(np.asarray(y_ret, float), -RET_CLIP, RET_CLIP)

        torch.manual_seed(seed)
        torch.set_num_threads(int(P["threads"]))
        torch.use_deterministic_algorithms(True)
        H, D = int(P["hidden"]), int(P["dense"])
        rnn = self._cell(nn, H)
        head = nn.Sequential(nn.Linear(H + Z.shape[1], D), nn.ReLU(), nn.Linear(D, 1))
        opt = torch.optim.Adam(list(rnn.parameters()) + list(head.parameters()),
                               lr=float(P["lr"]), weight_decay=float(P["weight_decay"]))
        loss_fn = nn.HuberLoss(delta=float(P["huber"]))
        tS, tZ, ty = (torch.tensor(a, dtype=torch.float32) for a in (S, Z, y))
        g = torch.Generator().manual_seed(seed)
        B = int(P["batch"])
        for _ in range(int(P["epochs"])):
            perm = torch.randperm(len(ty), generator=g)
            for i in range(0, len(ty), B):
                j = perm[i:i + B]
                _, h = rnn(tS[j])
                h = h[0] if isinstance(h, tuple) else h        # LSTM 은 (h, c), GRU 는 h
                loss = loss_fn(head(torch.cat([h[-1], tZ[j]], dim=1)).squeeze(1), ty[j])
                opt.zero_grad()
                loss.backward()
                opt.step()
        sd = {k: v.detach().numpy().copy() for k, v in rnn.state_dict().items()}
        hd = {k: v.detach().numpy().copy() for k, v in head.state_dict().items()}
        self.w = {**self._cell_weights(sd),
                  "W1": hd["0.weight"], "b1": hd["0.bias"], "W2": hd["2.weight"], "b2": hd["2.bias"]}
        return self

    def _cell(self, nn, H):
        return nn.LSTM(2, H, batch_first=True)

    def _cell_weights(self, sd):
        return {"W_ih": sd["weight_ih_l0"], "W_hh": sd["weight_hh_l0"], "b": sd["bias_ih_l0"] + sd["bias_hh_l0"]}

    # -------------------------------------------------------------------------
    # 기능  : numpy LSTM 순전파 (torch nn.LSTM 과 같은 식, 게이트 순서 i, f, g, o)
    # 수식  : i,f,o = σ(W_ih x_t + W_hh h_{t−1} + b),  g = tanh(·)
    #         c_t = f⊙c_{t−1} + i⊙g,  h_t = o⊙tanh(c_t)
    # output: h_L  (n, H)
    # -------------------------------------------------------------------------
    def _encode(self, S):
        w = self.w
        H = w["W_hh"].shape[1]
        h, c = np.zeros((len(S), H)), np.zeros((len(S), H))
        for t in range(S.shape[1]):
            a = S[:, t] @ w["W_ih"].T + h @ w["W_hh"].T + w["b"]
            i, f, g, o = _sigmoid(a[:, :H]), _sigmoid(a[:, H:2 * H]), np.tanh(a[:, 2 * H:3 * H]), _sigmoid(a[:, 3 * H:])
            c = f * c + i * g
            h = o * np.tanh(c)
        return h

    def predict(self, X):
        S, Z = self._norm(*self._arrays(X))
        w = self.w
        z = np.maximum(np.concatenate([self._encode(S), Z], axis=1) @ w["W1"].T + w["b1"], 0)
        return (z @ w["W2"].T + w["b2"])[:, 0]

    def state(self):
        return {"w": self.w, "mu_s": self.mu_s, "sd_s": self.sd_s, "mu_z": self.mu_z, "sd_z": self.sd_z}

    def load(self, st):
        self.w, self.mu_s, self.sd_s, self.mu_z, self.sd_z = (st[k] for k in ("w", "mu_s", "sd_s", "mu_z", "sd_z"))
        return self


def _sigmoid(a):
    return 1 / (1 + np.exp(-a))


# -----------------------------------------------------------------------------
# 기능  : GRU 회귀 신호 (M1). LSTMSignal 과 입력·head·loss·학습이 같고 순환 셀만 GRU
#         (게이트 2개 + cell state 없음 → 파라미터 약 3/4, 짧은 시퀀스·작은 표본에서 덜 과적합)
# 수식  : r = σ(W_ir x_t + b_ir + W_hr h_{t−1} + b_hr),  u = σ(W_iz x_t + b_iz + W_hz h_{t−1} + b_hz)
#         n = tanh(W_in x_t + b_in + r⊙(W_hn h_{t−1} + b_hn)),  h_t = (1−u)⊙n + u⊙h_{t−1}
# output: s = ŷ (%)
# -----------------------------------------------------------------------------
class GRUSignal(LSTMSignal):
    def _cell(self, nn, H):
        return nn.GRU(2, H, batch_first=True)

    def _cell_weights(self, sd):
        return {"W_ih": sd["weight_ih_l0"], "W_hh": sd["weight_hh_l0"], "b_ih": sd["bias_ih_l0"], "b_hh": sd["bias_hh_l0"]}

    def _encode(self, S):
        w = self.w
        H = w["W_hh"].shape[1]
        h = np.zeros((len(S), H))
        for t in range(S.shape[1]):
            gi = S[:, t] @ w["W_ih"].T + w["b_ih"]
            gh = h @ w["W_hh"].T + w["b_hh"]
            r, u = _sigmoid(gi[:, :H] + gh[:, :H]), _sigmoid(gi[:, H:2 * H] + gh[:, H:2 * H])
            n = np.tanh(gi[:, 2 * H:] + r * gh[:, 2 * H:])
            h = (1 - u) * n + u * h
        return h


# -----------------------------------------------------------------------------
# 기능  : 실적일 서프라이즈 방향 보정 신호 (M5). 갭에 "실적 발표 후 같은 방향으로 더 가는 몫(PEAD)" 을 더함
# 수식  : f = surp_sign (mode="sign")  또는  clip(surp_pct_w, ±c)/c (mode="pct"),  실적일이 아니면 f = 0
#         β  = OLS 기울기:  rest = ret_pct − gap_pct  ~  a + β·f    (fit 구간의 실적일 행만)
#         s  = gap_pct + β·f                                         (실적일 아닌 행은 gap 그대로)
# loss  : 제곱오차 (β 추정),  k 는 tune 구간 score 최대 (결정 규칙)
# input : X[gap_pct, earn, surp_sign, surp_pct_w],  params {mode, clip, shrink (β 에 곱하는 수축, 기본 1)}
# output: s (%)
# -----------------------------------------------------------------------------
class EarnSurpSignal:
    learnable = True
    kind = "scalar"

    def __init__(self, features=None, params=None):
        p = dict(params or {})
        self.mode, self.clip, self.shrink = p.get("mode", "sign"), float(p.get("clip", 20.0)), float(p.get("shrink", 1.0))
        if self.mode not in ("sign", "pct"):
            raise ValueError(f"mode 는 sign / pct 중 하나: {self.mode}")
        self.features = ["gap_pct", "earn", "surp_sign", "surp_pct_w"]
        self.beta, self.n_earn = 0.0, 0

    def _f(self, X):
        f = X["surp_sign"] if self.mode == "sign" else X["surp_pct_w"].clip(-self.clip, self.clip) / self.clip
        return np.where(X["earn"].to_numpy(float) > 0, np.nan_to_num(f.to_numpy(float)), 0.0)

    def fit(self, X, y_ret, y_label, seed=0):
        f = self._f(X)
        m = (X["earn"].to_numpy(float) > 0) & np.isfinite(X["gap_pct"].to_numpy(float))
        rest = np.asarray(y_ret, float)[m] - X["gap_pct"].to_numpy(float)[m]
        self.n_earn = int(m.sum())
        if self.n_earn >= 10 and np.var(f[m]) > 0:
            self.beta = float(np.polyfit(f[m], np.clip(rest, -RET_CLIP, RET_CLIP), 1)[0]) * self.shrink
        return self

    def predict(self, X):
        return X["gap_pct"].to_numpy(float) + self.beta * self._f(X)

    def state(self):
        return {"beta": self.beta, "n_earn": self.n_earn, "mode": self.mode, "clip": self.clip, "shrink": self.shrink}

    def load(self, st):
        self.beta, self.n_earn, self.mode, self.clip, self.shrink = (st[k] for k in ("beta", "n_earn", "mode", "clip", "shrink"))
        return self


SIGNALS = {"gap": ColumnSignal, "column": ColumnSignal, "regress": RegressSignal,
           "classify5": Classify5Signal, "decomp": DecompSignal,
           "xgb_reg": XGBRegSignal, "xgb_cls5": XGBCls5Signal, "lstm": LSTMSignal, "gru": GRUSignal,
           "earn_surp": EarnSurpSignal}


# =============================================================================
# 5. 결정 규칙
#    스칼라 신호용 규칙은 모두 경계 4개 b0 < b1 < b2 < b3 로 바뀜
#        label = #{b_k ≤ s}   (s = b 이면 위 구간. label_of 의 왼쪽 포함과 같음),  s = NaN 이면 보합
#    공통 인터페이스  fit(s, y, obj) → self,  predict(s) → label,  params() → dict
# =============================================================================

def _bounds_predict(s, b):
    s = np.asarray(s, float)
    out = np.searchsorted(np.asarray(b, float), np.nan_to_num(s, nan=0.0), side="right")
    out[np.isnan(s)] = FLAT
    return out.astype(int)


class _BoundsRule:
    """스칼라 신호 규칙의 공통 부분. 하위 클래스는 _search 만 구현함."""

    def __init__(self, params=None):
        self.cfg, self.b = dict(params or {}), None

    def fit(self, s, y, obj):
        self.b = np.asarray(self._search(np.asarray(s, float), np.asarray(y, int), obj), float)
        return self

    def predict(self, s):
        return _bounds_predict(s, self.b)

    def params(self):
        return {"bounds": [float(v) for v in self.b]}

    def state(self):
        return {"b": self.b, "cfg": self.cfg}

    def load(self, st):
        self.b, self.cfg = np.asarray(st["b"], float), st["cfg"]
        return self


def _k_grid(cfg):
    lo, hi, step = cfg.get("k_grid", (0.5, 8.0, 0.25))
    return np.round(np.arange(lo, hi + 1e-9, step), 4)


def _k_bounds(k_up, k_dn):
    return [-2.5 / k_dn, -0.6 / k_dn, 0.6 / k_up, 2.5 / k_up]


# -----------------------------------------------------------------------------
# 기능  : 전부 보합 (B0)
# output: label = 2
# -----------------------------------------------------------------------------
class FlatRule(_BoundsRule):
    def _search(self, s, y, obj):
        return [-np.inf, -np.inf, np.inf, np.inf]


# -----------------------------------------------------------------------------
# 기능  : 과제 구간 그대로 (B1)
# 수식  : label = label_of(s)  → 경계 (−2.5, −0.6, 0.6, 2.5)
# -----------------------------------------------------------------------------
class IdentityRule(_BoundsRule):
    def _search(self, s, y, obj):
        return [-2.5, -0.6, 0.6, 2.5]


# -----------------------------------------------------------------------------
# 기능  : 배율 1개 (B2)
# 수식  : label = label_of(k × s)  → 경계 (−2.5/k, −0.6/k, 0.6/k, 2.5/k)
# param : k_grid = (시작, 끝, 간격)  기본 (0.5, 8.0, 0.25)
# -----------------------------------------------------------------------------
class KRule(_BoundsRule):
    def _search(self, s, y, obj):
        best = max(_k_grid(self.cfg), key=lambda k: obj(y, _bounds_predict(s, _k_bounds(k, k))))
        self.k = float(best)
        return _k_bounds(best, best)

    def params(self):
        return {**super().params(), "k": self.k}

    def state(self):
        return {**super().state(), "k": self.k}

    def load(self, st):
        self.k = st["k"]
        return super().load(st)


# -----------------------------------------------------------------------------
# 기능  : 대칭 경계 2개 (B2-Sym)
# 수식  : |s| < a → 2,  a ≤ |s| < b → 1/3,  |s| ≥ b → 0/4    → 경계 (−b, −a, a, b)
# param : q_grid  a, b 후보로 쓸 |s| 의 분위수 목록 (scale 에 무관하게 탐색)
# -----------------------------------------------------------------------------
class Sym2Rule(_BoundsRule):
    def _search(self, s, y, obj):
        q = self.cfg.get("q_grid", list(np.round(np.arange(0.05, 0.99, 0.03), 3)))
        cand = np.unique(np.nanquantile(np.abs(s), q))
        best, arg = -np.inf, (0.6, 2.5)
        for i, a in enumerate(cand):
            for b in cand[i + 1:]:
                v = obj(y, _bounds_predict(s, [-b, -a, a, b]))
                if v > best:
                    best, arg = v, (a, b)
        return [-arg[1], -arg[0], arg[0], arg[1]]


# -----------------------------------------------------------------------------
# 기능  : 상승·하락 배율 분리 (B2-Asym)
# 수식  : s ≥ 0 이면 label_of(k_up × s),  s < 0 이면 label_of(k_dn × s)
# -----------------------------------------------------------------------------
class AsymKRule(_BoundsRule):
    def _search(self, s, y, obj):
        g = _k_grid(self.cfg)
        best, arg = -np.inf, (1.0, 1.0)
        for ku in g:
            for kd in g:
                v = obj(y, _bounds_predict(s, _k_bounds(ku, kd)))
                if v > best:
                    best, arg = v, (ku, kd)
        self.k_up, self.k_dn = map(float, arg)
        return _k_bounds(*arg)

    def params(self):
        return {**super().params(), "k_up": self.k_up, "k_dn": self.k_dn}

    def state(self):
        return {**super().state(), "k_up": self.k_up, "k_dn": self.k_dn}

    def load(self, st):
        self.k_up, self.k_dn = st["k_up"], st["k_dn"]
        return super().load(st)


# -----------------------------------------------------------------------------
# 기능  : 경계 4개 직접 최적화 (B2-4T, D0)
# 수식  : label = #{b_k ≤ s},  b0 < b1 < b2 < b3
# 방법  : AsymK 해에서 출발해, 경계 하나씩 s 의 분위수 후보 위에서 바꾸는 좌표 하강을 n_pass 번
# param : n_q (후보 분위수 개수, 기본 80),  n_pass (기본 3)
# -----------------------------------------------------------------------------
class FourTRule(_BoundsRule):
    def _search(self, s, y, obj):
        b = list(AsymKRule(self.cfg)._search(s, y, obj))
        cand = np.unique(np.nanquantile(s, np.linspace(0.005, 0.995, int(self.cfg.get("n_q", 80)))))
        for _ in range(int(self.cfg.get("n_pass", 3))):
            for k in range(4):
                lo = b[k - 1] if k else -np.inf
                hi = b[k + 1] if k < 3 else np.inf
                opts = [c for c in cand if lo < c < hi] + [b[k]]
                b[k] = max(opts, key=lambda c: obj(y, _bounds_predict(s, b[:k] + [c] + b[k + 1:])))
        return b


# -----------------------------------------------------------------------------
# 기능  : 분해 구조의 결정 규칙 (D1 / D2 / D2-C)
# 수식  : P(move) < t_move                      → 2
#         P(ext|move) ≥ t_ext                   → 0 / 4 (방향 = 1[P(up) ≥ 0.5])
#         그 외                                  → 1 / 3
#         conf_gate=True 이면 급등락이라도 |P(up) − 0.5| < c 일 때 1 / 3 으로 낮춤 (D2-C)
# param : q_grid (t_move, t_ext 후보 분위수),  c_grid (conf 후보),  conf_gate
# 입력  : s = DecompSignal.predict() 의 (n, 3) 배열
# -----------------------------------------------------------------------------
class DecompRule:
    def __init__(self, params=None):
        self.cfg = dict(params or {})
        self.t_move, self.t_ext, self.c = 0.5, 0.5, 0.0

    @staticmethod
    def _apply(s, t_move, t_ext, c):
        p_move, p_ext, p_up = s[:, 0], s[:, 1], s[:, 2]
        up = p_up >= 0.5
        ext = (p_ext >= t_ext) & (np.abs(p_up - 0.5) >= c)
        lab = np.where(up, np.where(ext, 4, 3), np.where(ext, 0, 1))
        return np.where(p_move < t_move, FLAT, lab).astype(int)

    def fit(self, s, y, obj):
        q = self.cfg.get("q_grid", list(np.round(np.arange(0.05, 0.96, 0.05), 2)))
        tm = np.unique(np.quantile(s[:, 0], q))
        te = np.unique(np.quantile(s[:, 1], q))
        cs = self.cfg.get("c_grid", [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]) if self.cfg.get("conf_gate") else [0.0]
        best = -np.inf
        for a in tm:
            for b in te:
                for c in cs:
                    v = obj(y, self._apply(s, a, b, c))
                    if v > best:
                        best, (self.t_move, self.t_ext, self.c) = v, (float(a), float(b), float(c))
        return self

    def predict(self, s):
        return self._apply(np.asarray(s, float), self.t_move, self.t_ext, self.c)

    def params(self):
        return {"t_move": self.t_move, "t_ext": self.t_ext, "c": self.c}

    def state(self):
        return {"cfg": self.cfg, **self.params()}

    def load(self, st):
        self.cfg, self.t_move, self.t_ext, self.c = st["cfg"], st["t_move"], st["t_ext"], st["c"]
        return self


DECISIONS = {"flat": FlatRule, "identity": IdentityRule, "k": KRule, "sym2": Sym2Rule,
             "asym": AsymKRule, "4t": FourTRule, "decomp": DecompRule}


# =============================================================================
# 6. regime 보정 (Test-Time Regime Calibration)
#    스칼라 신호의 크기만 바꿈:  s' = s × f(r)   → 부호(방향)는 그대로, 크기 판단만 regime 에 따라 바뀜
#    r 은 그날 universe 로 계산한 reg_* 열. 정답은 쓰지 않음 (학습 구간에서 f 만 정함)
#    공통 인터페이스  fit(s, r, y, decision, obj) → self,  apply(s, r) → s'
# =============================================================================

class NoRegime:
    """T0. f = 1"""
    learnable = False
    inputs = []

    def __init__(self, params=None, extra_groups=None):
        self.var = None

    def values(self, X):
        return None

    def fit(self, s, r, y, decision, obj):
        return self

    def apply(self, s, r):
        return s

    def params(self):
        return {}

    def state(self):
        return {}

    def load(self, st):
        return self


# -----------------------------------------------------------------------------
# 기능  : T1. regime 변수 삼분위별 배율
# 수식  : f = f_low (r < e1), 1 (e1 ≤ r < e2), f_high (r ≥ e2),  e1, e2 = tune 구간 날짜별 r 의 1/3, 2/3 분위
# param : var (regime 열 이름, 기본 reg_med_vol20),  f_grid (기본 0.6 ~ 1.6, 0.1 간격)
# -----------------------------------------------------------------------------
class TercileRegime:
    learnable = False

    def __init__(self, params=None, extra_groups=None):
        p = dict(params or {})
        self.var = p.get("var", "reg_med_vol20")
        self.grid = np.round(np.arange(*p.get("f_grid", (0.6, 1.61, 0.1))), 3)
        self.edges, self.f = None, (1.0, 1.0, 1.0)

    def _factor(self, r):
        t = np.digitize(np.nan_to_num(r, nan=np.nanmedian(r) if np.isfinite(r).any() else 0.0), self.edges)
        return np.asarray(self.f)[t]

    def fit(self, s, r, y, decision, obj):
        self.edges = np.nanquantile(np.unique(r[np.isfinite(r)]), [1 / 3, 2 / 3])
        best, keep = -np.inf, self.f
        for lo in self.grid:
            for hi in self.grid:
                self.f = (lo, 1.0, hi)
                v = obj(y, decision.predict(self.apply(s, r)))
                if v > best:
                    best, keep = v, self.f
        self.f = tuple(float(v) for v in keep)
        return self

    def apply(self, s, r):
        return s * self._factor(r)

    @property
    def inputs(self):
        return [self.var]

    def values(self, X):
        return X[self.var].to_numpy(float)

    def params(self):
        return {"var": self.var, "edges": [float(e) for e in self.edges], "f": list(self.f)}

    def state(self):
        return {"var": self.var, "edges": self.edges, "f": self.f, "grid": self.grid}

    def load(self, st):
        self.var, self.edges, self.f, self.grid = st["var"], st["edges"], st["f"], st["grid"]
        return self


# -----------------------------------------------------------------------------
# 기능  : T2. regime 값에 따른 연속 배율
# 수식  : f = clip(exp(α · (r − med) / iqr), 0.5, 2),   med·iqr 는 tune 구간 날짜별 r 로 계산
# param : var,  a_grid (기본 −1.0 ~ 1.0, 0.1 간격)
# -----------------------------------------------------------------------------
class LinearRegime:
    learnable = False

    def __init__(self, params=None, extra_groups=None):
        p = dict(params or {})
        self.var = p.get("var", "reg_med_vol20")
        self.grid = np.round(np.arange(*p.get("a_grid", (-1.0, 1.01, 0.1))), 3)
        self.med, self.iqr, self.alpha = 0.0, 1.0, 0.0

    def apply(self, s, r):
        z = np.nan_to_num((r - self.med) / self.iqr, nan=0.0)
        return s * np.clip(np.exp(self.alpha * z), 0.5, 2.0)

    def fit(self, s, r, y, decision, obj):
        u = np.unique(r[np.isfinite(r)])
        self.med = float(np.median(u))
        q1, q3 = np.quantile(u, [0.25, 0.75])
        self.iqr = float(q3 - q1) or 1.0
        self.alpha = float(max(self.grid, key=lambda a: obj(y, decision.predict(self._with(a).apply(s, r)))))
        return self

    @property
    def inputs(self):
        return [self.var]

    def values(self, X):
        return X[self.var].to_numpy(float)

    def _with(self, a):
        c = LinearRegime({"var": self.var})
        c.med, c.iqr, c.alpha = self.med, self.iqr, a
        return c

    def params(self):
        return {"var": self.var, "med": self.med, "iqr": self.iqr, "alpha": self.alpha}

    def state(self):
        return self.params()

    def load(self, st):
        self.var, self.med, self.iqr, self.alpha = st["var"], st["med"], st["iqr"], st["alpha"]
        return self




# -----------------------------------------------------------------------------
# 기능  : 텍스트 vectorizer 후보 (T5 ablation). 이름 → sklearn 변환기
#         count_uni   단어 1-gram 개수               tfidf_uni   단어 1-gram TF-IDF (sublinear)
#         tfidf_bi    단어 1~2-gram TF-IDF           tfidf_char  문자 3~5-gram (단어 경계) TF-IDF
#         tfidf_stop  tfidf_uni + 불용어 제거 (부정어는 남김)
#         tfidf_df    tfidf_uni + min_df 20, max_df 0.5 (희귀·흔한 단어 가지치기, w4-2 p25)
#         lm_vocab    LM 금융 사전 단어만 어휘로 쓰는 개수 벡터 (사전이 없으면 오류)
#         hashing     상태 없는 해싱 TF (어휘 학습 없음)
#         공통: 소문자화는 vectorizer 안에서만, 토큰 규칙은 $·%·숫자를 보존 (w4-2 p18, p21)
# input : name, params {min_df, max_df}
# output: sklearn vectorizer
# -----------------------------------------------------------------------------
VEC_TOKEN = r"(?u)\$?[A-Za-z]+(?:[.'&-][A-Za-z]+)*|\d+(?:[.,]\d+)*%?"


def make_vectorizer(name, params=None):
    from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfVectorizer
    p = dict(params or {})
    mn, mx = p.get("min_df", 5), p.get("max_df", 1.0)
    word = dict(token_pattern=VEC_TOKEN, lowercase=True, min_df=mn, max_df=mx)
    stop = sorted(STOPWORDS - NEGATORS)
    if name == "count_uni":
        return CountVectorizer(**word)
    if name == "tfidf_uni":
        return TfidfVectorizer(sublinear_tf=True, **word)
    if name == "tfidf_bi":
        return TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), **word)
    if name == "tfidf_char":
        return TfidfVectorizer(sublinear_tf=True, analyzer="char_wb", ngram_range=(3, 5), min_df=mn, max_df=mx)
    if name == "tfidf_stop":
        return TfidfVectorizer(sublinear_tf=True, stop_words=stop, **word)
    if name == "tfidf_df":
        return TfidfVectorizer(sublinear_tf=True, **{**word, "min_df": p.get("min_df", 20), "max_df": p.get("max_df", 0.5)})
    if name == "lm_vocab":
        lm = text_resources().get("lm")
        if not lm:
            raise ValueError("lm_vocab: artifacts/text_resources.json 에 LM 사전이 없음 (python -m exp text resources)")
        vocab = sorted({w.lower() for k in ("positive", "negative", "uncertainty") for w in lm[k]})
        return CountVectorizer(token_pattern=VEC_TOKEN, lowercase=True, vocabulary=vocab)
    if name == "hashing":
        return HashingVectorizer(token_pattern=VEC_TOKEN, lowercase=True, n_features=2 ** 18, alternate_sign=False)
    raise ValueError(f"vectorizer '{name}' 없음. 가능: count_uni tfidf_uni tfidf_bi tfidf_char tfidf_stop tfidf_df lm_vocab hashing")


# -----------------------------------------------------------------------------
# 기능  : T2~T5. 텍스트로 "오늘 이 종목이 급등락할 확률"을 학습하고, 그 확률의 삼분위로 갭 신호의 크기 배율을 정함
#         방향(부호)은 갭 그대로. 텍스트는 크기 판단만 바꿈 (docs/text_pipeline.md 3.7)
# 구성  : prefit  학습 구간 앞부분(fit)으로 분류기 학습      → P(M=1 | 텍스트)
#         fit     학습 구간 뒷부분(tune)으로 삼분위 배율 결정  (TercileRegime 과 같음)
# input : params
#         input       "text"    titles_r1 문자열 → vectorizer → 로지스틱 회귀
#                     "columns" 숫자 열 목록 → 결측 중앙값 대체 → 표준화 → 로지스틱 회귀
#         vectorizer  make_vectorizer 이름 (input="text")
#         columns     열 이름 또는 묶음 이름 목록 (input="columns")
#         C           로지스틱 회귀 규제 (기본 1.0),  min_df / max_df,  f_grid (배율 후보)
# 목표  : M = 1[label ∈ {0, 4}]   (급등락 여부)
# loss  : class_weight="balanced" 로지스틱 log loss (급등락이 17% 라 불균형 보정)
# output: values(X) = P(M=1 | 텍스트). 텍스트가 없는 행은 NaN → 가운데 삼분위(배율 1)
# -----------------------------------------------------------------------------
class TextModelRegime(TercileRegime):
    learnable = True

    def __init__(self, params=None, extra_groups=None):
        p = dict(params or {})
        super().__init__(p)
        self.input = p.get("input", "text")
        self.vec_name = p.get("vectorizer", "tfidf_uni")
        self.vec_params = {k: p[k] for k in ("min_df", "max_df") if k in p}
        self.C = float(p.get("C", 1.0))
        self.text_col = p.get("text_col", "titles_r1")             # titles_r1 / lead_r1 / doc_r1
        if self.text_col not in TEXT_COLS:
            raise ValueError(f"text_col 은 {sorted(TEXT_COLS)} 중 하나: {self.text_col}")
        self.cols = (resolve_features(p.get("columns", []), extra_groups) if self.input == "columns" else [self.text_col])
        self.var = f"textclf:{self.input}:{self.vec_name if self.input == 'text' else '+'.join(self.cols)}"
        self.model = None

    @property
    def inputs(self):
        return list(self.cols)

    def prefit(self, F, seed=0):
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        y = np.isin(F["label"].to_numpy(int), (0, 4)).astype(int)
        clf = LogisticRegression(C=self.C, class_weight="balanced", max_iter=2000, random_state=seed)
        if self.input == "text":
            m = F[self.text_col].fillna("").str.len().to_numpy() > 0
            self.model = make_pipeline(make_vectorizer(self.vec_name, self.vec_params), clf)
            self.model.fit(F[self.text_col].to_numpy()[m], y[m])
        else:
            self.model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), clf)
            self.model.fit(F[self.cols].to_numpy(float), y)
        return self

    def values(self, X):
        if self.input == "text":
            t = X[self.text_col].fillna("").to_numpy()
            out = np.full(len(X), np.nan)
            m = np.array([len(s) > 0 for s in t], bool)
            if m.any():
                out[m] = self.model.predict_proba(t[m])[:, 1]
            return out
        return self.model.predict_proba(X[self.cols].to_numpy(float))[:, 1]

    def params(self):
        return {**super().params(), "input": self.input, "vectorizer": self.vec_name, "columns": self.cols, "C": self.C}

    def state(self):
        return {**super().state(), "model": self.model, "input": self.input, "vec_name": self.vec_name,
                "cols": self.cols, "C": self.C, "text_col": self.text_col}

    def load(self, st):
        super().load(st)
        self.model, self.input, self.vec_name, self.cols, self.C = st["model"], st["input"], st["vec_name"], st["cols"], st["C"]
        self.text_col = st.get("text_col", "titles_r1")
        return self


# -----------------------------------------------------------------------------
# 기능  : H1~H4. HAR 식 선형 모형으로 "오늘 움직임 크기"를 예측하고, 그 삼분위로 갭 신호의 크기 배율을 정함
#         방향(부호)은 갭 그대로. TextModelRegime 과 같은 구조 (prefit → 삼분위 배율)
# 수식  : t_j = log(1 + max(x_j, 0))    (x = abs_gap, absr1, absr5, absr22, pre_rng, earn …, 결측은 fit 구간 중앙값)
#         y   = log(min(|ret_pct|, 15) + 0.1)
#         ŷ   = a + Σ_j b_j t_j          (OLS, Corsi 2009 HAR-RV 를 |r| 로 바꾼 것)
#         mode "level"   values = ŷ
#              "excess"  values = ŷ − b_gap · t_gap   (같은 |갭| 에서 갭 외 변수가 말하는 크기. abs_gap 필요)
# loss  : 제곱오차 (OLS),  배율은 tune 구간 score 최대
# input : params  columns (열 또는 묶음 이름), mode, f_grid
# output: values(X) = 위 값 → 삼분위 배율 f_low / 1 / f_high
# -----------------------------------------------------------------------------
class MagModelRegime(TercileRegime):
    learnable = True

    def __init__(self, params=None, extra_groups=None):
        p = dict(params or {})
        super().__init__(p)
        self.cols = resolve_features(p.get("columns", ["har"]), extra_groups)
        self.mode = p.get("mode", "level")
        if self.mode not in ("level", "excess"):
            raise ValueError(f"mode 는 level / excess 중 하나: {self.mode}")
        if self.mode == "excess" and "abs_gap" not in self.cols:
            raise ValueError("mode='excess' 는 columns 에 abs_gap 이 있어야 함")
        self.var = f"magmodel:{self.mode}:{'+'.join(self.cols)}"
        self.med, self.coef, self.icpt = None, None, 0.0

    @property
    def inputs(self):
        return list(self.cols)

    def _t(self, X):
        T = np.log1p(np.clip(X[self.cols].to_numpy(float), 0, None))
        return np.where(np.isnan(T), self.med, T)

    def prefit(self, F, seed=0):
        self.med = np.nan_to_num(np.nanmedian(np.log1p(np.clip(F[self.cols].to_numpy(float), 0, None)), axis=0))
        T = self._t(F)
        y = np.log(np.minimum(np.abs(F["ret_pct"].to_numpy(float)), RET_CLIP) + 0.1)
        A = np.column_stack([np.ones(len(T)), T])
        sol = np.linalg.lstsq(A, y, rcond=None)[0]
        self.icpt, self.coef = float(sol[0]), sol[1:]
        return self

    def values(self, X):
        T = self._t(X)
        v = self.icpt + T @ self.coef
        if self.mode == "excess":
            j = self.cols.index("abs_gap")
            v = v - self.coef[j] * T[:, j]
        return v

    def params(self):
        return {**super().params(), "mode": self.mode, "columns": self.cols,
                "coef": {c: float(b) for c, b in zip(self.cols, self.coef)} if self.coef is not None else None,
                "intercept": self.icpt}

    def state(self):
        return {**super().state(), "cols": self.cols, "mode": self.mode, "med": self.med, "coef": self.coef, "icpt": self.icpt}

    def load(self, st):
        super().load(st)
        self.cols, self.mode, self.med, self.coef, self.icpt = st["cols"], st["mode"], st["med"], st["coef"], st["icpt"]
        return self


REGIMES = {"none": NoRegime, "tercile": TercileRegime, "linear": LinearRegime, "textclf": TextModelRegime,
           "magmodel": MagModelRegime}


# =============================================================================
# 7. Pipeline  (spec 하나 = 실험 후보 하나 = 제출 모델 하나)
# =============================================================================

# -----------------------------------------------------------------------------
# 기능  : spec 검사. 이름이 registry 에 없거나 조합이 맞지 않으면 바로 멈춤
# input : spec dict
# output: 기본값을 채운 spec 사본
# -----------------------------------------------------------------------------
def validate_spec(spec):
    s = {"signal": {"type": "gap"}, "decision": {"type": "k"}, "regime": {"type": "none"},
         "selection": {"criterion": "pooled"}, "tune_frac": 0.2, "tune_on": "auto", "seed": 0, **spec}
    for part, reg in (("signal", SIGNALS), ("decision", DECISIONS), ("regime", REGIMES)):
        if s[part].get("type") not in reg:
            raise ValueError(f"{s.get('name')}: {part}.type '{s[part].get('type')}' 없음. 가능: {sorted(reg)}")
    decomp_sig = SIGNALS[s["signal"]["type"]].kind == "decomp"
    if decomp_sig != (s["decision"]["type"] == "decomp"):
        raise ValueError(f"{s.get('name')}: signal 'decomp' 와 decision 'decomp' 는 함께 써야 함")
    if decomp_sig and s["regime"]["type"] != "none":
        raise ValueError(f"{s.get('name')}: regime 보정은 스칼라 신호에만 씀")
    return s


class Pipeline:
    """신호 → 결정 규칙 → regime 보정을 묶은 모델 하나.

    fit:  학습 구간을 시간순으로 [fit | tune] 으로 나눔 (tune = 마지막 tune_frac 비율의 대상일)
          learnable 신호는 fit 부분으로 학습, 결정 규칙과 regime 은 tune 부분으로 고름
          학습 없는 신호(gap)는 tune_on="auto" 일 때 학습 구간 전체로 결정 규칙을 고름
    """

    # -------------------------------------------------------------------------
    # input : spec  {"name", "signal": {"type", "features", "params"}, "decision": {"type", "params"},
    #                "regime": {"type", "params"}, "selection": {...}, "tune_frac", "tune_on", "seed"}
    #         extra_groups  연구용 feature 묶음 {이름: [열]}
    # -------------------------------------------------------------------------
    def __init__(self, spec, extra_groups=None):
        self.spec = validate_spec(spec)
        self.extra_groups = dict(extra_groups or {})
        sig = self.spec["signal"]
        feats = resolve_features(sig.get("features", ["core"]), extra_groups)
        self.signal = SIGNALS[sig["type"]](feats, sig.get("params"))
        self.decision = DECISIONS[self.spec["decision"]["type"]](self.spec["decision"].get("params"))
        self.regime = REGIMES[self.spec["regime"]["type"]](self.spec["regime"].get("params"), extra_groups)

    @property
    def features(self):
        return list(dict.fromkeys(list(self.signal.features) + list(self.regime.inputs)))

    @property
    def groups(self):
        return groups_of(self.features)

    @property
    def deployable(self):
        return set(self.features) <= DEPLOYABLE

    def _time_split(self, X):
        t = X["target"].to_numpy(NS)
        edge = np.quantile(np.unique(t).astype("int64"), 1 - float(self.spec["tune_frac"]))
        tune = t.astype("int64") > edge
        return ~tune, tune

    def _split(self, X):
        if not self.signal.learnable and self.spec["tune_on"] == "auto":
            return np.zeros(len(X), bool), np.ones(len(X), bool)
        return self._time_split(X)

    # -------------------------------------------------------------------------
    # 기능  : 학습. X 는 feature + 정답(label, ret_pct) 을 가진 패널
    # input : X  DataFrame[target, label, ret_pct, <features>]
    # output: self
    # -------------------------------------------------------------------------
    def fit(self, X):
        X = X.sort_values(["target", "symbol"]).reset_index(drop=True)
        fit_m, tune_m = self._split(X)
        seed = int(self.spec["seed"])
        if self.signal.learnable:
            F = X[fit_m]
            self.signal.fit(F, F["ret_pct"].to_numpy(float), F["label"].to_numpy(int), seed)
        T = X[tune_m]
        y = T["label"].to_numpy(int)
        obj = Objective(self.spec["selection"], T["target"])
        s = self.signal.predict(T)
        self.decision.fit(s, y, obj)
        if self.regime.var and self.regime.learnable:
            # 학습형 regime(텍스트 분류기): 앞부분으로 분류기, 뒷부분으로 배율. 결정 규칙은 위에서 이미 고정
            rf, rt = self._time_split(X)
            self.regime.prefit(X[rf], seed)
            R = X[rt]
            self.regime.fit(self.signal.predict(R), self.regime.values(R), R["label"].to_numpy(int),
                            self.decision, Objective(self.spec["selection"], R["target"]))
        elif self.regime.var:
            self.regime.fit(s, self.regime.values(T), y, self.decision, obj)
        self.fit_info = {"n_fit": int(fit_m.sum()), "n_tune": int(tune_m.sum()),
                         "tune_from": str(T["target"].min().date()) if len(T) else None}
        return self

    # -------------------------------------------------------------------------
    # input : X  feature 표
    # output: label 배열 (0~4 정수)
    # -------------------------------------------------------------------------
    def predict(self, X):
        s = self.signal.predict(X)
        if self.regime.var:
            s = self.regime.apply(s, self.regime.values(X))
        return self.decision.predict(s)

    def params(self):
        return {"decision": self.decision.params(), "regime": self.regime.params(),
                **getattr(self, "fit_info", {})}

    # -------------------------------------------------------------------------
    # 기능  : 저장·복원. 클래스가 아니라 spec 과 학습 결과(sklearn 객체·숫자)만 저장함
    # -------------------------------------------------------------------------
    def state(self):
        return {"format": ARTIFACT_FORMAT, "spec": self.spec, "extra_groups": self.extra_groups,
                "signal": self.signal.state(),
                "decision": self.decision.state(), "regime": self.regime.state(),
                "params": self.params()}

    @classmethod
    def from_state(cls, st):
        if st.get("format") != ARTIFACT_FORMAT:
            raise ValueError(f"artifact format {st.get('format')} != {ARTIFACT_FORMAT}")
        p = cls(st["spec"], st.get("extra_groups"))
        p.signal.load(st["signal"])
        p.decision.load(st["decision"])
        p.regime.load(st["regime"])
        p.fit_info = {k: v for k, v in st.get("params", {}).items() if k.startswith(("n_", "tune"))}
        return p


# -----------------------------------------------------------------------------
# 기능  : artifact 가 없을 때 쓰는 기본 모델. 갭 규칙 label_of(DEFAULT_K × gap_pct)
# -----------------------------------------------------------------------------
def default_pipeline():
    p = Pipeline({"name": "default_gap_k", "signal": {"type": "gap"}, "decision": {"type": "k"}})
    p.decision.k, p.decision.b = DEFAULT_K, np.asarray(_k_bounds(DEFAULT_K, DEFAULT_K))
    return p


# =============================================================================
# 8. 제출 인터페이스
# =============================================================================

class Model:
    """Pipeline 을 감싸 day 하나를 예측함. feature 를 못 만든 종목은 보합으로 둠."""

    def __init__(self, pipeline=None):
        self.pipeline = pipeline or default_pipeline()

    # -------------------------------------------------------------------------
    # input : day  src.data.Day
    # output: DataFrame[symbol, label]  day.symbols 를 빠짐없이, 그것만
    # -------------------------------------------------------------------------
    def predict(self, day):
        out = pd.DataFrame({"symbol": day.symbols, "label": FLAT})
        try:
            x = day_features(day, self.pipeline.groups)
            if len(x):
                lab = pd.Series(self.pipeline.predict(x), index=x["symbol"].to_numpy())
                out["label"] = out["symbol"].map(lab).fillna(FLAT).astype(int)
        except Exception as e:      # 데이터 결손으로 예측 전체가 멈추지 않게 함
            print(f"[model] {day!r} 예측 실패, 전부 보합으로 대체: {type(e).__name__}: {e}", file=sys.stderr)
        return out


def load_model():
    """artifacts/pipeline.pkl 을 읽어 Model 을 돌려줌. 없으면 기본 갭 규칙."""
    if ARTIFACT.exists():
        with open(ARTIFACT, "rb") as f:
            return Model(Pipeline.from_state(pickle.load(f)))
    return Model()
