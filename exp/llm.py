"""E4-L LLM 스크리닝 (연구용, 제출 모델에서는 쓰지 않음).

기능  : 뉴스 제목 / Reddit 글을 LLM 으로 구조화하고, 그 결과가 갭 이후 움직임을 설명하는지 볼 feature 를 만듦
구성  : SCHEMAS·SYSTEM (출력 형식·지시문) · anonymize() · sample_news() · sample_reddit()
        · run_label() (silver label, 프롬프트 3종) · run_placebo() (시점 맞히기)
        · call() (캐시) · run_llm()
역할  : LLM 은 "주가에 좋은가"를 판단하지 않고 사실만 추출하는 parser 로만 씀. 방향 부호는 코드가 규칙으로 계산
        누수 통제: 회사명·티커·날짜를 지운 익명 입력(anon)과 원문 입력(raw)을 둘 다 돌려 비교
        재현성: 모든 호출을 (모델, 프롬프트 버전, 스키마, 입력) 해시로 캐시 → 다시 돌리면 API 를 부르지 않음
안전장치: dry_run=true (기본) 이면 API 를 부르지 않고 보낼 입력만 저장. max_calls 로 새 호출 수 상한

feature (대상일 단위)
    news   llm_<v>_sign       = sign(Σ 이벤트 부호),  beat/raise/upgrade = +1, miss/cut/downgrade = −1
           llm_<v>_severity   = 1~3,  llm_<v>_unexpected = 1~3,  llm_<v>_n = 처리한 기사 수
    reddit llm_reddit_bull    = mean(+1 bullish, −1 bearish, 0 그 외),  llm_reddit_hype = mean(1~3),  llm_reddit_n
"""

import hashlib
import json
import re
import time

import numpy as np
import pandas as pd

from src.model import map_to_target

from . import config as C
from .panel import calendar, load_tables, universe_panel
from .splits import partition

PROMPT_VERSION = "v1"
LLM_CACHE = C.CACHE / "llm"
ALIASES = C.ROOT / "configs" / "company_aliases.json"

SCHEMAS = {
    "news_event": {
        "type": "object", "additionalProperties": False,
        "required": ["event_type", "earnings_result", "guidance_action", "analyst_action", "legal_event",
                     "mna_event", "product_event", "management_event", "severity", "unexpectedness", "mixed_event"],
        "properties": {
            "event_type": {"type": "string", "enum": ["earnings", "guidance", "analyst", "legal_regulatory", "mna",
                                                      "product", "management", "macro", "other", "none"]},
            "earnings_result": {"type": "string", "enum": ["beat", "meet", "miss", "not_mentioned"]},
            "guidance_action": {"type": "string", "enum": ["raise", "maintain", "cut", "not_mentioned"]},
            "analyst_action": {"type": "string", "enum": ["upgrade", "maintain", "downgrade", "not_mentioned"]},
            "legal_event": {"type": "boolean"}, "mna_event": {"type": "boolean"},
            "product_event": {"type": "boolean"}, "management_event": {"type": "boolean"},
            "severity": {"type": "integer", "enum": [1, 2, 3]},
            "unexpectedness": {"type": "integer", "enum": [1, 2, 3]},
            "mixed_event": {"type": "boolean"},
        },
    },
    "reddit_sentiment": {
        "type": "object", "additionalProperties": False,
        "required": ["stance", "hype_intensity", "speculation", "short_squeeze", "event_reaction", "disagreement"],
        "properties": {
            "stance": {"type": "string", "enum": ["bullish", "bearish", "neutral", "unclear"]},
            "hype_intensity": {"type": "integer", "enum": [1, 2, 3]},
            "speculation": {"type": "boolean"}, "short_squeeze": {"type": "boolean"},
            "event_reaction": {"type": "boolean"}, "disagreement": {"type": "boolean"},
        },
    },
}
SYSTEM = {
    "news_event": ("You extract facts from one financial news headline. Report only what the headline states. "
                   "Do not judge whether the news is good or bad for any stock price, and do not use knowledge of "
                   "what happened after the headline. Use 'not_mentioned' or false when the headline does not say. "
                   "severity: 1 routine, 2 notable, 3 major. unexpectedness: 1 expected or scheduled, 2 somewhat, "
                   "3 described as surprising."),
    "reddit_sentiment": ("You label one Reddit post or comment about a stock. Report the author's stated stance "
                         "and tone only, not your own view of the stock. hype_intensity: 1 calm, 2 excited, 3 extreme."),
}


# -----------------------------------------------------------------------------
# 기능  : 익명화. 회사 이름·티커($XXX 포함)·날짜 표현을 지움
# input : text,  aliases  {티커: [이름]}
# output: 익명화한 문자열  (회사 → [COMPANY], 날짜 → [DATE])
# -----------------------------------------------------------------------------
_MONTHS = r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
_DATE_RX = re.compile(rf"\b{_MONTHS}\.?\s*\d{{0,2}}(?:,?\s*\d{{4}})?\b|\b(?:19|20)\d{{2}}\b|\b\d{{1,2}}/\d{{1,2}}(?:/\d{{2,4}})?\b|"
                      r"\b(?:Mon|Tues|Wednes|Thurs|Fri|Satur|Sun)day\b", re.I)


def anonymize(text, aliases):
    names = sorted({n for v in aliases.values() for n in v}, key=len, reverse=True)
    out = re.sub(r"\$[A-Za-z]{1,5}(?:[.\-][A-Za-z])?\b", "[COMPANY]", text)
    for n in names:
        out = re.sub(rf"(?<!\w){re.escape(n)}(?:'s)?(?!\w)", "[COMPANY]", out, flags=re.I)
    for t in aliases:
        out = re.sub(rf"\b{re.escape(t)}\b", "[COMPANY]", out)          # 티커는 대문자 그대로일 때만
    return re.sub(r"\s+", " ", _DATE_RX.sub("[DATE]", out)).strip()


def _aliases():
    return {k: v for k, v in json.loads(ALIASES.read_text(encoding="utf-8")).items() if not k.startswith("_")}


# -----------------------------------------------------------------------------
# 기능  : DEV 구간의 (종목, 대상일) 패널 (gap_pct, rest, label). 교차 feature 는 20종목 묶음으로 계산
# -----------------------------------------------------------------------------
def _dev_panel(cfg):
    syms = sorted(load_tables()["daily"]["symbol"].unique())
    unis = partition(syms, cfg["split"]["universe_size"], np.random.default_rng(C.derive_seed(cfg["seed"], "llm")))
    x = pd.concat([universe_panel(u, ["core"]) for u in unis], ignore_index=True)
    return x[x["target"] < pd.Timestamp(cfg["split"]["lockbox_start"])][["symbol", "target", "gap_pct", "rest", "label"]]


# -----------------------------------------------------------------------------
# 기능  : 갭 층화 뉴스 표본
# 방법  : 텍스트 파이프라인의 기사 표에서 관련도 R1 이상·비템플릿 기사만 씀 (태그만 붙은 무관한 기사 제외)
#         (종목, 대상일)마다 가장 많이 재게시된 클러스터의 첫 제목을 대표로 고름
#         |gap| 층 (small < 0.5, medium < 1.5, large ≥ 1.5 %) × 갭 부호 6칸에서 n_per_stratum 개씩 무작위 추출
# input : cfg [llm] n_per_stratum, seed
#         max_calls는 신규 item 처리 수 기준이며 retry 횟수는 별도
# output: DataFrame[symbol, target, text]  (universe ticker가 하나만 언급된 댓글만 사용)
# -----------------------------------------------------------------------------
def sample_news(cfg):
    from .text import articles, lockbox_safe
    L = cfg["llm"]
    x = _dev_panel(cfg)
    a = lockbox_safe(articles())
    a = a[(a["rel"] >= 1) & (a["is_template"] == 0)]
    cnt = a.groupby(["symbol", "target", "ckey"]).agg(dups=("title", "size"), title=("title", "first"),
                                                       first=("known_at", "min")).reset_index()
    rep = cnt.sort_values(["dups", "first", "ckey"], ascending=[False, True, True]).drop_duplicates(["symbol", "target"])
    rep = rep.merge(x, on=["symbol", "target"]).dropna(subset=["gap_pct"])
    size = pd.cut(rep["gap_pct"].abs(), [0, 0.5, 1.5, np.inf], right=False, labels=["small", "medium", "large"])
    rep["stratum"] = size.astype(str) + np.where(rep["gap_pct"] >= 0, "_pos", "_neg")
    rng = np.random.default_rng(C.derive_seed(cfg["seed"], "llm_news"))
    k = int(L.get("n_per_stratum", 50))
    pick = [g.iloc[np.sort(rng.choice(len(g), min(k, len(g)), replace=False))] for _, g in rep.groupby("stratum", sort=True)]
    return pd.concat(pick, ignore_index=True)[["symbol", "target", "title", "gap_pct", "stratum"]]


# -----------------------------------------------------------------------------
# 기능  : Reddit 표본. 시드로 섞은 row group 순서대로 읽으며 티커가 언급된 댓글을 모음
# input : cfg [llm] reddit_sub, reddit_n
# output: DataFrame[symbol, target, text]  (한 댓글에 티커가 여럿이면 종목마다 한 행)
# -----------------------------------------------------------------------------
def sample_reddit(cfg):
    import sys
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    sys.path.insert(0, str(C.ROOT / "eda"))
    import eda_utils as E

    L = cfg["llm"]

    syms = sorted(load_tables()["daily"]["symbol"].unique())
    pre, rx = E.ticker_patterns(syms)

    pf = pq.ParquetFile(
        C.ROOT
        / "dataset"
        / "reddit"
        / f"{L.get('reddit_sub', 'stocks')}.comments.parquet"
    )

    rng = np.random.default_rng(
        C.derive_seed(cfg["seed"], "llm_reddit")
    )

    lb = pd.Timestamp(cfg["split"]["lockbox_start"])
    cal = calendar()

    want = int(L.get("reddit_n", 300))

    # 너무 많이 메모리에 올리지 않기 위한 후보 pool
    pool_target = max(want * 5, 1000)

    rows = []

    # -------------------------------------------------------------------------
    # 1. ticker 언급 Reddit 댓글 후보 수집
    # -------------------------------------------------------------------------
    for i in rng.permutation(pf.num_row_groups):
        t = pf.read_row_group(
            int(i),
            columns=["created_et", "body"],
        )

        hit = pc.fill_null(
            pc.match_substring_regex(
                pc.fill_null(t.column("body"), ""),
                pre,
            ),
            False,
        ).to_numpy()

        if not hit.any():
            continue

        d = t.filter(hit).to_pandas()

        d["target"] = map_to_target(
            d["created_et"],
            cal,
        )

        d = d[
            d["target"].notna()
            & (d["target"] < lb)
        ]

        for _, r in d.iterrows():
            text = str(r["body"])

            found = {
                (a.upper() or b).replace(".", "-")
                for a, b in rx.findall(text)
            } & set(syms)

            if not found:
                continue

            # Reddit semantic feature는 종목별 stance를 해석해야 하므로
            # universe ticker가 하나만 언급된 글만 사용
            if len(found) != 1:
                continue

            symbol = next(iter(found))

            
            # 같은 댓글 안에서 ticker가 몇 번 반복되는지
            ticker_mentions = sum(
            1
            for a, b in rx.findall(text)
            if (a.upper() or b).replace(".", "-") == symbol
)

          

            rows.append(
                {
                    "symbol": symbol,
                    "target": r["target"],
                    "created_et": r["created_et"],
                    "text": text[:1500],
                    "text_len": len(text),
                    "ticker_mentions": ticker_mentions,
                }
)

            if len(rows) >= pool_target:
                break

        if len(rows) >= pool_target:
            break

    if not rows:
        return pd.DataFrame(
            columns=[
                "symbol",
                "target",
                "text",
                "sample_type",
            ]
        )

    pool = pd.DataFrame(rows)

    # 중복 제거
    pool = pool.drop_duplicates(
        subset=["target", "text"]
    ).reset_index(drop=True)

    # -------------------------------------------------------------------------
    # 2. 4종 representative sample
    #
    # random         : 전체 후보에서 무작위
    # long           : 긴 글
    # ticker_repeat  : ticker 반복 언급이 많은 글
    # recent         : 가장 최근 글
    # -------------------------------------------------------------------------
    n_each = max(want // 4, 1)

    chosen = []

    # random
    if len(pool):
        k = min(n_each, len(pool))

        random_part = pool.iloc[
            np.sort(
                rng.choice(
                    len(pool),
                    size=k,
                    replace=False,
                )
            )
        ].copy()

        random_part["sample_type"] = "random"
        chosen.append(random_part)

    # long
    remain = pool.copy()

    long_part = (
        remain
        .sort_values(
            ["text_len", "created_et"],
            ascending=[False, False],
        )
        .head(n_each)
        .copy()
    )

    if len(long_part):
        long_part["sample_type"] = "long"
        chosen.append(long_part)

    # ticker repeat
    repeat_part = (
        remain
        .sort_values(
            ["ticker_mentions", "created_et"],
            ascending=[False, False],
        )
        .head(n_each)
        .copy()
    )

    if len(repeat_part):
        repeat_part["sample_type"] = "ticker_repeat"
        chosen.append(repeat_part)

    # recent
    recent_part = (
        remain
        .sort_values(
            "created_et",
            ascending=False,
        )
        .head(n_each)
        .copy()
    )

    if len(recent_part):
        recent_part["sample_type"] = "recent"
        chosen.append(recent_part)

    # -------------------------------------------------------------------------
    # 3. 합치고 중복 제거
    # -------------------------------------------------------------------------
    out = pd.concat(
        chosen,
        ignore_index=True,
    )

    out = out.drop_duplicates(
        subset=["symbol", "target", "text"]
    )

    # 중복 제거 때문에 want보다 적어질 수 있으므로 남는 건 random으로 보충
    if len(out) < want:
        used = set(
            zip(
                out["symbol"],
                out["target"],
                out["text"],
            )
        )

        rest = pool[
            ~pool.apply(
                lambda r: (
                    r["symbol"],
                    r["target"],
                    r["text"],
                ) in used,
                axis=1,
            )
        ]

        need = min(
            want - len(out),
            len(rest),
        )

        if need > 0:
            extra = rest.iloc[
                np.sort(
                    rng.choice(
                        len(rest),
                        size=need,
                        replace=False,
                    )
                )
            ].copy()

            extra["sample_type"] = "random_fill"

            out = pd.concat(
                [out, extra],
                ignore_index=True,
            )

    return out[
        [
            "symbol",
            "target",
            "text",
            "sample_type",
        ]
    ].head(want)


# -----------------------------------------------------------------------------
# 기능  : LLM 호출 1건 (캐시 우선)
# input : schema  SCHEMAS 이름,  text,  L  [llm] 설정 (model, effort),  client  anthropic 클라이언트 또는 None
# output: (결과 dict 또는 None, 캐시에서 읽었는지)
# 캐시  : exp/cache/llm/<sha256>.json = {request(model, prompt_version, schema, system_sha, text), response, meta}
#         temperature 는 이 모델에서 지정할 수 없어 기록하지 않음 (모델 기본값)
# -----------------------------------------------------------------------------
def _key(schema, text, L):
    req = {"model": L["model"], "effort": L["effort"], "prompt_version": PROMPT_VERSION, "schema": schema,
           "system_sha": hashlib.sha256(SYSTEM[schema].encode()).hexdigest()[:12], "text": text}
    return hashlib.sha256(json.dumps(req, sort_keys=True).encode()).hexdigest(), req


def call(schema, text, L, client):
    k, req = _key(schema, text, L)
    path = LLM_CACHE / f"{k}.json"

    # -------------------------------------------------------------------------
    # 1. 기존 cache가 있으면 API를 다시 호출하지 않음
    # -------------------------------------------------------------------------
    if path.exists():
        try:
            cached = json.loads(path.read_text(encoding="utf-8"))
            return cached["response"], True
        except (json.JSONDecodeError, KeyError, OSError) as e:
            # cache 파일이 깨졌다면 API 호출을 다시 시도
            print(f"[llm] broken cache ignored: {path.name} ({e})")

    # dry_run이거나 API 호출 budget이 없는 경우
    if client is None:
        return None, False

    # -------------------------------------------------------------------------
    # 2. API 안정성 설정
    # -------------------------------------------------------------------------
    max_retries = int(L.get("max_retries", 3))
    base_delay = float(L.get("retry_base_delay", 2.0))

    last_error = None

    # 총 시도 횟수 = 최초 1회 + retry 횟수
    for attempt in range(max_retries + 1):
        try:
            # -----------------------------------------------------------------
            # 3. LLM 호출
            # -----------------------------------------------------------------
            resp = client.beta.messages.create(
                model=L["model"],
                max_tokens=1024,
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                system=SYSTEM[schema],
                messages=[{"role": "user", "content": text}],
                output_config={
                    "effort": L["effort"],
                    "format": {
                        "type": "json_schema",
                        "schema": SCHEMAS[schema],
                    },
                },
            )

            # -----------------------------------------------------------------
            # 4. 응답 parsing
            # -----------------------------------------------------------------
            out = None

            if resp.stop_reason != "refusal":
                text_blocks = [
                    b.text
                    for b in resp.content
                    if getattr(b, "type", None) == "text"
                ]

                if not text_blocks:
                    raise ValueError("LLM response contains no text block.")

                out = json.loads(text_blocks[0])

            # -----------------------------------------------------------------
            # 5. 성공한 결과는 cache 저장
            # -----------------------------------------------------------------
            meta = {
                "served_model": resp.model,
                "stop_reason": resp.stop_reason,
                "request_id": getattr(resp, "_request_id", None),
                "input_tokens": resp.usage.input_tokens,
                "output_tokens": resp.usage.output_tokens,
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "attempts": attempt + 1,
            }

            LLM_CACHE.mkdir(parents=True, exist_ok=True)

            path.write_text(
                json.dumps(
                    {
                        "request": req,
                        "response": out,
                        "meta": meta,
                    },
                    ensure_ascii=False,
                    indent=1,
                ),
                encoding="utf-8",
            )

            return out, False

        # ---------------------------------------------------------------------
        # 6. API / network / JSON parsing 등 실패 처리
        # ---------------------------------------------------------------------
        except Exception as e:
            last_error = e

            if attempt < max_retries:
                delay = base_delay * (2 ** attempt)

                print(
                    f"[llm] call failed "
                    f"(attempt {attempt + 1}/{max_retries + 1}): "
                    f"{type(e).__name__}: {e}"
                )
                print(f"[llm] retrying in {delay:.1f}s...")

                time.sleep(delay)

            else:
                print(
                    f"[llm] call permanently failed after "
                    f"{max_retries + 1} attempts: "
                    f"{type(e).__name__}: {e}"
                )

    # -------------------------------------------------------------------------
    # 7. 최종 실패 기록
    #    실패 결과는 정상 cache로 저장하지 않음 → 나중에 재실행 가능
    # -------------------------------------------------------------------------
    LLM_CACHE.mkdir(parents=True, exist_ok=True)

    fail_path = LLM_CACHE / "failures.jsonl"

    failure = {
        "key": k,
        "schema": schema,
        "model": L["model"],
        "prompt_version": PROMPT_VERSION,
        "error_type": type(last_error).__name__ if last_error else None,
        "error": str(last_error) if last_error else None,
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(fail_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(failure, ensure_ascii=False) + "\n")

    return None, False


# -----------------------------------------------------------------------------
# 기능  : 구조화 결과 → 숫자 feature (방향 부호는 규칙으로 계산)
# -----------------------------------------------------------------------------
def _news_row(r):
    s = {"beat": 1, "miss": -1}.get(r["earnings_result"], 0) + {"raise": 1, "cut": -1}.get(r["guidance_action"], 0) \
        + {"upgrade": 1, "downgrade": -1}.get(r["analyst_action"], 0)
    return {"sign": float(np.sign(s)), "severity": float(r["severity"]), "unexpected": float(r["unexpectedness"])}


def _reddit_row(r):
    return {
        "bull": {"bullish": 1.0, "bearish": -1.0}.get(r["stance"], 0.0),
        "hype": float(r["hype_intensity"]),
        "speculation": float(r["speculation"]),
        "short_squeeze": float(r["short_squeeze"]),
        "event_reaction": float(r["event_reaction"]),
        "disagreement": float(r["disagreement"]),
    }


# =============================================================================
# LLM silver label (수작업 라벨 대신) · placebo 시점 테스트
# =============================================================================
# 라벨 스키마: 대상 회사 관련도 + 투자자 관점 극성 (w5-1 p11, p20 entity-level)
_LABEL_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["relevance", "polarity", "confidence"],
    "properties": {
        "relevance": {"type": "string", "enum": ["about_target", "mentions_target", "not_about_target"]},
        "polarity": {"type": "string", "enum": ["positive", "neutral", "negative"]},
        "confidence": {"type": "integer", "enum": [1, 2, 3]},
    },
}
# 프롬프트 민감도 (w5-1 p38): 같은 과제를 문구만 다르게 3가지로 물음
_LABEL_SYSTEMS = [
    ("Label one financial news headline for the company written as [TARGET]. relevance: about_target if the "
     "headline is mainly about [TARGET], mentions_target if [TARGET] is only mentioned, not_about_target otherwise. "
     "polarity: from an investor's perspective on [TARGET], is the reported information positive, neutral or negative? "
     "Judge only what the headline says. [OTHER] marks other companies and [DATE] marks dates."),
    ("You are annotating headlines for a finance dataset. Target company = [TARGET]; other companies = [OTHER]. "
     "Decide whether the headline is about the target (about_target), merely mentions it (mentions_target) or is not "
     "about it (not_about_target). Then give the news polarity for a shareholder of the target: positive, neutral or "
     "negative. Use only the headline text."),
    ("Headline classification. Step 1, relevance of [TARGET] — about_target / mentions_target / not_about_target. "
     "Step 2, would a [TARGET] investor read this as good (positive), bad (negative) or neither (neutral) news? "
     "Ignore your knowledge of later events; read the words only. confidence 1 = unsure, 3 = clear."),
]
for _i, _s in enumerate(_LABEL_SYSTEMS, 1):
    SCHEMAS[f"headline_label_v{_i}"] = _LABEL_SCHEMA
    SYSTEM[f"headline_label_v{_i}"] = _s
# placebo (w5-1 p37): 제목만 보고 시점을 맞히면 모델이 사건을 기억하고 있다는 신호
SCHEMAS["date_guess"] = {
    "type": "object", "additionalProperties": False, "required": ["year", "month"],
    "properties": {"year": {"type": "integer"}, "month": {"type": "integer"}},
}
SYSTEM["date_guess"] = ("Guess when this news headline was published. Answer year (e.g. 2025) and month (1-12). "
                        "If you cannot tell, answer year 0 and month 0.")


# -----------------------------------------------------------------------------
# 기능  : 라벨용 익명화. 대상 회사 → [TARGET], 다른 회사 → [OTHER], 날짜 → [DATE]
# input : title, sym (대상 티커), aliases
# -----------------------------------------------------------------------------
def anonymize_target(title, sym, aliases):
    from src.model import symbol_pattern
    out = symbol_pattern(sym, aliases).sub("[TARGET]", title)
    for other in aliases:
        if other != sym:
            out = symbol_pattern(other, aliases).sub("[OTHER]", out)
    return re.sub(r"\s+", " ", _DATE_RX.sub("[DATE]", out)).strip()


def _label_sample(cfg, L):
    from .text import articles, lockbox_safe
    a = lockbox_safe(articles())
    a = a[a["is_template"] == 0].drop_duplicates(["symbol", "title_id"])
    rng = np.random.default_rng(C.derive_seed(cfg["seed"], "llm_label"))
    k = int(L.get("label_per_tier", 100))
    pick = []
    for rel, g in a.groupby("rel", sort=True):
        pick.append(g.iloc[np.sort(rng.choice(len(g), min(k, len(g)), replace=False))])
    return pd.concat(pick, ignore_index=True)


# -----------------------------------------------------------------------------
# 기능  : silver label 만들기와 평가
#         1) 관련도 R0/R1/R2 층에서 같은 수씩 표본 → 2) 3가지 프롬프트로 라벨 → 3) 다수결 silver label
#         4) 평가: 기업 연결(R1/R2)의 정밀도·재현율, 톤 측정(GDELT·LM·FinBERT)의 클래스별 recall,
#            프롬프트 간 일치율 (silver label 은 정답이 아니라 또 하나의 측정 도구임)
# output: label_items.csv, label_eval.json
# -----------------------------------------------------------------------------
def run_label(cfg, L, client, out, budget, pending):
    from src.model import lm_scores, text_resources
    aliases = _aliases()
    s = _label_sample(cfg, L)
    s["input"] = [anonymize_target(t, sym, aliases) for t, sym in zip(s["title"], s["symbol"])]
    for i in range(1, len(_LABEL_SYSTEMS) + 1):
        res = []
        for text in s["input"]:
            r, hit = call(f"headline_label_v{i}", text, L, client if budget[0] > 0 else None)
            if not hit and client is not None and budget[0] > 0:
                budget[0] -= 1
            if r is None and not hit:
                pending.append({"schema": f"headline_label_v{i}", "text": text})
            res.append(r or {})
        s[f"rel_v{i}"] = [r.get("relevance") for r in res]
        s[f"pol_v{i}"] = [r.get("polarity") for r in res]
    vs = [f"pol_v{i}" for i in range(1, len(_LABEL_SYSTEMS) + 1)]
    rs = [f"rel_v{i}" for i in range(1, len(_LABEL_SYSTEMS) + 1)]
    mode = lambda row: pd.Series(row).dropna().mode().iloc[0] if pd.Series(row).notna().any() else None
    s["pol_silver"] = s[vs].apply(lambda r: mode(r.values), axis=1)
    s["rel_silver"] = s[rs].apply(lambda r: mode(r.values), axis=1)
    s["lm_net"] = lm_scores(s["title"], text_resources().get("lm"))["lm_net"]
    s["pol_gdelt"] = np.select([s["tone"] > 1, s["tone"] < -1], ["positive", "negative"], "neutral")
    s["pol_lm"] = np.select([s["lm_net"] > 0, s["lm_net"] < 0], ["positive", "negative"],
                            np.where(s["lm_net"].isna(), None, "neutral"))
    try:
        from .text import FINBERT_SCORES
        fb = pd.read_parquet(FINBERT_SCORES, columns=["title_id", "p_pos", "p_neg", "p_neu"])
        s = s.merge(fb, on="title_id", how="left")
        s["pol_finbert"] = s[["p_pos", "p_neg", "p_neu"]].idxmax(axis=1).map(
            {"p_pos": "positive", "p_neg": "negative", "p_neu": "neutral"})
    except (FileNotFoundError, KeyError, ValueError):
        s["pol_finbert"] = None
    s.to_csv(out / "label_items.csv", index=False)

    ev = {"n": len(s), "labeled": int(s["pol_silver"].notna().sum())}
    lab = s[s["pol_silver"].notna()]
    if len(lab):
        pairs = [(a, b) for i, a in enumerate(vs) for b in vs[i + 1:]]
        ev["prompt_agreement_polarity"] = {f"{a}~{b}": float((lab[a] == lab[b]).mean()) for a, b in pairs}
        ev["prompt_agreement_relevance"] = {f"{a}~{b}": float((lab[a] == lab[b]).mean())
                                            for a, b in [(x.replace("pol", "rel"), y.replace("pol", "rel")) for x, y in pairs]}
        about = lab["rel_silver"] == "about_target"
        for r in (1, 2):
            pred = lab["rel"] >= r
            ev[f"entity_R{r}_precision"] = float(about[pred].mean()) if pred.any() else None
            ev[f"entity_R{r}_recall"] = float(pred[about].mean()) if about.any() else None
        rel_ok = lab[lab["rel_silver"] != "not_about_target"]
        for m in ("pol_gdelt", "pol_lm", "pol_finbert"):
            d = rel_ok[rel_ok[m].notna()]
            if len(d):
                ev[m] = {"n": len(d), "accuracy": float((d[m] == d["pol_silver"]).mean()),
                         "recall_by_class": {c: float((d.loc[d["pol_silver"] == c, m] == c).mean())
                                             for c in ("positive", "neutral", "negative") if (d["pol_silver"] == c).any()}}
        ev["silver_class_share"] = rel_ok["pol_silver"].value_counts(normalize=True).round(3).to_dict()
    (out / "label_eval.json").write_text(json.dumps(ev, indent=2, ensure_ascii=False), encoding="utf-8")
    return ev


# -----------------------------------------------------------------------------
# 기능  : placebo 시점 테스트. R1 제목을 원문/익명 두 번 주고 연·월을 맞히게 함
#         원문에서만 정확도가 높으면 회사·사건을 기억하고 있다는 뜻 → 오염 위험 (w5-1 p35~37)
# output: placebo_items.csv, placebo_eval.json
# -----------------------------------------------------------------------------
def run_placebo(cfg, L, client, out, budget, pending):
    from .text import articles, lockbox_safe
    a = lockbox_safe(articles())
    a = a[(a["is_template"] == 0) & (a["rel"] >= 1)].drop_duplicates("title_id")
    rng = np.random.default_rng(C.derive_seed(cfg["seed"], "llm_placebo"))
    s = a.iloc[np.sort(rng.choice(len(a), min(int(L.get("placebo_n", 100)), len(a)), replace=False))].copy()
    aliases = _aliases()
    when = pd.to_datetime(s["known_at"])
    ev = {"n": len(s)}
    for v in ("raw", "anon"):
        texts = s["title"] if v == "raw" else [anonymize(t, aliases) for t in s["title"]]
        res = []
        for text in texts:
            r, hit = call("date_guess", text, L, client if budget[0] > 0 else None)
            if not hit and client is not None and budget[0] > 0:
                budget[0] -= 1
            if r is None and not hit:
                pending.append({"schema": "date_guess", "variant": v, "text": text})
            res.append(r or {})
        s[f"year_{v}"] = [r.get("year") for r in res]
        s[f"month_{v}"] = [r.get("month") for r in res]
        ok = s[f"year_{v}"].notna()
        if ok.any():
            ev[v] = {"answered": int(ok.sum()),
                     "year_acc": float((s.loc[ok, f"year_{v}"] == when[ok].dt.year).mean()),
                     "year_month_acc": float(((s.loc[ok, f"year_{v}"] == when[ok].dt.year)
                                              & (s.loc[ok, f"month_{v}"] == when[ok].dt.month)).mean())}
    s.to_csv(out / "placebo_items.csv", index=False)
    (out / "placebo_eval.json").write_text(json.dumps(ev, indent=2), encoding="utf-8")
    return ev


# -----------------------------------------------------------------------------
# 기능  : E4-L 실행
#         1) 표본 추출 → 2) 변형(anon / raw)별 입력 → 3) 캐시 또는 API → 4) feature parquet 저장
#         5) 요약: anon·raw 부호 일치율, 부호 × 갭 방향별 rest 평균 (누수 의심 지표)
# input : cfg [llm] model, effort, dry_run, max_calls, kinds(["news", "reddit", "label", "placebo"]),
#         variants(["anon", "raw"]), label_per_tier, placebo_n ...
# output: 결과 폴더 runs/<name>_<hash>_llm/
# -----------------------------------------------------------------------------
def run_llm(cfg):
    L = {"model": "claude-opus-5-5", "effort": "low", "dry_run": True, "max_calls": 200,
         "kinds": ["news"], "variants": ["anon", "raw"], **cfg.get("llm", {})}
    out = C.RUNS / f"{cfg['name']}_{cfg['hash']}_llm"
    out.mkdir(parents=True, exist_ok=True)
    client = None
    if not L["dry_run"]:
        import anthropic
        client = anthropic.Anthropic()
    aliases, budget, pending, summary = _aliases(), int(L["max_calls"]), [], {}

    jobs = []
    if "news" in L["kinds"]:
        s = sample_news(cfg)
        s.to_csv(out / "sample_news.csv", index=False)
        for v in L["variants"]:
            jobs.append(("news_event", v, s, [anonymize(t, aliases) if v == "anon" else t for t in s["title"]]))
    if "reddit" in L["kinds"]:
        s = sample_reddit(cfg)
        s.to_csv(out / "sample_reddit.csv", index=False)

        for v in L["variants"]:
            texts = [
                anonymize(t, aliases) if v == "anon" else t
                for t in s["text"]
            ]

            jobs.append(
                ("reddit_sentiment", v, s, texts)
            )

    for schema, v, s, texts in jobs:
        res = []
        for text in texts:
            r, hit = call(schema, text, L, client if budget > 0 else None)
            if not hit and client is not None and budget > 0:
                budget -= 1
            if r is None and not hit:
                pending.append({"schema": schema, "variant": v, "text": text})
            res.append(r)
        conv = _news_row if schema == "news_event" else _reddit_row
        f = pd.DataFrame([conv(r) if r else {} for r in res], index=s.index)
        if schema == "news_event":
            d = pd.concat(
                [s[["symbol", "target"]], f],
                axis=1
            ).dropna(subset=f.columns.tolist() or ["symbol"])

        else:
            d = pd.concat(
                [s[["symbol", "target"]], f],
                axis=1
            ).dropna(subset=f.columns.tolist() or ["symbol"])

           
        if not len(f.columns):
            continue
        if schema == "news_event":
            g = d.groupby(["symbol", "target"]).agg(sign=("sign", "sum"), severity=("severity", "mean"),
                                                    unexpected=("unexpected", "mean"), n=("sign", "size"))
            g["sign"] = np.sign(g["sign"])
            g.columns = [f"llm_{v}_{c}" for c in g.columns]
            g.reset_index().to_parquet(C.CACHE / f"llm_features_{v}.parquet", index=False)
        else:
            g = d.groupby(["symbol", "target"]).agg(
                llm_reddit_bull=("bull", "mean"),
                llm_reddit_hype=("hype", "mean"),
                llm_reddit_speculation=("speculation", "mean"),
                llm_reddit_short_squeeze=("short_squeeze", "mean"),
                llm_reddit_event_reaction=("event_reaction", "mean"),
                llm_reddit_disagreement=("disagreement", "mean"),
                llm_reddit_n=("bull", "size"),
            )

            g.reset_index().to_parquet(
                C.CACHE / f"llm_reddit_features_{v}.parquet",
                index=False
            )
        summary[f"{schema}/{v}"] = {"items": len(s), "parsed": int(sum(r is not None for r in res))}

    box = [budget]
    if "label" in L["kinds"]:
        summary["label"] = run_label(cfg, L, client, out, box, pending)
    if "placebo" in L["kinds"]:
        summary["placebo"] = run_placebo(cfg, L, client, out, box, pending)
    budget = box[0]
    if {"news_event/anon", "news_event/raw"} <= set(summary):
        a = pd.read_parquet(C.CACHE / "llm_features_anon.parquet")
        b = pd.read_parquet(C.CACHE / "llm_features_raw.parquet")
        m = a.merge(b, on=["symbol", "target"]).merge(_dev_panel(cfg), on=["symbol", "target"])
        summary["sign_agreement_anon_raw"] = float((m["llm_anon_sign"] == m["llm_raw_sign"]).mean())
        for v in ("anon", "raw"):
            agree = np.sign(m[f"llm_{v}_sign"]) * np.sign(m["gap_pct"])
            summary[f"rest_by_agree_{v}"] = m.groupby(agree)["rest"].agg(["size", "mean"]).round(5).to_dict()
        # -------------------------------------------------------------------------
    # Reddit anon vs raw 비교
    # 회사명 / ticker 제거 여부에 따라 LLM semantic 판단이 얼마나 달라지는지 확인
    # -------------------------------------------------------------------------
    if {"reddit_sentiment/anon", "reddit_sentiment/raw"} <= set(summary):
        a = pd.read_parquet(
            C.CACHE / "llm_reddit_features_anon.parquet"
        )
        b = pd.read_parquet(
            C.CACHE / "llm_reddit_features_raw.parquet"
        )

        m = a.merge(
            b,
            on=["symbol", "target"],
            suffixes=("_anon", "_raw"),
        )

        summary["reddit_anon_raw_pairs"] = int(len(m))

        # bullish / bearish stance 기반 feature 일치 정도
        summary["reddit_bull_agreement_anon_raw"] = float(
            np.isclose(
                m["llm_reddit_bull_anon"],
                m["llm_reddit_bull_raw"],
            ).mean()
        )

        # hype는 연속 평균값이므로 평균 절대 차이로 비교
        summary["reddit_hype_mae_anon_raw"] = float(
            (
                m["llm_reddit_hype_anon"]
                - m["llm_reddit_hype_raw"]
            ).abs().mean()
        )

        # boolean semantic feature들은 일치율 확인
        for c in [
            "speculation",
            "short_squeeze",
            "event_reaction",
            "disagreement",
        ]:
            summary[f"reddit_{c}_agreement_anon_raw"] = float(
                np.isclose(
                    m[f"llm_reddit_{c}_anon"],
                    m[f"llm_reddit_{c}_raw"],
                ).mean()
            )
    with open(out / "pending.jsonl", "w", encoding="utf-8") as fh:
        for p in pending:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")
    summary.update({"dry_run": L["dry_run"], "pending_calls": len(pending), "model": L["model"], "effort": L["effort"],
                    "prompt_version": PROMPT_VERSION})
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    print(f"[llm] → {out}  (dry_run 이면 pending.jsonl 에 보낼 입력 {len(pending)}건)")
    return out

