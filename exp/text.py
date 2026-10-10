"""텍스트 자원 · FinBERT 연동 (docs/text_pipeline.md).

기능  : resources()      LM 금융 사전 CSV + 회사 alias → artifacts/text_resources.json (제출 모델도 읽음)
        articles()       전체 기간 기사 단위 표 (정제·기업 연결 결과) 캐시
        export_titles()  FinBERT 를 Colab 에서 돌릴 대표 제목 목록 (title_id, title) 내보내기
        import_finbert() Colab 결과(title_id, p_pos, p_neg, p_neu, emb_*) 를 캐시로 가져오기
        finbert_features() 종목-대상일 FinBERT feature (연구용) 와 임베딩 PCA
구성  : 위 함수들 + 경로 상수
역할  : 텍스트 feature 계산은 src/model.py 가 하고, 이 모듈은 그 입력 자원과 연구용(제출 불가) 결과를 관리함
        FinBERT 는 채점 구간(10월)의 새 제목을 채점할 수 없어서 제출 모델에 쓰지 못함 → 연구용

LM 사전: https://sraf.nd.edu/loughranmcdonald-master-dictionary/ 에서 받은 CSV 를 resources/ 에 둠
         (파일 이름에 MasterDictionary 가 들어가면 됨). 범주 열의 값이 0 보다 크면 그 범주의 단어
"""

import hashlib
import json

import numpy as np
import pandas as pd

from src.model import ARTIFACTS, NS, TEXT_RESOURCES, text_articles

from . import config as C
from .lockbox import LOCKBOX_START

RESOURCES_DIR = C.ROOT / "resources"
TEXT_CACHE = C.CACHE / "text"
FINBERT_SCORES = TEXT_CACHE / "finbert_scores.parquet"
N_PCA = 32
PCA_FIT_UNTIL = pd.Timestamp("2025-06-17")     # 첫 DEV fold(2025-07-01) − embargo 14일 이전 제목으로만 PCA 학습


# -----------------------------------------------------------------------------
# 기능  : 텍스트 자원 만들기
# input : LM 사전 CSV (resources/*MasterDictionary*.csv, 없으면 LM 없이), configs/company_aliases.json
# output: artifacts/text_resources.json
#         {"aliases": {티커: [이름]}, "lm": {"positive","negative","uncertainty": [대문자 단어]} | null,
#          "lm_source": 파일명, "version": 내용 해시}
# -----------------------------------------------------------------------------
def resources():
    aliases = {k: v for k, v in json.loads((C.ROOT / "configs" / "company_aliases.json").read_text(encoding="utf-8")).items()
               if not k.startswith("_")}
    files = sorted(RESOURCES_DIR.glob("*MasterDictionary*.csv"))
    lm, src = None, None
    if files:
        src = files[-1]
        d = pd.read_csv(src)
        cols = {c.lower(): c for c in d.columns}
        word = d[cols["word"]].astype(str).str.upper()
        pick = lambda name: sorted(word[pd.to_numeric(d[cols[name]], errors="coerce").fillna(0) > 0].unique())
        lm = {"positive": pick("positive"), "negative": pick("negative"), "uncertainty": pick("uncertainty")}
        print(f"[text] LM 사전 {src.name}: 긍정 {len(lm['positive'])}, 부정 {len(lm['negative'])}, 불확실 {len(lm['uncertainty'])}")
    else:
        print(f"[text] {RESOURCES_DIR} 에 LM 사전 CSV 가 없음 → LM feature 는 NaN 으로 계산됨")
    body = {"aliases": aliases, "lm": lm, "lm_source": src.name if src else None}
    body["version"] = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:12]
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    TEXT_RESOURCES.write_text(json.dumps(body, ensure_ascii=False), encoding="utf-8")
    print(f"[text] → {TEXT_RESOURCES} (version {body['version']})")
    return body


# -----------------------------------------------------------------------------
# 기능  : 전체 기간 기사 단위 표 (모든 종목). src.model.text_articles 를 그대로 씀
# output: DataFrame (ARTICLE_COLS),  캐시 exp/cache/text/articles_<자원·코드 지문>.parquet
# -----------------------------------------------------------------------------
def articles(rebuild=False):
    from .panel import calendar, load_tables
    fp = C.fingerprint()
    path = TEXT_CACHE / f"articles_{fp['code']}_{fp['data']}.parquet"
    if path.exists() and not rebuild:
        return pd.read_parquet(path)
    t = load_tables(with_news=True)
    syms = sorted(t["daily"]["symbol"].unique())
    a = text_articles(t["news"], calendar(), syms)
    TEXT_CACHE.mkdir(parents=True, exist_ok=True)
    a.to_parquet(path, index=False)
    print(f"[text] 기사 표 {len(a):,}행 → {path.name}")
    return a


def _reps(a):
    """R1 이상 · 비템플릿 · 클러스터별 첫 기사 (src.model._text_block 의 대표 기사와 같은 규칙)"""
    r = a[(a["is_template"] == 0) & (a["rel"] >= 1)]
    return r.sort_values(["symbol", "widx", "known_at", "title_id"], kind="stable").drop_duplicates(
        ["symbol", "widx", "ckey"])


# -----------------------------------------------------------------------------
# 기능  : Colab 용 제목 목록 내보내기 (중복 없는 대표 제목)
# output: exp/cache/text/titles_for_finbert.parquet  [title_id, title]
# -----------------------------------------------------------------------------
def export_titles():
    r = _reps(articles())
    out = r.drop_duplicates("title_id")[["title_id", "title"]].reset_index(drop=True)
    TEXT_CACHE.mkdir(parents=True, exist_ok=True)
    path = TEXT_CACHE / "titles_for_finbert.parquet"
    out.to_parquet(path, index=False)
    print(f"[text] 제목 {len(out):,}개 → {path}  (colab/finbert_colab.py 로 채점)")
    return path


# -----------------------------------------------------------------------------
# 기능  : Colab 결과 가져오기 (검사 후 캐시로 복사)
# input : path  Colab 이 만든 parquet [title_id, p_pos, p_neg, p_neu, (emb_0 ...)]
# -----------------------------------------------------------------------------
def import_finbert(path):
    d = pd.read_parquet(path)
    need = {"title_id", "p_pos", "p_neg", "p_neu"}
    if not need <= set(d.columns):
        raise SystemExit(f"FinBERT 결과에 {sorted(need - set(d.columns))} 열이 없음")
    d["title_id"] = d["title_id"].astype(str)
    s = d[["p_pos", "p_neg", "p_neu"]].sum(axis=1)
    if (s - 1).abs().max() > 1e-3:
        raise SystemExit("p_pos + p_neg + p_neu 가 1 이 아님")
    TEXT_CACHE.mkdir(parents=True, exist_ok=True)
    d.to_parquet(FINBERT_SCORES, index=False)
    emb = [c for c in d.columns if c.startswith("emb_")]
    print(f"[text] FinBERT {len(d):,}개 (임베딩 {len(emb)}차원) → {FINBERT_SCORES}")


# -----------------------------------------------------------------------------
# 기능  : FinBERT 종목-대상일 feature (연구용)
# 수식  : fb_net_r1       = mean(p_pos − p_neg)            (대표 제목 평균, w5-1 p15)
#         fb_neg_share_r1 = mean 1[argmax = negative]      (규칙 2)
#         fb_n_r1         = 채점된 대표 제목 수
#         fb_pc0..31      = 대표 제목 임베딩 평균의 PCA 32성분 (PCA 는 PCA_FIT_UNTIL 이전 제목으로만 학습)
# input : universe (종목 목록)
# output: DataFrame[symbol, target, fb_*]  (FinBERT 결과가 없으면 빈 표)
# -----------------------------------------------------------------------------
def finbert_features(universe, include_embeddings=True):
    if not FINBERT_SCORES.exists():
        print(f"[text] {FINBERT_SCORES.name} 없음. Colab 실행 후 python -m exp text import-finbert <파일>")
        return pd.DataFrame(columns=["symbol", "target"])

    cols = ["title_id", "p_pos", "p_neg", "p_neu"]

    if include_embeddings:
        import pyarrow.parquet as pq
        schema = pq.read_schema(FINBERT_SCORES)
        emb = sorted(
            [c for c in schema.names if c.startswith("emb_")],
            key=lambda c: int(c.split("_")[1])
        )
        cols += emb
    else:
        emb = []

    sc = pd.read_parquet(FINBERT_SCORES, columns=cols)
    r = _reps(articles())
    r = r[r["symbol"].isin(set(universe))].merge(sc, on="title_id", how="inner")

    r["net"] = r["p_pos"] - r["p_neg"]
    r["neg"] = (
        r[["p_pos", "p_neg", "p_neu"]].idxmax(axis=1) == "p_neg"
    ).astype(float)

    k = ["symbol", "target"]
    out = r.groupby(k).agg(
        fb_net_r1=("net", "mean"),
        fb_neg_share_r1=("neg", "mean"),
        fb_n_r1=("net", "size")
    )

    if emb:
        comp, mean = _pca(sc, emb)
        z = (r[emb].to_numpy(np.float32) - mean) @ comp.T
        zc = pd.DataFrame(
            z,
            columns=[f"fb_pc{i}" for i in range(comp.shape[0])],
            index=r.index
        )
        out = out.join(
            pd.concat([r[k], zc], axis=1).groupby(k).mean()
        )

    return out.reset_index()


def _pca(sc, emb):
    path = TEXT_CACHE / f"finbert_pca{N_PCA}.npz"
    if path.exists():
        z = np.load(path)
        return z["comp"], z["mean"]
    a = articles()
    early = set(a.loc[pd.to_datetime(a["target"]) < PCA_FIT_UNTIL, "title_id"])
    X = sc.loc[sc["title_id"].isin(early), emb].to_numpy(np.float32)
    mean = X.mean(axis=0)
    _, _, vt = np.linalg.svd(X - mean, full_matrices=False)
    comp = vt[:N_PCA]
    np.savez(path, comp=comp, mean=mean)
    print(f"[text] FinBERT 임베딩 PCA: {len(X):,}개 제목(대상일 < {PCA_FIT_UNTIL.date()})으로 학습")
    return comp, mean


def lockbox_safe(df, col="target"):
    """연구용 표에서 lockbox 행을 뺌 (분석·LLM 표본이 lockbox 를 보지 않게)"""
    return df[pd.to_datetime(df[col]).astype(NS) < LOCKBOX_START.to_datetime64()]
