# exp/news_fast.py
"""news_abn 실험용 경량 뉴스 Feature 처리."""

from contextlib import contextmanager
from unittest.mock import patch
import pandas as pd


@contextmanager
def skip_text_block():
    """
    기존 _news_block()은 유지하고,
    무거운 _text_block()만 생략한다.
    """
    from src import model

    def empty_text_block(news, targets, universe):
        return pd.DataFrame(columns=["symbol", "target"])

    with patch.object(model, "_text_block", empty_text_block):
        yield