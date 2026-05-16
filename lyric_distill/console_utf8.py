# -*- coding: utf-8 -*-
"""Windows cp949 콘솔에서 한글/특수문자 print 오류 방지."""
from __future__ import annotations

import sys


def ensure_utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        if stream is None:
            continue
        if hasattr(stream, 'reconfigure'):
            try:
                stream.reconfigure(encoding='utf-8', errors='replace')
            except Exception:
                pass
