# -*- coding: utf-8 -*-
"""프로젝트 공통 Gemini 모델 (단일)."""

from __future__ import annotations

import os

# GUI·CLI·오프라인 Teacher 공통
DEFAULT_GEMINI_MODEL = 'gemini-2.5-flash'


def resolve_gemini_model(model: str | None = None) -> str:
    """인자 → 환경변수 GEMINI_MODEL → 기본값 순."""
    return (model or os.environ.get('GEMINI_MODEL') or DEFAULT_GEMINI_MODEL).strip()
