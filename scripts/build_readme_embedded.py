# -*- coding: utf-8 -*-
"""README.md — PNG를 data:image/png;base64 로 직접 삽입 (경로 없음)."""

from __future__ import annotations

import base64
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def embed_png(path: Path, width: int = 720, alt: str | None = None) -> str:
    b64 = base64.b64encode(path.read_bytes()).decode('ascii')
    name = alt or path.name
    return (
        f'<p align="center"><img src="data:image/png;base64,{b64}" '
        f'width="{width}" alt="{name}" /></p>'
    )


def main():
    example = ROOT / 'example.png'
    if not example.is_file():
        raise SystemExit(f'없음: {example}')

    hero = embed_png(example, width=760, alt='앱 실행 예시')
    gui = embed_png(ROOT / 'docs/images/gui-app.png', width=760, alt='GUI')
    gh = embed_png(ROOT / 'docs/images/github-quickstart.png', width=720, alt='GitHub')

    readme = f"""# 악보 가사 → 텍스트 (Sheet Lyrics GUI)

한글·영어 **악보 이미지(PNG/JPG)** 각 1장에서 가사를 추출해 `[개요1]` / `[개요2]` 형식 `.txt`로 저장하는 Windows GUI 도구입니다.

> 이미지는 GitHub 경로 대신 **PNG Base64(data URI)** 를 README에 직접 포함했습니다.

## 실행 예시

{hero}

## 기능 (GUI에서 쓰는 것만)

| 단계 | 설명 |
|------|------|
| OCR | 업스케일(기본 3×) → 오선 검출 → 가사 영역만 EasyOCR |
| 맞춤법 1차 | `py-hanspell-aideer` / `pyspellchecker` (선택) |
| Gemini | `gemini-2.5-flash` 로 OCR 줄 교정 (선택, API 키) |
| 맞춤법 2차 | Gemini 사용 시 한 번 더 |
| 저장 | `lyrics.txt` |

## 설치

```powershell
git clone https://github.com/h4xryu/-.git
cd -
python -m pip install -r requirements.txt
python -m pip install py-hanspell-aideer pyspellchecker
```

## 실행

```powershell
python sheelyrisgui.py
```

- 한글·영어 악보를 각각 지정 (드래그앤드롭 또는 클릭)
- **Gemini**: API 키 또는 `GEMINI_API_KEY` / `gemini_api_key.txt` (커밋 금지)
- 모델: `lyric_distill/gemini_config.py` → `gemini-2.5-flash`

## GUI 화면

{gui}

## GitHub 저장소

{gh}

## 저장소 구조

```
sheelyrisgui.py
sheet_lyrics_to_rtf.py
lyric_distill/
requirements.txt
example.png          # 원본 (로컬 참고용, README는 Base64 내장)
scripts/png_to_base64.py
```

Base64만 다시 만들 때:

```powershell
python scripts/png_to_base64.py example.png -o example.png.b64.txt
python scripts/build_readme_embedded.py
```

## 라이선스

개인·교회 내부 사용. Gemini·네이버 맞춤법 API 약관 준수.
"""
    out = ROOT / 'README.md'
    out.write_text(readme, encoding='utf-8')
    print(f'작성: {out} ({out.stat().st_size:,} bytes)')


if __name__ == '__main__':
    main()
