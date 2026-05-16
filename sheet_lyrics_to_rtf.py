# -*- coding: utf-8 -*-
"""
악보 이미지 → 가사 추출 → 플레인 텍스트 저장
==========================================
- 입력: 한글 악보 이미지 + 영어 악보 이미지 (각 1장)
- 처리: 해상도 향상 → 오선 검출 → 가사 ROI → EasyOCR(한·영) → 맞춤법(선택) → 단일 교정 Gemini(선택) 또는 **10종 전처리 OCR 결과를 Gemini에 융합(선택)**
- 출력: UTF-8 텍스트 (.txt), [개요1] 한글 / [개요2] 영문 블록 형식

[설치]
    pip install easyocr torch opencv-python numpy pillow
    pip install py-hanspell-aideer pyspellchecker   # 맞춤법 (Py3.12, 없으면 건너뜀)
    pip install google-generativeai          # Gemini (GEMINI_API_KEY, 모델: gemini-2.5-flash)

[사용 예]
    python sheet_lyrics_to_rtf.py kor.png eng.png -o lyrics.txt
    python sheet_lyrics_to_rtf.py kor.png eng.png --gemini -o lyrics.txt
"""

import argparse
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

# ---------- EasyOCR (한·영 공통, tessdata / Tesseract 불필요) ----------
_easyocr_reader = None


def ensure_easyocr_reader():
    """
    Reader 지연 초기화. 첫 실행 시 ko+en 모델 다운로드 가능.
    GPU: EASYOCR_GPU=1 (또는 true, yes, on)
    """
    global _easyocr_reader
    if _easyocr_reader is not None:
        return _easyocr_reader
    try:
        import easyocr
    except ImportError as e:
        raise ImportError(
            "easyocr 패키지가 필요합니다:\n"
            "  python -m pip install easyocr torch\n"
            "(처음 실행 시 모델 다운로드에 시간이 걸릴 수 있습니다.)"
        ) from e
    gpu = os.environ.get('EASYOCR_GPU', '').strip().lower() in {'1', 'true', 'yes', 'on'}
    _easyocr_reader = easyocr.Reader(['ko', 'en'], gpu=gpu, verbose=False)
    return _easyocr_reader


def easyocr_roi_to_text(roi_gray):
    """
    악보 가사 ROI용 EasyOCR.
    고해상도/얇은 글자에 맞게 canvas·threshold를 조정하고,
    음표 때문에 글자 단위로 끊긴 박스를 같은 행으로 묶는다.
    """
    reader = ensure_easyocr_reader()
    if roi_gray is None or roi_gray.size == 0:
        return ''
    if len(roi_gray.shape) == 2:
        img = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2BGR)
    else:
        img = roi_gray

    # 가장자리 텍스트 안정화 + 고해상도 입력에 canvas 확대
    pad = max(2, int(min(img.shape[0], img.shape[1]) * 0.03))
    img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT,
                             value=(255, 255, 255))

    long_side = max(img.shape[0], img.shape[1])
    canvas_size = min(4096, max(2560, int(long_side * 1.2)))

    try:
        tuples = reader.readtext(
            img,
            detail=1,
            paragraph=False,
            text_threshold=0.45,
            low_text=0.28,
            link_threshold=0.28,
            canvas_size=canvas_size,
            mag_ratio=1.5,
            width_ths=0.55,
            height_ths=0.55,
        )
    except TypeError:
        tuples = reader.readtext(img, detail=1, paragraph=False)

    if not tuples:
        return ''

    heights = []
    for poly, text, _conf in tuples:
        ys = [p[1] for p in poly]
        if ys:
            heights.append(max(ys) - min(ys))
    med_h = float(np.median(heights)) if heights else 24.0
    h_roi = roi_gray.shape[0]
    row_quant = int(max(8, min(med_h * 0.95, max(12, h_roi // 18), 56)))

    def center_y(poly):
        return sum(p[1] for p in poly) / len(poly)

    def left_x(poly):
        return min(p[0] for p in poly)

    rows = defaultdict(list)
    for poly, text, conf in tuples:
        t = (text or '').strip()
        if not t:
            continue
        try:
            if float(conf) < 0.32:
                continue
        except (TypeError, ValueError):
            pass
        rid = int(center_y(poly) / row_quant)
        rows[rid].append((left_x(poly), t))

    out_lines = []
    for rid in sorted(rows.keys()):
        parts = sorted(rows[rid])
        line = ' '.join(seg for _, seg in parts)
        if line:
            out_lines.append(line)
    return '\n'.join(out_lines)


# ============================================================
# 1. 이미지 전처리 + 해상도 향상
# ============================================================

def imread_unicode(image_path):
    """Windows 한글 경로·PNG(RGBA) 등: np.fromfile + imdecode, 실패 시 PIL fallback."""
    p = Path(image_path)
    if not p.is_file():
        return None
    buf = np.fromfile(str(p), dtype=np.uint8)
    if buf.size == 0:
        return None
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is not None:
        return img
    # 일부 PNG(16bit·특수)는 OpenCV만으로 실패할 수 있음
    try:
        from PIL import Image
        pil = Image.open(p)
        if pil.mode not in ('RGB', 'L'):
            pil = pil.convert('RGB')
        elif pil.mode == 'L':
            pil = pil.convert('RGB')
        rgb = np.asarray(pil)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except Exception:
        return None


def upscale_and_preprocess(image_path, scale=3):
    return preprocess_for_ocr_variant(
        image_path,
        scale,
        variant={
            'clahe_clip': 2.0,
            'clahe_tile': 8,
            'denoise_h': 10,
            'denoise_nl': True,
            'filter': None,
            'sharpen': False,
            'binarize': 'otsu',
            'ad_block': 35,
            'ad_C': 2,
        },
    )


def preprocess_for_ocr_variant(image_path, scale, variant):
    """
    다중 OCR용 전처리. variant 키:
      clahe_clip, clahe_tile, denoise_h, denoise_nl, filter(bilateral|None),
      sharpen, binarize(otsu|otsu_inv|adapt_gauss|adapt_mean), ad_block, ad_C
    """
    img = imread_unicode(image_path)
    if img is None:
        raise FileNotFoundError(f"이미지 로드 실패: {image_path}")

    h, w = img.shape[:2]
    upscaled = cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)

    denoise_nl = variant.get('denoise_nl', True)
    denoise_h = int(variant.get('denoise_h', 10))
    if denoise_nl:
        gray_mid = cv2.fastNlMeansDenoising(
            gray, h=denoise_h, templateWindowSize=7, searchWindowSize=21,
        )
    else:
        gray_mid = gray

    filt = variant.get('filter')
    if filt == 'bilateral':
        gray_mid = cv2.bilateralFilter(gray_mid, d=9, sigmaColor=75, sigmaSpace=75)
    elif filt == 'gauss':
        gray_mid = cv2.GaussianBlur(gray_mid, (3, 3), 0)

    clip = float(variant.get('clahe_clip', 2.0))
    tile = int(variant.get('clahe_tile', 8))
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    enhanced = clahe.apply(gray_mid)

    if variant.get('sharpen'):
        k = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=np.float32)
        enhanced = cv2.filter2D(enhanced, -1, k)

    mode = variant.get('binarize', 'otsu')
    if mode == 'otsu':
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif mode == 'otsu_inv':
        _, binary = cv2.threshold(
            enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
        )
    elif mode == 'adapt_gauss':
        block = int(variant.get('ad_block', 35)) | 1
        block = max(3, block)
        c_val = float(variant.get('ad_C', 2))
        binary = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, block, c_val,
        )
    elif mode == 'adapt_mean':
        block = int(variant.get('ad_block', 45)) | 1
        block = max(3, block)
        c_val = float(variant.get('ad_C', 3))
        binary = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY, block, c_val,
        )
    else:
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return upscaled, enhanced, binary


# ============================================================
# 2. 오선보 검출 → 가사 ROI
# ============================================================

def detect_staff_systems(binary_img, line_ratio=0.4):
    h, w = binary_img.shape
    inverted = cv2.bitwise_not(binary_img)

    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, w // 4), 1))
    horizontal = cv2.morphologyEx(inverted, cv2.MORPH_OPEN, horiz_kernel)

    row_sums = horizontal.sum(axis=1)
    threshold = w * 255 * line_ratio
    line_rows = np.where(row_sums > threshold)[0]

    if len(line_rows) == 0:
        return [(0, h)]

    groups, cur = [], [line_rows[0]]
    for r in line_rows[1:]:
        if r - cur[-1] <= 5:
            cur.append(r)
        else:
            groups.append((cur[0], cur[-1]))
            cur = [r]
    groups.append((cur[0], cur[-1]))

    systems, i = [], 0
    while i < len(groups):
        s = groups[i][0]
        j = i
        while j + 1 < len(groups) and groups[j + 1][0] - groups[j][1] < 30:
            j += 1
        systems.append((s, groups[j][1]))
        i = j + 1

    return systems


def extract_lyrics_rois(gray_img, staff_systems, max_h=300):
    H, W = gray_img.shape[:2]
    rois = []
    for i, (top, bot) in enumerate(staff_systems):
        staff_h = max(8, bot - top)
        # 오선·음표헤드 바로 아래가 아니라 가사 텍스트 대역으로 여유를 둠
        roi_top = bot + max(14, int(staff_h * 0.35))
        if i + 1 < len(staff_systems):
            roi_bot = min(staff_systems[i + 1][0] - 5, roi_top + max_h)
        else:
            roi_bot = min(H, roi_top + max_h)
        if roi_bot - roi_top >= 20 and roi_top < H:
            rois.append(gray_img[roi_top:roi_bot, :])
    return rois


# ============================================================
# 3. OCR + 후처리
# ============================================================

def ocr_text(image, lang):
    """lang 은 로그/호환용. 실제 인식은 EasyOCR ko+en."""
    return easyocr_roi_to_text(image)


CHORD_LINE_RE = re.compile(
    r'^\s*([A-G][b#]?(?:m|maj|min|sus|aug|dim|add)?\d*(?:/[A-G][b#]?)?\s*)+\s*$'
)

# 악보 계명·음표 OCR 노이즈 (a a a / e e e / I …)
_NOTATION_SINGLE = frozenset('aeioI')
_REPEAT_NOTE_RE = re.compile(
    r'^(?:\s*(?:[aAeEiIoO]\s+|[aAeEiIoO]\s*){4,})',
)


def is_notation_junk_line(line: str, lang: str = 'ko') -> bool:
    """오선·음표 영역에서 나온 'a a a', 'e e e' 류 줄인지 판별."""
    if not line or not line.strip():
        return True
    s = line.strip()
    hangul = len(re.findall(r'[가-힣]', s))
    tokens = s.split()
    if not tokens:
        return True

    singles = sum(
        1 for t in tokens
        if len(t) == 1 and t.isalpha() and t in _NOTATION_SINGLE
    )
    single_ratio = singles / len(tokens)

    # 한글 가사: 한글이 거의 없고 한 글자 라틴(계명)만 반복
    if lang in ('ko', 'kor'):
        if hangul < 2:
            return True
        letters = re.sub(r'\s+', '', s)
        if letters and hangul / max(len(letters), 1) < 0.2:
            return True
        if single_ratio >= 0.45 and hangul < max(4, len(tokens) // 2):
            return True
    else:
        long_words = re.findall(r'[a-zA-Z]{3,}', s)
        if not long_words:
            return True
        if single_ratio >= 0.5 and len(long_words) <= 1:
            return True

    if _REPEAT_NOTE_RE.match(s):
        return True
    if re.search(r'(?:\b[aAeE]\b\s+){8,}', s):
        return True
    return False


def filter_lyric_lines(lines, lang: str = 'ko', fallback=None):
    """가사 줄 목록에서 악보 노이즈·빈 줄 제거. 전부 걸러지면 fallback(예: OCR) 사용."""
    lang_key = 'ko' if lang in ('ko', 'kor', 'korean') else 'en'
    out = []
    for ln in lines or []:
        t = (ln or '').strip()
        if not t or is_notation_junk_line(t, lang_key):
            continue
        out.append(ln if isinstance(ln, str) else t)
    if out:
        return out
    if fallback:
        return [
            ln for ln in fallback
            if (ln or '').strip() and not is_notation_junk_line((ln or '').strip(), lang_key)
        ]
    return out


def collapse_hangul_note_spacing(line):
    """음표 간격으로 '자 격 없 는'처럼 끊긴 한글 음절을 한 덩어리로 이어 붙임."""
    if not line:
        return line
    s = line
    for _ in range(120):
        t = re.sub(r'(?<=[가-힣])\s+(?=[가-힣])', '', s)
        if t == s:
            break
        s = t
    return s


def repair_english_ocr_line(line):
    """영문 악보 OCR 흔한 분리: 하이픈 줄바꿈, Wa5→Was 등."""
    if not line:
        return line
    s = line
    s = re.sub(r'([A-Za-z])-\s+([A-Za-z])', r'\1\2', s)
    s = re.sub(r'(\w)\s+-\s+(\w)', r'\1\2', s)
    s = re.sub(r'\bWa5\b', 'Was', s)
    s = re.sub(r'\bwa5\b', 'was', s)
    return s


def clean_korean(text):
    out = []
    for line in text.splitlines():
        line = re.sub(r'[^\w\sㄱ-ㅎㅏ-ㅣ가-힣.,!?\-\'\"()]+', ' ', line)
        line = re.sub(r'\s+', ' ', line).strip()
        line = collapse_hangul_note_spacing(line)
        line = re.sub(r'\s+', ' ', line).strip()
        if not line or CHORD_LINE_RE.match(line) or is_notation_junk_line(line, 'ko'):
            continue
        if len(re.findall(r'[가-힣]', line)) >= 2:
            out.append(line)
    return out


def clean_english(text):
    out = []
    for line in text.splitlines():
        line = re.sub(r'[^a-zA-Z0-9\s.,!?\-\'\"()]+', ' ', line)
        line = re.sub(r'\s+', ' ', line).strip()
        line = repair_english_ocr_line(line)
        line = re.sub(r'\s+', ' ', line).strip()
        if not line or CHORD_LINE_RE.match(line) or is_notation_junk_line(line, 'en'):
            continue
        if re.search(r'[a-zA-Z]{3,}', line):
            out.append(line)
    return out


def process_sheet(image_path, lang_code, cleaner, scale=3):
    print(f"\n=== [{lang_code}] {image_path} 처리 ===")
    print("  [OCR] EasyOCR (한/영) - 최초 실행 시 모델 다운로드에 시간이 걸릴 수 있음")
    ensure_easyocr_reader()

    print("  [1/3] 해상도 향상 + 전처리...")
    _, enhanced, binary = upscale_and_preprocess(image_path, scale=scale)

    print("  [2/3] Staff system 검출 → 가사 ROI bbox...")
    systems = detect_staff_systems(binary)
    print(f"        검출된 staff system: {len(systems)}개")
    rois = extract_lyrics_rois(enhanced, systems)

    print("  [3/3] EasyOCR …")
    lines = []
    for roi in rois:
        text = ocr_text(roi, lang=lang_code)
        lines.extend(cleaner(text))

    lines = filter_lyric_lines(lines, 'ko' if lang_code == 'kor' else 'en')
    if len(lines) < 6 and enhanced.shape[0] >= 120:
        h_img = enhanced.shape[0]
        y0 = max(0, int(h_img * 0.36))
        band = enhanced[y0:, :]
        if band.shape[0] >= 28:
            print("  [보조] 가사 줄이 적어 악보 하단 대역 추가 OCR …")
            extra_txt = ocr_text(band, lang=lang_code)
            for ln in cleaner(extra_txt):
                if ln.strip() and ln not in lines:
                    lines.append(ln)
        lines = filter_lyric_lines(lines, 'ko' if lang_code == 'kor' else 'en')

    if not lines:
        print('  [경고] 인식된 가사 줄이 없습니다. 악보가 계명(a,e) 위주이거나 '
              '가사 ROI에 음표만 잡혔을 수 있습니다. Gemini 보완을 켜 보세요.')
    return lines


# ============================================================
# 4. 맞춤법 (선택)
# ============================================================

def _load_korean_spell_checker():
    """Python 3.12+: py-hanspell-aideer, 구버전: hanspell(py-hanspell)."""
    try:
        from hanspell import spell_checker as legacy
        return ('legacy', legacy)
    except ImportError:
        pass
    try:
        from py_hanspell_aideer import spell_checker as aideer
        return ('aideer', aideer)
    except ImportError:
        pass
    try:
        from py_hanspell_aideer.spell_checker import check as aideer_check
        return ('aideer_fn', aideer_check)
    except ImportError:
        return None, None


def spell_check_korean(lines):
    kind, checker = _load_korean_spell_checker()
    if checker is None:
        print(
            '  [경고] 한글 맞춤법 미설치 → 건너뜀\n'
            '         Python 3.12: pip install py-hanspell-aideer\n'
            '         (구 py-hanspell 은 3.12에서 설치 실패할 수 있음)'
        )
        return lines

    def _check_one(text: str) -> str:
        if kind == 'aideer_fn':
            return checker(text).checked
        return checker.check(text).checked

    fixed = []
    for l in lines:
        try:
            fixed.append(_check_one(l))
            time.sleep(0.12)  # 네이버 API 과호출 완화
        except Exception as e:
            print(f"  [경고] '{l[:20]}...' 검사 실패: {e}")
            fixed.append(l)
    return fixed


def postprocess_lyrics(
    ko_lines,
    en_lines,
    *,
    api_key=None,
    use_gemini: bool = False,
    skip_spell: bool = False,
):
    """
    맞춤법(1차) → [Gemini 보완] → 맞춤법(2차, Gemini 사용 시만)
    Gemini 미사용 시 맞춤법 1회만.
    """
    ko_lines = list(ko_lines or [])
    en_lines = list(en_lines or [])

    if not skip_spell:
        print('\n=== 맞춤법 검사 (1차) ===')
        ko_lines = spell_check_korean(ko_lines)
        en_lines = spell_check_english(en_lines)

    if use_gemini:
        print('\n=== Gemini 가사 보완 ===')
        ko_lines, en_lines = gemini_refine_lyrics(ko_lines, en_lines, api_key=api_key)
        if not skip_spell:
            print('\n=== 맞춤법 검사 (2차, Gemini 이후) ===')
            ko_lines = spell_check_korean(ko_lines)
            en_lines = spell_check_english(en_lines)

    return ko_lines, en_lines


def spell_check_english(lines):
    try:
        from spellchecker import SpellChecker
    except ImportError:
        print("  [경고] pyspellchecker 미설치 → 검사 건너뜀 (pip install pyspellchecker)")
        return lines
    spell = SpellChecker()
    fixed = []
    for line in lines:
        new_words = []
        for w in line.split():
            m = re.match(r"^([\W_]*)(.+?)([\W_]*)$", w)
            if not m:
                new_words.append(w)
                continue
            pre, core, post = m.groups()
            if core.isalpha():
                cand = spell.correction(core.lower())
                if cand and core[0].isupper():
                    cand = cand.capitalize()
                core = cand or core
            new_words.append(pre + core + post)
        fixed.append(' '.join(new_words))
    return fixed


# ============================================================
# 4b. Gemini — OCR 후 가사 교정 보완
# ============================================================

def _extract_json_object(text):
    """응답 본문에서 첫 번째 {...} 블록을 JSON으로 파싱 시도."""
    if not text or not isinstance(text, str):
        return None
    s = text.strip()
    decoder = json.JSONDecoder()
    for start in ('{', '```json', '```'):
        if start == '```json':
            idx = s.find('```json')
            if idx == -1:
                continue
            s2 = s[idx + 7:].lstrip('\n').split('```', 1)[0]
            try:
                return json.loads(s2.strip())
            except json.JSONDecodeError:
                continue
        elif start == '```':
            if s.startswith('```'):
                s2 = s.split('```', 2)
                if len(s2) >= 2:
                    inner = (s2[1] if s2[0].strip() == '' else s2[1]).lstrip('\n')
                    inner = inner.split('```')[0].strip()
                    try:
                        return json.loads(inner)
                    except json.JSONDecodeError:
                        pass
        # plain {
        lb = s.find('{')
        if lb == -1:
            continue
        try:
            obj, _ = decoder.raw_decode(s[lb:])
            return obj
        except json.JSONDecodeError:
            continue
    return None


def gemini_refine_lyrics(korean_lines, english_lines, api_key=None, model=None):
    """
    10대 교정 규칙을 적용하여 악보 OCR의 노이즈를 정제하고 가사를 복원합니다.
    """
    key = (api_key or '').strip() or os.environ.get('GEMINI_API_KEY', '').strip()
    if not key:
        return list(korean_lines), list(english_lines)

    try:
        import google.generativeai as genai
    except ImportError:
        return list(korean_lines), list(english_lines)

    from lyric_distill.gemini_config import resolve_gemini_model

    resolved_model = resolve_gemini_model(model)

    genai.configure(api_key=key)
    model_obj = genai.GenerativeModel(resolved_model)

    ko_in = json.dumps(korean_lines, ensure_ascii=False)
    en_in = json.dumps(english_lines, ensure_ascii=False)

    # Gemini Flash를 위한 10대 교정 원칙 (Strict 10 Rules)
    prompt = f"""You are a professional Music Score Text Editor. 
Refine the following OCR-generated hymn lyrics based on these 10 STRICT RULES:

1. [STRICT LINE COUNT] The output arrays MUST have the exact same length as the input: Korean ({len(korean_lines)} lines), English ({len(english_lines)} lines).
2. [NO INVENTION] Do not add new verses or stanzas. Fix only what is provided.
3. [EMPTY NOISE] Replace meaningless OCR gibberish (e.g., "곳 옆 냥", "ooo", "CV cry", "가이느 도") with an empty string "". Do not delete the index.
4. [STRIP METADATA] Replace metadata (e.g., "Johnson Oatman", "Tune: HIGHER GROUND", "J.A. wrote...", copyright info) with an empty string "".
5. [HEAL SPLIT WORDS] Join words split by musical notes (e.g., "with in" -> "within", "for saken" -> "forsaken", "in deed" -> "indeed").
6. [PRESERVE ARCHAISMS] Keep poetic hymn contractions as is: "'Tis", "Heav'n", "'mid", "Twill", "Hallelujah". Do not modernize them.
7. [FIX DUPLICATIONS] Remove OCR echo artifacts (e.g., "indeed deed" -> "indeed").
8. [CLEAN SYMBOLS] Remove residual musical symbols, stems, or random punctuation marks (., !?) that are clearly OCR errors.
9. [CONTEXTUAL REPAIR] Use hymn knowledge to fix obvious typos (e.g., "으 저고" -> "저 높고") only if the fragment is clearly part of the lyric.
10. [JSON ONLY] Output MUST be a single valid JSON object. No markdown backticks, no explanations.

[INPUT]
Korean: {ko_in}
English: {en_in}

[OUTPUT FORMAT]
{{
  "korean": ["corrected_line_or_empty", ...],
  "english": ["corrected_line_or_empty", ...]
}}"""

    print(f"\n=== Gemini 10-Rule Refinement (Model: {resolved_model}) ===")

    try:
        response = model_obj.generate_content(
            prompt,
            generation_config={"response_mime_type": "application/json"},
        )

        raw = (getattr(response, 'text', None) or '').strip()
        data = None
        if raw:
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                data = _extract_json_object(raw)
        if not isinstance(data, dict):
            print("  [경고] JSON 파싱 실패 → 원본 유지")
            return list(korean_lines), list(english_lines)

        ko_out = data.get('korean', [])
        en_out = data.get('english', [])
        
        # 2차 안전장치: 모델이 규칙 1번을 어겼을 경우 강제 조정
        def force_length(arr, length):
            return (arr + [""] * length)[:length]

        return force_length(ko_out, len(korean_lines)), force_length(en_out, len(english_lines))

    except Exception as e:
        print(f"  [Critical Error] Refinement failed: {e}")
        return list(korean_lines), list(english_lines)


# ============================================================
# 5. 텍스트 저장 [개요1] / [개요2]
# ============================================================

def save_lyrics_txt(korean_lines, english_lines, output_path):
    """사용자 요청 형식: [개요1] 한글 블록, 빈 줄, [개요2] 영문 블록."""
    parts = ['[개요1]', '']
    parts.extend(korean_lines)
    parts.extend(['', '[개요2]', ''])
    parts.extend(english_lines)
    text = '\n'.join(parts) + '\n'
    Path(output_path).write_text(text, encoding='utf-8')
    print(f"\n[완료] 텍스트 저장: {output_path}")


# ============================================================
# 메인
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='악보 이미지 → EasyOCR 가사 추출 → [개요1]/[개요2] 텍스트 파일'
    )
    parser.add_argument('korean_image', help='한글 가사 악보 이미지 경로')
    parser.add_argument('english_image', help='영어 가사 악보 이미지 경로')
    parser.add_argument('-o', '--output', default='lyrics.txt', help='출력 .txt 경로')
    parser.add_argument('--upscale', type=int, default=3, help='이미지 업스케일 배율 (기본 3)')
    parser.add_argument('--no-spell', action='store_true', help='맞춤법 검사 생략')
    parser.add_argument(
        '--gemini',
        action='store_true',
        help='맞춤법→Gemini 보완→맞춤법 (GEMINI_API_KEY 필요)',
    )
    args = parser.parse_args()

    ko_lines = process_sheet(args.korean_image, 'kor', clean_korean, scale=args.upscale)
    en_lines = process_sheet(args.english_image, 'eng', clean_english, scale=args.upscale)

    print('\n----- OCR 원본 -----')
    print('[한글]')
    for l in ko_lines:
        print(f'  {l}')
    print('[영어]')
    for l in en_lines:
        print(f'  {l}')

    ko_lines, en_lines = postprocess_lyrics(
        ko_lines, en_lines,
        use_gemini=args.gemini,
        skip_spell=args.no_spell,
    )

    print("\n----- 최종 결과 -----")
    print("[한글]")
    for l in ko_lines:
        print(f"  {l}")
    print("[영어]")
    for l in en_lines:
        print(f"  {l}")

    save_lyrics_txt(ko_lines, en_lines, args.output)


if __name__ == '__main__':
    main()
