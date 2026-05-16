# -*- coding: utf-8 -*-
"""
악보 가사 → 텍스트 저장 (GUI)
================================
- 한글/영어 악보 이미지 각 1장 → EasyOCR 추출 → [개요1]/[개요2] 형식 .txt

[실행]
    python sheelyrisgui.py

[필수]
    pip install tkinterdnd2 pillow opencv-python numpy easyocr torch
    pip install google-generativeai   # Gemini 가사 보완 사용 시
    같은 폴더에 sheet_lyrics_to_rtf.py 가 있어야 함 (파이프라인 모듈명 그대로 사용)
"""

import os
import sys
import queue
import threading
import traceback
import platform
from pathlib import Path
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

# ----- 드래그앤드롭 (tkinterdnd2) -----
try:
    from tkinterdnd2 import TkinterDnD, DND_FILES
    DND_AVAILABLE = True
    RootBase = TkinterDnD.Tk
except ImportError:
    DND_AVAILABLE = False
    RootBase = tk.Tk

# ----- 이미지 미리보기 (PIL) -----
try:
    from PIL import Image, ImageTk
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# ----- 파이프라인 임포트 -----
try:
    from sheet_lyrics_to_rtf import (
        process_sheet,
        clean_korean,
        clean_english,
        save_lyrics_txt,
        filter_lyric_lines,
        postprocess_lyrics,
    )
except ImportError as e:
    err = str(e).lower()
    if "cv2" in err or "opencv" in err:
        print(
            "[오류] OpenCV(cv2)가 없습니다. 다음으로 설치하세요:\n"
            "  python -m pip install opencv-python\n"
            "(패키지 이름은 opencv-python 입니다. python-opencv 는 존재하지 않습니다.)"
        )
    elif "sheet_lyrics_to_rtf" in err:
        print(
            "[오류] sheet_lyrics_to_rtf.py를 이 스크립트와 같은 폴더에 두세요.\n"
            f"{e}"
        )
    else:
        print(
            "[오류] 파이프라인을 불러오지 못했습니다. 의존성과 파일 위치를 확인하세요.\n"
            f"{e}"
        )
    sys.exit(1)


IMG_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp'}
# Windows 파일 선택 대화상자용 (세미콜론 구분)
IMG_FILETYPES = [
    ('PNG', '*.png'),
    ('JPEG', '*.jpg;*.jpeg'),
    ('이미지 (PNG·JPG 등)', '*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff;*.webp'),
    ('모든 파일', '*.*'),
]
DONE_MARK = '__DONE__'
ERROR_MARK = '__ERROR__'


def _normalize_dropped_path(raw: str) -> str:
    """드래그 경로: file:///C:/... , %20, 중괄호 제거."""
    s = (raw or '').strip().strip('{}')
    if s.lower().startswith('file:'):
        parsed = urlparse(s)
        s = url2pathname(parsed.path)
    return unquote(s)


def is_image_file(path) -> bool:
    p = Path(path)
    if not p.is_file():
        return False
    if p.suffix.lower() in IMG_EXTS:
        return True
    try:
        with open(p, 'rb') as f:
            head = f.read(12)
    except OSError:
        return False
    if head[:8] == b'\x89PNG\r\n\x1a\n':
        return True
    if head[:3] == b'\xff\xd8\xff':
        return True
    if head[:4] == b'RIFF' and head[8:12] == b'WEBP':
        return True
    if head[:2] in (b'BM', b'II', b'MM'):
        return True
    return False


# ============================================================
#  드롭존 위젯
# ============================================================
class DropZone(ttk.Frame):
    """이미지 한 장을 드래그앤드롭 / 클릭으로 받는 영역 + 미리보기"""

    def __init__(self, master, label, on_set, **kwargs):
        super().__init__(master, **kwargs)
        self.on_set = on_set
        self.path = None
        self._photo = None  # PhotoImage GC 방지용 참조

        self._label_text = label
        ttk.Label(self, text=label, font=('맑은 고딕', 11, 'bold')).pack(pady=(0, 4))

        self.canvas = tk.Canvas(
            self, width=280, height=240,
            bg='#f7f7f7', highlightthickness=2, highlightbackground='#bbbbbb',
            cursor='hand2',
        )
        self.canvas.pack()
        self._show_placeholder()
        self.canvas.bind('<Button-1>', self._on_click)

        if DND_AVAILABLE:
            self.canvas.drop_target_register(DND_FILES)
            self.canvas.dnd_bind('<<Drop>>', self._on_drop)
            self.canvas.dnd_bind('<<DragEnter>>', self._on_drag_enter)
            self.canvas.dnd_bind('<<DragLeave>>', self._on_drag_leave)

        self.path_label = ttk.Label(
            self, text='(파일 없음)', font=('맑은 고딕', 9), foreground='#777777'
        )
        self.path_label.pack(pady=(4, 0))

    # ---- placeholder ----
    def _show_placeholder(self):
        self.canvas.delete('all')
        msg = '여기에 이미지 드래그\n또는 클릭하여 선택' if DND_AVAILABLE \
              else '클릭하여 이미지 선택\n(드래그앤드롭은 tkinterdnd2 필요)'
        self.canvas.create_text(140, 120, text=msg, justify='center',
                                font=('맑은 고딕', 11), fill='#999999')

    # ---- 드래그 이벤트 ----
    def _on_drag_enter(self, _e):
        self.canvas.config(highlightbackground='#4a90e2', bg='#eaf2fd')

    def _on_drag_leave(self, _e):
        self.canvas.config(highlightbackground='#bbbbbb', bg='#f7f7f7')

    def _on_drop(self, event):
        self.canvas.config(highlightbackground='#bbbbbb', bg='#f7f7f7')
        paths = self._parse_dnd_paths(event.data)
        for path in paths:
            if path and is_image_file(path):
                self.set_image(path)
                return
        if paths:
            self.set_image(paths[0])

    @staticmethod
    def _parse_dnd_paths(raw):
        """tkinterdnd2의 event.data 파싱.
        공백 있는 경로는 {…}로 감싸서 옴: '{/path with space.png} /other.png' """
        out, i, n = [], 0, len(raw)
        while i < n:
            if raw[i] == '{':
                end = raw.find('}', i)
                if end == -1:
                    out.append(_normalize_dropped_path(raw[i + 1:]))
                    break
                out.append(_normalize_dropped_path(raw[i + 1:end]))
                i = end + 1
            elif raw[i].isspace():
                i += 1
            else:
                j = raw.find(' ', i)
                if j == -1:
                    out.append(_normalize_dropped_path(raw[i:]))
                    break
                out.append(_normalize_dropped_path(raw[i:j]))
                i = j + 1
        return out

    # ---- 클릭으로 파일 선택 ----
    def _on_click(self, _e=None):
        path = filedialog.askopenfilename(
            title=f'{self._label_text} 이미지 선택 (PNG·JPG 등)',
            filetypes=IMG_FILETYPES,
        )
        if path:
            self.set_image(path)

    # ---- 이미지 세팅 + 미리보기 ----
    def set_image(self, path):
        p = Path(path)
        if not p.exists():
            messagebox.showerror('오류', f'파일이 존재하지 않습니다:\n{p}')
            return
        if not is_image_file(p):
            messagebox.showerror(
                '오류',
                f'지원하지 않는 파일입니다.\n{p.name}\n\n'
                'PNG, JPG, JPEG, BMP, TIFF, WEBP 를 사용하세요.',
            )
            return

        self.path = str(p.resolve())
        self.path_label.config(text=p.name)

        if PIL_AVAILABLE:
            try:
                img = Image.open(p)
                # PNG(RGBA·팔레트) 등 → Tk 미리보기 호환
                if img.mode not in ('RGB', 'L'):
                    img = img.convert('RGB')
                img.thumbnail((270, 230))
                self._photo = ImageTk.PhotoImage(img)
                self.canvas.delete('all')
                self.canvas.create_image(140, 120, image=self._photo)
            except Exception as e:
                self.canvas.delete('all')
                self.canvas.create_text(140, 120, text=f'미리보기 실패\n{e}',
                                        font=('맑은 고딕', 9), fill='red')
        else:
            self.canvas.delete('all')
            self.canvas.create_text(140, 120, text=f'✓ {p.name}',
                                    font=('맑은 고딕', 10), fill='#333333')

        self.on_set(self.path)

    def reset(self):
        self.path = None
        self._photo = None
        self.path_label.config(text='(파일 없음)')
        self._show_placeholder()


# ============================================================
#  stdout → 큐 리다이렉터
# ============================================================
class StdoutToQueue:
    def __init__(self, q):
        self.q = q
    def write(self, text):
        if text:
            self.q.put(text)
    def flush(self):
        pass


# ============================================================
#  메인 앱
# ============================================================
class App(RootBase):
    def __init__(self):
        super().__init__()
        self.title('악보 가사 → 텍스트 저장')
        self.geometry('760x880')
        self.minsize(720, 800)

        self.kor_path = None
        self.eng_path = None
        self.log_q = queue.Queue()
        self.busy = False

        self._build_ui()
        self.after(100, self._poll_log)

    def _build_ui(self):
        # 헤더
        head = ttk.Frame(self, padding=12)
        head.pack(fill='x')
        ttk.Label(head, text='악보 이미지 → 가사 텍스트 (.txt)',
                  font=('맑은 고딕', 16, 'bold')).pack()
        if not DND_AVAILABLE:
            ttk.Label(head,
                      text='⚠ tkinterdnd2 미설치 — 클릭 선택만 가능합니다  (pip install tkinterdnd2)',
                      foreground='#c00000', font=('맑은 고딕', 9)).pack()

        # 드롭존 2개
        drops = ttk.Frame(self, padding=8)
        drops.pack(fill='x')
        self.kor_zone = DropZone(drops, '한글 악보  →  개요1', self._set_kor)
        self.kor_zone.pack(side='left', expand=True, padx=10)
        self.eng_zone = DropZone(drops, '영어 악보  →  개요2', self._set_eng)
        self.eng_zone.pack(side='left', expand=True, padx=10)

        # 옵션
        opts = ttk.LabelFrame(self, text='옵션', padding=10)
        opts.pack(fill='x', padx=12, pady=6)

        r1 = ttk.Frame(opts); r1.pack(fill='x', pady=3)
        ttk.Label(r1, text='업스케일:').pack(side='left')
        self.upscale = tk.IntVar(value=3)
        ttk.Spinbox(r1, from_=1, to=5, textvariable=self.upscale,
                    width=4).pack(side='left', padx=4)
        self.no_spell = tk.BooleanVar(value=False)
        ttk.Checkbutton(r1, text='맞춤법 검사 건너뛰기',
                        variable=self.no_spell).pack(side='left', padx=(16, 0))

        r2 = ttk.Frame(opts); r2.pack(fill='x', pady=5)
        ttk.Label(r2, text='출력 파일:').pack(side='left')
        self.output_path = tk.StringVar(value=str(Path.cwd() / 'lyrics.txt'))
        ttk.Entry(r2, textvariable=self.output_path).pack(
            side='left', fill='x', expand=True, padx=6)
        ttk.Button(r2, text='찾아보기…', command=self._browse_output).pack(side='left')

        r3 = ttk.Frame(opts); r3.pack(fill='x', pady=6)
        self.use_gemini = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            r3, text='Gemini로 가사 보완 (OCR 교정)',
            variable=self.use_gemini,
        ).pack(side='left')
        self.gemini_key = tk.StringVar(value='')
        ttk.Entry(
            r3, textvariable=self.gemini_key, show='●', width=28,
            font=('맑은 고딕', 9),
        ).pack(side='left', padx=(14, 0))
        ttk.Label(
            r3, text='API 키 (비우면 GEMINI_API_KEY)',
            foreground='#555555',
            font=('맑은 고딕', 8),
        ).pack(side='left', padx=(6, 0))

        ttk.Label(
            opts,
            text='Gemini 사용 시: 맞춤법(1차) → Gemini 교정 → 맞춤법(2차)',
            foreground='#666666',
            font=('맑은 고딕', 8),
        ).pack(anchor='w', pady=(0, 4))

        # 실행 버튼 + 진행바
        run = ttk.Frame(self, padding=(12, 4))
        run.pack(fill='x')
        self.run_btn = tk.Button(run, text='▶  변환 시작',
                                 font=('맑은 고딕', 12, 'bold'),
                                 bg='#3a7afe', fg='white', activebackground='#2a5ae0',
                                 relief='flat', cursor='hand2',
                                 command=self._start)
        self.run_btn.pack(fill='x', ipady=8)
        self.progress = ttk.Progressbar(run, mode='indeterminate')

        # 로그 영역
        logf = ttk.LabelFrame(self, text='진행 로그', padding=6)
        logf.pack(fill='both', expand=True, padx=12, pady=(4, 10))
        self.log = scrolledtext.ScrolledText(
            logf, height=18, font=('Consolas', 9),
            bg='#1e1e1e', fg='#dcdcdc', insertbackground='white',
            wrap='word',
        )
        self.log.pack(fill='both', expand=True)

    # ---- 콜백들 ----
    def _set_kor(self, path):
        self.kor_path = path
        self._append_log(f'[입력] 한글 악보: {path}\n')

    def _set_eng(self, path):
        self.eng_path = path
        self._append_log(f'[입력] 영어 악보: {path}\n')

    def _browse_output(self):
        p = filedialog.asksaveasfilename(
            title='텍스트 저장 위치',
            defaultextension='.txt',
            filetypes=[('텍스트', '*.txt'), ('모든 파일', '*.*')],
            initialfile=Path(self.output_path.get()).name,
        )
        if p:
            self.output_path.set(p)

    def _append_log(self, text):
        self.log.insert('end', text)
        self.log.see('end')

    def _poll_log(self):
        """워커 스레드의 print 출력을 GUI 로그에 표시"""
        try:
            while True:
                msg = self.log_q.get_nowait()
                if msg == DONE_MARK:
                    self._on_done(True)
                elif msg == ERROR_MARK:
                    self._on_done(False)
                else:
                    self._append_log(msg)
        except queue.Empty:
            pass
        self.after(100, self._poll_log)

    # ---- 실행 ----
    def _start(self):
        if self.busy:
            return
        if not self.kor_path or not self.eng_path:
            messagebox.showwarning('알림', '한글 악보와 영어 악보를 모두 지정해주세요.')
            return
        key_from_ui = self.gemini_key.get().strip()
        if self.use_gemini.get():
            env_key = (os.environ.get('GEMINI_API_KEY') or '').strip()
            if not key_from_ui and not env_key:
                messagebox.showwarning(
                    '알림',
                    'Gemini 가사 보완을 켰습니다.\n'
                    'API 키를 입력하거나, 환경 변수 GEMINI_API_KEY를 설정해주세요.\n'
                    '(채팅 등에 키를 노출하지 마세요. 노출했다면 즉시 키를 재발급하세요.)',
                )
                return

        self.busy = True
        self.run_btn.config(state='disabled', text='⏳  처리 중…',
                            bg='#888888')
        self.progress.pack(fill='x', pady=(6, 0))
        self.progress.start(12)
        self.log.delete('1.0', 'end')

        t = threading.Thread(target=self._worker, daemon=True)
        t.start()

    def _worker(self):
        old_stdout = sys.stdout
        sys.stdout = StdoutToQueue(self.log_q)
        try:
            print('========================================')
            print('  악보 가사 추출 파이프라인 시작')
            print('========================================')

            scale = max(1, min(5, self.upscale.get()))
            key_arg = self.gemini_key.get().strip() or None

            ko_lines = process_sheet(self.kor_path, 'kor', clean_korean, scale=scale)
            en_lines = process_sheet(self.eng_path, 'eng', clean_english, scale=scale)

            print('\n----- OCR 원본 -----')
            print('[한글]'); [print(f'  {l}') for l in ko_lines]
            print('[영어]'); [print(f'  {l}') for l in en_lines]

            ko_snap = list(ko_lines)
            en_snap = list(en_lines)
            try:
                ko_lines, en_lines = postprocess_lyrics(
                    ko_lines,
                    en_lines,
                    api_key=key_arg,
                    use_gemini=self.use_gemini.get(),
                    skip_spell=self.no_spell.get(),
                )
            except Exception as e:
                print(f'\n  [경고] 후처리 실패: {e}')
                print(traceback.format_exc())

            n_before = len(ko_lines) + len(en_lines)
            ko_lines = filter_lyric_lines(ko_lines, 'ko', fallback=ko_snap)
            en_lines = filter_lyric_lines(en_lines, 'en', fallback=en_snap)
            if not ko_lines and not en_lines:
                print('\n  [경고] 가사로 인정할 줄이 없습니다. '
                      '음표/계명(a,e)만 OCR됐을 수 있습니다. '
                      '「Gemini 가사 보완」을 켜거나 악보 이미지·해상도를 확인하세요.')
            elif n_before > 0 and len(ko_lines) + len(en_lines) < n_before // 2:
                print('\n  [경고] 악보 노이즈(계명 a/e 등)가 많아 대부분의 줄을 제외했습니다.')

            print('\n----- 최종 결과 -----')
            print('[한글]'); [print(f'  {l}') for l in ko_lines]
            print('[영어]'); [print(f'  {l}') for l in en_lines]

            save_lyrics_txt(ko_lines, en_lines, self.output_path.get())
            print('\n✅ 모든 처리 완료!')
            self.log_q.put(DONE_MARK)
        except Exception:
            print('\n❌ 오류 발생:')
            print(traceback.format_exc())
            self.log_q.put(ERROR_MARK)
        finally:
            sys.stdout = old_stdout

    def _on_done(self, success):
        self.busy = False
        self.run_btn.config(state='normal', text='▶  변환 시작',
                            bg='#3a7afe')
        self.progress.stop()
        self.progress.pack_forget()

        if success:
            out = self.output_path.get()
            if messagebox.askyesno('완료',
                                   f'텍스트 저장 완료:\n{out}\n\n저장 폴더를 열까요?'):
                self._open_folder(out)
        else:
            messagebox.showerror('오류', '처리 중 오류가 발생했습니다. 로그를 확인하세요.')

    @staticmethod
    def _open_folder(path):
        folder = str(Path(path).parent.resolve())
        try:
            sysname = platform.system()
            if sysname == 'Windows':
                os.startfile(folder)
            elif sysname == 'Darwin':
                os.system(f'open "{folder}"')
            else:
                os.system(f'xdg-open "{folder}"')
        except Exception:
            pass


def main():
    app = App()
    # 다크모드 등에서 ttk 테마 가독성 보완
    try:
        style = ttk.Style(app)
        if 'clam' in style.theme_names():
            style.theme_use('clam')
    except Exception:
        pass
    app.mainloop()


if __name__ == '__main__':
    main()