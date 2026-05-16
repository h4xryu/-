# -*- coding: utf-8 -*-
"""PNG → Base64 문자열 / README용 data-URI 조각 생성."""

from __future__ import annotations

import argparse
import base64
from pathlib import Path


def png_to_base64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode('ascii')


def main():
    parser = argparse.ArgumentParser(description='PNG를 Base64(data URI)로 변환')
    parser.add_argument('png', type=Path, help='입력 PNG 경로')
    parser.add_argument(
        '-o', '--out',
        type=Path,
        help='.b64.txt 저장 (한 줄 Base64). 기본: 입력파일명.b64.txt',
    )
    parser.add_argument(
        '--markdown',
        action='store_true',
        help='README용 HTML img 태그(data URI) stdout 출력',
    )
    parser.add_argument('--width', type=int, default=720, help='--markdown 시 img 너비')
    args = parser.parse_args()

    if not args.png.is_file():
        raise SystemExit(f'파일 없음: {args.png}')

    b64 = png_to_base64(args.png)
    out_txt = args.out or args.png.with_suffix(args.png.suffix + '.b64.txt')
    out_txt.write_text(b64, encoding='ascii')
    print(f'저장: {out_txt} ({len(b64)} chars)')

    if args.markdown:
        uri = f'data:image/png;base64,{b64}'
        print(
            f'<img src="{uri}" width="{args.width}" alt="{args.png.name}" />\n'
            f'<!-- GitHub README는 data: URI 이미지를 막을 수 있음. '
            f'이 경우 docs/images/ 상대경로를 쓰세요. -->'
        )


if __name__ == '__main__':
    main()
