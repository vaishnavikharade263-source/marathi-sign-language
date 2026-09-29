"""
text_overlay.py
-----------------
cv2.putText() can only draw OpenCV's built-in Latin glyph set -- it cannot
draw Devanagari (or any non-Latin script) text at all, under any
circumstances. This isn't a font-configuration issue, it's a hard limit of
that specific function. This module works around it by rendering text with
Pillow (which can use any installed TrueType font) onto the frame instead,
with a safe ASCII fallback if no Devanagari-capable font can be found on
the system (so this never crashes -- it just won't show the Devanagari
glyph without a suitable font installed).
"""

import numpy as np
import cv2
from PIL import ImageFont, ImageDraw, Image

# Common install locations for a Devanagari-capable font, checked in order.
# Nirmala UI ships with every Windows install since Windows 8, so this will
# succeed out of the box for essentially every Windows user.
_CANDIDATE_FONT_PATHS = [
    "C:/Windows/Fonts/Nirmala.ttc",        # Windows 11 (font collection file)
    "C:/Windows/Fonts/Nirmala.ttf",        # Windows 8 / 10
    "C:/Windows/Fonts/NirmalaB.ttf",       # Nirmala UI Bold
    "C:/Windows/Fonts/mangal.ttf",          # Mangal (older Windows)
    "C:/Windows/Fonts/Aparaj.ttf",          # Aparajita (ships with Windows)
    "C:/Windows/Fonts/utsaah.ttf",          # Utsaah (ships with Windows)
    "/System/Library/Fonts/Kohinoor.ttc",    # macOS
    "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf",   # Linux (Noto, if installed)
    "/usr/share/fonts/truetype/lohit-devanagari/Lohit-Devanagari.ttf",  # Linux (Lohit, if installed)
]

_font_cache = {}
_font_load_attempted = set()


def _load_font(size: int):
    if size in _font_cache:
        return _font_cache[size]
    if size in _font_load_attempted:
        return None
    _font_load_attempted.add(size)

    for path in _CANDIDATE_FONT_PATHS:
        try:
            font = ImageFont.truetype(path, size)
            _font_cache[size] = font
            return font
        except Exception:
            continue
    return None


def draw_text(frame_bgr: np.ndarray, text: str, position=(10, 40), font_size: int = 32,
              color=(0, 255, 0)) -> np.ndarray:
    """
    Draws `text` (may contain Devanagari) onto a BGR OpenCV frame.

    Uses a system Devanagari font via Pillow if one can be found; otherwise
    falls back to cv2.putText with the text's ASCII-safe substitute (so
    this never crashes on a machine without such a font -- it just won't
    render the non-Latin characters).
    """
    font = _load_font(font_size)
    if font is None:
        ascii_safe = text.encode("ascii", "replace").decode("ascii")
        # cv2.putText treats `position` as the text BASELINE (bottom-left), so
        # shift down by roughly the text height; otherwise it clips at the top.
        baseline = (position[0], position[1] + int(font_size * 0.9))
        cv2.putText(frame_bgr, ascii_safe, baseline, cv2.FONT_HERSHEY_SIMPLEX,
                    font_size / 40, color, 2)
        return frame_bgr

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil_img)
    pil_color = (color[2], color[1], color[0])  # PIL wants RGB, frame color tuples here are BGR
    draw.text(position, text, font=font, fill=pil_color)
    result_rgb = np.array(pil_img)
    return cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
