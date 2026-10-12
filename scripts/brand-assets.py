"""Build the web logo and icons from the brand originals (docs/images/brand/, transparent PNGs).

usage (from the repo root): uv run --no-project --with pillow python scripts/brand-assets.py
Writes frontend/public/: xcrs-logo.webp (wordmark), favicon.ico, icon-192.png, apple-touch-icon.png.
"""

from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
BRAND = ROOT / "docs/images/brand"
PUBLIC = ROOT / "frontend/public"


def trimmed(path: Path) -> Image.Image:
    """The image cropped to its visible pixels (the originals carry near-invisible noise around the art)."""
    img = Image.open(path).convert("RGBA")
    visible = img.getchannel("A").point(lambda a: 255 if a > 8 else 0)
    return img.crop(visible.getbbox())


def square(img: Image.Image, size: int, pad: float, background=(0, 0, 0, 0)) -> Image.Image:
    canvas = Image.new("RGBA", (size, size), background)
    inner = round(size * (1 - 2 * pad))
    scale = inner / max(img.size)
    art = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
    canvas.alpha_composite(art, ((size - art.width) // 2, (size - art.height) // 2))
    return canvas


# Wordmark: up to 112 px tall on the home page; 240 px keeps it sharp on 2x screens.
wordmark = trimmed(BRAND / "XCRS_logo.png")
wordmark.resize((round(wordmark.width * 240 / wordmark.height), 240), Image.LANCZOS).save(
    PUBLIC / "xcrs-logo.webp", quality=90, method=6
)

# The X mark for small places: browser tab, home-screen icons.
mark = trimmed(BRAND / "X_logo.png")
square(mark, 256, 0.0).save(PUBLIC / "favicon.ico", sizes=[(16, 16), (32, 32), (48, 48)])
square(mark, 192, 0.04).save(PUBLIC / "icon-192.png", optimize=True)
# iOS fills transparency with black: a white tile with some room around the mark.
square(mark, 180, 0.12, (255, 255, 255, 255)).convert("RGB").save(PUBLIC / "apple-touch-icon.png", optimize=True)
print("wordmark", wordmark.size, "mark", mark.size)
