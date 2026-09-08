"""
Generate Fadhili AI's social preview image.

The card is drawn programmatically from the project's own palette and
wordmark. It contains no photography, no stock imagery and nothing
AI-generated — it is typography and vector shapes rendered with PIL, so
its provenance is this script and nothing else.

Run:
    backend/.venv/Scripts/python.exe frontend/scripts/make_og_image.py
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parents[1] / "public" / "og-image.png"

W, H = 1200, 630

INK = (20, 24, 26)
PAPER = (242, 239, 231)
PAPER_DIM = (154, 160, 158)
OCHRE = (200, 155, 60)
TEAL = (127, 179, 160)

SERIF_BOLD = "C:/Windows/Fonts/georgiab.ttf"
SANS = "C:/Windows/Fonts/segoeui.ttf"
SANS_SEMI = "C:/Windows/Fonts/seguisb.ttf"


def rounded_hand(draw, x, y, size):
    """
    The Fadhili mark: a simplified signing hand, matching favicon.svg
    and the Logo component.
    """
    s = size / 32.0

    draw.rounded_rectangle(
        [x, y, x + size, y + size],
        radius=int(8 * s),
        fill=INK,
        outline=(52, 58, 60),
        width=max(1, int(0.6 * s)),
    )

    width = max(2, int(2.3 * s))

    # Two raised fingers.
    draw.line(
        [(x + 11 * s, y + 19.5 * s), (x + 11 * s, y + 11.2 * s)],
        fill=PAPER, width=width, joint="curve",
    )
    draw.line(
        [(x + 14.2 * s, y + 16.4 * s), (x + 14.2 * s, y + 9.4 * s)],
        fill=PAPER, width=width, joint="curve",
    )

    # Thumb / palm sweep in ochre.
    draw.line(
        [
            (x + 17.4 * s, y + 12.2 * s),
            (x + 20.5 * s, y + 12.2 * s),
            (x + 20.5 * s, y + 20.3 * s),
            (x + 16.0 * s, y + 25.0 * s),
            (x + 10.5 * s, y + 22.0 * s),
            (x + 7.6 * s, y + 18.2 * s),
        ],
        fill=OCHRE,
        width=width,
        joint="curve",
    )


def main():
    image = Image.new("RGB", (W, H), INK)
    draw = ImageDraw.Draw(image)

    # A restrained accent rule rather than a gradient wash.
    draw.rectangle([0, 0, W, 6], fill=OCHRE)
    draw.rectangle([0, H - 6, W, H], fill=(31, 68, 56))

    title = ImageFont.truetype(SERIF_BOLD, 92)
    tagline = ImageFont.truetype(SANS_SEMI, 40)
    body = ImageFont.truetype(SANS, 29)

    left = 96

    rounded_hand(draw, left, 108, 96)

    draw.text((left, 246), "Fadhili AI", font=title, fill=PAPER)
    draw.text((left, 366), "Sign. Speak. Connect.", font=tagline, fill=OCHRE)

    draw.text(
        (left, 444),
        "Kenyan Sign Language recognition and learning.",
        font=body,
        fill=PAPER_DIM,
    )
    draw.text(
        (left, 484),
        "Experimental research software — not an interpreter.",
        font=body,
        fill=PAPER_DIM,
    )

    # Small vocabulary marker, kept factual.
    draw.line([(left, 540), (left + 88, 540)], fill=TEAL, width=3)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUT, "PNG", optimize=True)
    print(f"Wrote {OUT}  ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
