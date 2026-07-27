#!/usr/bin/env python3
"""Generate a beautiful banner PNG for dzyy-vllm GitHub README."""

import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np

W, H = 960, 280
OUT = os.path.join(os.path.dirname(__file__), "banner.png")

# -- Canvas & gradient background (dark navy → deep purple) --
img = Image.new("RGBA", (W, H), (0, 0, 0, 255))
draw = ImageDraw.Draw(img)

# Create background gradient
for y in range(H):
    t = y / H
    r = int(15 + t * 15)       # 15→30
    g = int(12 + t * 10)       # 12→22
    b = int(41 + t * 21)       # 41→62
    draw.rectangle([(0, y), (W, y + 1)], fill=(r, g, b, 255))

# -- Glow / accent circles --
for cx, cy, rad, alpha in [
    (150, 80, 200, 18),
    (820, 170, 180, 12),
    (700, 60, 120, 8),
    (200, 210, 100, 6),
]:
    glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    gdraw = ImageDraw.Draw(glow)
    for i in range(rad, 0, -2):
        a = int(alpha * (i / rad))
        gdraw.ellipse([cx - i, cy - i, cx + i, cy + i],
                      fill=(124, 77, 255, a))
    img = Image.alpha_composite(img, glow)

# -- Grid pattern (subtle tech dots) --
for x in range(30, W, 40):
    for y in range(30, H, 40):
        draw.ellipse([x - 1, y - 1, x + 1, y + 1], fill=(255, 255, 255, 6))

# -- Corner decorations --
corner_color = (124, 77, 255, 40)
# Top-left
draw.line([(35, 28), (115, 28)], fill=corner_color, width=2)
draw.line([(35, 28), (35, 75)], fill=corner_color, width=2)
# Bottom-right
draw.line([(925, 252), (845, 252)], fill=(0, 229, 255, 35), width=2)
draw.line([(925, 252), (925, 205)], fill=(0, 229, 255, 35), width=2)
# Small corner squares
draw.rectangle([(33, 26), (37, 30)], fill=(124, 77, 255, 80))
draw.rectangle([(923, 250), (927, 254)], fill=(0, 229, 255, 70))

# -- Font loading --
font_paths = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf",
    "/usr/share/fonts/truetype/ubuntu/UbuntuMono-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeMonoBold.ttf",
]
title_path, subtitle_path = None, None
for p in font_paths:
    if os.path.exists(p):
        title_path = p
        break
for p in font_paths:
    if os.path.exists(p):
        subtitle_path = p
        break

# Fallback to default
if title_path is None:
    title_path = subtitle_path = None

# -- Draw title with manual gradient effect (top half → bottom half) --
title = "dzyy-vllm"
title_size = 58
try:
    title_font = ImageFont.truetype(title_path, title_size) if title_path else ImageFont.load_default()
except Exception:
    title_font = ImageFont.load_default()

# Measure text to center it
bbox = draw.textbbox((0, 0), title, font=title_font)
tw = bbox[2] - bbox[0]
th = bbox[3] - bbox[1]
tx = (W - tw) // 2
ty = 65

# Draw title with alpha gradient overlay for a subtle effect
# Main title in white
draw.text((tx, ty), title, font=title_font, fill=(220, 220, 255, 255))

# Overlay gradient colored version (slightly offset alpha)
# We create a mask and composite
title_overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
to_draw = ImageDraw.Draw(title_overlay)

# Gradient colors: cyan → purple → pink
colors = [
    (0, 229, 255, 220),    # cyan
    (100, 180, 255, 220),  # light blue
    (124, 77, 255, 220),   # purple
    (180, 100, 255, 220),  # light purple
    (224, 64, 251, 220),   # pink
]

# Apply color gradient across title by drawing overlapping text
for i, color in enumerate(colors):
    xoff = int((i - 2) * 0.5)  # micro offset for glow effect
    to_draw.text((tx + xoff, ty + 1), title, font=title_font, fill=color)

# Blend overlay with original
img = Image.alpha_composite(img, title_overlay)

# -- Underline below title --
line_y = ty + th + 18
# Gradient underline
for x in range(int(tx), int(tx + tw)):
    t = (x - tx) / tw
    r = int(0 + t * (124 - 0) + (1 - t) * 0)
    g = int(229 * (1 - t) + 77 * t)
    b = int(255 * (1 - t) + 255 * t)
    draw.line([(x, line_y), (x, line_y + 2)], fill=(r, g, b, 180))

# Small diamond at left end of underline
draw.polygon([
    (tx - 6, line_y + 1),
    (tx, line_y - 4),
    (tx + 6, line_y + 1),
    (tx, line_y + 6),
], fill=(0, 229, 255, 150))

# Small diamond at right end
draw.polygon([
    (tx + tw - 6, line_y + 1),
    (tx + tw, line_y - 4),
    (tx + tw + 6, line_y + 1),
    (tx + tw, line_y + 6),
], fill=(224, 64, 251, 150))

# -- Subtitle --
subtitle = "Multi-version vLLM from Scratch"
sub_size = 18
try:
    sub_font = ImageFont.truetype(subtitle_path, sub_size) if subtitle_path else ImageFont.load_default()
except Exception:
    sub_font = ImageFont.load_default()

bbox2 = draw.textbbox((0, 0), subtitle, font=sub_font)
sw = bbox2[2] - bbox2[0]
sx = (W - sw) // 2
sy = line_y + 30
draw.text((sx, sy), subtitle, font=sub_font, fill=(179, 136, 255, 220))

# -- Tech stack line --
tech = "KV Cache  •  Continuous Batching  •  FlashInfer  •  CUDA Graph  •  Vision  •  MoE  •  TP/PP"
tech_size = 12
try:
    tech_font = ImageFont.truetype(subtitle_path, tech_size) if subtitle_path else ImageFont.load_default()
except Exception:
    tech_font = ImageFont.load_default()

bbox3 = draw.textbbox((0, 0), tech, font=tech_font)
tsw = bbox3[2] - bbox3[0]
tsx = (W - tsw) // 2
tsy = sy + 38
draw.text((tsx, tsy), tech, font=tech_font, fill=(92, 107, 192, 160))

# -- Footer: version info --
footer_size = 11
try:
    footer_font = ImageFont.truetype(subtitle_path, footer_size) if subtitle_path else ImageFont.load_default()
except Exception:
    footer_font = ImageFont.load_default()

draw.text((40, H - 30), "v0  →  v5", font=footer_font, fill=(92, 107, 192, 110))
draw.text((W - 40, H - 30), "PyTorch  +  CUDA", font=footer_font,
         fill=(92, 107, 192, 110), anchor="ra")

# Save
img.save(OUT, "PNG")
print(f"Banner saved to {OUT}  ({os.path.getsize(OUT) / 1024:.1f} KB)")
