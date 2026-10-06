"""Leaf and crop textures drawn by code: RGBA images whose transparent parts Gazebo cuts away."""

import math

import numpy as np

SUPERSAMPLE = 4        # draw 4x larger, then shrink: smooth leaf edges


def _canvas(width, height):
    """Return a transparent RGBA image and a drawing context, both supersampled."""
    from PIL import Image, ImageDraw   # Pillow (python3-pil), only needed to draw textures

    image = Image.new('RGBA', (width * SUPERSAMPLE, height * SUPERSAMPLE), (0, 0, 0, 0))
    return image, ImageDraw.Draw(image)


def _finish(image, rng, noise=14):
    """Shrink a supersampled image and add a little colour noise to the opaque parts."""
    from PIL import Image

    small = image.resize((image.width // SUPERSAMPLE, image.height // SUPERSAMPLE),
                         Image.LANCZOS)
    pixels = np.asarray(small).astype(float)
    pixels[:, :, :3] += rng.normal(0.0, noise, pixels.shape[:2])[:, :, None]
    return Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8))


def _blade(draw, cx, base, tip, half_widths, colour, s=SUPERSAMPLE):
    """Fill a leaf blade between y = base and y = tip; half_widths samples its width profile."""
    n = len(half_widths)
    ys = [base + (tip - base) * k / (n - 1) for k in range(n)]
    left = [(s * (cx - w), s * y) for w, y in zip(half_widths, ys)]
    right = [(s * (cx + w), s * y) for w, y in zip(half_widths, ys)]
    draw.polygon(left + right[::-1], fill=colour)


def _profile(n, power_base, power_tip, width):
    """Return n half-widths of a leaf: 0 at both ends, widest in between."""
    t = np.linspace(0.0, 1.0, n)
    shape = np.sin(math.pi * t ** power_base) ** power_tip
    return list(width * shape / shape.max())


def sugar_beet_leaf(rng):
    """Return a sugar beet leaf: a long pale stalk (petiole) and a broad, wavy dark blade."""
    w, h = 128, 256
    image, draw = _canvas(w, h)
    s = SUPERSAMPLE
    draw.rectangle([s * (w / 2 - 4), s * h * 0.55, s * (w / 2 + 4), s * h],
                   fill=(150, 175, 95, 255))                        # petiole at the bottom
    widths = _profile(40, 0.8, 0.8, w * 0.45)
    widths = [x * (1 + 0.05 * math.sin(k * 1.7)) for k, x in enumerate(widths)]  # wavy edge
    base, tip = h * 0.62, h * 0.02
    _blade(draw, w / 2, base, tip, widths, (52, 105, 38, 255))
    for k in range(1, 7):                                           # side veins, inside
        y0 = base - (base - tip) * 0.13 * k
        y1 = y0 - h * 0.07
        reach = 0.75 * widths[int(round((base - y1) / (base - tip) * (len(widths) - 1)))]
        for side in (-1, 1):
            draw.line([s * w / 2, s * y0, s * (w / 2 + side * reach), s * y1],
                      fill=(95, 140, 70, 255), width=s)
    draw.line([s * w / 2, s * h * 0.62, s * w / 2, s * h * 0.05], fill=(140, 170, 95, 255),
              width=2 * s)                                          # midrib
    return _finish(image, rng)


def maize_leaf(rng):
    """Return a maize leaf: a long strap, widest past the middle, with a pale midrib."""
    w, h = 64, 512
    image, draw = _canvas(w, h)
    s = SUPERSAMPLE
    _blade(draw, w / 2, h, 0, _profile(60, 0.9, 0.5, w * 0.46), (70, 125, 45, 255))
    draw.line([s * w / 2, s * h, s * w / 2, s * h * 0.03], fill=(150, 185, 110, 255),
              width=2 * s)
    return _finish(image, rng)


def potato_leaf(rng):
    """Return a potato leaf: a stalk with three pairs of oval leaflets and one at the tip."""
    w, h = 192, 256
    image, draw = _canvas(w, h)
    s = SUPERSAMPLE
    colour, dark = (60, 110, 40, 255), (45, 90, 32, 255)
    draw.line([s * w / 2, s * h, s * w / 2, s * h * 0.1], fill=(110, 140, 70, 255),
              width=3 * s)
    for k, y in enumerate([0.75, 0.52, 0.30]):
        for side in (-1, 1):
            cx, cy, rx, ry = w / 2 + side * w * 0.21, h * y, w * 0.20, h * 0.07
            draw.ellipse([s * (cx - rx), s * (cy - ry), s * (cx + rx), s * (cy + ry)],
                         fill=colour if k % 2 else dark)
    draw.ellipse([s * w * 0.32, s * h * 0.0, s * w * 0.68, s * h * 0.22], fill=colour)
    return _finish(image, rng)


def soybean_leaf(rng):
    """Return a soybean leaf: three pointed oval leaflets on a short stalk."""
    w, h = 192, 192
    image, draw = _canvas(w, h)
    s = SUPERSAMPLE
    draw.line([s * w / 2, s * h, s * w / 2, s * h * 0.55], fill=(115, 145, 70, 255), width=3 * s)
    widths = _profile(30, 0.9, 0.7, w * 0.13)
    _blade(draw, w / 2, h * 0.55, h * 0.02, widths, (65, 120, 45, 255))
    for side in (-1, 1):
        leaflet, rotated = _canvas(w, h)
        _blade(rotated, w / 2, h * 0.55, h * 0.12, widths, (55, 108, 40, 255))
        leaflet = leaflet.rotate(-side * 55, center=(s * w / 2, s * h * 0.55))
        image.alpha_composite(leaflet)
    return _finish(image, rng)


def wheat_row(rng):
    """Return a side view of a wheat row, 1 m long: many thin blades and stems with ears."""
    w, h = 512, 384
    image, draw = _canvas(w, h)
    s = SUPERSAMPLE
    for _ in range(140):                                            # leaf blades
        x, lean = rng.uniform(0, w), rng.uniform(-0.5, 0.5)
        top = h * rng.uniform(0.35, 0.75)
        draw.line([s * x, s * h, s * (x + lean * (h - top)), s * (h - top)],
                  fill=(int(rng.uniform(60, 90)), int(rng.uniform(115, 150)), 45, 255),
                  width=int(s * rng.uniform(1.5, 3.0)))
    for _ in range(70):                                             # stems with ears on top
        x, lean = rng.uniform(4, w - 4), rng.uniform(-0.15, 0.15)
        top = h * rng.uniform(0.75, 0.95)
        tx, ty = x + lean * top, h - top
        draw.line([s * x, s * h, s * tx, s * ty], fill=(110, 150, 70, 255), width=s * 2)
        draw.ellipse([s * (tx - 3), s * (ty - 22), s * (tx + 3), s * (ty + 2)],
                     fill=(170, 175, 85, 255))
    return _finish(image, rng, noise=10)


def wheat_canopy(rng):
    """Return a view from above of a wheat canopy, 1 m x 1 m: ears and leaf tips, gaps between."""
    w = h = 384
    image, draw = _canvas(w, h)
    s = SUPERSAMPLE
    for _ in range(900):
        x, y, angle = rng.uniform(0, w), rng.uniform(0, h), rng.uniform(0, math.pi)
        length = rng.uniform(4, 12)
        dx, dy = length * math.cos(angle), length * math.sin(angle)
        ear = rng.random() < 0.35
        colour = ((165, 172, 85, 255) if ear else
                  (int(rng.uniform(55, 95)), int(rng.uniform(110, 150)), 45, 255))
        draw.line([s * (x - dx), s * (y - dy), s * (x + dx), s * (y + dy)], fill=colour,
                  width=int(s * (3.5 if ear else 2.0)))
    return _finish(image, rng, noise=10)


TEXTURES = {'sugar_beet_leaf': sugar_beet_leaf, 'maize_leaf': maize_leaf,
            'potato_leaf': potato_leaf, 'soybean_leaf': soybean_leaf, 'wheat_row': wheat_row,
            'wheat_canopy': wheat_canopy}


def draw_texture(name, seed=1):
    """Return texture 'name' as an RGBA image; the same seed draws the same picture."""
    return TEXTURES[name](np.random.default_rng(seed))
