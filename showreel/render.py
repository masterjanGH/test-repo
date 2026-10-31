#!/usr/bin/env python3
"""
ANFAL — "The Uncommon is Earned" showreel.

Every frame is generated procedurally (light, sand, embers, woven cloth,
cable knit, stone arches, typography). No people, no faces, no footage.
Audio is sound design only (wind, sand, fire, hooves, fabric, stone steps),
synthesised from filtered noise: no music, no tones, no instruments.

Usage:
  python3 render.py                 # full render -> anfal_showreel.mp4 (+ _silent)
  python3 render.py --preview 3,9   # write PNG stills at those seconds to out/preview/
"""
import os, sys, math, wave, subprocess, functools, argparse
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

cv2.setNumThreads(1)

HERE = os.path.dirname(os.path.abspath(__file__))
FONT_DIR = os.path.join(HERE, "fonts")
OUT_DIR = os.path.join(HERE, "out")
W, H, FPS = 1080, 1920, 30

F_SERIF_IT = os.path.join(FONT_DIR, "CormorantGaramond-Italic.ttf")
F_SERIF = os.path.join(FONT_DIR, "CormorantGaramond.ttf")
F_CAPS = os.path.join(FONT_DIR, "Cinzel.ttf")
F_AR = os.path.join(FONT_DIR, "Amiri-Regular.ttf")

GOLD = np.array([0xAE, 0x8D, 0x47], np.float32) / 255.0
GREEN = np.array([0x0A, 0x4F, 0x2E], np.float32) / 255.0
CREAM = np.array([0.94, 0.90, 0.82], np.float32)

YY, XX = np.mgrid[0:H, 0:W].astype(np.float32)
TAU = 2 * math.pi


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def sstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def clamp01(x):
    return min(max(x, 0.0), 1.0)


def ease_out(x):
    x = clamp01(x)
    return 1 - (1 - x) ** 3


def ease_io(x):
    x = clamp01(x)
    return x * x * (3 - 2 * x)


def fbm(h, w, scale, seed, octaves=5, persistence=0.5):
    rng = np.random.default_rng(seed)
    out = np.zeros((h, w), np.float32)
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        gh = max(3, int(h / scale * 2 ** o) + 3)
        gw = max(3, int(w / scale * 2 ** o) + 3)
        g = rng.random((gh, gw)).astype(np.float32)
        out += amp * cv2.resize(g, (w, h), interpolation=cv2.INTER_CUBIC)
        tot += amp
        amp *= persistence
    out -= out.min()
    out /= out.max() + 1e-6
    return out


def splat(xs, ys, vals, sigma, h=H, w=W):
    buf = np.zeros((h, w), np.float32)
    m = (xs >= 0) & (xs < w - 1) & (ys >= 0) & (ys < h - 1) & (vals > 1e-5)
    if m.any():
        np.add.at(buf, (ys[m].astype(np.int32), xs[m].astype(np.int32)), vals[m].astype(np.float32))
    if sigma > 0.3:
        buf = cv2.GaussianBlur(buf, (0, 0), sigma) * (2 * math.pi * sigma * sigma)
    return buf


def normals_from_height(h, strength):
    gx = cv2.Sobel(h, cv2.CV_32F, 1, 0, ksize=3) * (strength / 8.0)
    gy = cv2.Sobel(h, cv2.CV_32F, 0, 1, ksize=3) * (strength / 8.0)
    nz = 1.0 / np.sqrt(gx * gx + gy * gy + 1.0)
    return -gx * nz, -gy * nz, nz


def unit(v):
    v = np.array(v, np.float32)
    return v / np.linalg.norm(v)


def rgb(a, c):
    """broadcast scalar field a (HxW) times colour c (3,) -> HxWx3"""
    return a[..., None] * np.asarray(c, np.float32)


# ----------------------------------------------------------------------------
# typography
# ----------------------------------------------------------------------------
@functools.lru_cache(maxsize=None)
def _font(path, size, wght):
    f = ImageFont.truetype(path, size, layout_engine=ImageFont.Layout.RAQM)
    if wght is not None:
        try:
            f.set_variation_by_axes([wght])
        except Exception:
            pass
    return f


@functools.lru_cache(maxsize=512)
def text_alpha(text, path, size, tracking=0, wght=None, rtl=False):
    fnt = _font(path, size, wght)
    pad = 70
    if rtl or tracking == 0:
        kw = dict(direction="rtl", language="ar") if rtl else {}
        bb = fnt.getbbox(text, **kw)
        im = Image.new("L", (bb[2] - bb[0] + 2 * pad, bb[3] - bb[1] + 2 * pad), 0)
        ImageDraw.Draw(im).text((pad - bb[0], pad - bb[1]), text, font=fnt, fill=255, **kw)
    else:
        adv = [fnt.getlength(c) for c in text]
        total = sum(adv) + tracking * (len(text) - 1)
        asc, desc = fnt.getmetrics()
        im = Image.new("L", (int(total) + 2 * pad, asc + desc + 2 * pad), 0)
        d = ImageDraw.Draw(im)
        x = pad
        for c, a in zip(text, adv):
            d.text((x, pad), c, font=fnt, fill=255)
            x += a + tracking
    a = np.asarray(im, np.float32) / 255.0
    ys, xs = np.nonzero(a > 0.002)
    a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return np.pad(a, pad)


def gold_fill(h, w):
    v = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    top = np.array([1.00, 0.90, 0.64], np.float32)
    mid = GOLD * 1.15
    bot = np.array([0.52, 0.40, 0.19], np.float32)
    k1 = np.clip(v / 0.5, 0, 1)[..., None]
    k2 = np.clip((v - 0.5) / 0.5, 0, 1)[..., None]
    c = top * (1 - k1) + mid * k1
    c = c * (1 - k2) + bot * k2
    return np.broadcast_to(c, (h, w, 3))


def comp_text(img, a, cx, cy, op=1.0, color=CREAM, gold=False, blur=0.0,
              shadow=0.55, shimmer=None):
    if op <= 0.004:
        return
    if blur > 0.3:
        a = cv2.GaussianBlur(a, (0, 0), blur)
    h, w = a.shape
    x0, y0 = int(round(cx - w / 2)), int(round(cy - h / 2))
    sx0, sy0 = max(0, -x0), max(0, -y0)
    x0c, y0c = max(0, x0), max(0, y0)
    x1c, y1c = min(W, x0 + w), min(H, y0 + h)
    if x1c <= x0c or y1c <= y0c:
        return
    a = a[sy0:sy0 + (y1c - y0c), sx0:sx0 + (x1c - x0c)]
    reg = img[y0c:y1c, x0c:x1c]
    if shadow > 0:
        sh = cv2.GaussianBlur(a, (0, 0), 14)[..., None] * (shadow * op)
        reg *= 1 - np.clip(sh * 1.4, 0, 0.85)
    A = (a * op)[..., None]
    if gold:
        col = gold_fill(*a.shape).copy()
        if shimmer is not None:
            xx = np.linspace(0, 1, a.shape[1], dtype=np.float32)[None, :]
            yy = np.linspace(0, 1, a.shape[0], dtype=np.float32)[:, None]
            band = np.exp(-(((xx + yy * 0.25) - (shimmer * 1.6 - 0.3)) / 0.07) ** 2)
            col = col + band[..., None] * np.array([0.9, 0.75, 0.45], np.float32)
    else:
        col = np.asarray(color, np.float32)
    reg[:] = reg * (1 - A) + col * A


def fade_window(t, t0, t1, fin=0.7, fout=0.35):
    if t < t0 or t > t1:
        return 0.0
    return ease_out((t - t0) / fin) * (1 - ease_io((t - (t1 - fout)) / fout))


def line_item(img, t, text, style, t0, t1, cy, cx=W / 2, gold=False, shimmer=None):
    op = fade_window(t, t0, t1)
    if op <= 0:
        return
    k = ease_out((t - t0) / 0.9)
    blur = (1 - k) * 7
    dy = (1 - k) * 22
    if style == "en":
        a = text_alpha(text, F_SERIF_IT, 76, 0, 500)
    elif style == "ar":
        a = text_alpha(text, F_AR, 56, 0, None, True)
        op *= 0.88
    elif style == "big":
        a = text_alpha(text, F_SERIF_IT, 150, 0, 500)
    elif style == "caps":
        a = text_alpha(text, F_CAPS, 54, 26, 500)
    elif style == "small":
        a = text_alpha(text, F_CAPS, 26, 10, 500)
    else:
        raise ValueError(style)
    comp_text(img, a, cx, cy + dy, op=op, blur=blur, gold=gold, shimmer=shimmer)


# ----------------------------------------------------------------------------
# scenes  (each: T seconds, render(t) -> float32 HxWx3, roughly display-referred)
# ----------------------------------------------------------------------------
class Beam:
    """Darkness. A single shaft of golden light. Dust suspended in it."""
    T = 4.2
    LINES = [("Not everything is for everyone.", "en", 0.9, 0.50),
             ("مش كل شي للكل.", "ar", 1.35, 0.50 + 0.048)]

    def __init__(self):
        self.haze = fbm(H + 400, W, 420, 11, 5)
        self.haze2 = fbm(H + 400, W, 110, 12, 4)
        rng = np.random.default_rng(5)
        N = 1400
        self.px, self.py, self.pz = rng.random(N) * W, rng.random(N) * H, rng.random(N)
        self.vx, self.vy = rng.normal(-3, 5, N), rng.normal(5, 5, N)
        self.ph, self.br = rng.random(N) * TAU, 0.35 + 0.65 * rng.random(N)

    def render(self, t):
        T = self.T
        ang = math.radians(116 + 3.0 * t / T)
        dx, dy = math.cos(ang), math.sin(ang)
        sx, sy = W * 0.95, -H * 0.10
        rx, ry = XX - sx, YY - sy
        along = rx * dx + ry * dy
        perp = np.abs(-rx * dy + ry * dx)
        width = 26 + np.maximum(along, 0) * 0.19
        off = int(400 - t * 55)
        hz = self.haze[off:off + H] * 0.7 + self.haze2[off:off + H] * 0.5
        reveal = ease_out(t / 2.2)
        core = np.exp(-(perp / width) ** 2 * 1.7) * np.exp(-np.maximum(along, 0) / 2400.0) * (along > 0)
        wide = np.exp(-(perp / (width * 3.2)) ** 2) * np.exp(-np.maximum(along, 0) / 1800.0) * 0.18
        beam = (core * (0.30 + 0.85 * hz) + wide) * reveal
        img = np.empty((H, W, 3), np.float32)
        img[:] = (0.010, 0.008, 0.007)
        img += rgb(beam, (1.0, 0.72, 0.40))
        # floor glow where the beam lands
        land = np.exp(-(((XX - W * 0.18) / 380) ** 2 + ((YY - H * 1.02) / 160) ** 2)) * 0.25 * reveal
        img += rgb(land, (1.0, 0.65, 0.35))
        # dust
        x = self.px + self.vx * t + 7 * np.sin(0.6 * t + self.ph)
        y = self.py + self.vy * t + 6 * np.cos(0.45 * t + self.ph * 1.3)
        xi = np.clip(x, 0, W - 1).astype(np.int32)
        yi = np.clip(y, 0, H - 1).astype(np.int32)
        b = beam[yi, xi] + 0.015
        tw = 0.65 + 0.35 * np.sin(3.0 * t + self.ph * 5)
        v = self.br * b * tw * 1.5
        z = self.pz
        dust = (splat(x[z < 0.65], y[z < 0.65], v[z < 0.65], 0.8)
                + splat(x[(z >= 0.65) & (z < 0.92)], y[(z >= 0.65) & (z < 0.92)],
                        v[(z >= 0.65) & (z < 0.92)] * 0.8, 2.0)
                + splat(x[z >= 0.92], y[z >= 0.92], v[z >= 0.92] * 0.22, 9.0))
        img += rgb(dust, (1.0, 0.86, 0.62))
        return img


class Ashes:
    """Embers rising out of the dark. Smoke. It started here."""
    T = 4.2
    LINES = [("It started in the ashes.", "en", 0.6, 0.47),
             ("بلّشت من الرماد.", "ar", 1.05, 0.47 + 0.048)]

    def __init__(self):
        self.smoke = fbm(H + 900, W, 330, 21, 6)
        self.smoke2 = fbm(H + 900, W, 130, 22, 5)
        self.glow = np.exp(-(H - YY) / (H * 0.30)).astype(np.float32)
        rng = np.random.default_rng(23)
        N = 520
        self.x0 = rng.random(N) * W
        self.sp = 90 + rng.random(N) ** 1.5 * 300
        self.ph = rng.random(N)
        self.wa = 8 + rng.random(N) * 50
        self.wf = 0.6 + rng.random(N) * 2.2
        self.wp = rng.random(N) * TAU
        self.dr = rng.normal(-12, 10, N)
        self.z = rng.random(N)
        self.br = 0.4 + 0.6 * rng.random(N)
        self.fl = 8 + rng.random(N) * 20

    def pos(self, t):
        span = H + 300
        y = H + 120 - ((t * self.sp + self.ph * span) % span)
        x = self.x0 + self.wa * np.sin(self.wf * t + self.wp) + self.dr * t
        return x, y

    def render(self, t):
        o1 = int(880 - t * 120)
        o2 = int(880 - t * 170)
        s = self.smoke[o1:o1 + H] * 0.75 + self.smoke2[o2:o2 + H] * 0.45
        sm = np.clip(s - 0.48, 0, 1) ** 1.4 * 1.6
        g = self.glow
        img = np.empty((H, W, 3), np.float32)
        img[:] = (0.008, 0.006, 0.006)
        img += rgb(g, (0.36, 0.085, 0.018))
        img += rgb(sm * (0.12 + g * 0.9), (0.30, 0.20, 0.15))
        x, y = self.pos(t)
        xp, yp = self.pos(t - 0.055)
        heat = np.clip(y / H, 0, 1) ** 0.6
        fl = 0.6 + 0.4 * np.sin(self.fl * t + self.wp * 7)
        val = heat * fl * self.br
        sharp = np.zeros((H, W), np.float32)
        bok = np.zeros((H, W), np.float32)
        for i in range(len(x)):
            if not (-20 < y[i] < H + 20):
                continue
            tgt = bok if self.z[i] > 0.88 else sharp
            th = 1 if self.z[i] < 0.5 else 2
            cv2.line(tgt, (int(xp[i]), int(yp[i])), (int(x[i]), int(y[i])), float(val[i]), th)
        core = cv2.GaussianBlur(sharp, (0, 0), 0.9) * 2.2
        halo = cv2.GaussianBlur(sharp, (0, 0), 7) * 7.0
        bk = cv2.GaussianBlur(bok, (0, 0), 16) * 18.0
        img += rgb(core, (1.0, 0.82, 0.52)) + rgb(halo, (1.0, 0.36, 0.07)) + rgb(bk, (1.0, 0.42, 0.10))
        return img


class Dunes:
    """Golden hour. Backlit dunes, wind lifting sand off the crests."""
    T = 4.4
    LINES = [("Every step, a battle.", "en", 0.5, 0.24),
             ("كل خطوة، معركة.", "ar", 0.95, 0.24 + 0.048)]

    def __init__(self):
        rng = np.random.default_rng(51)
        self.hy = hy = 0.55 * H
        cfg = [(hy + 6, 16, 520, 300, 6, 0.86),
               (hy + 50, 42, 700, 380, 12, 0.66),
               (hy + 165, 95, 900, 520, 24, 0.44),
               (hy + 380, 175, 1250, 700, 40, 0.22),
               (hy + 720, 300, 1700, 950, 62, 0.06)]
        self.layers = []
        for base, amp, p1, p2, sp, hz in cfg:
            n1 = cv2.resize(rng.random((1, 48)).astype(np.float32), (W * 4, 1),
                            interpolation=cv2.INTER_CUBIC)[0]
            self.layers.append(dict(base=base, amp=amp, f1=TAU / p1, f2=TAU / p2,
                                    ph1=rng.random() * TAU, ph2=rng.random() * TAU,
                                    sp=sp, hz=hz, n1=n1, off=rng.random() * W))
        self.rip = fbm(H, W, 170, 52, 4)
        k = np.clip(YY / hy, 0, 1) ** 2.0
        top = np.array([0.16, 0.08, 0.07], np.float32)
        hor = np.array([1.0, 0.64, 0.34], np.float32)
        self.sky = top * (1 - k[..., None]) + hor * k[..., None]
        N = 1600
        self.sx0 = rng.random(N) * W
        self.sph = rng.random(N)
        self.srate = 0.6 + rng.random(N) * 0.8
        self.sjit = rng.normal(0, 26, N)
        self.sbr = rng.random(N)

    def ridge(self, L, t):
        xs = np.arange(W, dtype=np.float32) + L["off"] + t * L["sp"]
        n = L["n1"][(xs.astype(np.int64)) % len(L["n1"])]
        f = (0.62 * (1 - np.abs(np.sin(xs * L["f1"] + L["ph1"]))) ** 1.3
             + 0.38 * (0.5 + 0.5 * np.sin(xs * L["f2"] + L["ph2"])) + 0.25 * (n - 0.5))
        return (L["base"] - L["amp"] * f).astype(np.float32)

    def render(self, t):
        T, hy = self.T, self.hy
        img = self.sky.copy()
        sunx, suny = 0.63 * W, hy - 150 + 40 * t / T
        d = np.sqrt((XX - sunx) ** 2 + (YY - suny) ** 2)
        img += rgb(np.exp(-d / 300) * 0.75 + np.exp(-d / 70) * 0.5, (1.0, 0.55, 0.26))
        img += rgb(sstep(76, 70, d), (1.6, 1.25, 0.85))
        haze = np.array([0.98, 0.64, 0.38], np.float32)
        xs = np.arange(W, dtype=np.float32)
        ridge3 = None
        for li, L in enumerate(self.layers):
            r = self.ridge(L, t)
            if li == 3:
                ridge3 = r
            y0 = max(0, int(r.min()) - 3)
            dy = YY[y0:] - r[None, :]
            mask = np.clip(dy + 0.5, 0, 1)
            dr = np.gradient(r)
            nx, ny = dr, -np.ones_like(dr)
            nn = np.sqrt(nx * nx + 1)
            lx, ly = sunx - xs, (suny - r) * 0.6
            ll = np.sqrt(lx * lx + ly * ly) + 1e-3
            lit1 = np.clip((nx * lx + ny * ly) / (nn * ll), 0, 1) ** 1.5
            lit1 = cv2.GaussianBlur(lit1[None, :].astype(np.float32), (0, 0), 4)[0]
            # slip faces run diagonally down from the crest, not straight down
            sl = (0.55 * np.tanh((xs - sunx) / 140.0))[None, :]
            idx = np.clip((xs[None, :] + np.maximum(dy, 0) * sl).astype(np.int32), 0, W - 1)
            litf = lit1[idx]
            lit = lit1
            fall = np.exp(-np.maximum(dy, 0) / (25 + L["amp"] * 0.55))
            near = 1 - L["hz"]
            litc = np.array([1.0, 0.60, 0.30], np.float32)
            shc = np.array([0.20, 0.09, 0.055], np.float32) * (0.6 + 0.4 * L["hz"])
            shade = litf * (0.25 + 0.75 * fall) * (0.35 + 0.65 * fall)
            col = shc + rgb(shade, litc - shc) * 0.9
            rim = np.exp(-np.maximum(dy, 0) / 2.2) * (0.25 + 1.4 * lit[None, :])
            col += rgb(rim, (1.0, 0.72, 0.42))
            if li >= 3:
                rp = np.sin(dy * 0.13 + XX[y0:] * 0.011 + self.rip[y0:] * 9.0)
                col += rgb((rp * 0.5 + 0.5) * np.exp(-np.maximum(dy, 0) / 260) * 0.10 * near, litc)
            col = col * (1 - L["hz"]) + haze * L["hz"]
            img[y0:] = img[y0:] * (1 - mask[..., None]) + col * mask[..., None]
        # sand lifting off the crests of layer 3
        u = (t * self.srate + self.sph) % 1.0
        x = self.sx0 - u * 260
        ri = ridge3[np.clip(self.sx0.astype(int), 0, W - 1)]
        y = ri - 3 - u * 45 + self.sjit * u
        v = np.sin(math.pi * u) * self.sbr * 0.12
        s = splat(x, y, v, 0.7)
        s = cv2.blur(s, (41, 1)) * 9
        img += rgb(s, (1.0, 0.78, 0.48))
        sc = 1 + 0.06 * ease_io(t / T)
        M = np.float32([[sc, 0, (1 - sc) * W / 2], [0, sc, (1 - sc) * hy]])
        return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


class Hooves:
    """Raking light over rippled sand. Hoofprints pressing in, one stride at a time."""
    T = 4.2
    LINES = [("Nothing was handed to us.", "en", 0.6, 0.50),
             ("ما حدا عطانا شي.", "ar", 1.05, 0.50 + 0.048)]

    def __init__(self):
        self.W2, self.Hc, self.v = 1600, 2300, 250.0
        W2, Hc = self.W2, self.Hc
        self.Hw = Hw = Hc + int(self.v * self.T) + 260
        y, x = np.mgrid[0:Hw, 0:W2].astype(np.float32)
        nb = fbm(Hw, W2, 520, 41, 4)
        nm = fbm(Hw, W2, 90, 42, 4)
        wp = fbm(Hw, W2, 280, 43, 3)
        th = 0.22
        ph = (y * math.cos(th) + x * math.sin(th) + 80 * wp) * TAU / 36
        rip = np.sin(ph) + 0.28 * np.sin(2 * ph + 0.6)
        g = np.random.default_rng(44).standard_normal((Hw, W2)).astype(np.float32)
        g = cv2.GaussianBlur(g, (0, 0), 0.8)
        self.base = (2.6 * rip * (0.5 + 0.5 * nm) + 34 * nb + 0.35 * g).astype(np.float32)
        self.alb = (0.90 + 0.22 * (fbm(Hw, W2, 200, 45, 4) - 0.5)).astype(np.float32)
        tw, tht = 160, 180
        v, u = np.mgrid[-tht // 2:tht // 2, -tw // 2:tw // 2].astype(np.float32)
        e = np.sqrt((u / 50) ** 2 + (v / 58) ** 2)
        inside = sstep(1.04, 0.94, e)
        ring = sstep(0.68, 0.78, e) * sstep(1.0, 0.9, e)
        heel = sstep(14, 32, v)
        shoe = ring * (1 - heel)
        frog = sstep(12, 3, np.abs(u) - (v - 4) * 0.33) * sstep(-8, 8, v) * inside
        rimb = np.exp(-((e - 1.13) / 0.09) ** 2)
        tm = -(7.5 * inside + 6.0 * shoe - 3.0 * frog) + 2.4 * rimb
        self.tmpl = cv2.GaussianBlur(tm.astype(np.float32), (0, 0), 1.4)
        C0 = Hw - Hc
        self.C0 = C0
        self.prints = []
        trig = 0.20 * Hc
        for k in range(0, 60):
            wy = C0 + Hc - 140 - k * 150
            if wy < 0:
                break
            side = 1 if k % 2 == 0 else -1
            wx = W2 / 2 + side * 60 + 18 * math.sin(k * 0.55)
            tau = (trig - wy + C0) / self.v
            self.prints.append((wx, wy, tau))
        self.alb_print = 0.86

    def thud_times(self):
        return [p[2] for p in self.prints if 0 <= p[2] <= self.T]

    def render(self, t):
        W2, Hc = self.W2, self.Hc
        c = int(round(self.C0 - self.v * t))
        hmap = self.base[c:c + Hc].copy()
        alb = self.alb[c:c + Hc].copy()
        th, tw = self.tmpl.shape
        for wx, wy, tau in self.prints:
            d = ease_out((t - tau) / 0.09)
            if d <= 0:
                continue
            sy = int(wy - c - th / 2)
            sx = int(wx - tw / 2)
            if sy + th < 0 or sy >= Hc:
                continue
            a0, a1 = max(0, sy), min(Hc, sy + th)
            hmap[a0:a1, sx:sx + tw] += self.tmpl[a0 - sy:a1 - sy] * d
            alb[a0:a1, sx:sx + tw] *= 1 - (1 - self.alb_print) * np.clip(-self.tmpl[a0 - sy:a1 - sy] / 6, 0, 1) * d
        nx, ny, nz = normals_from_height(hmap, 1.0)
        L = unit((0.62, -0.55, 0.38))
        dif = np.clip(nx * L[0] + ny * L[1] + nz * L[2], 0, 1) ** 1.25
        sand = np.array([0.84, 0.64, 0.44], np.float32)
        light = rgb(dif, (1.25, 0.92, 0.60)) + np.array([0.16, 0.13, 0.15], np.float32)
        img = alb[..., None] * sand * light
        src = np.float32([[0, 0], [W2, 0], [W2 / 2 + W * 0.40, Hc], [W2 / 2 - W * 0.40, Hc]])
        dst = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
        P = cv2.getPerspectiveTransform(src, dst)
        img = cv2.warpPerspective(img, P, (W, H), flags=cv2.INTER_LINEAR)
        far = sstep(0.42, 0.0, YY / H)
        bl = cv2.GaussianBlur(img, (0, 0), 5)
        img = img * (1 - far[..., None]) + bl * far[..., None]
        img += rgb(far ** 1.6 * 0.30, (1.0, 0.70, 0.42))
        img *= 0.92
        return img


def weave_texture(N=2600, P=30):
    y, x = np.mgrid[0:N, 0:N].astype(np.float32)
    i = (x // P).astype(np.int32)
    j = (y // P).astype(np.int32)
    fx, fy = (x % P) / P, (y % P) / P
    rw = 0.45
    pw = np.sqrt(np.clip(1 - ((fx - 0.5) / rw) ** 2, 0, 1))
    pf = np.sqrt(np.clip(1 - ((fy - 0.5) / rw) ** 2, 0, 1))
    over = ((i - j) % 3) != 0
    cy_ = np.cos((fy - 0.5) * math.pi)
    cx_ = np.cos((fx - 0.5) * math.pi)
    uw = np.where(over, 0.5 + 0.45 * cy_, 0.5 - 0.22 * cy_)
    uf = np.where(~over, 0.5 + 0.45 * cx_, 0.5 - 0.22 * cx_)
    rng = np.random.default_rng(31)
    fv = cv2.resize(rng.random((N // 25, N // 2)).astype(np.float32), (N, N), interpolation=cv2.INTER_CUBIC)
    fh = cv2.resize(rng.random((N // 2, N // 25)).astype(np.float32), (N, N), interpolation=cv2.INTER_CUBIC)
    hw = pw * uw + pw * (0.035 * np.sin((y + x * 2.4) * TAU / 6) + 0.09 * (fv - 0.5))
    hf = pf * uf + pf * (0.035 * np.sin((x + y * 2.4) * TAU / 6) + 0.09 * (fh - 0.5))
    isw = hw >= hf
    h = cv2.GaussianBlur(np.maximum(hw, hf).astype(np.float32), (0, 0), 0.8)
    nx, ny, nz = normals_from_height(h, 13.0)
    var = fbm(N, N, 400, 33, 3)
    warpc = np.array([0.80, 0.68, 0.50], np.float32)
    weftc = np.array([0.70, 0.58, 0.41], np.float32)
    fib = np.where(isw, fv, fh)
    alb = np.where(isw[..., None], warpc, weftc) * (0.86 + 0.24 * fib[..., None]) * (0.92 + 0.16 * var[..., None])
    ao = 0.30 + 0.70 * np.clip(h / 0.9, 0, 1)
    A = np.dstack([nx, ny, nz, ao]).astype(np.float32)
    return A, alb.astype(np.float32)


class Weave:
    """Macro of woven cloth. Light travels across the threads."""
    T = 4.0
    LINES = [("Built thread by thread.", "en", 0.9, 0.50),
             ("خيط بخيط.", "ar", 1.35, 0.50 + 0.048)]

    def __init__(self):
        self.N = 2600
        self.A, self.B = weave_texture(self.N)

    def render(self, t):
        T, N = self.T, self.N
        ang = 14 + 5 * t / T
        sc = 1.36 - 0.14 * ease_io(t / T)
        cxs, cys = N / 2 + 30 * t, N / 2 - 25 * t
        M = cv2.getRotationMatrix2D((cxs, cys), ang, sc)
        M[0, 2] += W / 2 - cxs
        M[1, 2] += H / 2 - cys
        A = cv2.warpAffine(self.A, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        B = cv2.warpAffine(self.B, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        nx, ny, nz, ao = A[..., 0], A[..., 1], A[..., 2], A[..., 3]
        az = math.radians(205 + 85 * t / T)
        el = math.radians(26)
        L = np.array([math.cos(az) * math.cos(el), math.sin(az) * math.cos(el), math.sin(el)], np.float32)
        dif = np.clip(nx * L[0] + ny * L[1] + nz * L[2], 0, 1)
        hv = unit(L + np.array([0, 0, 1], np.float32))
        spec = np.clip(nx * hv[0] + ny * hv[1] + nz * hv[2], 0, 1) ** 30 * 0.18
        img = B * (rgb(0.10 * ao, (0.85, 0.85, 0.95)) + rgb(1.15 * dif * ao, (1.0, 0.88, 0.70))) + rgb(spec, (1.0, 0.9, 0.75))
        bl = cv2.GaussianBlur(img, (0, 0), 7)
        m = np.maximum(sstep(0.12, 0.42, np.abs(YY / H - 0.5)), 1 - ease_out(t / 1.3))
        img = img * (1 - m[..., None]) + bl * m[..., None]
        return img * 0.95


def cable_texture(N=2400):
    y, x = np.mgrid[0:N, 0:N].astype(np.float32)
    rep, pw, c = 300.0, 220.0, 110.0
    lx = x % rep
    A, P, R = 40.0, 280.0, 52.0
    ph = y * TAU / P
    s1, s2 = c + A * np.cos(ph), c - A * np.cos(ph)
    h1 = np.sqrt(np.clip(1 - ((lx - s1) / R) ** 2, 0, 1))
    h2 = np.sqrt(np.clip(1 - ((lx - s2) / R) ** 2, 0, 1))
    top1 = 0.5 + 0.5 * np.tanh(3 * np.sin(ph))
    H1 = h1 * (0.62 + 0.38 * top1)
    H2 = h2 * (0.62 + 0.38 * (1 - top1))
    which = H1 >= H2
    sw = np.where(which, s1, s2)
    cab = np.maximum(H1, H2)
    st = np.abs(np.sin(math.pi * (y / 15 + np.abs(lx - sw) / 26)))
    cab = cab * (1 + 0.09 * st)
    bgst = 0.16 + 0.04 * np.abs(np.sin(math.pi * (y / 15 + np.abs((lx % 22) - 11) / 22)))
    panel = np.maximum(cab, bgst)
    q = lx - pw
    rib1 = np.sqrt(np.clip(1 - ((q - 22) / 13) ** 2, 0, 1)) * (0.42 + 0.06 * np.abs(np.sin(math.pi * y / 13)))
    rib2 = np.sqrt(np.clip(1 - ((q - 58) / 13) ** 2, 0, 1)) * (0.42 + 0.06 * np.abs(np.sin(math.pi * y / 13 + 1)))
    purl = 0.12 + 0.05 * np.sin(TAU * y / 11) * np.sin(math.pi * np.clip(q / 80, 0, 1))
    chan = np.maximum(np.maximum(rib1, rib2), purl)
    h = np.where(lx < pw, panel, chan).astype(np.float32)
    h += (fbm(N, N, 5, 61, 2) - 0.5) * 0.04
    h += (fbm(N, N, 3, 63, 2) - 0.5) * 0.05
    h = cv2.GaussianBlur(h, (0, 0), 1.6)
    nx, ny, nz = normals_from_height(h, 42.0)
    ao = 0.25 + 0.75 * np.clip(h / 0.9, 0, 1)
    fib = fbm(N, N, 30, 62, 3)
    return np.dstack([nx, ny, nz, ao]).astype(np.float32), fib.astype(np.float32)


class Cable:
    """Cable knit, three cuts. Quiet. Heavy. Intentional."""
    T = 4.2
    CUTS = [dict(alb=(0.90, 0.86, 0.78), sc=0.92, ang=0, word="Quiet.", az0=200, dof=0.0),
            dict(alb=(0.075, 0.075, 0.08), sc=1.45, ang=90, word="Heavy.", az0=250, dof=0.25),
            dict(alb=(0.90, 0.86, 0.78), sc=2.3, ang=-9, word="Intentional.", az0=300, dof=0.5)]
    LINES = []

    def __init__(self):
        self.N = 2400
        self.A, self.fib = cable_texture(self.N)

    def render(self, t):
        N = self.N
        k = min(int(t / 1.4), 2)
        lt = t - 1.4 * k
        C = self.CUTS[k]
        cxs, cys = N / 2, N / 2 + 110 * lt
        M = cv2.getRotationMatrix2D((cxs, cys), C["ang"] + 1.5 * lt, C["sc"] * (1 + 0.03 * lt))
        M[0, 2] += W / 2 - cxs
        M[1, 2] += H / 2 - cys
        A = cv2.warpAffine(self.A, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        fib = cv2.warpAffine(self.fib, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        nx, ny, nz, ao = A[..., 0], A[..., 1], A[..., 2], A[..., 3]
        az = math.radians(C["az0"] + 40 * lt)
        el = math.radians(30)
        L = np.array([math.cos(az) * math.cos(el), math.sin(az) * math.cos(el), math.sin(el)], np.float32)
        dif = np.clip(nx * L[0] + ny * L[1] + nz * L[2], 0, 1)
        sheen = (1 - nz) ** 2 * dif
        alb = np.array(C["alb"], np.float32)
        dark = alb.mean() < 0.3
        kd = 2.4 if dark else 1.1
        img = rgb((0.85 + 0.3 * fib) * ao * (0.10 + kd * dif), 1) * alb
        img = img * np.array([1.0, 0.94, 0.85], np.float32)
        img += rgb(sheen * (0.22 if dark else 0.08), (1.0, 0.9, 0.75))
        if C["dof"] > 0:
            bl = cv2.GaussianBlur(img, (0, 0), 9)
            m = sstep(0.10, 0.10 + 0.35 * (1 - C["dof"]), np.abs(YY / H - 0.5))
            img = img * (1 - m[..., None]) + bl * m[..., None]
        if not dark:
            img *= 0.88
        word = C["word"]
        a = text_alpha(word, F_SERIF_IT, 150, 0, 500)
        op = ease_out(lt / 0.35) * (1 - ease_io((lt - 1.15) / 0.25))
        kk = ease_out(lt / 0.6)
        comp_text(img, a, W / 2, H * 0.5 + (1 - kk) * 20, op=op, blur=(1 - kk) * 6, shadow=0.75)
        return img


class Arches:
    """A stone hall at sunset. Light pours through arched windows."""
    T = 4.6
    LINES = [("For the ones who earned it.", "en", 0.7, 0.125),
             ("للّي تعبوا ووصلوا.", "ar", 1.15, 0.125 + 0.048)]
    CENTERS = (-2.6, 0.0, 2.6)
    HW, SILL, SPRING, TRANSOM = 0.8, 1.2, 5.0, 3.8

    def __init__(self):
        self.f, self.cx, self.cy, self.camY, self.Zw = 1100.0, W / 2, H / 2, 1.6, 10.0
        ppm = 120
        self.ppm = ppm
        wt = fbm(1500, 1500, 260, 71, 5) * 0.6 + fbm(1500, 1500, 30, 72, 4) * 0.4
        yy, xx = np.mgrid[0:1500, 0:1500].astype(np.float32)
        row = (yy / (0.55 * ppm)).astype(np.int32)
        bx = (xx / (1.1 * ppm) + 0.5 * (row % 2))
        mort = np.minimum(np.abs((yy / (0.55 * ppm)) % 1 - 0.0), np.abs(bx % 1))
        mort = np.minimum(mort, 1 - ((yy / (0.55 * ppm)) % 1))
        mort = np.minimum(mort, 1 - (bx % 1))
        self.walltex = (wt * (0.55 + 0.45 * sstep(0.0, 0.03, mort))).astype(np.float32)
        ft = fbm(1500, 1500, 200, 73, 5) * 0.7 + fbm(1500, 1500, 25, 74, 3) * 0.3
        self.floortex = ft.astype(np.float32)
        rng = np.random.default_rng(75)
        N = 900
        self.dX = rng.uniform(-4.5, 4.5, N)
        self.dY = rng.uniform(0.0, 6.5, N)
        self.dZ = rng.uniform(1.5, 9.8, N)
        self.dph = rng.random(N) * TAU
        self.dq = smallq = (W // 4, H // 4)
        yq, xq = np.mgrid[0:smallq[1], 0:smallq[0]].astype(np.float32)
        self.xq, self.yq = xq * 4 + 2, yq * 4 + 2
        self.dens = fbm(smallq[1], smallq[0], 60, 76, 4)

    def win(self, X, Y, soft=0.025):
        m = np.zeros_like(X)
        for c in self.CENTERS:
            dx = np.abs(X - c)
            rect = sstep(self.HW + soft, self.HW - soft, dx) * sstep(self.SILL - soft, self.SILL + soft, Y) \
                * sstep(self.SPRING + soft, self.SPRING - soft, Y)
            r = np.sqrt(dx * dx + (Y - self.SPRING) ** 2)
            arch = sstep(self.HW + soft, self.HW - soft, r) * (Y >= self.SPRING)
            o = np.maximum(rect, arch)
            o *= sstep(0.025, 0.055, dx)
            o *= sstep(0.025, 0.055, np.abs(Y - self.TRANSOM))
            m = np.maximum(m, o)
        return m

    def sun(self, t):
        return (-0.42 + 0.20 * t / self.T, 1.0, 1.0)

    def render(self, t):
        f, cx, cy, camY, Zw, ppm = self.f, self.cx, self.cy, self.camY, self.Zw, self.ppm
        camZ = 0.9 * ease_io(t / self.T)
        D = Zw - camZ
        Sx, Sy, Sz = self.sun(t)
        img = np.zeros((H, W, 3), np.float32)
        yfl = cy + f * camY / D
        # wall
        Xw = (XX - cx) * D / f
        Yw = camY + (cy - YY) * D / f
        wm = self.win(Xw, Yw)
        mx = ((Xw + 6) * ppm).astype(np.float32)
        my = ((12 - Yw) * ppm).astype(np.float32)
        wt = cv2.remap(self.walltex, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        wall = rgb(0.35 + 0.9 * wt, (0.070, 0.052, 0.040))
        wall += rgb(np.exp(-np.maximum(Yw, 0) / 1.4) * 0.05, (1.0, 0.6, 0.3))
        k = np.clip((Yw - self.SILL) / (self.SPRING + self.HW - self.SILL), 0, 1)[..., None]
        sky = np.array([2.1, 1.35, 0.75], np.float32) * (1 - k) + np.array([1.0, 0.48, 0.30], np.float32) * k
        wallimg = wall * (1 - wm[..., None]) + sky * wm[..., None]
        # glow spill around the windows
        sm = cv2.resize(wm, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
        spill = cv2.resize(cv2.GaussianBlur(sm, (0, 0), 14), (W, H))
        wallimg += rgb(spill * 0.22, (1.0, 0.62, 0.32))
        # floor
        dyf = np.maximum(YY - cy, 1.0)
        Zf = camZ + f * camY / dyf
        Xf = (XX - cx) * (Zf - camZ) / f
        s = (Zw - Zf) / Sz
        patch = self.win(Xf + s * Sx, s * Sy, soft=0.05)
        fx = ((Xf + 6) * ppm).astype(np.float32)
        fz = ((Zf) * ppm).astype(np.float32)
        ft = cv2.remap(self.floortex, fx, fz, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        gx = np.abs(((Xf + 0.5) % 1.0) - 0.5)
        gz = np.abs(((Zf + 0.5) % 1.0) - 0.5)
        grout = np.minimum(gx, gz)
        groutk = 0.75 + 0.25 * sstep(0.003, 0.012 * (Zf - camZ) / 3 + 0.004, grout)
        floor = rgb((0.5 + 0.8 * ft) * groutk, (0.050, 0.040, 0.034))
        floor += rgb(patch * (0.55 + 0.6 * ft) * groutk, (1.55, 1.02, 0.58))
        # floor reflection of the bright windows (polished stone)
        flm = (YY > yfl)
        img[:] = np.where(flm[..., None], floor, wallimg)
        aa = np.clip(YY - yfl + 0.5, 0, 1)[..., None]
        img = wallimg * (1 - aa) + img * aa
        # volumetric shafts at quarter resolution
        xq, yq = self.xq, self.yq
        dyq = np.maximum(yq - cy, 1.0)
        depthq = np.where(yq > cy + f * camY / D, camZ + f * camY / dyq, Zw)
        acc = np.zeros_like(xq)
        n = 30
        zs = np.linspace(camZ + 0.6, Zw - 0.05, n)
        for Zs in zs:
            dZ = Zs - camZ
            X = (xq - cx) * dZ / f
            Y = camY + (cy - yq) * dZ / f
            valid = (Y > 0) & (Zs < depthq)
            sS = (Zw - Zs) / Sz
            acc += self.win(X + sS * Sx, Y + sS * Sy, soft=0.06) * valid
        acc *= (zs[1] - zs[0]) * (0.55 + 0.9 * self.dens)
        acc = cv2.GaussianBlur(acc, (0, 0), 1.2)
        shaft = cv2.resize(acc, (W, H), interpolation=cv2.INTER_CUBIC)
        img += rgb(shaft * 0.085, (1.0, 0.70, 0.40))
        # dust motes in the beams
        X = self.dX + 0.05 * np.sin(0.4 * t + self.dph)
        Y = self.dY + 0.06 * np.cos(0.3 * t + self.dph) - 0.02 * t
        Z = self.dZ
        dZ = Z - camZ
        ok = dZ > 0.3
        sx_ = cx + X * f / dZ
        sy_ = cy - (Y - camY) * f / dZ
        sS = (Zw - Z) / Sz
        lit = self.win(X + sS * Sx, Y + sS * Sy, soft=0.04) * ok
        v = lit * (0.6 + 0.4 * np.sin(2 * t + self.dph * 3)) * 0.9
        near = dZ < 3.2
        dust = splat(sx_[~near], sy_[~near], v[~near], 1.0) + splat(sx_[near], sy_[near], v[near] * 0.25, 6.0)
        img += rgb(dust, (1.0, 0.85, 0.6))
        return img


class Palette:
    """The colourways, as cloth. Hard cuts."""
    T = 4.0
    COLS = [("OLIVE", (0.29, 0.32, 0.18), 1.4), ("BEIGE", (0.76, 0.64, 0.47), 1.45),
            ("NAVY", (0.075, 0.10, 0.20), 1.9), ("CHARCOAL", (0.17, 0.17, 0.18), 1.6),
            ("OFF-WHITE", (0.90, 0.87, 0.80), 1.40)]
    LINES = []

    def __init__(self):
        self.twill = (0.9 * np.sin((XX + YY) * TAU / 8.0)).astype(np.float32)
        self.fib = fbm(H, W, 14, 81, 3)
        self.big = fbm(H, W, 600, 82, 3)

    def render(self, t):
        k = min(int(t / 0.8), 4)
        lt = t - 0.8 * k
        name, alb, gain = self.COLS[k]
        a = 0.5 + k * 1.1
        ca, sa = math.cos(a), math.sin(a)
        p = XX * ca + YY * sa
        q = -XX * sa + YY * ca
        w1 = p / (145 + 20 * k) + 0.8 * np.sin(q / 380 + k) + lt * 0.9 + k
        w2 = q / 260 + p / 700 + 1.3 - lt * 0.6
        hbig = (-190 * np.abs(np.sin(w1)) ** 0.75 + 50 * np.sin(w2)
                + 160 * self.big)
        h = hbig + self.twill + (self.fib - 0.5) * 0.8
        nx, ny, nz = normals_from_height(h.astype(np.float32), 1.0)
        L = unit((-0.55, -0.70, 0.45))
        dif = np.clip(nx * L[0] + ny * L[1] + nz * L[2], 0, 1)
        fill = np.clip(nx * 0.6 + ny * 0.2 + nz * 0.75, 0, 1) * 0.18
        ao = 0.40 + 0.60 * sstep(-260, 120, hbig)
        sheen = (1 - nz) ** 3 * dif * 0.10
        light = (dif * 1.05 + fill + 0.08) * ao * gain
        img = rgb(light * (0.9 + 0.2 * self.fib), 1) * np.array(alb, np.float32) * np.array([1.0, 0.96, 0.9], np.float32)
        img += rgb(sheen, (1.0, 0.92, 0.8))
        sc = 1 + 0.04 * lt
        M = np.float32([[sc, 0, (1 - sc) * W / 2], [0, sc, (1 - sc) * H / 2]])
        img = cv2.warpAffine(img, M, (W, H), borderMode=cv2.BORDER_REFLECT)
        op = ease_out(lt / 0.12)
        comp_text(img, text_alpha("—  Nº 0%d  —" % (k + 1), F_CAPS, 26, 8, 500), W / 2, H * 0.665, op=op * 0.85, shadow=0.7)
        comp_text(img, text_alpha(name, F_CAPS, 58, 28, 500), W / 2, H * 0.705, op=op, shadow=0.7)
        return img


class Earned:
    """Deep green. A gold line. The tagline."""
    T = 4.2
    LINES = []

    def __init__(self):
        self.suede = fbm(H, W, 9, 91, 3)
        self.big = fbm(H, W, 500, 92, 3)
        r = np.sqrt(((XX - W / 2) / (W * 0.75)) ** 2 + ((YY - H * 0.48) / (H * 0.55)) ** 2)
        self.rad = np.clip(1 - r, 0, 1) ** 1.3

    def render(self, t):
        g = (0.20 + 0.85 * self.rad) * (0.9 + 0.12 * self.suede) * (0.85 + 0.3 * self.big)
        img = rgb(g, GREEN * 0.95)
        cyl = H * 0.5
        lw = 470 * ease_out((t - 0.25) / 1.1)
        if lw > 1:
            ln = sstep(1.6, 0.4, np.abs(YY - cyl)) * sstep(lw, lw - 30, np.abs(XX - W / 2))
            img = img * (1 - ln[..., None]) + gold_fill(1, 1)[0, 0] * ln[..., None] * 1.1
            glow = cv2.GaussianBlur(ln, (0, 0), 6) * 0.6
            img += rgb(glow, GOLD)
        sh = (t - 2.4) / 1.1 if t > 2.4 else None
        sh = sh if sh is not None and sh < 1 else None
        for text, path, size, trk, rtl, t0, cy in [
                ("THE UNCOMMON", F_CAPS, 80, 20, False, 0.55, cyl - 92),
                ("is Earned.", F_SERIF_IT, 136, 0, False, 1.15, cyl + 105),
                ("النادر… بينكسب.", F_AR, 60, 0, True, 1.75, cyl + 245)]:
            op = ease_out((t - t0) / 0.9)
            if op <= 0:
                continue
            kk = ease_out((t - t0) / 1.0)
            a = text_alpha(text, path, size, trk, 500 if not rtl else None, rtl)
            comp_text(img, a, W / 2, cy + (1 - kk) * 18, op=op * (0.9 if rtl else 1), blur=(1 - kk) * 7,
                      gold=True, shimmer=sh, shadow=0.5)
        return img


class Logo:
    """ANFAL. أنفال. Few pieces remain."""
    T = 6.0
    LINES = []

    def __init__(self):
        rng = np.random.default_rng(101)
        N = 260
        self.px, self.py = rng.random(N) * W, rng.random(N) * H
        self.vy, self.ph = -rng.random(N) * 18 - 4, rng.random(N) * TAU
        self.br, self.z = rng.random(N), rng.random(N)
        r = np.sqrt(((XX - W / 2) / W) ** 2 + ((YY - H * 0.42) / (H * 0.6)) ** 2)
        self.bg = np.exp(-r * 2.2).astype(np.float32)

    def render(self, t):
        img = rgb(self.bg * 0.075, (0.9, 0.68, 0.40)) + np.array([0.006, 0.005, 0.004], np.float32)
        x = self.px + 6 * np.sin(0.5 * t + self.ph)
        y = self.py + self.vy * t
        v = self.br * 0.35 * (0.6 + 0.4 * np.sin(2 * t + self.ph * 4)) * ease_out(t / 1.5)
        dust = splat(x[self.z < 0.8], y[self.z < 0.8], v[self.z < 0.8], 0.9) + \
            splat(x[self.z >= 0.8], y[self.z >= 0.8], v[self.z >= 0.8] * 0.2, 8)
        img += rgb(dust, GOLD * 1.2)
        cy = H * 0.40
        trk = int(round(130 - 80 * ease_out(t / 2.6)))
        a = text_alpha("ANFAL", F_CAPS, 196, trk, 500)
        k = ease_out(t / 1.6)
        sh = (t - 3.0) / 1.2 if 3.0 < t < 4.2 else None
        comp_text(img, a, W / 2, cy, op=k, blur=(1 - k) * 12, gold=True, shimmer=sh, shadow=0)
        for text, path, size, tk, rtl, t0, dy, gold, opm in [
                ("أنفال", F_AR, 92, 0, True, 0.95, 190, True, 0.95),
                ("Few pieces remain.", F_SERIF_IT, 66, 0, False, 2.25, 395, False, 1.0),
                ("ما ضلّ إلا كم قطعة.", F_AR, 48, 0, True, 2.65, 475, False, 0.85),
                ("@ANFAL.CO    ·    ANFALCO.COM", F_CAPS, 32, 8, False, 3.3, 640, False, 0.80)]:
            op = ease_out((t - t0) / 0.9) * opm
            if op <= 0:
                continue
            kk = ease_out((t - t0) / 1.0)
            al = text_alpha(text, path, size, tk, None if rtl else 500, rtl)
            comp_text(img, al, W / 2, cy + dy + (1 - kk) * 16, op=op, blur=(1 - kk) * 6, gold=gold, shadow=0)
        lw = 90 * ease_out((t - 1.6) / 0.9)
        if lw > 1:
            ly = cy + 300
            ln = sstep(1.4, 0.3, np.abs(YY - ly)) * sstep(lw, lw - 20, np.abs(XX - W / 2))
            img = img * (1 - ln[..., None]) + GOLD * ln[..., None]
        return img


SCENES = [Beam, Ashes, Dunes, Hooves, Weave, Cable, Arches, Palette, Earned, Logo]
STARTS = np.cumsum([0] + [S.T for S in SCENES]).tolist()
TOTAL = STARTS[-1]
NFRAMES = int(round(TOTAL * FPS))


# ----------------------------------------------------------------------------
# post
# ----------------------------------------------------------------------------
_VIG = None
_GRAIN = None


def _post_init():
    global _VIG, _GRAIN
    r = np.sqrt(((XX - W / 2) / (W / 2)) ** 2 * 0.85 + ((YY - H / 2) / (H / 2)) ** 2)
    _VIG = (1 - 0.38 * sstep(0.45, 1.45, r)).astype(np.float32)
    rng = np.random.default_rng(999)
    g = []
    for _ in range(8):
        n = rng.standard_normal((int(H / 1.4), int(W / 1.4))).astype(np.float32)
        n = cv2.resize(n, (W, H), interpolation=cv2.INTER_CUBIC)
        g.append(n / n.std())
    _GRAIN = g


def post(img, fi):
    if _VIG is None:
        _post_init()
    small = cv2.resize(img, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
    br = np.clip(small - 0.72, 0, None)
    bl = cv2.GaussianBlur(br, (0, 0), 14) * 0.7 + cv2.GaussianBlur(br, (0, 0), 4) * 0.4
    img = img + cv2.resize(bl, (W, H), interpolation=cv2.INTER_LINEAR) * 0.55
    k = 0.82
    hi = img > k
    img = np.where(hi, k + (1 - k) * (1 - np.exp(-(img - k) / (1 - k))), img)
    img = img * _VIG[..., None]
    img = img * 0.975 + np.array([0.012, 0.009, 0.007], np.float32)
    lum = img.mean(axis=2, keepdims=True)
    amt = 0.022 * (0.35 + 0.65 * np.sqrt(np.clip(lum * (1.4 - lum) * 2.2, 0, 1)))
    img = img + _GRAIN[fi % 8][..., None] * amt
    return (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)


# ----------------------------------------------------------------------------
# frame dispatch
# ----------------------------------------------------------------------------
_inst = {}


def scene_at(tg):
    for i, S in enumerate(SCENES):
        if tg < STARTS[i + 1] or i == len(SCENES) - 1:
            return i, S, tg - STARTS[i]


def render_frame(fi):
    tg = fi / FPS
    i, S, t = scene_at(tg)
    if S not in _inst:
        _inst.clear()  # one scene's precompute in memory per worker at a time
        _inst[S] = S()
    sc = _inst[S]
    img = sc.render(t).astype(np.float32)
    for (text, style, t0, cy) in S.LINES:
        line_item(img, t, text, style, t0, S.T - 0.1, H * cy)
    # blink-dips between scenes
    fin = 1.0 if i == 0 else 0.10
    fout = 0.9 if i == len(SCENES) - 1 else 0.10
    a = ease_io(t / fin) * (1 - ease_io((t - (S.T - fout)) / fout))
    img *= a
    return post(img, fi).tobytes()


# ----------------------------------------------------------------------------
# audio: sound design only, all filtered noise. No pitch, no music.
# ----------------------------------------------------------------------------
def build_audio(path):
    from scipy.signal import butter, sosfilt, lfilter, fftconvolve
    sr = 48000
    n = int(TOTAL * sr) + sr // 2
    tt = np.arange(n) / sr
    rng = np.random.default_rng(7)

    def nz(m=n):
        return rng.standard_normal(m)

    def filt(x, kind, f):
        return sosfilt(butter(2, f, kind, fs=sr, output="sos"), x)

    def envpts(pts):
        ts, vs = zip(*pts)
        return np.interp(tt, ts, vs)

    def gusts(rate, seed):
        r = np.random.default_rng(seed)
        m = int(TOTAL * rate) + 4
        pts = r.random(m)
        return np.interp(tt, np.arange(m) / rate, pts)

    def decays(times, amps, tau, m=n):
        imp = np.zeros(m)
        for t0, a in zip(times, amps):
            i = int(t0 * sr)
            if 0 <= i < m:
                imp[i] += a
        return lfilter([1.0], [1.0, -math.exp(-1.0 / (sr * tau))], imp)

    S = dict(zip([c.__name__ for c in SCENES], STARTS))
    out = np.zeros((2, n))

    # wind bed, level follows the scenes
    lvl = envpts([(0, 0.05), (S["Ashes"], 0.10), (S["Dunes"] - 0.3, 0.12), (S["Dunes"] + 0.4, 0.85),
                  (S["Hooves"], 0.55), (S["Weave"], 0.14), (S["Cable"], 0.10), (S["Arches"], 0.18),
                  (S["Palette"], 0.10), (S["Earned"], 0.08), (S["Logo"], 0.06), (TOTAL - 1.0, 0.03), (TOTAL + 1, 0)])
    for ch in range(2):
        w = filt(filt(nz(), "low", 380), "low", 900) * 1.3
        g = 0.35 + 0.65 * gusts(0.7, 10 + ch)
        out[ch] += w * g * lvl
        hiss = filt(nz(), "high", 3200) * (0.25 + 0.75 * gusts(1.3, 20 + ch))
        hl = envpts([(0, 0), (S["Dunes"] - 0.2, 0), (S["Dunes"] + 0.5, 0.20), (S["Hooves"] + 0.3, 0.08),
                     (S["Weave"], 0.0), (TOTAL + 1, 0)])
        out[ch] += hiss * hl

    # fire: low roar + crackle
    fl = envpts([(0, 0), (S["Ashes"] - 0.1, 0), (S["Ashes"] + 0.3, 1), (S["Dunes"] - 0.2, 1), (S["Dunes"] + 0.1, 0), (TOTAL + 1, 0)])
    for ch in range(2):
        roar = filt(filt(nz(), "low", 160), "high", 40) * 0.9
        times = np.sort(rng.uniform(S["Ashes"], S["Dunes"], 140))
        amps = rng.random(140) ** 3 * 2.2
        cr = filt(nz(), "high", 1800) * decays(times, amps, 0.0035)
        out[ch] += (roar + cr) * fl

    # hooves on sand
    hv = SCENES[3]()
    for tau in hv.thud_times():
        t0 = S["Hooves"] + tau
        body = filt(nz(), "low", 140) * decays([t0], [1.0], 0.07) * 2.6
        crunch = filt(filt(nz(), "high", 1400), "low", 5500) * decays([t0 + 0.004], [1.0], 0.05) * 0.7
        pan = 0.5 + 0.12 * (1 if int(tau / 0.6) % 2 else -1)
        out[0] += (body + crunch) * (1 - pan) * 2
        out[1] += (body + crunch) * pan * 2
    del hv

    # fabric: rustle swells over the weave, swishes on the cuts
    def swish(t0, dur, amp, lo=700, hi=6000):
        a, d = dur * 0.35, dur * 0.65
        e = np.clip((tt - t0) / a, 0, 1) ** 2 * np.exp(-np.clip(tt - t0 - a, 0, None) / (d / 3))
        e[tt < t0] = 0
        x = filt(filt(nz(), "high", lo), "low", hi)
        return x * e * amp

    rust = envpts([(0, 0), (S["Weave"], 0), (S["Weave"] + 0.6, 0.20), (S["Cable"] + 4.0, 0.15), (S["Arches"], 0), (TOTAL + 1, 0)])
    for ch in range(2):
        out[ch] += filt(filt(nz(), "high", 900), "low", 5000) * (0.2 + 0.8 * gusts(2.2, 30 + ch)) ** 2 * rust
    cuts = [S["Weave"] + 0.05, S["Cable"], S["Cable"] + 1.4, S["Cable"] + 2.8]
    for i, c in enumerate(cuts):
        x = swish(c - 0.12, 0.45, 0.55)
        out[0] += x * (0.6 if i % 2 else 0.4)
        out[1] += x * (0.4 if i % 2 else 0.6)
    for i in range(5):
        c = S["Palette"] + 0.8 * i
        x = swish(c - 0.06, 0.25, 0.75, 300, 3500)
        out[0] += x * (0.55 if i % 2 else 0.45)
        out[1] += x * (0.45 if i % 2 else 0.55)

    # stone hall: room tone + leather soles on stone, with a long tail
    rt = envpts([(0, 0), (S["Arches"] - 0.1, 0), (S["Arches"] + 0.3, 0.12), (S["Palette"] - 0.2, 0.12), (S["Palette"], 0), (TOTAL + 1, 0)])
    for ch in range(2):
        out[ch] += filt(nz(), "low", 260) * rt
    steps = [S["Arches"] + x for x in (0.55, 1.45, 2.35, 3.25)]
    dry = np.zeros(n)
    for i, st in enumerate(steps):
        heel = filt(nz(), "high", 900) * decays([st], [1.0], 0.012) * 0.9
        knock = filt(nz(), "low", 300) * decays([st + 0.002], [1.0], 0.03) * 1.2
        dry += (heel + knock) * (0.7 + 0.1 * i)
    ir_t = np.arange(int(1.8 * sr)) / sr
    for ch, g in ((0, 0.55), (1, 0.45)):
        ir = filt(nz(len(ir_t)), "low", 3000) * np.exp(-ir_t / 0.45)  # independent tail per channel
        ir /= np.sqrt((ir ** 2).sum())
        out[ch] += dry * g + fftconvolve(dry, ir)[:n] * 0.6

    # a single breath of air under the logo
    x = swish(S["Logo"] - 0.2, 2.2, 0.35, 250, 2500)
    out += x

    # end in silence
    fade = np.clip((TOTAL - tt) / 1.2, 0, 1)
    out *= fade
    out *= np.clip(tt / 0.6, 0, 1)
    out = filt(out, "high", 25)
    peak = np.abs(out).max()
    out = out / peak * 0.70
    pcm = (np.clip(out.T, -1, 1) * 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", default=None, help="comma separated seconds")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    if args.preview:
        pdir = os.path.join(OUT_DIR, "preview")
        os.makedirs(pdir, exist_ok=True)
        for s in args.preview.split(","):
            fi = int(round(float(s) * FPS))
            buf = render_frame(fi)
            im = np.frombuffer(buf, np.uint8).reshape(H, W, 3)
            p = os.path.join(pdir, "f_%05.2f.jpg" % float(s))
            Image.fromarray(im).save(p, quality=88)
            print(p)
        return

    import multiprocessing as mp
    raw = os.path.join(OUT_DIR, "video_only.mp4")
    wav = os.path.join(OUT_DIR, "sound_design.wav")
    print("audio ...", flush=True)
    build_audio(wav)
    print("frames: %d (%.1fs)" % (NFRAMES, TOTAL), flush=True)
    enc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                            "-s", "%dx%d" % (W, H), "-r", str(FPS), "-i", "-",
                            "-c:v", "libx264", "-preset", "slow", "-crf", "20", "-maxrate", "10M",
                            "-bufsize", "20M", "-tune", "film",
                            "-pix_fmt", "yuv420p", "-profile:v", "high", "-movflags", "+faststart", raw],
                           stdin=subprocess.PIPE)
    ctx = mp.get_context("fork")
    with ctx.Pool(args.workers) as pool:
        # contiguous chunks so each worker keeps one scene's textures warm
        for k, buf in enumerate(pool.imap(render_frame, range(NFRAMES), chunksize=6)):
            enc.stdin.write(buf)
            if k % 60 == 0:
                print("  %d/%d" % (k, NFRAMES), flush=True)
    enc.stdin.close()
    enc.wait()
    final = os.path.join(HERE, "anfal_showreel.mp4")
    silent = os.path.join(HERE, "anfal_showreel_silent.mp4")
    subprocess.check_call(["ffmpeg", "-y", "-loglevel", "error", "-i", raw, "-i", wav, "-map", "0:v", "-map", "1:a",
                           "-c:v", "copy", "-af", "loudnorm=I=-18:TP=-1.5:LRA=14", "-ar", "48000",
                           "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", final])
    subprocess.check_call(["ffmpeg", "-y", "-loglevel", "error", "-i", raw, "-c:v", "copy", "-an",
                           "-movflags", "+faststart", silent])
    os.remove(raw)
    print("done:", final, silent)


if __name__ == "__main__":
    main()
