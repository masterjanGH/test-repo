#!/usr/bin/env python3
"""
ANFAL — "اسأل اللي جرّبوا · Ask the ones who wore it"  (showreel v2, customer-led)

Built only from Anfal's own material: customer photos and campaign shots from
anfalco.com, the logo, and the packaging photo. Nothing is invented: no quotes,
no numbers, no generated faces. Lines on screen come from Anfal's own site copy.
Audio is sound design only (paper, ribbon, card flicks, sea air) from filtered
noise: no music, no instruments, no pitched tones.

  python3 render_v2.py                 # -> anfal_showreel_v2.mp4 (+ _silent)
  python3 render_v2.py --preview 1,5,9 # stills in out/preview_v2/
"""
import os, math, functools, argparse, subprocess, wave
import numpy as np
import cv2
import render as R
from render import (W, H, FPS, TAU, YY, XX, sstep, ease_out, ease_io, fbm, splat, rgb,
                    text_alpha, comp_text, gold_fill, F_SERIF_IT, F_CAPS, F_AR, GOLD, GREEN, CREAM)

HERE = os.path.dirname(os.path.abspath(__file__))
A = os.path.join(HERE, "assets")
OUT = os.path.join(HERE, "out")


# ----------------------------------------------------------------------------
# photos
# ----------------------------------------------------------------------------
@functools.lru_cache(maxsize=None)
def photo(name, flip=False, grade=True):
    im = cv2.imread(os.path.join(A, name), cv2.IMREAD_COLOR)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    if flip:
        im = im[:, ::-1]
    h, w = im.shape[:2]
    if h > 2700:  # pre-shrink big masters so the moves never alias
        im = cv2.resize(im, (int(w * 2700 / h), 2700), interpolation=cv2.INTER_AREA)
    im = im.astype(np.float32) / 255.0
    if grade:  # one warm, slightly muted grade so phones and camera shots sit together
        lum = (im @ np.array([0.299, 0.587, 0.114], np.float32))[..., None]
        im = lum + (im - lum) * 0.84
        im = im * np.array([1.035, 1.0, 0.925], np.float32)
        im = 0.5 + (im - 0.5) * 1.07
        im = np.clip(im, 0, 1) * 0.92
    return np.ascontiguousarray(im)


def cover(img, z, cx, cy, ow=W, oh=H):
    ih, iw = img.shape[:2]
    s = max(ow / iw, oh / ih) * z
    hw, hh = ow / (2 * s * iw), oh / (2 * s * ih)
    cx = min(max(cx, hw), 1 - hw)
    cy = min(max(cy, hh), 1 - hh)
    M = np.float32([[s, 0, ow / 2 - s * cx * iw], [0, s, oh / 2 - s * cy * ih]])
    return cv2.warpAffine(img, M, (ow, oh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


@functools.lru_cache(maxsize=None)
def blurred_bg(name, flip=False):
    img = photo(name, flip)
    sm = cover(img, 1.25, 0.5, 0.5, W // 4, H // 4)
    sm = cv2.GaussianBlur(sm, (0, 0), 9)
    bg = cv2.resize(sm, (W, H), interpolation=cv2.INTER_CUBIC)
    bg = bg * 0.26 * np.array([0.85, 1.0, 0.92], np.float32) + GREEN * 0.16
    return bg


def card_frame(name, lt, dur, flip=False, box=(860, 1240), cy=0.49, zoom=(1.0, 1.06), settle=True):
    img = photo(name, flip)
    ih, iw = img.shape[:2]
    f = min(box[0] / iw, box[1] / ih)
    cw, ch = int(iw * f) // 2 * 2, int(ih * f) // 2 * 2
    out = blurred_bg(name, flip).copy()
    z = zoom[0] + (zoom[1] - zoom[0]) * ease_io(lt / dur)
    c = cover(img, z, 0.5, 0.45, cw, ch)
    dy = int((1 - ease_out(lt / 0.28)) * 34) if settle else 0
    x0, y0 = (W - cw) // 2, int(H * cy - ch / 2) + dy
    # soft shadow
    sh = np.zeros((H, W), np.float32)
    sh[y0 + 24:y0 + ch + 24, x0 + 6:x0 + cw - 6] = 1
    sh = cv2.GaussianBlur(sh, (0, 0), 28)
    out *= 1 - sh[..., None] * 0.65
    out[y0:y0 + ch, x0:x0 + cw] = c
    g = tuple(float(v) for v in GOLD * 1.15)
    cv2.rectangle(out, (x0 - 16, y0 - 16), (x0 + cw + 15, y0 + ch + 15), g, 2)
    return out, (x0, y0, cw, ch)


# ----------------------------------------------------------------------------
# text
# ----------------------------------------------------------------------------
def txt(img, t, text, font, size, t0, t1, cy, gold=False, track=0, rtl=False, op=1.0,
        shadow=0.6, color=CREAM, fin=0.6, fout=0.25, shimmer=None, wght=500, rise=18):
    if t < t0 or t > t1:
        return
    o = ease_out((t - t0) / fin) * (1 - ease_io((t - (t1 - fout)) / fout)) * op
    if o <= 0.003:
        return
    k = ease_out((t - t0) / (fin + 0.25))
    a = text_alpha(text, font, size, track, None if rtl else wght, rtl)
    comp_text(img, a, W / 2, cy + (1 - k) * rise, op=o, blur=(1 - k) * 6, gold=gold,
              shadow=shadow, color=color, shimmer=shimmer)


def light_spot(cx, cy, r):
    return np.exp(-(((XX - cx) / r) ** 2 + ((YY - cy) / (r * 1.25)) ** 2)).astype(np.float32)


# ----------------------------------------------------------------------------
# scenes
# ----------------------------------------------------------------------------
class Hook:
    """Dark. Light finds the horse embroidered on the collar. اسأل اللي جرّبوا."""
    T = 3.4

    def __init__(self):
        rng = np.random.default_rng(3)
        N = 500
        self.px, self.py = rng.random(N) * W, rng.random(N) * H
        self.vx, self.vy = rng.normal(-4, 5, N), rng.normal(6, 5, N)
        self.ph, self.br, self.z = rng.random(N) * TAU, rng.random(N), rng.random(N)

    def render(self, t):
        T = self.T
        img = cover(photo("prod_DSC09036-2400.jpg"), 1.95 - 0.30 * ease_io(t / T), 0.69, 0.43)
        k = ease_out(t / 1.5)
        lx = -0.25 * W + (0.52 * W + 0.25 * W) * k
        ly = 0.40 * H + 0.06 * H * k
        sp = light_spot(lx, ly, 360 + 40 * math.sin(t * 1.3))
        warm = np.array([1.12, 0.98, 0.80], np.float32)
        out = img * (0.22 + rgb(sp * 1.25, warm))
        x = self.px + self.vx * t + 6 * np.sin(0.6 * t + self.ph)
        y = self.py + self.vy * t
        xi, yi = np.clip(x, 0, W - 1).astype(int), np.clip(y, 0, H - 1).astype(int)
        v = self.br * sp[yi, xi] * 0.8
        d = R.splat(x[self.z < 0.8], y[self.z < 0.8], v[self.z < 0.8], 0.9) + \
            R.splat(x[self.z >= 0.8], y[self.z >= 0.8], v[self.z >= 0.8] * 0.2, 8)
        out += rgb(d, (1.0, 0.86, 0.6))
        txt(out, t, "اسأل اللي جرّبوا", F_AR, 112, 0.45, T, H * 0.715, gold=True, rtl=True,
            shimmer=(t - 1.7) / 1.0 if 1.7 < t < 2.7 else None, fout=0.01, shadow=0.75)
        txt(out, t, "Ask the ones who wore it.", F_SERIF_IT, 60, 0.95, T, H * 0.715 + 120, fout=0.01, shadow=0.75)
        return out


class OneOrder:
    """The green bag. It starts with one order."""
    T = 2.6

    def render(self, t):
        out = cover(photo("packaging.png"), 1.32 + 0.16 * ease_io(t / self.T), 0.51, 0.40)
        out = out * 0.95
        txt(out, t, "It starts with one order.", F_SERIF_IT, 76, 0.15, self.T, H * 0.70, fout=0.01, shadow=0.7)
        txt(out, t, "كلّو بيبلّش بطلبيّة وحدة.", F_AR, 56, 0.45, self.T, H * 0.70 + 92, rtl=True, op=0.9,
            fout=0.01, shadow=0.7)
        return out


class Arrives:
    """Customers holding the bag (mirror shots flipped so ANFAL reads right)."""
    T = 2.4
    SHOTS = ["site_Screenshot_2026-05-02_at_5.39.20_PM.jpg", "site_Screenshot_2026-05-02_at_5.43.31_PM.jpg"]

    def render(self, t):
        k = 0 if t < 1.2 else 1
        lt = t - 1.2 * k
        out, _ = card_frame(self.SHOTS[k], lt, 1.2, flip=True, box=(860, 1080), cy=0.43, zoom=(1.0, 1.05))
        txt(out, t, "It arrives like it should.", F_SERIF_IT, 70, 0.1, self.T, H * 0.765, fout=0.01)
        txt(out, t, "وبتوصل متل ما لازم.", F_AR, 52, 0.4, self.T, H * 0.765 + 86, rtl=True, op=0.9, fout=0.01)
        return out


class Wall:
    """Nine real customers. A counter that keeps going."""
    SHOTS = [("cust_mvqrbmyq-jDRo9leK-sm.jpg", 1.7), ("site_59584A49-4693-4DE5-8F0C-06334011554F.jpg", 1.7),
             ("site_E892EDC1-9170-4E2F-9360-261759DB7D39.jpg", 1.6), ("cust_mvqrbv65-OO89Dgji-sm.jpg", 1.3),
             ("site_03452287-d132-489d-8bfc-b25b9e72fbbe.jpg", 1.3), ("site_855966bd-7a68-42dc-8b42-0b3d278de352.jpg", 1.2),
             ("site_feedback1.jpg", 1.1), ("site_fabd8da2-4377-46aa-9ded-714bfaeef9ea_2.jpg", 0.9),
             ("site_ca102277-7024-4b91-9505-bc5abd2f004e.jpg", 1.7)]
    T = sum(d for _, d in SHOTS)

    @classmethod
    def cut_times(cls):
        return np.cumsum([0] + [d for _, d in cls.SHOTS])[:-1].tolist()

    def render(self, t):
        starts = self.cut_times()
        k = max(i for i, s in enumerate(starts) if t >= s - 1e-9)
        name, dur = self.SHOTS[k]
        lt = t - starts[k]
        out, (x0, y0, cw, ch) = card_frame(name, lt, dur, box=(860, 1230), cy=0.505)
        hd = text_alpha("WORN BY THE ONES WHO EARNED IT", F_CAPS, 30, 9, 500)
        comp_text(out, hd, W / 2, H * 0.072, op=ease_out(t / 0.5), shadow=0.4)
        n = text_alpha("Nº %02d" % (k + 1), F_CAPS, 30, 6, 500)
        comp_text(out, n, W / 2, H * 0.112, op=ease_out(lt / 0.15), gold=True, shadow=0.3)
        return out


class Gift:
    """The card inside. For your brother. Your father. Your friend."""
    T = 4.2

    def render(self, t):
        T = self.T
        k = ease_io(t / T)
        out = cover(photo("packaging.png"), 1.72 + 0.18 * k, 0.60 - 0.02 * k, 0.80 + 0.01 * k) * 0.95
        for i, (line, t0) in enumerate([("For your brother.", 0.25), ("Your father.", 1.05), ("Your friend.", 1.85)]):
            txt(out, t, line, F_SERIF_IT, 84, t0, T, H * (0.12 + 0.058 * i), fout=0.01, shadow=0.8)
        txt(out, t, "لخيّك. لبيّك. لرفيقك.", F_AR, 58, 2.55, T, H * 0.12 + 0.058 * 3 * H + 6, rtl=True,
            op=0.9, fout=0.01, shadow=0.8)
        return out


class Lebanon:
    """Campaign cuts. Made 100% in Lebanon."""
    SHOTS = [("prod_3FE3C476-4783-42BA-BAC7-D0F711B94FCA.jpg", 0.5, 0.42),
             ("prod_DSC08842_1.jpg", 0.55, 0.40),
             ("prod_DSC09240.jpg", 0.45, 0.42),
             ("prod_poloBrown1.jpg", 0.45, 0.40)]
    T = 3.8

    def render(self, t):
        k = min(int(t / 0.95), 3)
        lt = t - 0.95 * k
        name, cx, cy = self.SHOTS[k]
        out = cover(photo(name), 1.04 + 0.05 * lt, cx, cy)
        sh = sstep(0.62, 0.95, YY / H)[..., None]
        out = out * (1 - 0.45 * sh)
        txt(out, t, "Made 100% in Lebanon.", F_SERIF_IT, 76, 0.1, self.T, H * 0.765, fout=0.01, shadow=0.85)
        txt(out, t, "معمولة ١٠٠٪ بلبنان.", F_AR, 56, 0.4, self.T, H * 0.765 + 90, rtl=True, op=0.92, fout=0.01,
            shadow=0.85)
        return out


@functools.lru_cache(maxsize=4)
def logo_alpha(width):
    im = cv2.imread(os.path.join(A, "logo_white.png"), cv2.IMREAD_UNCHANGED)
    a = im[..., 3].astype(np.float32) / 255.0
    ys, xs = np.nonzero(a > 0.04)
    a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h = int(a.shape[0] * width / a.shape[1])
    return cv2.resize(a, (width, h), interpolation=cv2.INTER_AREA)


class EndCard:
    """Anfal green. The gold mark draws in. Your turn. WhatsApp."""
    T = 6.5

    def __init__(self):
        self.suede = fbm(H, W, 9, 91, 3)
        self.big = fbm(H, W, 500, 92, 3)
        r = np.sqrt(((XX - W / 2) / (W * 0.75)) ** 2 + ((YY - H * 0.40) / (H * 0.55)) ** 2)
        self.rad = np.clip(1 - r, 0, 1) ** 1.3

    def render(self, t):
        g = (0.20 + 0.85 * self.rad) * (0.9 + 0.12 * self.suede) * (0.85 + 0.3 * self.big)
        out = rgb(g, GREEN * 0.95)
        a = logo_alpha(560)
        h, w = a.shape
        yy = np.linspace(0, 1, h, dtype=np.float32)[:, None]
        xx = np.linspace(0, 1, w, dtype=np.float32)[None, :]
        p = yy * 0.75 + xx * 0.25
        r = -0.1 + 1.25 * ease_io(t / 1.6)
        m = sstep(r + 0.06, r - 0.06, p)
        al = a * m
        blur = (1 - ease_out(t / 1.8)) * 5
        cy = H * 0.31
        sh = (t - 2.0) / 1.2 if 2.0 < t < 3.2 else None
        comp_text(out, np.pad(al, 40), W / 2, cy, op=1.0, gold=True, blur=blur, shimmer=sh, shadow=0.35)
        txt(out, t, "The Uncommon is Earned.", F_SERIF_IT, 70, 1.3, self.T + 1, cy + h / 2 + 95, gold=True, shadow=0.3)
        txt(out, t, "Your turn.", F_SERIF_IT, 96, 2.4, self.T + 1, H * 0.585, shadow=0.3)
        txt(out, t, "دورك.", F_AR, 64, 2.7, self.T + 1, H * 0.585 + 95, rtl=True, op=0.9, shadow=0.3)
        txt(out, t, "ORDER ON WHATSAPP", F_CAPS, 28, 3.3, self.T + 1, H * 0.70, gold=True, track=10, shadow=0.2, rise=8)
        txt(out, t, "+961 76 790 486", F_CAPS, 44, 3.45, self.T + 1, H * 0.735, track=5, shadow=0.2, rise=8)
        txt(out, t, "ANFALCO.COM   ·   @ANFAL.CO", F_CAPS, 24, 3.7, self.T + 1, H * 0.775, track=7, op=0.8,
            shadow=0.2, rise=8)
        return out


SCENES = [Hook, OneOrder, Arrives, Wall, Gift, Lebanon, EndCard]
STARTS = np.cumsum([0] + [S.T for S in SCENES]).tolist()
TOTAL = STARTS[-1]
NFRAMES = int(round(TOTAL * FPS))
_inst = {}


def render_frame(fi):
    tg = fi / FPS
    i = max(j for j in range(len(SCENES)) if tg >= STARTS[j] - 1e-9)
    S = SCENES[i]
    t = tg - STARTS[i]
    if S not in _inst:
        _inst[S] = S() if "__init__" in S.__dict__ else S()
    img = _inst[S].render(t).astype(np.float32)
    a = 1.0
    if i == 0:
        a = ease_io(t / 0.2)
    if S is EndCard:
        a = ease_io(t / 0.3) * (1 - ease_io((t - (S.T - 0.5)) / 0.5) * 0.0)
    if i == len(SCENES) - 2:  # dip into the end card
        a *= 1 - ease_io((t - (S.T - 0.2)) / 0.2)
    img *= a
    return R.post(img, fi).tobytes()


# ----------------------------------------------------------------------------
# sound design: paper, ribbon, card flicks, sea air. Filtered noise only.
# ----------------------------------------------------------------------------
def build_audio(path):
    from scipy.signal import butter, sosfilt, lfilter
    sr = 48000
    n = int(TOTAL * sr) + sr // 2
    tt = np.arange(n) / sr
    rng = np.random.default_rng(17)
    S = dict(zip([c.__name__ for c in SCENES], STARTS))

    def nz(m=n):
        return rng.standard_normal(m)

    def filt(x, kind, f):
        return sosfilt(butter(2, f, kind, fs=sr, output="sos"), x)

    def envpts(pts):
        ts, vs = zip(*pts)
        return np.interp(tt, ts, vs)

    def decays(times, amps, tau):
        imp = np.zeros(n)
        for t0, a in zip(times, amps):
            i = int(t0 * sr)
            if 0 <= i < n:
                imp[i] += a
        return lfilter([1.0], [1.0, -math.exp(-1.0 / (sr * tau))], imp)

    def swell(t0, dur, amp, lo, hi):
        a, d = dur * 0.35, dur * 0.65
        e = np.clip((tt - t0) / a, 0, 1) ** 2 * np.exp(-np.clip(tt - t0 - a, 0, None) / (d / 3))
        e[tt < t0] = 0
        return filt(filt(nz(), "high", lo), "low", hi) * e * amp

    out = np.zeros((2, n))
    # quiet air under everything
    air = envpts([(0, 0.05), (S["Lebanon"], 0.05), (S["Lebanon"] + 0.2, 0.0), (S["EndCard"], 0.04),
                  (TOTAL - 1, 0.02), (TOTAL + 1, 0)])
    for ch in range(2):
        out[ch] += filt(filt(nz(), "low", 500), "low", 900) * 1.4 * air
    # hook: a slow brush of air as the light arrives
    out += swell(0.0, 1.8, 0.35, 300, 3000)
    # paper bag: crinkle bursts
    for ch in range(2):
        times = np.sort(rng.uniform(S["OneOrder"] + 0.05, S["OneOrder"] + 1.4, 70))
        amps = rng.random(70) ** 2 * 0.9
        out[ch] += filt(filt(nz(), "high", 1500), "low", 9000) * decays(times, amps, 0.006)
    out += swell(S["OneOrder"] - 0.1, 0.6, 0.35, 600, 6000)
    # ribbon / handle swish on each bag shot
    for c in (S["Arrives"], S["Arrives"] + 1.2):
        out += swell(c - 0.08, 0.35, 0.5, 900, 7000)
    # card flicks on every customer
    for i, c in enumerate(Wall.cut_times()):
        t0 = S["Wall"] + c
        flick = filt(filt(nz(), "high", 2200), "low", 9000) * decays([t0], [1.0], 0.012) * 0.9
        body = filt(filt(nz(), "high", 250), "low", 1100) * decays([t0 + 0.003], [1.0], 0.035) * 0.8
        pan = 0.5 + (0.08 if i % 2 else -0.08)
        out[0] += (flick + body) * (1 - pan) * 2
        out[1] += (flick + body) * pan * 2
    # gift card slides out of the bag
    out += swell(S["Gift"] + 0.05, 0.9, 0.45, 500, 5000)
    out += swell(S["Gift"] + 1.0, 0.7, 0.18, 1200, 6000)
    # Lebanon: sea and wind off the Corniche, swishes on the cuts
    sea = envpts([(0, 0), (S["Lebanon"] - 0.05, 0), (S["Lebanon"] + 0.1, 1), (S["EndCard"] - 0.2, 1),
                  (S["EndCard"] + 0.6, 0), (TOTAL + 1, 0)])
    wave_env = 0.45 + 0.55 * (0.5 + 0.5 * np.sin(TAU * (tt - S["Lebanon"]) / 2.6 - 1.2)) ** 2
    for ch in range(2):
        out[ch] += filt(filt(nz(), "low", 700), "high", 60) * wave_env * sea * 0.9
        out[ch] += filt(nz(), "high", 2500) * wave_env * sea * 0.12
    for k in range(1, 4):
        out += swell(S["Lebanon"] + 0.95 * k - 0.1, 0.3, 0.3, 700, 6000)
    # end card: one breath of air as the mark draws in
    out += swell(S["EndCard"] - 0.05, 1.8, 0.30, 250, 2500)
    fade = np.clip((TOTAL + 0.1 - tt) / 1.0, 0, 1)
    out *= fade
    out = filt(out, "high", 30)
    out = out / np.abs(out).max() * 0.7
    pcm = (np.clip(out.T, -1, 1) * 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", default=None)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if args.preview:
        from PIL import Image
        pdir = os.path.join(OUT, "preview_v2")
        os.makedirs(pdir, exist_ok=True)
        for s in args.preview.split(","):
            fi = int(round(float(s) * FPS))
            im = np.frombuffer(render_frame(fi), np.uint8).reshape(H, W, 3)
            p = os.path.join(pdir, "f_%05.2f.jpg" % float(s))
            Image.fromarray(im).save(p, quality=88)
        print("total %.1fs" % TOTAL)
        return
    import multiprocessing as mp
    raw = os.path.join(OUT, "v2_video_only.mp4")
    wav = os.path.join(OUT, "v2_sound.wav")
    build_audio(wav)
    print("frames %d (%.1fs)" % (NFRAMES, TOTAL), flush=True)
    enc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                            "-s", "%dx%d" % (W, H), "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow",
                            "-crf", "20", "-maxrate", "10M", "-bufsize", "20M", "-tune", "film", "-pix_fmt", "yuv420p",
                            "-profile:v", "high", "-movflags", "+faststart", raw], stdin=subprocess.PIPE)
    with mp.get_context("fork").Pool(args.workers) as pool:
        for k, buf in enumerate(pool.imap(render_frame, range(NFRAMES), chunksize=6)):
            enc.stdin.write(buf)
            if k % 120 == 0:
                print("  %d/%d" % (k, NFRAMES), flush=True)
    enc.stdin.close()
    enc.wait()
    final = os.path.join(HERE, "anfal_showreel_v2.mp4")
    silent = os.path.join(HERE, "anfal_showreel_v2_silent.mp4")
    subprocess.check_call(["ffmpeg", "-y", "-loglevel", "error", "-i", raw, "-i", wav, "-map", "0:v", "-map", "1:a",
                           "-c:v", "copy", "-af", "loudnorm=I=-18:TP=-1.5:LRA=14", "-ar", "48000", "-c:a", "aac",
                           "-b:a", "192k", "-shortest", "-movflags", "+faststart", final])
    subprocess.check_call(["ffmpeg", "-y", "-loglevel", "error", "-i", raw, "-c:v", "copy", "-an",
                           "-movflags", "+faststart", silent])
    os.remove(raw)
    print("done:", final, silent)


if __name__ == "__main__":
    main()
