# ANFAL — "The Uncommon is Earned" (showreel)

`anfal_showreel.mp4` — 44 s, 1080×1920 (9:16, Reels/TikTok/Stories), 30 fps, with sound design.
`anfal_showreel_silent.mp4` — same picture, no audio track, ready for a Lebanese-dialect voiceover.

Everything is generated in code (`render.py`): no footage, no people, no faces.
Audio is natural sound only (wind, sand, fire, hooves on sand, fabric, footsteps on stone),
synthesised from filtered noise. No music, no instruments, no pitched tones.

| Time | Scene | Line |
|---|---|---|
| 0:00 | A shaft of gold light, dust in the dark | Not everything is for everyone. / مش كل شي للكل. |
| 0:04 | Embers rising | It started in the ashes. / بلّشت من الرماد. |
| 0:08 | Backlit dunes at golden hour | Every step, a battle. / كل خطوة، معركة. |
| 0:13 | Hoofprints pressing into sand | Nothing was handed to us. / ما حدا عطانا شي. |
| 0:17 | Macro woven cloth, light sweep | Built thread by thread. / خيط بخيط. |
| 0:21 | Cable knit, 3 cuts (off-white / black / macro) | Quiet. Heavy. Intentional. |
| 0:25 | Stone hall, sunset through arched windows | For the ones who earned it. / للّي تعبوا ووصلوا. |
| 0:30 | The colourways as cloth, hard cuts | Olive · Beige · Navy · Charcoal · Off-White |
| 0:34 | Anfal green, gold line | The Uncommon is Earned. / النادر… بينكسب. |
| 0:38 | Logo | ANFAL · أنفال · Few pieces remain. / ما ضلّ إلا كم قطعة. · @ANFAL.CO · ANFALCO.COM |

Re-render: `pip install numpy opencv-python-headless pillow scipy` (Pillow with libraqm for Arabic), ffmpeg, then
`python3 render.py` (~8 min on 4 cores). `python3 render.py --preview 3,11,36` writes stills to `out/preview/`.
Copy lives in each scene's `LINES` / text calls in `render.py`.

Fonts (SIL Open Font License, from Google Fonts): Cormorant Garamond, Cinzel, Amiri.

---

# v2 — "اسأل اللي جرّبوا · Ask the ones who wore it" (customer-led)

`anfal_showreel_v2.mp4` (27 s, sound, −17.7 LUFS) and `anfal_showreel_v2_silent.mp4`. Rendered by `render_v2.py`.

Built only from Anfal's own material in `assets/`: customer and campaign photos from anfalco.com, the logo, the
packaging photo, the packing shot and the riding shot. No invented quotes or numbers.

Sound: horses lead — distant gallops through the whole clip (one rider closes in and passes), far-off neighs, snorts and
steps — with far-off birds and a little wind underneath and light paper/ribbon foley. All synthesised; no music.

| Time | Scene | Line |
|---|---|---|
| 0:00 | Light finds the horse embroidered on the Harrington collar | اسأل اللي جرّبوا / Ask the ones who wore it. |
| 0:02.6 | The green bag | It starts with one order. / كلّو بيبلّش بطلبيّة وحدة. |
| 0:04.4 | Orders lined up, tagged, ready to go out | Then another. And another. / وبعدها تانية… وتالتة. |
| 0:06.2 | Two customers with their bags | It arrives like it should. / وبتوصل متل ما لازم. |
| 0:07.6 | Nine real customers, gold-framed, Nº 01 → 09, cuts accelerating | WORN BY THE ONES WHO EARNED IT |
| 0:15.7 | The card inside the bag | For your brother. Your father. Your friend. / لخيّك. لبيّك. لرفيقك. |
| 0:18.7 | Campaign cuts: Corniche, white car, coffee | Made 100% in Lebanon. / معمولة ١٠٠٪ بلبنان. |
| 0:20.8 | The white horse (no words), cut to the horse in the logo | |
| 0:22.4 | Gold logo draws in on Anfal green | The Uncommon is Earned. · Your turn. / دورك. · ORDER AT ANFALCO.COM · @ANFAL.CO |
