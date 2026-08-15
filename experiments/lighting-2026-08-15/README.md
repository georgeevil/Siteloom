# CLD-129 spike: IR vs colour is a free measurement

Question (CLD-129): can a model-free statistic over stored crops separate
IR frames from colour frames, before building lighting-condition profiles?

Answer: **yes, and with a margin far wider than hoped.**

## Method

`measure_ir.py <media_dir>` computes, per detection crop (plate sub-crops
skipped): mean per-pixel channel spread `mean(max(B,G,R) - min(B,G,R))`
(0..255) and mean HSV saturation. Run over two corpora:

* the live site's media dir, 2026-08-15 — 4,906 crops, hours 13–20 UTC,
  three cameras (all daylight/colour)
* the re-ID race's corpus (CLD-118, 2026-08-12/13, scratchpad media) —
  3,092 crops including genuine night footage, hours 00–03 UTC

## Findings

1. **IR frames are exactly grayscale.** Every night-hour crop (877 of
   877, both cameras, hours 00–03 UTC) measures channel spread **0.00** —
   B=G=R survives UniFi's H.264 encode and our JPEG re-encode bit-exact.
2. **Colour frames never reach zero.** The lowest colour crop across
   both corpora is **5.5** (a white car roof on gravel — an achromatic
   *subject*, which is exactly the case a saturation heuristic was
   feared to confuse). Typical colour crops sit at 15–40.
3. **The gap (0, 5.5) is empty** over all 7,998 crops. A threshold of
   `spread < 2` classifies every crop in both corpora correctly. No
   rolling window is needed to separate the two clean states; the
   window (the operator's "1 to 5 minutes" instinct) is still right for
   the *transition* band — dusk will flap between modes frame to frame,
   and the profile must not flap with it.
4. **"front-yard is always-IR even at midday" is not true in this
   data.** Its daytime crops are colour in both corpora (spread ~17–25,
   visibly colour when eyeballed). Its footage is washed-out /
   low-saturation colour by day and true IR (spread 0) by night. The
   CLD-118 AUC gap on that camera is real but the cause needs the
   condition column to be diagnosed properly — which is this issue's
   point.

## Recommendation for the build

* Condition per frame: `IR` when mean channel spread < 2 (measure on
  the sampled frame, not the crop — cheaper still and subject-independent),
  else `colour`; smooth over a rolling window (~1–2 min) with hysteresis
  so dusk cannot flap profiles mid-event.
* Record the condition on the `Detection`/`Event` row so "/stats splits
  by condition" and "which profile was in force?" become queries.
* Profiles cover what CLD-128 made per-camera: the plate floors, plus
  identity thresholds/margins.

## Reproduce

```
python experiments/lighting-2026-08-15/measure_ir.py <media_dir> out.csv
```

The two CSVs are regenerable from the media dirs and are not committed.
