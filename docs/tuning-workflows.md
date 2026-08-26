# Tuning a camera, step by step

This guide is for whoever is standing in front of a camera that isn't
behaving — no tracker vocabulary required. Everything below happens on
the **Tuning** page (Jobs → Tuning). The one rule to hold onto:

> **A trial changes nothing.** It reads footage, draws what it would
> have tracked, and writes a report. Settings only move when you apply a
> result you have reviewed — and every apply can be reverted from the
> Config history panel.

## Workflow 1 — a new camera just went up

**Desired outcome:** the camera tracks people and vehicles as *one track
per subject*, and you have the evidence pictures proving it.

1. Tuning → **Tune a camera from its own footage** → pick the camera.
2. Pick a ten-minute window where something actually happened — someone
   walking through, a car arriving. (The pre-filled window ends two
   minutes ago because the recorder often cannot export the newest
   minute.)
3. Pick the scene card that *describes* the camera — "Close doorway",
   "Wide driveway", "Garden with moving foliage", "Car park at night".
   Cards marked *reasoned starting point* haven't been measured on this
   site yet; that is fine, that is what the trial is for.
4. Run. A few minutes later, open the report. Look at the **pictures
   first**: outlined frames are moments a new track appeared — each one
   should be a real person or vehicle *entering*, not someone who was
   already there getting a second track.
5. Happy? **Apply to camera.** Not sure? Run a second trial with a
   different scene card over the *same window* and use **Compare** — the
   verdict sentence says which run did better and why.

## Workflow 2 — something looks wrong on a camera

**Desired outcome:** you can name the failure, and a trial shows a
setting that fixes it without breaking something else.

- **One person shows up as several tracks / several events?** That is
  *splitting*. Run a trial over a window showing the problem, with the
  lean slider one notch toward **merge**.
- **Two people (or a person and a car) sharing one track?** That is
  *merging*. Slider one notch toward **split**.
- Compare the trial against one run with current settings on the same
  window. The comparison refuses a fix that only *looks* cleaner by
  splitting subjects into more tracks — that guard is built in.
- Apply only when the pictures agree with you.

## Workflow 3 — try a clip you downloaded

For a camera that isn't set up yet, or when the recorder refuses to
export (see below).

1. In UniFi Protect (or any camera app), export the clip as **MP4**.
2. Tuning → **Try settings on a downloaded clip** → choose the file, and
   say which camera the footage came from — that picks the settings the
   trial starts from, and where an accepted result would be applied.
3. Scene card, run, review — exactly as above.

## Reading a report

| You see | It means |
|---|---|
| "Tracked 2 people over 90 s with no tracking problems visible" | What it says — apply-worthy. |
| "…where two subjects may have been merged" | One track absorbed two subjects: the *merge* failure. Lean split, or try the trial's suggestion. |
| "…where one subject may have been split or a phantom track appeared" | The *split* failure — often two people overlapping. Lean merge, or accept it: splitting is the cheaper failure and the system stitches fragments back. |
| "the sampling looks too slow for how fast subjects move" | Raise frames-per-second — no other setting fixes this. |
| "footage reads as night (IR)" | Colour-based recognition is weak here; the "Car park at night" card is the honest starting bundle. |
| *measured* vs *heuristic* on a suggestion | *Measured* rules were proven on this site's own footage; *heuristic* ones are reasoned starting points. |

## Workflow 4 — a camera that goes infrared at night

Cameras look completely different in IR — colour vanishes, and the
settings that win in daylight aren't the ones that win at night. If a
trial's report says *"footage reads as night (IR)"*:

1. Apply that trial to the camera's **night profile** (the apply
   dropdown has a "— night profile" entry). Daytime settings stay
   untouched.
2. From then on the camera switches automatically: the system measures
   each frame's colour and flips profiles when the footage actually goes
   IR — never by the clock, so re-processing old footage stays correct.
3. The "Car park at night" scene card is the honest starting bundle for
   a night trial.

## Workflow 5 — let the search look for you

When you don't know which scene card fits: Tuning → **Search for better
settings** → pick the camera and a few saved clips showing real
activity. Every named bundle is tried; losers are dropped after one
clip, survivors earn more, and the winner — if anything actually beats
the current settings — appears under *Proposals waiting for review*
with side-by-side evidence pictures. **It never applies itself**; "no
winner" is a real answer meaning the camera is as tuned as those clips
can show. Budget: expect an hour-plus of detector time, cancellable
from Jobs.

## When the recorder refuses to export

If a trial fails with a message about the NVR refusing the export: this
is the Protect console itself declining (it happens; a console restart
usually clears it). Nothing is wrong with your window or settings — use
Workflow 3 with a manually downloaded clip in the meantime.

## What you can't break

- Trials never create events, never identify anyone, never teach the
  recognition galleries anything.
- Applying writes the *smallest possible* per-camera change, snapshots
  the config first, and the snapshot list (bottom of the Tuning page)
  reverts it in one click.
- A few settings are deliberately not tunable here (crop margin, the
  tracked classes, the compute device) — changing those is a bigger
  operation with its own warnings, not a slider.
