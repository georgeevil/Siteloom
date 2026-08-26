"""Post-tracker analysis layers.

The tracker itself is a black box inside ultralytics (`modules/
detection.py` merely materializes its config); what lives here consumes
its per-frame output — (track_id, bbox) — and adds judgement the tracker
cannot: occlusion awareness today, nothing else yet.
"""
