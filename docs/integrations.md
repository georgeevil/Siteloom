# Integrations: Frigate, MQTT, Double Take, CompreFace

Siteloom can slot into the popular self-hosted NVR stack instead of (or
alongside) running its own camera ingest. The reference wiring people
build with off-the-shelf parts looks like this:

```
UniFi/RTSP cameras → Frigate (motion + object detection)
Frigate → MQTT (frigate/events)
MQTT → Double Take → CompreFace (/recognize) → match/unknown → webhooks
photo backfill ─────────────────────────────↗ (same CompreFace collection)
```

Siteloom implements both sides of that pattern:

1. **As the consumer** — `siteloom frigate` plays the Double Take +
   CompreFace role on top of an existing Frigate install.
2. **As the recognizer** — a CompreFace-compatible REST API, so Double
   Take itself (or anything else that speaks CompreFace) can use
   Siteloom as its face service.

Either way there is exactly **one identity collection**: faces enrolled
from the photo library backfill, verified in the review UI, matched on
Siteloom's own cameras, or enrolled through the REST API are all the
same store. That is the point of the whole design — a five-year-old
vacation photo and a live camera frame resolve against identical data.

## 1. Consuming Frigate events

Frigate keeps doing what it is good at: RTSP ingest, motion gating, and
first-pass object detection on cheap hardware. Siteloom subscribes to
its event stream and adds the recognition layer.

Step by step, mirroring the reference stack:

1. **Frigate → MQTT.** Frigate publishes to `frigate/events` on your
   broker (Mosquitto typically). No Siteloom involvement.
2. **MQTT → Siteloom.** `siteloom frigate` subscribes to that topic.
   On a `new` event (and rate-limited `update`s, since later frames
   often carry a better face view) for a configured label, it pulls the
   triggering snapshot from **Frigate's own HTTP API**
   (`/api/events/<id>/snapshot.jpg`) — it never touches the camera.
3. **Recognition.** The snapshot runs through the identity pipeline:
   face pipeline for persons, appearance + optional plate OCR for
   vehicles, matched against the shared collection.
4. **Results out.** The event and identity land in Siteloom's store
   (visible in the web UI, deduped per Frigate event id), the result is
   republished on `siteloom/identity`, and configured webhooks fire.

### Configuration

```yaml
integrations:
  mqtt:
    enabled: true
    host: 192.168.1.10      # your Mosquitto
    port: 1883
    username: siteloom
    password: "..."
  frigate:
    enabled: true
    api_url: http://192.168.1.10:5000
    mqtt_topic: frigate/events
    labels: [person, car, truck, motorcycle, bus]
    cameras: []             # empty = all
    min_score: 0.6
    update_interval_s: 10   # max one snapshot per event per N seconds
  webhooks:
    - url: https://example.com/hook
      events: [identity.match, identity.unknown, identity.new_plate]
      token: "optional-bearer-token"
```

Run it:

```bash
siteloom frigate --config site.yaml
```

### Published topics

| Topic | When |
|---|---|
| `siteloom/events` | every stored event update |
| `siteloom/identity` | every identity resolution (match or new unknown) |

Identity payloads carry `name` (null while unknown), `known`,
`new_identity`, `similarity`, `plate`, `camera`, and `event_id` — enough
for a Home Assistant automation to act on without a follow-up query.
The same topics are published by Siteloom's own camera ingest when
`integrations.mqtt.enabled` is true, once per event+identity pairing
(one visit = one notification, not one per frame).

### Webhook events

| Event | Meaning |
|---|---|
| `identity.match` | a labeled (known) identity was recognized |
| `identity.unknown` | an unknown identity appeared or reappeared |
| `identity.new_plate` | an existing vehicle identity just gained a plate from OCR |

Delivery is fire-and-forget on a worker thread with a 10 s timeout — a
dead endpoint degrades to a log line, never blocked recognition.

## 2. CompreFace-compatible recognition API

Enabled by default on the web server (`siteloom serve`). The subset of
CompreFace's Recognition Service API that Double Take and enrollment
scripts actually use:

| Endpoint | Purpose |
|---|---|
| `POST /api/v1/recognition/recognize` | multipart `file` → boxes + subject matches |
| `GET  /api/v1/recognition/subjects` | list known subjects |
| `POST /api/v1/recognition/subjects` | create a subject (`{"subject": "Name"}`) |
| `POST /api/v1/recognition/faces?subject=Name` | enroll a face example |
| `GET  /api/v1/recognition/faces` | subjects with example counts |

Auth is CompreFace's own convention — the `x-api-key` header — enforced
when `integrations.recognition_api.api_key` is set.

Pointing Double Take at it:

```yaml
# double-take config
detectors:
  compreface:
    url: http://siteloom-host:8000
    key: your-api-key
```

A "subject" is a labeled face Identity. Unknown-bucket identities are
never reported as subjects — CompreFace semantics are "known people
only", and unknowns surface through Siteloom's own UI and MQTT instead.

Example:

```bash
curl -F file=@snapshot.jpg \
  "http://localhost:8000/api/v1/recognition/recognize?prediction_count=3"
# {"result":[{"box":{...,"probability":0.92},
#   "subjects":[{"subject":"George Chigrichenko","similarity":0.46}, ...]}]}
```

Compare `similarity` against your configured face threshold
(`identity.identifiers.face.threshold`) — after a fine-tune,
`siteloom train face` reports and can apply the right value.

### While the vector store is held (ingest running)

The embedded vector store allows one client per data directory, and
`siteloom run` holds it for hours — the *normal* condition on a live
site. The API stays up through that, by decision (CLD-110): **reads
degrade, writes refuse.**

- `POST /recognize` answers `200` in its ordinary shape with zero
  subjects — Double Take treats it as "nobody recognised" and keeps
  working — plus an additive `"degraded": "vector-store-busy"` body
  field and a `Retry-After` header, so a client that cares can tell
  "nobody matched" from "nobody was looked for". Face boxes are still
  reported; only the subject search is skipped.
- `POST /faces` (enrolment) answers `503` with a CompreFace-shaped
  error body (`{"message": ..., "code": 41}`) and enrolls **nothing** —
  a write that pretended success would silently drop the face. Retry it
  after the holding job finishes (see `/jobs`).
- `GET/POST /subjects` and `GET /faces` read and write only the SQL
  database and are unaffected.

Recovery is automatic: the moment the holding process exits, the next
request serves matches again.

## 3. Backfill enrolls the same collection

Two equivalent paths, per source of photos:

- **Siteloom-native**: `siteloom takeout import` / `siteloom library
  index` + review at `/training`, then `siteloom train enroll` (also runs
  automatically on each confirmation) — embeddings land under labeled
  identities in the shared collection.
- **CompreFace-style scripts**: anything that enrolls into CompreFace
  (an Immich exporter, a folder-per-person script) works unchanged
  against `POST /api/v1/recognition/faces?subject=Name`.

## Operational notes

- **One process per vector store.** The embedded Qdrant allows a single
  client per data directory. `siteloom serve`, `siteloom frigate`, and
  enrollment sweeps each open it — run them against the same directory
  one at a time, or move to a Qdrant server (config change) when you
  need true concurrency. While another process holds it, the recognition
  API degrades to no-match on reads and refuses enrolments with a 503
  (CLD-110, section 2 above); the console answers 503 with the same
  guidance. A stale `.lock` after a crash can be deleted
  once you've confirmed no Siteloom process is running.
- **Broker down ≠ pipeline down.** MQTT publishing and webhooks degrade
  to a warning; recognition and storage continue (NFR1).
- **Frigate label mapping** is explicit (`LABEL_MAP` in
  `siteloom/integrations/frigate.py`); labels not in the map pass
  through by name and get a dynamic identifier if `auto_add_classes` is
  on.
