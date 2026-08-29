<!-- Generated 2026-08-28 by a 13-agent research workflow run from the Siteloom repo: 6 cluster surveys, then an independent fidelity check of every cited claim per cluster, then one synthesis. Verdicts across the checks: confirmed 105, corrected 18 (per cluster: {"frigate": {"corrected": 4, "confirmed": 14}, "doubletake-compreface": {"corrected": 6, "confirmed": 13}, "immich-photoprism": {"confirmed": 21, "corrected": 1}, "backbones-quality": {"confirmed": 19, "corrected": 2}, "restoration": {"confirmed": 21, "corrected": 2}, "nvr-watchlist": {"confirmed": 17, "corrected": 3}}). Only confirmed/corrected findings are stated as fact; section 9 lists what could not be verified. -->

# Face identification beyond one frame — what other open-source projects do

Across the fifteen projects surveyed, four patterns recur. **Cheap quality proxies, never a quality model**: everyone gates on face size and detector score; two add Laplacian blur (Frigate, Ente) and one adds a landmark pose check (Ente); no shipping project uses SER-FIQ/MagFace/CR-FIQA in the loop, and the two dedicated FIQA models are CC BY-NC anyway. **Consensus is rare and, where it exists, is a weighted vote or a clustering rule**: Frigate is the only NVR that decides per track (area-weighted vote, tie means no identity); the photo libraries (Immich, PhotoPrism, Ente) decide per face but refuse to found an identity without N mutually-close neighbours or a runner-up margin; every other NVR/integration decides per frame or keeps best-of-N. **Re-processing is throw-away-and-recompute the gallery**: nobody migrates vectors in place, and only PhotoPrism stamps provenance per row and refuses to mix spaces. **The licence-safe backbone is the one Siteloom already runs** (YuNet MIT + SFace Apache-2.0); every InsightFace-derived weight is non-commercial. The one thing nobody does: put restoration or super-resolution *before* the embedder — FaceFusion, the only project shipping restorers, enhances *after* face selection — and the measured evidence on surveillance footage says that is the right call.

## 1. How each project decides — comparison table

| Project | Detector / Recognizer (dims, license) | Frame selection & quality gate | Per-frame or per-track decision | Re-processing after changes | Alert gating |
|---|---|---|---|---|---|
| Frigate dev a745070b / v0.16.0 [1][2][3] | YuNet (`facedet.onnx`) inside person box, or `face` attribute of a Frigate+ model; FaceNet TFLite 160 px (`small`, default) or ArcFace ONNX 112 px (`large`); dims —; code MIT, weights from NickM-27/facenet-onnx (Apache-2.0 repo, provenance undocumented) [4] | Largest face ≥ `detection_threshold` 0.7 (YuNet path only); face area ≥ `min_area` 750 px²; Laplacian blur penalty (`blur_confidence_filter` true); attempts only on tracked-object updates (5 s cadence; 1 s fast-track dev-only, standalone-YuNet path only, #22673 2026-03-29) | Per track: `weighted_average` over ≤12 attempts (6 after recognition), weight `min(area,4000)·(score−unknown_score)·10`, unknowns excluded, `min_faces` 1, tie → none, publish at `recognition_threshold` 0.9 | No vector DB; class means rebuilt from `/media/frigate/clips/faces/<name>/` on train/register/delete/rename (not `create`); no retroactive re-recognition | `sub_label` at threshold; MQTT `tracked_object_update` after *every* attempt regardless of threshold; review categorised by object label only |
| Double Take 92521a0 [5][6][7] | None of its own; forwards to CompreFace/DeepStack/Facebox/Rekognition; MIT | Polls Frigate `latest.jpg` ×10 and `snapshot.jpg` ×10 in parallel; `match.confidence` 60, `match.min_area` 10000 px², `unknown.confidence` 40; optional Haar face-count pre-gate (fails closed before OpenCV loads) | Per event, best-of-N: highest confidence per subject name kept; `stop_on_match` true | None; retraining delegated to the detector | Event id frozen after first match; per-subject MQTT topic; `unknown` topic below `unknown.confidence` |
| CompreFace ddf32da [8][9][10][11] | MTCNN (`det_prob_threshold` 0.85) / RetinaFace; FaceNet `20180402-114759` TF .pb (default; docs say embeddings are "an array of 512 or 128 numbers") or InsightFace MXNet; code Apache-2.0, FaceNet weights no explicit licence, InsightFace non-commercial | Largest faces first (`limit` 0); downscale to `IMG_LENGTH_LIMIT` 640; no pose/blur gate | Per image, stateless; top `prediction_count` 1, no built-in threshold (docs: ">0.5" for high-security) | `POST /api/v1/migrate` re-embeds gallery when calculator version differs; needs `save_images_to_db=true`; "wasn't tested enough" | — |
| CodeProject.AI / DeepStack [12][13] | YOLOv5 `face.pt` at 320 px; IR-SE-50, 512-d; module GPL-3.0, weights unlicensed | `min_confidence` doubles as detector and similarity cutoff (code 0.67, `modulesettings.json` 0.4) | Per image, argmax; register stores `torch.mean` of uploads | None | — |
| Immich 469a870 [14][15][16] | SCRFD-style decode at 640 px + ArcFace 5-point align 112 px; `buffalo_l` (InsightFace pack), 512-d pgvector cosine; weights `license: other (insightface)`, permission "does not extend to the redistribution or commercial use of their models by third parties" [17] | `minScore` 0.7 only; no size/blur/pose | Per face; DBSCAN-derived: core point iff ≥ `minFaces` 3 neighbours (incl. itself) within `maxDistance` 0.5 and asset in timeline; else deferred | Model change ⇒ re-run Face Detection (force deletes ML faces); recognition `force` re-clusters stored vectors; re-detect keeps matched ML faces with their *old* vector | — |
| PhotoPrism develop 6c9a369 [18][19][20][21] | YuNet MIT (default) or SCRFD non-free, 640 px input on 720 px thumbs; `sface` ONNX 128-d Apache-2.0 (default), `auraface` 512-d Apache-2.0, `facenet` 512-d unknown, `arcface_r50/mbf` non-free | `FACE_SIZE` 25 px marker, `FACE_CLUSTER_SIZE` 60 px to cluster; `FACE_SCORE`/`FACE_CLUSTER_SCORE` per detector (YuNet 65/70); "Confidence is a weak quality signal" | Per marker: DBSCAN (`FACE_CLUSTER_CORE` 4) then centroid match ≤ radius + `MatchDist`; `FACE_MATCH_MARGIN` 0.05 over runner-up | `embed_model`/`detect_model` per row; mismatch pauses; `faces migrate` (checkpointed, resumable); `faces reset` re-clusters without decoding files | — |
| Ente 150be48 [22][23][24] | `yolov5s_face_640_640_static_b1.onnx`; `mobilefacenet_portable_static_b1.onnx`; dims and licences — | Entry: `blur > 10 && score > 0.8`; `isBadFace` = blur < 50, or blur < 200 && score < 0.85, or not "straight" (5-landmark pose) | Per face, linear NN linking; join at cosine ≥ 0.76 (0.84 if neighbour is bad); rejected clusters never re-applied | `faceIndexingVersion` constant; lower-versioned remote index ignored | — |
| Viseron b4f7098 [25][26][27] | Object box → `face_recognition` (dlib) KNN, or CompreFace/DeepStack/CodeProject backends; MIT | None; `Queue(maxsize=1)` drops frames while busy | Per frame; dlib `distance_threshold` 0.6 hard-coded; CompreFace `similarity_threshold` 0.5 | KNN rebuilt from folder at start; backends re-upload with `train` | Event per frame; `expire_after` 5 s gates storage only |
| ZoneMinder zmes / pyzm [28][29] | dlib via pyzm `FaceDlib`, KNN, `face_recog_dist_threshold` 0.6; pyzm GPL-2.0, hook MIT | Per ZM event: `frame_set` `snapshot,alarm`, `frame_strategy` `most_models`, `resize` 800 | Per frame; one frame's result wins | `faces.dat` must be deleted to retrain; no re-run over past events | Once per ZM event by construction |
| Shinobi plugins 7ce248b [30][31][32] | TF.js: ssdMobilenetv1 + face-api.js descriptors, `distanceThreshold` 0.6 (fork MIT). ONNX: RFB-320 (MIT) + `w600k_mbf.onnx` (non-commercial), cosine `similarityThreshold` 0.4; plugin AGPL-3.0/enterprise | `faceMinConfidence` 0.5 / `detectorConfidence` 0.7; ONNX plugin: bbox crop, no landmark alignment | Per frame | Sidecar caches per model (`.faceDescriptor` vs `.faceEmbeddingOnnx`) | `trigger` per frame; unmatched face minted as `UNKN_<epoch>` and added to matcher |
| Scrypted b63b7df [33][34] | `scrypted_yolov9t_relu_face` 320 px, `minThreshold` 0.5; `inception_resnet_v1` 160 px, dims —; HF card MIT | None | Embedding → label matching lives in closed NVR plugin | — | Smart Motion Sensor: `labelDistance` 2, `labelScore` 0 |
| DeepFace stream a2a24a7 [35][36] | VGG-Face default; per-model cosine thresholds (SFace 0.593, ArcFace 0.68, Buffalo_L 0.55); MIT, model licences inherited | Face width > 130 px (function default) | Recognition runs once when `frame_threshold` 5 consecutive frames *had a face*, then freezes `time_threshold` 5 s — not N matches | Pickle named by model/detector/alignment; recompute | Top hit `df.iloc[0]`; no unknown record |

## 2. Frame selection and quality gating (Q1) — per project, the concrete mechanism, with config keys/defaults

- **Frigate** [2][3]: YuNet path takes the *largest* face ≥ `detection_threshold` 0.7; Frigate+ path takes the highest-scoring `face` attribute (`min_score` 0.5 applies instead). Both require area ≥ `min_area` 750 px² (reference config wrongly says 500 [37]). `blur_confidence_filter` subtracts 0.06/0.04/0.02/0.01 from the frame score at Laplacian variance < 120/160/200/250. With a Frigate+ model the saved snapshot is the frame with the largest visible face [38]. Dev docs source: "Avoid training on infrared (gray-scale)... will not be able to extract features" — the live 0.16 page omits "not" [39].
- **Double Take** [5][6]: `detect.match.min_area` 10000 px² at `frigate.image.height` 500; Haar cascade `opencv_face_required` false for self-hosted detectors.
- **CompreFace** [9]: sorts by area, `limit` 0; plugins (pose, mask) append DTOs, gate nothing.
- **Immich** [14]: `minScore` 0.7 before NMS 0.4; representative face is the cluster founder or `getRandomFace` [15].
- **PhotoPrism** [18][19]: `FACE_SIZE` 25 px (min 10), `FACE_CLUSTER_SIZE` 60 px on the Fit720 thumbnail; `FACE_SIZE_RETRY` 10 px only when a picture yields no face (changed 19 of 861 pictures, added 1163 faces, 1149 from twelve crowd shots). Cover: `ORDER BY (subj_src <> auto) DESC, size DESC, score DESC` [20].
- **Ente** [22]: constants `kLaplacianSoftThreshold` 50, `kLaplacianVerySoftThreshold` 200, `kMinimumQualityFaceScore` 0.80, `kMediumQualityFaceScore` 0.85.
- **DeepFace stream** [35]: `grab_facial_areas(threshold=130)` px width, `expand_percentage=0`.
- **InsightFace / opencv_zoo** [40][41]: `det_thresh` 0.5, no size gate; YuNet defaults `conf_threshold` 0.9, `nms_threshold` 0.3, detects "faces of pixels between around 10x10 to 300x300".
- **Quality models** (none shipped in any product): MagFace quality = L2 norm of the un-normalised 512-d feature [42]; SER-FIQ = 100 dropout passes, `alpha` 130, `r` 0.88 [43]; CR-FIQA = one `nn.Linear(512,1)` head on the recognizer [44]. Forensic evidence: "using higher-quality frames leads to lower Cllr values... incorporating lower-quality frames actually led to a worsened Cllr" [45].

## 3. Multi-frame consensus (Q2)

- **Frigate** [2]: the only NVR with a track-level decision. Per-frame score = sigmoid-mapped cosine to a class mean (ArcFace median 0.3, FaceNet 0.5, range 0.6, slope 12) minus blur penalty; ≤ `unknown_score` 0.8 → "unknown". Vote weight `min(face_area, 4000)·(score − 0.8)·10`; tie → no identity; caps `MAX_FACE_ATTEMPTS` 12 / `MAX_FACES_ATTEMPTS_AFTER_REC` 6 (frames with no face don't consume one); history dropped on `expire_object`. Dev docs: "If Frigate sees a person as 'Tom, Tom, Sam, Tom, Tom,' it will still conclude the person was Tom" and "A single very high confidence match will not by itself assign a sub label" [39]. The checker's arithmetic: `recognition_threshold` 0.9 ≈ cosine 0.41 for ArcFace — not comparable to a raw-cosine threshold.
- **Double Take** [7]: best-of-N, no vote, no embeddings.
- **Immich** [15]: core point needs `minFaces` 3 (itself included) within `maxDistance` 0.5; non-core faces deferred and retried nightly.
- **PhotoPrism** [21][19]: DBSCAN core 4; runner-up margin `FACE_MATCH_MARGIN` 0.05 — "a marker inside that margin is left unassigned - which is recoverable, where a wrong assignment is not"; collisions narrow clusters (`Epsilon` 0.01) or retire them as `AmbiguousFace`.
- **Ente** [22]: quality-dependent join threshold 0.76 / 0.84.
- **Set-aggregation prior art**: InsightFace `eval_ijbc.py` keeps features un-normalised by default (`use_norm_score = True`) so norm weights frames, multiplies by faceness score, mean-pools per video, sums per template [46]; AdaFace IJB-S fuses `features * norms` [47]; NAN reports IJB-A TAR@FAR=1e-3 74.54 vs 64.20 for averaging (ResNet34/WebFace softmax) [48]; CAFace (MIT) is an order-invariant streaming aggregator on an AdaFace backbone [49].
- **FaceFusion** [50]: `average_face_identity` = numpy mean of embeddings; for swap consistency, not identification.
- Everyone else (Viseron, zmes, Shinobi, HA, DeepStack, CodeProject): per frame.

## 4. Restoration, super-resolution, frontalization before embedding (Q3) — lead with the EVIDENCE for/against, then the licenses

**Evidence.** No project in the survey runs restoration before embedding; FaceFusion calls `select_faces` (ArcFace on the raw frame) *then* `enhance_face` [51]. Published measurements:
- Against, surveillance domain: NFI et al. 2024 — "preprocessing facial images with the super resolution CodeFormer, it unexpectedly increased Cllr, undermining evidence reliability, advising against its use" (ArcFace/FaceNet/QMagFace on ENFSI 2015, SCFace, XQLFW, ChokePoint, ForenFace) [45]. QMUL-SurvFace — five SR methods "often bring slightly negative effect to surveillance FR"; CentreFace TPIR@20%FPIR 21.0% → 20.0% with SRCNN/FSRCNN [52].
- Against, frontalization: arXiv 2512.03199 on CASIA-WebFace (491,414 images) — ArcFace rank-1 89.5% → 43.4% NextFace, 80.4% CFR-GAN, 77.8% CFR-GAN+CodeFormer universal; only a failure-predicting gate (91.6% precision, 51.8% recall) reached 92.8% [53]. Rotate-and-Render uses frontalization only as training augmentation; IJB-A TAR@FAR=.001 95.39 → 95.63 on MS1MV2 [54].
- For, synthetic degradation: arXiv 2308.07967 on XQLFW — GFP-GAN v1.3 raised AdaFace 0.879 → 0.896, MagFace 0.835 → 0.863, ArcFace 0.737 → 0.865; GPEN hurt AdaFace (0.857), SGPN hurt AdaFace and MagFace; restored images show "non-existent glasses or changes in eye color" that "yield to a potential loss of identity information" [55].
- Self-admissions: GFPGAN V1.3 "have a slight change on identity" [56]; CodeFormer IDS 0.60 vs 0.32 input vs 0.89 ground truth [57].

**Licences.** GFPGAN Apache-2.0 [58]; RestoreFormer++ Apache-2.0 [59]; Real-ESRGAN BSD-3-Clause [57]; facexlib MIT (SORT sub-module GPL-3.0) [60]; CFR-GAN Apache-2.0, 3DDFA_V2 MIT, Rotate-and-Render CC-BY-4.0; CodeFormer NTU S-Lab 1.0 — "Redistribution and use for non-commercial purpose" only [61]; GPEN no LICENSE file, API `license: null` [62]; DECA non-commercial research [63]; InsightFace ArcFace weights non-commercial [17]. FaceFusion's own licence page marks ArcFace, GPEN, RetinaFace/SCRFD Non-Commercial and YOLO Face GPL-3.0 [64].

## 5. Backbones (Q4) — table: model, dims, threshold convention, ONNX, license, low-quality robustness evidence

| Model | Dims | Threshold convention | ONNX | License | Low-quality evidence |
|---|---|---|---|---|---|
| SFace (opencv_zoo) [41][65] | 128 | cosine ≥ 0.363 / normL2 ≤ 1.128 on LFW (99.60%); CALFW 0.340, CPLFW 0.275, AgeDB-30 0.277 | yes (+int8) | Apache-2.0 (upstream zhongyy/SFace has no licence file) | None published; PhotoPrism marks TAR/FAR "n/a" after it "admitted roughly a quarter of cross-sibling comparisons" [19] |
| YuNet (detector) [41] | — | score 0.9, nms 0.3 | yes | MIT | "10x10 to 300x300" px |
| AuraFace (PhotoPrism registry) [21][19] | 512 | Euclidean on unit vectors: ClusterDist/Radius/MatchDist 0.98/0.76/0.35 | — | Apache-2.0 | TAR 0.9308 at 0.14% FAR on PhotoPrism's benchmark |
| InsightFace buffalo_l / antelopev2 [17][66] | 512 (training default; ONNX output unverified) | cosine; no published deployment threshold (GUI DBSCAN 0.48) | yes | Code MIT; weights "non-commercial research purposes only" | IJB-C(E4) 97.25 (buffalo_l) |
| AdaFace R100 [67][68] | 512 | cosine of normalised features; returns `(feature, norm)`; BGR, mean/std 0.5; no published threshold | no official (issue #43: "currently I do not have a plan to provide the onnx code") | MIT | IJB-S Sur-to-Single Rank-1 65.26 vs ArcFace 57.35 (MS1MV2); 71.35 with WebFace12M; TinyFace 72.29 |
| MagFace iResNet100 [42][69] | 512 | cosine; norm = quality | third-party only | Apache-2.0 (no weights statement) | — (paper tables not read) |
| Frigate ArcFace `arcface.onnx` / FaceNet `facenet.tflite` [4][3] | — | sigmoid-mapped cosine, accept 0.9 | yes / TFLite | distributed Apache-2.0; provenance undocumented | — |
| FaceNet `20180402-114759` (CompreFace default) [11] | docs: "512 or 128" | tanh-mapped Euclidean, coefs (1.1817961, 5.291995557) | no (TF .pb) | repo MIT, weights no explicit licence | LFW 99.63% |
| IR-SE-50 (CodeProject/DeepStack) [12] | 512 | `(cos+1)/2` ≥ 0.67 | no | GPL-3.0 module; weights unstated | — |
| dlib `face_recognition` (Viseron, zmes, HA) [26][29] | 128 (library property; not asserted in cited files) | Euclidean ≤ 0.6 | no | code MIT | — |
| `w600k_mbf` (Shinobi ONNX) [31] | — | cosine ≥ 0.4 | yes | non-commercial (InsightFace) | — |
| Quality-only: SER-FIQ [70], CR-FIQA [71] | — | score in [0,1] | no | CC BY-NC-SA 4.0 / CC BY-NC 4.0 | FNMR plots; NIST FATE rank claim — no tables in README |

## 6. Re-processing after model or threshold changes (Q5)

- **Frigate** [3][2]: no vectors stored; class means rebuilt from crop files on every clear; dev branch uses `build_class_mean` (trim 0.15, cosine-outlier rejection 0.30, floor max(5, ⌈0.7n⌉)), v0.16.0 plain `trim_mean(0.15)`. `POST /faces/reprocess` re-scores one saved attempt image and never touches an event; docs refuse bulk reprocess ("only images from a similar angle will have its score affected"). `save_attempts` 200 dev / 100 v0.16.0.
- **CompreFace** [10]: async `/api/v1/migrate` re-embeds rows whose `calculator` differs; no re-scoring of past queries.
- **Immich** [16]: "you must re-run the Face Detection job for all images upon changing a model"; detection `force` deletes ML faces and vacuums with `reindexVectors: true`; recognition `force` unassigns and re-clusters; the better-clusters guide runs Reset at `minFaces` 20/10 then Missing at 10 then 3 [72].
- **PhotoPrism** [19]: `faces.embed_model`, `markers.embed_model`, `markers.detect_model` per row — the *detector* is part of the space; mismatch "pauses embedding work"; `faces migrate --dry-run|--yes` re-embeds at `FACE_MIGRATE_SIZE` 10 px / `FACE_MIGRATE_SCORE` 50, seeds clusters from surviving assignments, checkpoints, refuses finalize past a failure ratio, writes the model to `options.yml`; `faces reset` (default / `--all` / `--force`) keeps `embeddings_json` so re-clustering decodes no files.
- **Ente** [24]: `faceIndexingVersion` bump. **DeepFace** [36]: pickle name encodes model/detector/alignment/normalization/expand. **Shinobi** [32]: per-model sidecar extensions. **zmes** [29]: delete `faces.dat`. **Viseron** [27]: rebuild on start.
- Nobody re-embeds stored *event* crops; Siteloom's rebuild-vectors command is the only one described that does.

## 7. Watchlist / alert gating (Q6)

- **Frigate** [8][73]: no watchlist; `sub_label` published when the weighted average reaches 0.9, re-published on later qualifying attempts; MQTT `tracked_object_update` "Published after each recognition attempt, regardless of whether the score meets `recognition_threshold`" with `score` the running weighted average — a consumer must gate on `score >= recognition_threshold` and dedupe on `id`. Review alerts categorised by `label.split(": ")[0]`; push notification only appends `sub_labels`.
- **Double Take** [7]: `IDS.push(id)` after the first match freezes the event; `update` while processing is dropped; `end` ignored; person-count reset after 30 s.
- **Viseron** [27]: `EVENT_FACE_DETECTED` every recognised frame; `expire_after` 5 s only gates storage and the binary sensor.
- **Scrypted** [34]: `labelDistance` 2 (docs: set 0 for faces), `labelScore` 0; matching and once-per-event logic are closed-source.
- **Shinobi TF.js** [30]: `trigger` per frame; `face_save_unknown` mints `UNKN_<epoch>` guarded by `alreadyCompiling`, a refresh timeout and `loadingUnknown`.
- **Home Assistant** [74]: `dlib_face_identify` polled every 10 s, fired `image_processing.detect_face` per matched face per scan, `confidence` 0.6; removed in 2025.12 (PR #155450).
- **zmes**: one hook per ZoneMinder event by construction [28]. Immich/PhotoPrism/Ente/CompreFace: no alerting layer.

## 8. Options for Siteloom

**Option 1 — Quality-gated, area-weighted per-visit consensus (do first). Size: S–M.** What: collect per-frame face resolutions over an event and decide once — Frigate's shape: exclude unknowns, weight by `min(face_area, cap) × (score − floor)`, require the winner to have ≥ N votes and no vote-count tie, publish only when the aggregate clears the threshold; gate frames first by face area (Frigate 750 px², PhotoPrism 60 px on a 720 px thumbnail, DeepFace 130 px width) and a Laplacian blur penalty (Frigate's tiers, Ente's 50/200). Shown working: Frigate v0.16.0+ [2]; the NFI paper for "fewer, better frames" [45]. Touches: a per-visit consensus step in the resolver over the per-frame results the event already collects, composing with CLD-41 — the runner-up margin is PhotoPrism's `FACE_MATCH_MARGIN` 0.05 [19] and stays; scored in the replay lab against `EventIdentity.verdict`. Licence: none. Why first: no new model, no vector migration, and it addresses the stated failure (one low-quality visit collecting several names) exactly where Frigate addresses it. Caveat: Frigate's thresholds live in sigmoid space (0.9 ≈ cosine 0.41 for ArcFace) — copy the mechanism, tune the numbers in the lab.

**Option 2 — Feed the consensus from alert gating. Size: S.** Publish once per event+identity only from the consensus result, never from a per-frame hit (Frigate's MQTT contract puts that burden on the consumer [73]; Double Take freezes on first match [7]). Touches: the webhook/MQTT publish point. Licence: none.

**Option 3 — Detector-aware embedding-space stamp. Size: S.** PhotoPrism records `detect_model` alongside `embed_model` and pauses on mismatch [19] — matches Siteloom's rule that `crop_margin` is a vector-space migration. Touches: the CLD-106 stamp and `doctor`. Accuracy gain: none directly; it prevents silent mixed-space galleries when Options 4–5 land.

**Option 4 — A second face identifier algo (AdaFace or AuraFace). Size: L.** AdaFace R100 is the strongest permissively licensed low-quality backbone in the material (MIT; IJB-S 65.26 vs ArcFace 57.35 [67]) and its returned `norm` is a free per-frame quality weight for Option 1 (AdaFace's own IJB-S fusion is `features * norms` [47]). Missing: no official ONNX [68], no published deployment threshold, BGR mean/std-0.5 preprocessing unlike SFace. AuraFace is 512-d Apache-2.0 with PhotoPrism-measured thresholds [21], runtime not stated in the material. Touches: a new identifier in the embedder registry with its own cosine threshold, a rebuild-vectors migration (galleries and stored crops), lab re-tuning. Not options: InsightFace buffalo/w600k weights (non-commercial [17]), Frigate's `arcface.onnx` (provenance undocumented [4]). Note Frigate's dev docs warn colour-trained models put IR faces in "a different feature distribution" [39] — relevant to an always-IR camera.

**Option 5 — A lab-only "refine" job with GFPGAN v1.4. Size: M; expected gain: uncertain, likely negative on surveillance footage.** Evidence is for on XQLFW (+2 to +13 points [55]) and against on real surveillance video [45][52]; frontalization degrades without a failure gate [53]. GFPGAN is the only permissively licensed restorer with a positive published recognition result [58]; CodeFormer [61], GPEN [62] and DECA [63] are licence-blocked. Do it as a refine job over stored crops replayed in the lab and scored against verdicts, never as a live pipeline stage — and only after Option 1 has set the per-frame quality gate the head-pose paper shows is required.

## 9. Could not verify

- Frigate: embedding dims of `facenet.tflite`/`arcface.onnx`; upstream licence/provenance of all four weight files (NickM-27/facenet-onnx has only a LICENSE); YuNet/LBF licences within Frigate; HA integration side of notifications; `min_area` docs 500 vs code 750 drift unexplained.
- Frigate docs: two quotes exist only in the dev docs source; the live 0.16 page says "will be able to extract features from gray-scale images".
- CompreFace: FaceNet dims (512-d and "Inception-ResNet-v1/VGGFace2" come from davidsandberg's README, not CompreFace); whether query images are stored; age/gender weights are likely rude-carnie InceptionV3 checkpoints, not GilLevi's; which RetinaFace variant the Mobilenet build pins.
- CodeProject/DeepStack: `facerec-high.model` and `face.pt` provenance/licence.
- Immich: exact recognizer inside `buffalo_l` (recalled as w600k_r50, unconfirmed); server query for "hidden below minFaces"; whether the live index is VectorChord.
- PhotoPrism: values are from `develop` 6c9a369 ("Last Updated: August 28, 2026"); tagged releases may differ; "default for new libraries" is a paraphrase of `Default: true`.
- Ente: embedding dims, weight licences, blur computation site; server-side migration.
- Backbones: buffalo ONNX output shape; SFace low-resolution benchmarks; SER-FIQ/CR-FIQA/MagFace numeric tables (papers not read); MagFace's quality-assessment checkpoint not inspected; NAN official code absent (search negative); jinyanxu NAN (35 stars, no licence) not checked.
- dlib `face_recognition` and face-api.js 128-d: library properties, not asserted in the cited files.
- Restoration: Pleško et al. 2025 occlusion paper paywalled; CodeFormer Table 1 competitor IDS (GFP-GAN 0.42, GPEN 0.54) read from a fetch summary, not the table; 2308.07967 prose says SGPN declines 4–6% while its table shows an ArcFace gain; line numbers were checked at specific commits (GFPGAN 7552a77, CodeFormer b33cc7d, RestoreFormer++ 59d250f, facexlib 260620a) and will drift.
- Scrypted: matching threshold, once-per-event gating, `inception_resnet_v1` dims — closed plugin.
- Shinobi: core alert throttling; whether the ONNX plugin mints `UNKN_` identities; detector preprocessing `(x−127.0)/128.0` differs from embedder.
- pyzm: inference reads unprefixed `upsample_times`/`num_jitters`, training reads `face_`-prefixed keys — mapping not traced.
- Home Assistant: `facebox`, `microsoft_face_detect` not read.
- Q3 for every application project: no issue thread or benchmark on restoration was found in their trackers (code searched, trackers not exhaustively).

## Sources

1. https://github.com/blakeblackshear/frigate/blob/a745070b76276ef7865bf6513d627196ff1c6d10/frigate/camera/state.py
2. https://github.com/blakeblackshear/frigate/blob/a745070b76276ef7865bf6513d627196ff1c6d10/frigate/data_processing/real_time/face.py
3. https://github.com/blakeblackshear/frigate/blob/a745070b76276ef7865bf6513d627196ff1c6d10/frigate/data_processing/common/face/model.py
4. https://github.com/NickM-27/facenet-onnx
5. https://github.com/jakowenko/double-take/blob/92521a0bc8ba70f64c4f794332d48387663ba20e/api/src/constants/defaults.js
6. https://github.com/jakowenko/double-take/blob/92521a0bc8ba70f64c4f794332d48387663ba20e/api/src/util/process.util.js
7. https://github.com/jakowenko/double-take/blob/92521a0bc8ba70f64c4f794332d48387663ba20e/api/src/util/frigate.util.js
8. https://github.com/blakeblackshear/frigate/blob/a745070b76276ef7865bf6513d627196ff1c6d10/frigate/comms/webpush.py
9. https://github.com/exadel-inc/CompreFace/blob/ddf32da8245e2e6688c3ab6f60587fd3e31538c2/embedding-calculator/src/services/facescan/plugins/mixins.py
10. https://github.com/exadel-inc/CompreFace/blob/ddf32da8245e2e6688c3ab6f60587fd3e31538c2/java/api/src/main/java/com/exadel/frs/core/trainservice/component/migration/MigrationComponent.java
11. https://github.com/exadel-inc/CompreFace/blob/ddf32da8245e2e6688c3ab6f60587fd3e31538c2/custom-builds/README.md
12. https://github.com/codeproject/CodeProject.AI-FaceProcessing/blob/0189c53b0cfa054d69001d06caaa6448b4c537a7/intelligencelayer/face.py
13. https://github.com/johnolafenwa/DeepStack/blob/73b6f7ab0f9a8acc8f7a49c6f13594fe470de042/server/server.go
14. https://github.com/immich-app/immich/blob/469a870a2233e7361bcb855b183fd41272cfd056/machine-learning/immich_ml/models/facial_recognition/detection.py
15. https://github.com/immich-app/immich/blob/469a870a2233e7361bcb855b183fd41272cfd056/server/src/services/person.service.ts
16. https://github.com/immich-app/immich/blob/469a870a2233e7361bcb855b183fd41272cfd056/machine-learning/immich_ml/models/facial_recognition/_ops.py
17. https://github.com/deepinsight/insightface/blob/master/README.md
18. https://github.com/photoprism/photoprism/blob/6c9a3699a052812074f4ccf13bed8833cb16dec6/internal/ai/face/config.go
19. https://github.com/photoprism/photoprism/blob/6c9a3699a052812074f4ccf13bed8833cb16dec6/internal/ai/face/README.md
20. https://github.com/photoprism/photoprism/blob/6c9a3699a052812074f4ccf13bed8833cb16dec6/internal/entity/query/covers.go
21. https://github.com/photoprism/photoprism/blob/6c9a3699a052812074f4ccf13bed8833cb16dec6/internal/ai/face/models.go
22. https://github.com/ente-io/ente/blob/150be487bcb78c1f84041f27ddb0bc092077daaf/web/packages/new/photos/services/ml/cluster.ts
23. https://github.com/ente-io/ente/blob/150be487bcb78c1f84041f27ddb0bc092077daaf/mobile/apps/photos/lib/services/machine_learning/ml_model_assets.dart
24. https://github.com/ente-io/ente/blob/150be487bcb78c1f84041f27ddb0bc092077daaf/web/packages/new/photos/services/ml/worker.ts
25. https://github.com/roflcoopter/viseron/blob/b4f7098a7ee351e82fbff043bebd6577d982ab26/viseron/domains/post_processor/__init__.py
26. https://github.com/roflcoopter/viseron/blob/b4f7098a7ee351e82fbff043bebd6577d982ab26/viseron/components/dlib/predict.py
27. https://github.com/roflcoopter/viseron/blob/b4f7098a7ee351e82fbff043bebd6577d982ab26/viseron/domains/face_recognition/__init__.py
28. https://github.com/ZoneMinder/zmeventnotification/blob/bb6c9943df1f1304cffa72d392b0c65e2f5a3b5c/hook/objectconfig.ini
29. https://github.com/ZoneMinder/pyzm/blob/dc778e10665073a9d344ebab5f843fc1a9ae7f50/pyzm/ml/face_dlib.py
30. https://gitlab.com/Shinobi-Systems/shinobi-plugins/-/blob/7ce248b2532956846525b1a0b26411aed0580919/plugins/face-logger-groups-tfjs-gpu-2-0-0/shinobi-face.js
31. https://gitlab.com/Shinobi-Systems/shinobi-plugins/-/blob/7ce248b2532956846525b1a0b26411aed0580919/plugins/face-recognition-onnx/README.md
32. https://gitlab.com/Shinobi-Systems/shinobi-plugins/-/blob/7ce248b2532956846525b1a0b26411aed0580919/plugins/face-recognition-onnx/libs/faceStore.js
33. https://github.com/koush/scrypted/blob/b63b7df286998d528773decee0a5c15808c23237/plugins/openvino/src/predict/face_recognize.py
34. https://github.com/koush/scrypted/blob/b63b7df286998d528773decee0a5c15808c23237/plugins/objectdetector/src/smart-motionsensor.ts
35. https://github.com/serengil/deepface/blob/a2a24a748d67ab7ba19ee8813584f123c8b48747/deepface/modules/streaming.py
36. https://github.com/serengil/deepface/blob/a2a24a748d67ab7ba19ee8813584f123c8b48747/deepface/modules/recognition.py
37. https://github.com/blakeblackshear/frigate/blob/a745070b76276ef7865bf6513d627196ff1c6d10/docs/docs/configuration/advanced/reference.md
38. https://github.com/blakeblackshear/frigate/blob/a745070b76276ef7865bf6513d627196ff1c6d10/frigate/util/image.py
39. https://github.com/blakeblackshear/frigate/blob/a745070b76276ef7865bf6513d627196ff1c6d10/docs/docs/configuration/face_recognition.md
40. https://github.com/deepinsight/insightface/blob/master/python-package/insightface/app/face_analysis.py
41. https://github.com/opencv/opencv_zoo/blob/main/models/face_detection_yunet/README.md
42. https://github.com/IrvingMeng/MagFace/blob/main/README.md
43. https://github.com/pterhoer/FaceImageQuality/blob/master/face_image_quality.py
44. https://github.com/fdbtrs/CR-FIQA/blob/main/backbones/iresnet.py
45. https://pmc.ncbi.nlm.nih.gov/articles/PMC10938076/
46. https://github.com/deepinsight/insightface/blob/master/recognition/arcface_torch/eval_ijbc.py
47. https://github.com/mk-minchul/AdaFace/blob/master/validation_lq/validate_IJB_S.py
48. https://github.com/YirongMao/NAN/blob/master/README.md
49. https://github.com/mk-minchul/CAFace/blob/master/README.md
50. https://github.com/facefusion/facefusion/blob/master/facefusion/face_creator.py
51. https://github.com/facefusion/facefusion/blob/master/facefusion/processors/modules/face_enhancer/core.py
52. https://ar5iv.labs.arxiv.org/html/1804.09691
53. https://arxiv.org/html/2512.03199
54. https://ar5iv.labs.arxiv.org/html/2003.08124
55. https://arxiv.org/pdf/2308.07967
56. https://github.com/TencentARC/GFPGAN/blob/master/README.md
57. https://github.com/sczhou/CodeFormer/blob/master/inference_codeformer.py
58. https://github.com/TencentARC/GFPGAN/blob/master/gfpgan/utils.py
59. https://github.com/wzhouxiff/RestoreFormerPlusPlus/blob/main/scripts/metrics/cal_identity_distance.py
60. https://github.com/xinntao/facexlib/blob/master/README.md
61. https://github.com/sczhou/CodeFormer/blob/master/LICENSE
62. https://github.com/yangxy/GPEN/blob/main/README.md
63. https://github.com/yfeng95/DECA/blob/master/LICENSE
64. https://docs.facefusion.io/introduction/licenses
65. https://github.com/opencv/opencv_zoo/blob/main/models/face_recognition_sface/sface.py
66. https://github.com/deepinsight/insightface/blob/master/python-package/README.md
67. https://github.com/mk-minchul/AdaFace/blob/master/README.md
68. https://github.com/mk-minchul/AdaFace/issues/43
69. https://github.com/IrvingMeng/MagFace/issues/32
70. https://github.com/pterhoer/FaceImageQuality/blob/master/README.md
71. https://github.com/fdbtrs/CR-FIQA/blob/main/README.md
72. https://github.com/immich-app/immich/blob/469a870a2233e7361bcb855b183fd41272cfd056/docs/docs/features/facial-recognition.md
73. https://github.com/blakeblackshear/frigate/blob/a745070b76276ef7865bf6513d627196ff1c6d10/docs/docs/integrations/mqtt.md
74. https://github.com/home-assistant/core/blob/2025.6.0/homeassistant/components/dlib_face_identify/image_processing.py
