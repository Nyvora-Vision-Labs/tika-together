/*
 * Tika placement engine — runs entirely in the browser.
 *
 * Algorithm
 * ---------
 * 1. Load    Decode the photo onto a canvas, downscaled so the long side is ≤ 2048 px
 *            (browsers apply EXIF orientation when drawing).
 * 2. Detect  MediaPipe Face Landmarker returns a 468-point face mesh per face; Pose
 *            Landmarker returns body points (used for the shoulders).
 * 3. Measure For each face, derive from the mesh:
 *              • eye line       33 → 263 (outer eye corners) → roll angle + inter-ocular distance (IOD)
 *              • up vector      perpendicular to the eye line, pointing to the forehead
 *              • glabella       landmark 9 (between the eyebrows) → bottom anchor of the tika
 *              • forehead top   landmark 10 → available forehead length
 *              • yaw            nose tip (1) vs. face edges (234, 454) → horizontal foreshortening
 * 4. Size    tika: column length = 0.72 × forehead length, clamped to 0.30–0.60 × IOD;
 *                  dot radius = 0.08 × IOD (× foreshortening horizontally)
 *            mala: MediaPipe Pose Landmarker finds each person's shoulders (pose 11, 12); each
 *                  face is matched to the body whose nose is nearest. The shoulder line gives
 *                  the garland's position, tilt and size, so it rests on that person's own
 *                  shoulders even when their head leans onto a sibling. With no shoulders in
 *                  view it falls back to hanging straight down from just below the chin.
 * 5. Check   Compare forehead colour with cheek colour (landmarks 50, 280). A large difference
 *            means hair, a hat or a hand probably covers the forehead → warning.
 *            Forehead luminance sets a shade factor so the tika matches the photo's lighting.
 * 6. Paint   Mala first: two long ropes per person that start at the sides of the neck and
 *            hang almost straight down past the chest (a deep cubic U whose bottom is usually out
 *            of frame). Outer rope: marigold balls just touching, each a lit dome of ~90
 *            ruffled petals with dark crevices; ends near the neck shrink and darken as the
 *            rope goes back over the shoulders. Inner rope: oval makhamali heads of papery magenta bracts, turned
 *            along the string. Each rope casts a soft shadow on the body. The whole layer is
 *            masked by MediaPipe's body segmentation (body, clothes, hair — never faces or
 *            background) and blended with slight softness to match the photo.
 *            Then the tika: seven round powder dots, one per saptarangi colour, stacked bottom
 *            to top from between the eyebrows, each slightly irregular in size and position,
 *            plus a few white rice grains. Powder grain and the shade factor are added
 *            off-screen, then it is composited rotated by the roll angle, softly blurred, at
 *            88 % opacity so skin texture shows through.
 * 6b. Edit  After creation every mala is editable. A mala is stored as its two end points, a
 *            size unit and fixed rope lengths. Dragging an end re-solves the hang depth (bisection
 *            on the curve's arc length) so the rope keeps its length: closer ends hang lower,
 *            wider ends rise, and an end stops once the rope is pulled tight. Dragging the
 *            flowers moves the whole mala; the size control (slider, −/+ buttons, +/− keys, pinch)
 *            scales the whole mala uniformly about its neck point, 40–250 %.
 *            Flowers are cached as sprites so editing stays smooth.
 * 7. Compose
 *            separate  → each photo must contain a face (largest wins when several); crop a 4:5
 *                        portrait around each face and lay both side by side on a keepsake card.
 *            together  → keep faces at least 35 % the size of the largest (drops background
 *                        people); need ≥ 2; each mala is kept on its owner's side of the midpoint
 *                        between neighbouring siblings (slid over, then shrunk if needed); mala + tika on every kept face; frame the full photo.
 */
(function () {
  'use strict';

  const VISION_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14';
  const MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';
  const POSE_MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task';
  const SEGMENT_MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/1/selfie_multiclass_256x256.tflite';
  const MAX_SIDE = 2048;
  const MIN_RELATIVE_FACE_AREA = 0.35;
  const LM = { top: 10, glabella: 9, chin: 152, nose: 1, eyeL: 33, eyeR: 263, edgeL: 234, edgeR: 454, cheekL: 50, cheekR: 280 };
  // Saptarangi (seven-colour) tika dots, bottom → top.
  const TIKA_COLORS = ['#f2c318', '#c4161c', '#2f8f4e', '#f3eee2', '#2357b5', '#e2507a', '#ef7d1a'];
  // Marigold hue range (deep orange, occasionally golden) and makhamali (globe amaranth) magenta.
  const MARIGOLD_HUES = [[24, 30], [24, 30], [24, 30], [32, 38]];
  const MAKHAMALI_HUE = [306, 320];

  class TikaError extends Error {}

  // ---------- helpers ----------
  const sub = (a, b) => ({ x: a.x - b.x, y: a.y - b.y });
  const add = (a, b) => ({ x: a.x + b.x, y: a.y + b.y });
  const mul = (a, k) => ({ x: a.x * k, y: a.y * k });
  const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
  const norm = a => mul(a, 1 / (Math.hypot(a.x, a.y) || 1));
  const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
  function mulberry32(seed) {
    return function () {
      seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
      let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  // ---------- 1. load ----------
  async function loadCanvas(url) {
    const img = new Image();
    img.src = url;
    await img.decode();
    const scale = Math.min(1, MAX_SIDE / Math.max(img.naturalWidth, img.naturalHeight));
    const canvas = document.createElement('canvas');
    canvas.width = Math.round(img.naturalWidth * scale);
    canvas.height = Math.round(img.naturalHeight * scale);
    canvas.getContext('2d').drawImage(img, 0, 0, canvas.width, canvas.height);
    return canvas;
  }

  // ---------- 2. detect ----------
  let modelsPromise = null;
  function getModels() {
    if (!modelsPromise) {
      modelsPromise = (async () => {
        const { FaceLandmarker, PoseLandmarker, ImageSegmenter, FilesetResolver } = await import(`${VISION_URL}/vision_bundle.mjs`);
        const fileset = await FilesetResolver.forVisionTasks(`${VISION_URL}/wasm`);
        const create = async (Task, modelAssetPath, extra) => {
          const options = delegate => ({ baseOptions: { modelAssetPath, delegate }, runningMode: 'IMAGE', ...extra });
          try { return await Task.createFromOptions(fileset, options('GPU')); }
          catch { return Task.createFromOptions(fileset, options('CPU')); }
        };
        const [face, pose, segmenter] = await Promise.all([
          create(FaceLandmarker, MODEL_URL, { numFaces: 6, minFaceDetectionConfidence: 0.5, minFacePresenceConfidence: 0.5 }),
          // Pose is only used to find shoulders for the mala; without it we estimate from the chin.
          create(PoseLandmarker, POSE_MODEL_URL, { numPoses: 6, minPoseDetectionConfidence: 0.5 }).catch(() => null),
          // Body outline (hair / body skin / face skin / clothes) keeps garlands off faces and the
          // background. Optional: without it the mala is simply drawn unmasked.
          create(ImageSegmenter, SEGMENT_MODEL_URL, { outputCategoryMask: true, outputConfidenceMasks: false }).catch(() => null),
        ]);
        return { face, pose, segmenter };
      })().catch(() => {
        modelsPromise = null;
        throw new TikaError('The face finder could not load. Check your connection and try again.');
      });
    }
    return modelsPromise;
  }

  async function detectFaces(canvas) {
    const { face, pose } = await getModels();
    const faceSets = face.detect(canvas).faceLandmarks || [];
    let poses = [];
    if (pose) { try { poses = pose.detect(canvas).landmarks || []; } catch { poses = []; } }
    const owners = matchPoses(faceSets, poses, canvas);
    return faceSets
      .map((lm, i) => measureFace(lm, canvas, owners[i]))
      .sort((a, b) => b.area - a.area);
  }

  // Give each face the body whose pose-nose is closest to the face's nose (greedy, one body per
  // face), as long as it is within 0.4 face-widths (hugging siblings can produce a merged pose
  // between two faces; that must not be matched). Faces with no body fall back to estimates.
  function matchPoses(faceSets, poses, canvas) {
    const W = canvas.width, H = canvas.height;
    const pairs = [];
    faceSets.forEach((lm, fi) => {
      const nose = { x: lm[LM.nose].x * W, y: lm[LM.nose].y * H };
      const faceW = Math.hypot((lm[LM.edgeL].x - lm[LM.edgeR].x) * W, (lm[LM.edgeL].y - lm[LM.edgeR].y) * H);
      poses.forEach((p, pi) => {
        const d = Math.hypot(p[0].x * W - nose.x, p[0].y * H - nose.y);
        if (d < 0.4 * faceW) pairs.push({ fi, pi, d });
      });
    });
    pairs.sort((a, b) => a.d - b.d);
    const owners = [], used = new Set();
    for (const { fi, pi } of pairs) {
      if (owners[fi] || used.has(pi)) continue;
      owners[fi] = poses[pi];
      used.add(pi);
    }
    return owners;
  }

  // Where a garland may appear: on the body (body skin, clothes, accessories) and hair below
  // the head, never over a face or the background. Feathered so edges look tucked, not cut.
  // Categories of selfie_multiclass: 0 background, 1 hair, 2 body skin, 3 face skin, 4 clothes, 5 other.
  async function bodyMask(canvas) {
    const { segmenter } = await getModels();
    if (!segmenter) return null;
    try {
      const result = segmenter.segment(canvas);
      const cat = result.categoryMask;
      const data = cat.getAsUint8Array();
      const mw = cat.width, mh = cat.height;
      const small = document.createElement('canvas');
      small.width = mw; small.height = mh;
      const sctx = small.getContext('2d');
      const img = sctx.createImageData(mw, mh);
      for (let i = 0; i < data.length; i++) {
        const c = data[i];
        img.data[i * 4 + 3] = c === 2 || c === 4 || c === 5 || c === 1 ? 255 : 0;
      }
      sctx.putImageData(img, 0, 0);
      cat.close();
      if (result.close) result.close();
      const mask = document.createElement('canvas');
      mask.width = canvas.width; mask.height = canvas.height;
      const m = mask.getContext('2d');
      m.filter = `blur(${Math.max(1, Math.round(Math.max(canvas.width, canvas.height) / 500))}px)`;
      m.drawImage(small, 0, 0, mask.width, mask.height);
      return mask;
    } catch { return null; }
  }

  // ---------- 3–5. measure, size, check ----------
  function measureFace(lm, canvas, pose) {
    const W = canvas.width, H = canvas.height;
    const P = i => ({ x: lm[i].x * W, y: lm[i].y * H });
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (const p of lm) {
      minX = Math.min(minX, p.x * W); maxX = Math.max(maxX, p.x * W);
      minY = Math.min(minY, p.y * H); maxY = Math.max(maxY, p.y * H);
    }
    const bbox = { x: minX, y: minY, w: maxX - minX, h: maxY - minY };

    const eyeL = P(LM.eyeL), eyeR = P(LM.eyeR);
    const eyeDir = norm(sub(eyeR, eyeL));
    const roll = Math.atan2(eyeDir.y, eyeDir.x);
    const up = { x: eyeDir.y, y: -eyeDir.x };
    const iod = dist(eyeL, eyeR);

    const nose = P(LM.nose);
    const dl = dist(nose, P(LM.edgeL)), dr = dist(nose, P(LM.edgeR));
    const yaw = (dl - dr) / (dl + dr || 1);
    const foreshorten = clamp(Math.sqrt(1 - yaw * yaw), 0.45, 1);

    const glabella = P(LM.glabella);
    const foreheadLen = dist(glabella, P(LM.top));
    const length = clamp(0.72 * foreheadLen, 0.30 * iod, 0.60 * iod);
    const dotR = 0.08 * iod;
    const center = add(glabella, mul(up, 0.04 * iod + length / 2));

    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    const r = Math.max(2, iod * 0.07);
    const forehead = sampleColor(ctx, center, r);
    const cheeks = averageColor(sampleColor(ctx, P(LM.cheekL), r), sampleColor(ctx, P(LM.cheekR), r));
    const covered = colorDistance(forehead, cheeks) > 80;
    const lum = 0.2126 * forehead.r + 0.7152 * forehead.g + 0.0722 * forehead.b;
    const shade = clamp(lum / 185, 0.55, 1);

    return { bbox, area: bbox.w * bbox.h, iod, roll, yaw, tika: { center, length, dotR, foreshorten, shade }, mala: measureMala(P, up, pose, W, H), covered };
  }

  // Where the mala sits: a "neck base" point, a down/side frame and a size unit (shoulder width).
  // Best source is the person's own shoulders from the pose model (pose landmarks 11 and 12):
  // they follow the body even when the head leans onto a sibling. Otherwise estimate from the
  // face: garlands hang with gravity, so mostly straight down, from just below the chin.
  function malaFrame(P, up, pose, W, H) {
    const chin = P(LM.chin), top = P(LM.top);
    const faceW = dist(P(LM.edgeL), P(LM.edgeR));
    const faceH = dist(top, chin);
    if (pose) {
      const l = pose[11], r = pose[12];
      const seen = p => (p.visibility ?? 1) > 0.5;
      const L = { x: l.x * W, y: l.y * H }, R = { x: r.x * W, y: r.y * H };
      const shoulders = dist(L, R);
      if (seen(l) && seen(r) && shoulders > 1.1 * faceW && shoulders < 3.5 * faceW) {
        let side = norm(sub(L, R));
        if (side.x < 0) side = mul(side, -1);
        const down = { x: -side.y, y: side.x };
        const mid = mul(add(L, R), 0.5);
        // The garland rests on the shoulders just beside the neck, a little above the joint line,
        // but never higher than just under the chin.
        // MediaPipe's shoulder points are the joints; the garland starts higher, where the neck
        // meets the shoulders, but never above just under the chin.
        let neckBase = add(mid, mul(down, -0.22 * shoulders));
        if (neckBase.y < chin.y + 0.12 * faceH) neckBase = { x: neckBase.x, y: chin.y + 0.12 * faceH };
        return { neckBase, down, side, unit: shoulders };
      }
    }
    const down = norm(add(mul({ x: 0, y: 1 }, 0.85), mul(up, -0.15)));
    const side = { x: -down.y, y: down.x };
    // The head pivots on the neck: when it tilts, the top swings far and the chin only a little,
    // so the base of the neck sits slightly back from the chin, opposite the tilt.
    const neckBase = { x: chin.x + (chin.x - top.x) * 0.45, y: chin.y + 0.15 * faceH };
    return { neckBase, down, side, unit: 1.9 * faceW };
  }

  function measureMala(P, up, pose, W, H) {
    const { neckBase, down, side, unit } = malaFrame(P, up, pose, W, H);
    // Each strand starts on the shoulders either side of the neck, flares out and dips into a U
    // over the chest. All sizes are fractions of the shoulder width.
    const strand = (neck, flare, depth, flowerR) => ({
      a: add(neckBase, mul(side, -neck * unit)),
      b: add(neckBase, mul(side, neck * unit)),
      flare: flare * unit,
      depth: depth * unit,
      r: flowerR * unit,
    });
    // Like a real Bhai Tika mala: thick ropes that start right at the sides of the neck and hang
    // almost straight down past the chest; the bottom of the loop is usually out of frame.
    return {
      down, side, neckBase, unit,
      outer: strand(0.21, 0.12, 1.3, 0.075),
      inner: strand(0.125, 0.09, 1.2, 0.04),
    };
  }

  function sampleColor(ctx, p, r) {
    const x = Math.round(p.x - r), y = Math.round(p.y - r), s = Math.round(r * 2);
    const sx = clamp(x, 0, ctx.canvas.width - 1), sy = clamp(y, 0, ctx.canvas.height - 1);
    const sw = clamp(s, 1, ctx.canvas.width - sx), sh = clamp(s, 1, ctx.canvas.height - sy);
    const d = ctx.getImageData(sx, sy, sw, sh).data;
    let R = 0, G = 0, B = 0, n = 0;
    for (let i = 0; i < d.length; i += 4) { R += d[i]; G += d[i + 1]; B += d[i + 2]; n++; }
    return { r: R / n, g: G / n, b: B / n };
  }
  const averageColor = (a, b) => ({ r: (a.r + b.r) / 2, g: (a.g + b.g) / 2, b: (a.b + b.b) / 2 });
  const colorDistance = (a, b) => Math.hypot(a.r - b.r, a.g - b.g, a.b - b.b);

  // ---------- 6. paint ----------
  function raggedCircle(path, cx, cy, rx, ry, rand, rough = 0.12, steps = 28) {
    for (let i = 0; i <= steps; i++) {
      const t = (i / steps) * Math.PI * 2;
      const k = 1 + (rand() - 0.5) * rough;
      const x = cx + Math.cos(t) * rx * k, y = cy + Math.sin(t) * ry * k;
      i === 0 ? path.moveTo(x, y) : path.lineTo(x, y);
    }
    path.closePath();
  }

  function paintTika(ctx, face, seed) {
    const { center, length: l, dotR, foreshorten, shade } = face.tika;
    const rand = mulberry32(seed);
    const rx = dotR * foreshorten;
    const pad = Math.ceil(dotR * 2 + 4);
    const off = document.createElement('canvas');
    off.width = Math.ceil(rx * 2 + pad * 2);
    off.height = Math.ceil(l + dotR * 2 + pad * 2);
    const o = off.getContext('2d');
    o.translate(off.width / 2, off.height / 2);

    // Seven round dots stacked bottom (between the eyebrows) → top.
    const n = TIKA_COLORS.length;
    const step = l / (n - 1);
    const all = new Path2D();
    TIKA_COLORS.forEach((color, i) => {
      const k = 0.85 + rand() * 0.3;
      const cx = (rand() - 0.5) * rx * 0.3;
      const cy = l / 2 - i * step + (rand() - 0.5) * step * 0.2;
      const dot = new Path2D();
      raggedCircle(dot, cx, cy, rx * k, dotR * k, rand);
      o.fillStyle = color;
      o.fill(dot);
      all.addPath(dot);
    });
    // A few rice grains (akshata) stuck around the column.
    for (let i = 0; i < 4; i++) {
      o.save();
      o.translate((rand() - 0.5) * rx * 2.4, (rand() - 0.5) * l);
      o.rotate(rand() * Math.PI);
      o.fillStyle = '#fbf8ef';
      o.beginPath();
      o.ellipse(0, 0, dotR * 0.38, dotR * 0.17, 0, 0, Math.PI * 2);
      o.fill();
      o.restore();
    }
    // Powder grain, only on the dots.
    o.save();
    o.clip(all);
    const grains = Math.round((rx * 2 * l) / 4);
    const grainR = Math.max(0.5, dotR * 0.06);
    for (let i = 0; i < grains; i++) {
      o.fillStyle = rand() < 0.5 ? `rgba(255,255,255,${0.1 + rand() * 0.2})` : `rgba(0,0,0,${0.08 + rand() * 0.18})`;
      o.beginPath();
      o.arc((rand() - 0.5) * rx * 2, (rand() - 0.5) * (l + dotR * 2), grainR * (0.5 + rand()), 0, Math.PI * 2);
      o.fill();
    }
    o.restore();

    // Match the photo's lighting on the forehead.
    o.globalCompositeOperation = 'source-atop';
    o.fillStyle = `rgba(0,0,0,${1 - shade})`;
    o.fillRect(-off.width / 2, -off.height / 2, off.width, off.height);

    ctx.save();
    ctx.translate(center.x, center.y);
    ctx.rotate(face.roll);
    ctx.filter = `blur(${Math.max(0.4, dotR * 0.04).toFixed(2)}px)`;
    ctx.globalAlpha = 0.88;
    ctx.drawImage(off, -off.width / 2, -off.height / 2);
    ctx.restore();
  }

  // ---------- mala (flower garland) ----------
  // Points spaced evenly (by arc length) along a strand, each with the strand's direction there.
  // The strand's curve: a cubic from a to b whose control points hang `depth` × 4/3 below the
  // anchors (so a level strand dips exactly `depth`) and flare outward by `flare`.
  function curve(strand, down, side) {
    const { a, b, depth, flare } = strand;
    const k = (depth * 4) / 3;
    const c1 = add(add(a, mul(down, k)), mul(side, -flare)), c2 = add(add(b, mul(down, k)), mul(side, flare));
    return t => {
      const u = 1 - t;
      return {
        x: u * u * u * a.x + 3 * u * u * t * c1.x + 3 * u * t * t * c2.x + t * t * t * b.x,
        y: u * u * u * a.y + 3 * u * u * t * c1.y + 3 * u * t * t * c2.y + t * t * t * b.y,
      };
    };
  }

  function curveLength(strand, down, side, n = 48) {
    const at = curve(strand, down, side);
    let len = 0, prev = at(0);
    for (let i = 1; i <= n; i++) { const p = at(i / n); len += dist(prev, p); prev = p; }
    return len;
  }

  function pointsAlong(strand, down, side, spacing) {
    const at = curve(strand, down, side);
    const samples = [];
    let prev = at(0), len = 0;
    for (let i = 0; i <= 400; i++) {
      const p = at(i / 400);
      len += dist(prev, p);
      samples.push({ p, len });
      prev = p;
    }
    const out = [];
    for (let d = 0, j = 0; d <= len; d += spacing) {
      while (j < samples.length - 1 && samples[j].len < d) j++;
      const q = samples[Math.min(j + 1, samples.length - 1)].p, o = samples[Math.max(j - 1, 0)].p;
      out.push({ x: samples[j].p.x, y: samples[j].p.y, angle: Math.atan2(q.y - o.y, q.x - o.x) });
    }
    return out;
  }

  const hsl = (h, sat, l, a = 1) => `hsla(${h.toFixed(1)},${sat.toFixed(0)}%,${clamp(l, 0, 100).toFixed(1)}%,${a})`;
  const LIGHT = norm({ x: -0.45, y: -0.9 }); // light from above, slightly left

  // One marigold: a ruffled ball of ~90 small petals packed in a dome, lit from above, with
  // dark crevices between petals and a darker rim where it meets its neighbours.
  function paintMarigold(ctx, c, r, rand, dim = 0) {
    const [h0, h1] = MARIGOLD_HUES[Math.floor(rand() * MARIGOLD_HUES.length)];
    const hue = h0 + rand() * (h1 - h0);
    ctx.save();
    ctx.translate(c.x, c.y);
    // Contact shadow / crevice around the ball.
    const rim = ctx.createRadialGradient(0, r * 0.12, r * 0.55, 0, r * 0.12, r * 1.12);
    rim.addColorStop(0, 'rgba(50,15,0,0.55)');
    rim.addColorStop(1, 'rgba(50,15,0,0)');
    ctx.fillStyle = rim;
    ctx.beginPath(); ctx.arc(0, r * 0.12, r * 1.12, 0, Math.PI * 2); ctx.fill();
    // Dark body so gaps between petals read as depth.
    ctx.fillStyle = hsl(hue - 6, 95, 22 - dim * 8);
    ctx.beginPath(); ctx.arc(0, 0, r * 0.92, 0, Math.PI * 2); ctx.fill();
    // Petals scattered over the dome, outermost first so the top of the dome overlaps them.
    const petals = [];
    for (let i = 0; i < 90; i++) {
      const t = rand() * Math.PI * 2, d = Math.sqrt(rand()) * 0.9;
      petals.push({ t, d });
    }
    petals.sort((p, q) => q.d - p.d);
    for (const { t, d } of petals) {
      const px = Math.cos(t) * d * r, py = Math.sin(t) * d * r;
      // Dome lighting: facing the light → brighter; centre (higher) → brighter; rim → darker.
      const facing = -(Math.cos(t) * LIGHT.x + Math.sin(t) * LIGHT.y) * d;
      const light = 50 + facing * 12 - d * d * 14 + (rand() - 0.5) * 12 - dim * 14;
      const pr = r * (0.14 + rand() * 0.1) * (1.1 - d * 0.3);
      ctx.save();
      ctx.translate(px, py);
      ctx.rotate(t + Math.PI / 2 + (rand() - 0.5) * 1.2);
      ctx.beginPath();
      const steps = 9;
      for (let k = 0; k <= steps; k++) {
        const u = (k / steps) * Math.PI * 2;
        const wave = 1 + Math.sin(u * 3 + rand() * 6) * 0.18;
        const x = Math.cos(u) * pr * wave, y = Math.sin(u) * pr * 0.7 * wave;
        k ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
      }
      ctx.closePath();
      ctx.fillStyle = hsl(hue + (rand() - 0.5) * 6, 100, light);
      ctx.fill();
      // Curled petal edge catches the light.
      ctx.strokeStyle = hsl(hue + 4, 100, light + 12, 0.35);
      ctx.lineWidth = Math.max(0.3, pr * 0.18);
      ctx.beginPath(); ctx.arc(0, 0, pr * 0.8, Math.PI * 1.1, Math.PI * 1.9); ctx.stroke();
      ctx.restore();
    }
    ctx.restore();
  }

  // One makhamali head: a small oval cone of papery bracts, oriented along the string.
  function paintMakhamali(ctx, c, r, rand, dim = 0) {
    const [h0, h1] = MAKHAMALI_HUE;
    const hue = h0 + rand() * (h1 - h0);
    const rx = r * 1.3, ry = r; // longer along the string
    ctx.save();
    ctx.translate(c.x, c.y);
    ctx.rotate(c.angle + (rand() - 0.5) * 0.5);
    const body = ctx.createRadialGradient(-rx * 0.2, -ry * 0.35, r * 0.1, 0, 0, rx);
    body.addColorStop(0, hsl(hue, 55, 50 - dim * 12));
    body.addColorStop(0.6, hsl(hue, 62, 32 - dim * 10));
    body.addColorStop(1, hsl(hue - 4, 70, 16 - dim * 6));
    ctx.fillStyle = body;
    ctx.beginPath(); ctx.ellipse(0, 0, rx, ry, 0, 0, Math.PI * 2); ctx.fill();
    // Overlapping bract scales with pale tips.
    for (let i = 0; i < 26; i++) {
      const t = rand() * Math.PI * 2, d = Math.sqrt(rand()) * 0.85;
      const x = Math.cos(t) * rx * d, y = Math.sin(t) * ry * d;
      const facing = -(Math.cos(t) * LIGHT.x + Math.sin(t) * LIGHT.y) * d;
      ctx.fillStyle = hsl(hue + 4, 50, 44 + facing * 14 + (rand() - 0.5) * 12 - dim * 14, 0.75);
      ctx.beginPath(); ctx.ellipse(x, y, r * 0.2, r * 0.13, t, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = `rgba(255,225,245,${0.15 + rand() * 0.25})`;
      ctx.beginPath(); ctx.arc(x + Math.cos(t) * r * 0.1, y + Math.sin(t) * r * 0.08, r * 0.06, 0, Math.PI * 2); ctx.fill();
    }
    ctx.restore();
  }

  // Shadow the rope casts on the body, so the garland sits on the person instead of floating.
  function paintRopeShadow(ctx, points, r, down) {
    if (points.length < 2) return;
    ctx.save();
    ctx.filter = `blur(${(r * 0.45).toFixed(1)}px)`;
    ctx.strokeStyle = 'rgba(25,10,0,0.42)';
    ctx.lineWidth = r * 2;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.beginPath();
    points.forEach((p, i) => {
      const x = p.x + down.x * r * 0.28 + r * 0.08, y = p.y + down.y * r * 0.28;
      i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    });
    ctx.stroke();
    ctx.restore();
  }

  // Horizontal extent of a mala, including the flowers' radius.
  function malaExtent(mala) {
    let lo = Infinity, hi = -Infinity;
    for (const s of [mala.outer, mala.inner]) {
      for (const p of pointsAlong(s, mala.down, mala.side, s.r * 0.5)) {
        lo = Math.min(lo, p.x - s.r); hi = Math.max(hi, p.x + s.r);
      }
    }
    return { lo, hi };
  }

  function transformMala(mala, dx, k) {
    const c = mala.neckBase;
    const move = p => ({ x: c.x + (p.x - c.x) * k + dx, y: c.y + (p.y - c.y) * k });
    for (const s of [mala.outer, mala.inner]) {
      s.a = move(s.a); s.b = move(s.b);
      s.flare *= k; s.depth *= k; s.r *= k;
    }
    mala.neckBase = { x: c.x + dx, y: c.y };
  }

  // Together photos: siblings' bodies don't overlap, so each garland must stay between the
  // midpoints to its neighbours' necks. Slide it toward its owner's side first (up to 0.3 of
  // a shoulder width), then shrink it (down to 75 %) if it still doesn't fit.
  function keepMalasApart(faces) {
    const necks = faces.map(f => f.mala.neckBase.x);
    faces.forEach((face, i) => {
      const left = i > 0 ? (necks[i - 1] + necks[i]) / 2 : -Infinity;
      const right = i < faces.length - 1 ? (necks[i] + necks[i + 1]) / 2 : Infinity;
      const mala = face.mala;
      let { lo, hi } = malaExtent(mala);
      const over = Math.max(0, left - lo) - Math.max(0, hi - right);
      const maxShift = 0.3 * mala.unit;
      const dx = clamp(over, -maxShift, maxShift);
      if (dx) transformMala(mala, dx, 1);
      ({ lo, hi } = malaExtent(mala));
      const c = mala.neckBase.x;
      const k = Math.min(1,
        lo < left ? (c - left) / (c - lo) : 1,
        hi > right ? (right - c) / (hi - c) : 1);
      if (k < 1) transformMala(mala, 0, Math.max(0.75, k));
    });
  }

  // ---------- editable mala ----------
  // A mala is stored as its two end points (where it sits at the sides of the neck), a size unit,
  // the gravity direction and the fixed rope lengths (in units). Moving an end never changes the
  // length: the hang depth is solved so the rope keeps its length, like a real garland, and an
  // end can't be pulled further than the rope allows.
  const OUTER = { inset: 0, flare: 0.12, r: 0.075 };
  const INNER = { inset: 0.085, flare: 0.09, r: 0.04 };

  function strandsOf(m) {
    const span = dist(m.a, m.b);
    const side = span > 1e-6 ? norm(sub(m.b, m.a)) : { x: -m.down.y, y: m.down.x };
    const inset = Math.min(INNER.inset * m.unit, span * 0.3);
    const mk = (a, b, spec, lenU) => ({ a, b, flare: spec.flare * m.unit, r: spec.r * m.unit, len: lenU * m.unit, depth: 0 });
    return {
      side,
      outer: mk(m.a, m.b, OUTER, m.lenOuter),
      inner: mk(add(m.a, mul(side, inset)), sub(m.b, mul(side, inset)), INNER, m.lenInner),
    };
  }

  function solveDepth(st, down, side) {
    st.depth = 0;
    if (curveLength(st, down, side) >= st.len) return st; // stretched tight
    let lo = 0, hi = st.len;
    st.depth = hi;
    while (curveLength(st, down, side) < st.len && hi < st.len * 8) { lo = hi; hi *= 2; st.depth = hi; }
    for (let i = 0; i < 28; i++) {
      st.depth = (lo + hi) / 2;
      if (curveLength(st, down, side) < st.len) lo = st.depth; else hi = st.depth;
    }
    st.depth = (lo + hi) / 2;
    return st;
  }

  function malaGeometry(m) {
    const g = strandsOf(m);
    solveDepth(g.outer, m.down, g.side);
    solveDepth(g.inner, m.down, g.side);
    return { down: m.down, side: g.side, outer: g.outer, inner: g.inner };
  }

  function feasible(m) {
    const { side, outer, inner } = strandsOf(m);
    return curveLength(outer, m.down, side) <= outer.len && curveLength(inner, m.down, side) <= inner.len;
  }

  function toEditable(face, seed) {
    const { down, side, outer, inner } = face.mala;
    const unit = outer.r / OUTER.r;
    const m = {
      a: { ...outer.a }, b: { ...outer.b }, down, unit, unit0: unit, seed,
      lenOuter: curveLength(outer, down, side) / unit,
      lenInner: curveLength(inner, down, side) / unit,
    };
    m.initial = { a: { ...m.a }, b: { ...m.b }, unit };
    return m;
  }

  const lerp = (p, q, t) => ({ x: p.x + (q.x - p.x) * t, y: p.y + (q.y - p.y) * t });

  // Move one end ('a' or 'b'). If the rope can't reach, stop where it is pulled tight.
  function moveEndpoint(m, which, p) {
    const old = { ...m[which] };
    m[which] = { ...p };
    if (feasible(m)) return;
    let lo = 0, hi = 1;
    for (let i = 0; i < 22; i++) {
      const t = (lo + hi) / 2;
      m[which] = lerp(old, p, t);
      if (feasible(m)) lo = t; else hi = t;
    }
    m[which] = lerp(old, p, lo);
  }

  function moveMala(m, dx, dy) {
    m.a = { x: m.a.x + dx, y: m.a.y + dy };
    m.b = { x: m.b.x + dx, y: m.b.y + dy };
  }

  // Bigger/smaller: the whole mala scales uniformly about the middle of its two ends (the neck),
  // so flowers, rope length and the spread of the ends all grow or shrink together and the
  // garland keeps its shape. (Uniform scaling keeps the rope feasible; the fallback below is a
  // safety net for rounding.)
  const SIZE_MIN = 0.4, SIZE_MAX = 2.5;
  function resizeMala(m, factor) {
    const unit = m.unit0 * clamp(factor, SIZE_MIN, SIZE_MAX);
    const k = unit / m.unit;
    const c = lerp(m.a, m.b, 0.5);
    m.a = add(c, mul(sub(m.a, c), k));
    m.b = add(c, mul(sub(m.b, c), k));
    m.unit = unit;
    if (feasible(m)) return;
    const mid = lerp(m.a, m.b, 0.5), a0 = { ...m.a }, b0 = { ...m.b };
    let lo = 0, hi = 1;
    for (let i = 0; i < 22; i++) {
      const t = (lo + hi) / 2;
      m.a = lerp(a0, mid, t); m.b = lerp(b0, mid, t);
      if (feasible(m)) hi = t; else lo = t;
    }
    m.a = lerp(a0, mid, hi); m.b = lerp(b0, mid, hi);
  }

  function resetMala(m) {
    m.a = { ...m.initial.a }; m.b = { ...m.initial.b }; m.unit = m.initial.unit;
  }

  function malaPoints(m) {
    const g = malaGeometry(m);
    return {
      g,
      marigolds: pointsAlong(g.outer, g.down, g.side, g.outer.r * 1.3),
      makhamali: pointsAlong(g.inner, g.down, g.side, g.inner.r * 1.7),
    };
  }

  // Which mala (index) is under point p (image pixels), or -1.
  function hitTest(malas, p, slop = 0) {
    let best = -1, bestD = Infinity;
    malas.forEach((m, i) => {
      const { g, marigolds, makhamali } = malaPoints(m);
      for (const [pts, r] of [[marigolds, g.outer.r], [makhamali, g.inner.r]]) {
        for (const q of pts) {
          const d = dist(p, q) - r * 1.1 - slop;
          if (d < 0 && d < bestD) { bestD = d; best = i; }
        }
      }
    });
    return best;
  }

  // Flowers are painted once into small cached sprites (a few looks per kind, size bucket and
  // shade), then stamped along the rope, so dragging stays smooth.
  const SPRITE_VARIANTS = 8;
  const sprites = new Map();
  function sprite(kind, rPx, variant, dimLevel) {
    const R = Math.max(4, Math.round(rPx / 2) * 2);
    const key = `${kind}:${R}:${variant}:${dimLevel}`;
    let c = sprites.get(key);
    if (c) return c;
    if (sprites.size > 600) sprites.clear();
    const size = Math.ceil(R * 2.8);
    c = document.createElement('canvas');
    c.width = c.height = size;
    const x = c.getContext('2d');
    const rand = mulberry32(variant * 131 + (kind === 'marigold' ? 1 : 7));
    const dim = dimLevel / 3;
    if (kind === 'marigold') paintMarigold(x, { x: size / 2, y: size / 2 }, R, rand, dim);
    else paintMakhamali(x, { x: size / 2, y: size / 2, angle: 0 }, R, rand, dim);
    c.R = R;
    sprites.set(key, c);
    return c;
  }

  function stamp(ctx, kind, p, r, scale, variant, angle, dim) {
    const sp = sprite(kind, r * scale, variant, Math.round(dim * 3));
    const k = r / sp.R; // sprite pixels → image pixels
    ctx.save();
    ctx.translate(p.x, p.y);
    ctx.rotate(angle);
    ctx.drawImage(sp, (-sp.width / 2) * k, (-sp.height / 2) * k, sp.width * k, sp.height * k);
    ctx.restore();
  }

  // Draw malas onto ctx. `scale` maps image pixels to ctx pixels. `mask` keeps flowers on bodies;
  // with `ghost`, the hidden parts show faintly (used while editing so the mala never vanishes).
  function drawMalas(ctx, malas, { scale = 1, mask = null, ghost = false, soft = false } = {}) {
    const W = ctx.canvas.width, H = ctx.canvas.height;
    const layer = document.createElement('canvas');
    layer.width = W; layer.height = H;
    const l = layer.getContext('2d');
    l.setTransform(scale, 0, 0, scale, 0, 0);
    let softness = 0;
    for (const m of malas) {
      const { g, marigolds, makhamali } = malaPoints(m);
      const { outer, inner, down } = g;
      paintRopeShadow(l, marigolds, outer.r, down);
      paintRopeShadow(l, makhamali, inner.r, down);
      // Near the neck the rope goes back over the shoulders: smaller and darker there.
      const recede = (p, st) => Math.max(0, 1 - Math.min(dist(p, st.a), dist(p, st.b)) / (st.r * 4));
      makhamali.forEach((p, i) => {
        const h = mulberry32(m.seed * 7919 + i * 31 + 5);
        const k = recede(p, inner);
        stamp(l, 'makhamali', p, inner.r * (0.92 + h() * 0.16) * (1 - 0.3 * k), scale, Math.floor(h() * SPRITE_VARIANTS), p.angle + (h() - 0.5) * 0.5, k);
      });
      marigolds.forEach((p, i) => {
        const h = mulberry32(m.seed * 7919 + i * 31 + 11);
        const k = recede(p, outer);
        stamp(l, 'marigold', p, outer.r * (0.92 + h() * 0.16) * (1 - 0.3 * k), scale, Math.floor(h() * SPRITE_VARIANTS), h() * Math.PI * 2, k);
      });
      softness = Math.max(softness, outer.r * scale * 0.02);
    }
    l.setTransform(1, 0, 0, 1, 0, 0);
    if (mask) {
      if (ghost) {
        ctx.save();
        ctx.setTransform(1, 0, 0, 1, 0, 0);
        ctx.globalAlpha = 0.35;
        ctx.drawImage(layer, 0, 0);
        ctx.restore();
      }
      l.globalCompositeOperation = 'destination-in';
      l.drawImage(mask, 0, 0, mask.width * scale, mask.height * scale);
    }
    ctx.save();
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    if (soft) ctx.filter = `blur(${clamp(softness, 0.3, 1.2).toFixed(2)}px)`;
    ctx.drawImage(layer, 0, 0);
    ctx.restore();
  }

  // ---------- scenes: one photo + its malas, editable after creation ----------
  // base = photo with tika (no mala). The mala is painted under the tika in reality, but they
  // never overlap, so drawing it on top of the tika'd photo is equivalent.
  function makeScene(id, base, mask, faces, seed) {
    return { id, base, mask, malas: faces.map((f, i) => toEditable(f, seed + i * 13)) };
  }

  function composeScene(scene) {
    const c = document.createElement('canvas');
    c.width = scene.base.width; c.height = scene.base.height;
    const ctx = c.getContext('2d');
    ctx.drawImage(scene.base, 0, 0);
    drawMalas(ctx, scene.malas, { mask: scene.mask, soft: true });
    return c;
  }

  // Editor view: ctx is the on-screen canvas; scale maps image pixels to it.
  function drawEditor(ctx, scene, scale, selected) {
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    ctx.drawImage(scene.base, 0, 0, scene.base.width * scale, scene.base.height * scale);
    drawMalas(ctx, scene.malas, { scale, mask: scene.mask, ghost: true });
    const m = scene.malas[selected];
    if (!m) return;
    const hr = Math.max(9, 11 * (window.devicePixelRatio || 1));
    ctx.save();
    ctx.setLineDash([6, 5]);
    ctx.strokeStyle = 'rgba(255,255,255,0.85)';
    ctx.lineWidth = 1.5 * (window.devicePixelRatio || 1);
    ctx.beginPath();
    ctx.moveTo(m.a.x * scale, m.a.y * scale);
    ctx.lineTo(m.b.x * scale, m.b.y * scale);
    ctx.stroke();
    ctx.setLineDash([]);
    for (const p of [m.a, m.b]) {
      ctx.beginPath();
      ctx.arc(p.x * scale, p.y * scale, hr, 0, Math.PI * 2);
      ctx.fillStyle = '#fffdf8';
      ctx.shadowColor = 'rgba(0,0,0,0.35)'; ctx.shadowBlur = 6;
      ctx.fill();
      ctx.shadowColor = 'transparent';
      ctx.lineWidth = 3 * (window.devicePixelRatio || 1);
      ctx.strokeStyle = '#803b38';
      ctx.stroke();
    }
    ctx.restore();
  }

  function faceWarnings(face, label) {
    const out = [];
    if (face.covered) out.push(`${label}: the forehead looks partly covered (hair or a hat?), so the tika may sit on top of it.`);
    if (face.iod < 28) out.push(`${label}: the face is small in this photo, so the tika may look soft. A closer photo works best.`);
    if (Math.abs(face.yaw) > 0.45) out.push(`${label}: the face is turned to the side. A photo facing the camera gives a neater tika.`);
    return out;
  }

  // ---------- 7. compose ----------
  const PAPER = '#faf7f0', INK = '#40382f', WINE = '#803b38', MUTED = '#817a70';
  const SERIF = "'Playfair Display', Georgia, serif";
  const SANS = "'DM Sans', system-ui, sans-serif";
  // What each sibling is called on the keepsake. Users pick दिदी र भाइ or बहिनी र दाइ after creating.
  const LABELS = {
    bahini: { word: 'बहिनी', sub: 'little sister' },
    didi: { word: 'दिदी', sub: 'big sister' },
    bhai: { word: 'भाइ', sub: 'little brother' },
    dai: { word: 'दाइ', sub: 'big brother' },
  };
  const DEFAULT_LABELS = { sister: 'didi', brother: 'bhai' };

  function portraitCrop(canvas, face, aspect) {
    const { x, y, w, h } = face.bbox;
    let ch = h * 2.9, cw = ch * aspect;
    const fit = Math.min(1, canvas.width / cw, canvas.height / ch);
    cw *= fit; ch *= fit;
    const cx = x + w / 2, cy = y + h / 2 + h * 0.55;
    return {
      sx: clamp(cx - cw / 2, 0, canvas.width - cw),
      sy: clamp(cy - ch / 2, 0, canvas.height - ch),
      sw: cw, sh: ch,
    };
  }

  function drawPolaroid(ctx, canvas, face, { cx, cy, w, angle, label, sub }) {
    const photoW = w - 36, photoH = photoW * 1.25;
    const h = photoH + 110;
    const crop = portraitCrop(canvas, face, 0.8);
    ctx.save();
    ctx.translate(cx, cy);
    ctx.rotate(angle);
    ctx.shadowColor = 'rgba(97,68,39,0.18)'; ctx.shadowBlur = 40; ctx.shadowOffsetY = 14;
    ctx.fillStyle = '#fffdf8';
    ctx.fillRect(-w / 2, -h / 2, w, h);
    ctx.shadowColor = 'transparent';
    ctx.drawImage(canvas, crop.sx, crop.sy, crop.sw, crop.sh, -photoW / 2, -h / 2 + 18, photoW, photoH);
    ctx.fillStyle = '#8c4a3f';
    ctx.font = `500 36px ${SERIF}`;
    ctx.textBaseline = 'middle';
    ctx.textAlign = 'left';
    ctx.fillText(label, -photoW / 2, h / 2 - 46);
    ctx.fillStyle = MUTED;
    ctx.font = `italic 24px ${SERIF}`;
    ctx.textAlign = 'right';
    ctx.fillText(sub, photoW / 2, h / 2 - 46);
    ctx.restore();
  }

  function drawFooter(ctx, W, y, pair) {
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillStyle = WINE;
    ctx.font = `500 52px ${SERIF}`;
    ctx.fillText('शुभ भाइटीका', W / 2, y);
    ctx.fillStyle = MUTED;
    ctx.font = `500 20px ${SANS}`;
    ctx.fillText(pair ? `${pair.toUpperCase()}  ·  A LIFETIME OF LOVE  ·  tika together` : 'A LITTLE TRADITION. A LIFETIME OF LOVE.  ·  tika together', W / 2, y + 52);
  }

  async function createSeparate(sisterUrl, brotherUrl) {
    const [sisCanvas, broCanvas] = await Promise.all([loadCanvas(sisterUrl), loadCanvas(brotherUrl)]);
    const warnings = [];
    const people = [];
    const scenes = [];
    for (const [canvas, who, label] of [[sisCanvas, 'sister', 'Sister'], [broCanvas, 'brother', 'Brother']]) {
      const faces = await detectFaces(canvas);
      if (!faces.length) throw new TikaError(`We couldn’t find a face in the ${who}’s photo. Try one where the face is clear and well lit.`);
      if (faces.length > 1) warnings.push(`${label}: we found ${faces.length} faces and used the largest one.`);
      const face = faces[0];
      const mask = await bodyMask(canvas);
      paintTika(canvas.getContext('2d'), face, who === 'sister' ? 11 : 23);
      warnings.push(...faceWarnings(face, label));
      people.push({ face });
      scenes.push(makeScene(who, canvas, mask, [face], who === 'sister' ? 7 : 19));
    }

    await document.fonts.ready;
    // Faces are found and painted once; render() only redraws the card, so changing
    // बहिनी/दिदी or भाइ/दाइ afterwards is instant.
    function render(labels = DEFAULT_LABELS) {
      const sis = LABELS[labels.sister] || LABELS.bahini, bro = LABELS[labels.brother] || LABELS.bhai;
      const out = document.createElement('canvas');
      out.width = 1600; out.height = 1200;
      const ctx = out.getContext('2d');
      ctx.fillStyle = PAPER;
      ctx.fillRect(0, 0, out.width, out.height);
      ctx.strokeStyle = '#dfcaa9';
      ctx.beginPath(); ctx.arc(800, 520, 430, 0, Math.PI * 2); ctx.stroke();
      drawPolaroid(ctx, composeScene(scenes[0]), people[0].face, { cx: 520, cy: 500, w: 520, angle: -0.07, label: sis.word, sub: sis.sub });
      drawPolaroid(ctx, composeScene(scenes[1]), people[1].face, { cx: 1080, cy: 520, w: 520, angle: 0.07, label: bro.word, sub: bro.sub });
      drawFooter(ctx, out.width, 1070);
      return out;
    }
    return { canvas: render(), render, warnings, labelled: true, scenes };
  }

  async function createTogether(url) {
    const canvas = await loadCanvas(url);
    const all = await detectFaces(canvas);
    if (!all.length) throw new TikaError('We couldn’t find any faces in this photo. Try one where both faces are clear and well lit.');
    const faces = all.filter(f => f.area >= all[0].area * MIN_RELATIVE_FACE_AREA).sort((a, b) => a.bbox.x - b.bbox.x);
    if (faces.length < 2) throw new TikaError('We could only find one clear face in this photo. Try a photo where you’re both facing the camera, or switch to separate photos.');
    const warnings = [];
    if (all.length > faces.length) warnings.push(`We skipped ${all.length - faces.length} smaller face${all.length - faces.length > 1 ? 's' : ''} in the background.`);
    if (faces.length > 2) warnings.push(`We found ${faces.length} siblings and gave each of them a tika.`);
    const ctx = canvas.getContext('2d');
    keepMalasApart(faces);
    const scene = makeScene('together', canvas, await bodyMask(canvas), faces, 53);
    faces.forEach((face, i) => {
      paintTika(ctx, face, 101 + i * 17);
      warnings.push(...faceWarnings(face, `Person ${i + 1} from the left`));
    });

    await document.fonts.ready;
    // With exactly two siblings the footer names them (e.g. "दिदी र भाइ"); we don't guess who is who in the photo.
    function render(labels = DEFAULT_LABELS) {
      const sis = LABELS[labels.sister] || LABELS.bahini, bro = LABELS[labels.brother] || LABELS.bhai;
      const pair = faces.length === 2 ? `${sis.word} र ${bro.word}` : null;
      const margin = Math.round(Math.max(canvas.width, canvas.height) * 0.035);
      const scale = Math.max(canvas.width, canvas.height) / 1600;
      const footer = Math.round(170 * scale);
      const out = document.createElement('canvas');
      out.width = canvas.width + margin * 2;
      out.height = canvas.height + margin + footer;
      const o = out.getContext('2d');
      o.fillStyle = '#fffdf8';
      o.fillRect(0, 0, out.width, out.height);
      o.drawImage(composeScene(scene), margin, margin);
      o.scale(scale, scale);
      drawFooter(o, out.width / scale, (canvas.height + margin + footer * 0.42) / scale, pair);
      return out;
    }
    return { canvas: render(), render, warnings, labelled: faces.length === 2, scenes: [scene] };
  }

  window.Tika = {
    LABELS, createSeparate, createTogether, preload: () => getModels().catch(() => {}), TikaError,
    // Mala editor
    drawEditor, hitTest, moveEndpoint, moveMala, resizeMala, resetMala,
  };
})();
