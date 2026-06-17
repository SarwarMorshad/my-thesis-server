import { COCO_CLASSES, CLASS_GROUPS } from "../constants/coco.js";

const NUM_CLASSES = 80;

// IoU of two [x,y,w,h] boxes
function iou(a, b) {
  const x1 = Math.max(a[0], b[0]);
  const y1 = Math.max(a[1], b[1]);
  const x2 = Math.min(a[0] + a[2], b[0] + b[2]);
  const y2 = Math.min(a[1] + a[3], b[1] + b[3]);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const union = a[2] * a[3] + b[2] * b[3] - inter;
  return union <= 0 ? 0 : inter / union;
}

// Class-aware non-maximum suppression
function nms(boxes, nmsThreshold) {
  boxes.sort((p, q) => q.confidence - p.confidence);
  const kept = [];
  const dead = new Set();
  for (let i = 0; i < boxes.length; i++) {
    if (dead.has(i)) continue;
    kept.push(boxes[i]);
    for (let j = i + 1; j < boxes.length; j++) {
      if (dead.has(j)) continue;
      if (boxes[i].classId !== boxes[j].classId) continue;
      if (iou(boxes[i].bbox, boxes[j].bbox) > nmsThreshold) dead.add(j);
    }
  }
  return kept;
}

function applyClassFilter(boxes, classFilter) {
  const group = CLASS_GROUPS[classFilter];
  if (!group) return boxes; // "all"
  const allowed = new Set(group);
  return boxes.filter((b) => allowed.has(b.className));
}

// Map a box from letterboxed input space back to original pixels
function unletterbox(xC, yC, w, h, meta) {
  const { origW, origH, scale, padX, padY } = meta;
  let x = (xC - w / 2 - padX) / scale;
  let y = (yC - h / 2 - padY) / scale;
  const ww = w / scale;
  const hh = h / scale;
  x = Math.max(0, Math.min(origW - 1, x));
  y = Math.max(0, Math.min(origH - 1, y));
  return [x, y, Math.max(0, Math.min(origW - x, ww)), Math.max(0, Math.min(origH - y, hh))];
}

/** Decode YOLOv8 output [1,84,N] (or [1,N,84]) — no objectness, 80 class scores. */
export function decodeYolov8(output, meta, confidenceThreshold) {
  const data = output.data;
  const dims = output.dims;
  const C = 84;
  let layout, N;
  if (dims.length === 3 && dims[1] === C) {
    layout = "CHW";
    N = dims[2];
  } else if (dims.length === 3 && dims[2] === C) {
    layout = "HWC";
    N = dims[1];
  } else {
    return [];
  }

  const boxes = [];
  for (let i = 0; i < N; i++) {
    let xC, yC, w, h, maxScore = 0, maxClass = -1;
    if (layout === "CHW") {
      xC = data[i]; yC = data[N + i]; w = data[2 * N + i]; h = data[3 * N + i];
      for (let c = 0; c < NUM_CLASSES; c++) {
        const s = data[(4 + c) * N + i];
        if (s > maxScore) { maxScore = s; maxClass = c; }
      }
    } else {
      const base = i * C;
      xC = data[base]; yC = data[base + 1]; w = data[base + 2]; h = data[base + 3];
      for (let c = 0; c < NUM_CLASSES; c++) {
        const s = data[base + 4 + c];
        if (s > maxScore) { maxScore = s; maxClass = c; }
      }
    }
    if (maxScore < confidenceThreshold) continue;
    boxes.push({
      classId: maxClass,
      className: COCO_CLASSES[maxClass] ?? String(maxClass),
      confidence: maxScore,
      bbox: unletterbox(xC, yC, w, h, meta),
    });
  }
  return boxes;
}

/** Decode classic YOLOv5 output [1,25200,85] — objectness + 80 class scores. */
export function decodeYolov5(output, meta, confidenceThreshold) {
  const data = output.data;
  const rows = output.dims[1];
  const stride = 85;
  const boxes = [];
  for (let i = 0; i < rows; i++) {
    const off = i * stride;
    const obj = data[off + 4];
    if (obj < confidenceThreshold) continue;
    let maxScore = 0, maxClass = -1;
    for (let c = 0; c < NUM_CLASSES; c++) {
      const s = data[off + 5 + c];
      if (s > maxScore) { maxScore = s; maxClass = c; }
    }
    const confidence = obj * maxScore;
    if (confidence < confidenceThreshold) continue;
    boxes.push({
      classId: maxClass,
      className: COCO_CLASSES[maxClass] ?? String(maxClass),
      confidence,
      bbox: unletterbox(data[off], data[off + 1], data[off + 2], data[off + 3], meta),
    });
  }
  return boxes;
}

/** Full postprocess: decode -> NMS -> class filter. */
export function postprocess(family, output, meta, { confidenceThreshold = 0.25, nmsThreshold = 0.55, classFilter = "all" } = {}) {
  const raw = family === "yolov5" ? decodeYolov5(output, meta, confidenceThreshold) : decodeYolov8(output, meta, confidenceThreshold);
  return applyClassFilter(nms(raw, nmsThreshold), classFilter);
}
