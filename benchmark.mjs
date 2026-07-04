// benchmark.mjs
// Runs all four models over a folder of images via the backend /detect endpoint
// and prints averaged metrics per model (for the thesis Results table).
//
// Usage:
//   1) Start the backend:   npm run dev   (models present in models/)
//   2) Put your test images in a folder, e.g. ./test-images
//   3) Run:                 node benchmark.mjs test-images
//
// Optional: BACKEND=http://host:port node benchmark.mjs <folder>

import fs from "node:fs";
import path from "node:path";

const BACKEND = process.env.BACKEND || "http://localhost:3001";
const IMAGES_DIR = process.argv[2] || "test-images";

const MODELS = ["yolov8", "yolov5", "coco-ssd", "mobilenet-ssd"];
const NAMES = {
  yolov8: "YOLOv8x",
  yolov5: "YOLOv5l",
  "coco-ssd": "COCO-SSD",
  "mobilenet-ssd": "MobileNet SSD",
};

const IMG_EXT = new Set([".jpg", ".jpeg", ".png", ".bmp", ".webp"]);

const files = fs
  .readdirSync(IMAGES_DIR)
  .filter((f) => IMG_EXT.has(path.extname(f).toLowerCase()))
  .map((f) => path.join(IMAGES_DIR, f));

if (files.length === 0) {
  console.error(`No images found in "${IMAGES_DIR}".`);
  process.exit(1);
}

console.log(`Benchmarking ${files.length} images against ${BACKEND}\n`);

async function detect(modelId, buffer) {
  const res = await fetch(`${BACKEND}/detect?modelId=${modelId}`, {
    method: "POST",
    headers: { "Content-Type": "application/octet-stream" },
    body: buffer,
  });
  if (!res.ok) throw new Error(`${modelId}: HTTP ${res.status} ${await res.text()}`);
  return res.json();
}

const rows = [];

for (const modelId of MODELS) {
  let n = 0, sumDet = 0, sumConf = 0, sumTotal = 0, sumInf = 0, sumClasses = 0;

  // Warm-up run (loads the ONNX session; excluded from the averages)
  try { await detect(modelId, fs.readFileSync(files[0])); } catch { /* ignore */ }

  for (const file of files) {
    let r;
    try {
      r = await detect(modelId, fs.readFileSync(file));
    } catch (e) {
      console.error("  " + e.message);
      continue;
    }
    const dets = r.detections || [];
    const t = r.timings || {};
    const inf = t.inferenceMs ?? r.inferenceMs ?? 0;
    const total = (t.preprocessMs || 0) + inf + (t.postprocessMs || 0);
    const conf = dets.length
      ? dets.reduce((s, d) => s + (d.confidence || 0), 0) / dets.length
      : 0;
    const classes = new Set(dets.map((d) => d.className)).size;

    n++; sumDet += dets.length; sumConf += conf;
    sumTotal += total; sumInf += inf; sumClasses += classes;
  }

  if (n === 0) { console.log(`${NAMES[modelId]}: no successful runs`); continue; }

  const avgTotal = sumTotal / n;
  rows.push({
    name: NAMES[modelId],
    det: (sumDet / n).toFixed(1),
    conf: (100 * sumConf / n).toFixed(1),
    total: avgTotal.toFixed(0),
    infer: (sumInf / n).toFixed(0),
    fps: (1000 / avgTotal).toFixed(2),
    classes: (sumClasses / n).toFixed(1),
  });
}

// Print a readable summary
console.log("\nAveraged over", files.length, "images (CPU):\n");
console.log(
  "Model".padEnd(15) + "Det".padStart(7) + "Conf%".padStart(9) +
  "Latency".padStart(11) + "Infer".padStart(9) + "FPS".padStart(8) + "Classes".padStart(9)
);
for (const r of rows) {
  console.log(
    r.name.padEnd(15) + r.det.padStart(7) + r.conf.padStart(9) +
    `${r.total} ms`.padStart(11) + `${r.infer} ms`.padStart(9) +
    r.fps.padStart(8) + r.classes.padStart(9)
  );
}

// Print LaTeX-ready rows for the thesis table
console.log("\nLaTeX rows for tab:quant_results:\n");
for (const r of rows) {
  console.log(`${r.name} & ${r.det} & ${r.conf}\\% & $\\sim$${r.total} ms & ${r.fps} & ${Math.round(r.classes)} \\\\`);
}
