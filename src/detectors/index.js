import ort from "onnxruntime-node";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { preprocess, preprocessSsd } from "../lib/preprocess.js";
import { postprocess, postprocessSsd } from "../lib/postprocess.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
// Models live in <repo>/models by default; the bundled Electron app overrides
// this via the MODELS_DIR env var to point at its packaged resources.
const MODELS_DIR = process.env.MODELS_DIR || path.join(__dirname, "..", "..", "models");

// Model registry — id -> file + output family. SSD models are added in a later phase.
export const MODELS = {
  yolov8: { id: "yolov8", name: "YOLOv8x", family: "yolov8", file: "yolov8x.onnx", inputSize: 640 },
  yolov5: { id: "yolov5", name: "YOLOv5l", family: "yolov5", file: "yolov5l.onnx", inputSize: 640 },
  "coco-ssd": { id: "coco-ssd", name: "COCO-SSD", family: "ssd", file: "coco_ssd.onnx", inputSize: 300 },
  "mobilenet-ssd": { id: "mobilenet-ssd", name: "MobileNet SSD", family: "ssd", file: "mobilenet_ssd.onnx", inputSize: 300 },
};

// onnxruntime sessions are cached and not re-entrant — serialize runs per session.
const sessions = new Map(); // id -> { session, lock }

async function getSession(modelId) {
  if (sessions.has(modelId)) return sessions.get(modelId);
  const cfg = MODELS[modelId];
  if (!cfg) throw new Error(`Unknown model: ${modelId}`);

  const modelPath = path.join(MODELS_DIR, cfg.file);
  const session = await ort.InferenceSession.create(modelPath, {
    executionProviders: ["cpu"], // GPU (CUDA) provider added at deploy time
    graphOptimizationLevel: "all",
  });
  const entry = { session, lock: Promise.resolve() };
  sessions.set(modelId, entry);
  console.log(`✅ Loaded ${cfg.name} (${cfg.file})`);
  return entry;
}

function runExclusive(entry, feeds) {
  const result = entry.lock.then(() => entry.session.run(feeds));
  entry.lock = result.then(() => {}, () => {});
  return result;
}

/**
 * Detect objects in an image buffer.
 * @returns {{detections:Array, inferenceMs:number, imageSize:{width,height}}}
 */
export async function detect(modelId, buffer, options = {}) {
  const cfg = MODELS[modelId];
  if (!cfg) throw new Error(`Unknown model: ${modelId}`);

  const entry = await getSession(modelId);
  const inputName = entry.session.inputNames[0];

  let preprocessMs, inferenceMs, postprocessMs, detections, meta;

  if (cfg.family === "ssd") {
    const tPre = performance.now();
    const pre = await preprocessSsd(buffer);
    preprocessMs = performance.now() - tPre;
    meta = pre.meta;

    const tensor = new ort.Tensor("uint8", pre.data, pre.dims);
    const tInf = performance.now();
    const raw = await runExclusive(entry, { [inputName]: tensor });
    inferenceMs = performance.now() - tInf;

    // Normalize output names — tf2onnx-converted models suffix them with ":0".
    const outputs = {};
    for (const k of Object.keys(raw)) outputs[k.replace(/:0$/, "")] = raw[k];

    const tPost = performance.now();
    detections = postprocessSsd(outputs, meta, options); // NMS is in the graph
    postprocessMs = performance.now() - tPost;
  } else {
    const tPre = performance.now();
    const pre = await preprocess(buffer);
    preprocessMs = performance.now() - tPre;
    meta = pre.meta;

    const tensor = new ort.Tensor("float32", pre.data, [1, 3, meta.inputSize, meta.inputSize]);
    const tInf = performance.now();
    const outputs = await runExclusive(entry, { [inputName]: tensor });
    inferenceMs = performance.now() - tInf;

    const tPost = performance.now();
    detections = postprocess(cfg.family, outputs[entry.session.outputNames[0]], meta, options);
    postprocessMs = performance.now() - tPost;
  }

  return {
    detections,
    inferenceMs, // kept for backward compatibility
    timings: { preprocessMs, inferenceMs, postprocessMs },
    inputSize: cfg.inputSize,
    imageSize: { width: meta.origW, height: meta.origH },
  };
}
