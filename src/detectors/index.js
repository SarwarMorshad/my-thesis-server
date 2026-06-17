import ort from "onnxruntime-node";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { preprocess } from "../lib/preprocess.js";
import { postprocess } from "../lib/postprocess.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const MODELS_DIR = path.join(__dirname, "..", "..", "models");

// Model registry — id -> file + output family. SSD models are added in a later phase.
export const MODELS = {
  yolov8: { id: "yolov8", name: "YOLOv8x", family: "yolov8", file: "yolov8x.onnx" },
  yolov5: { id: "yolov5", name: "YOLOv5l", family: "yolov5", file: "yolov5l.onnx" },
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
  const { data, meta } = await preprocess(buffer);

  const tensor = new ort.Tensor("float32", data, [1, 3, meta.inputSize, meta.inputSize]);
  const inputName = entry.session.inputNames[0];

  const t0 = performance.now();
  const outputs = await runExclusive(entry, { [inputName]: tensor });
  const inferenceMs = performance.now() - t0;

  const output = outputs[entry.session.outputNames[0]];
  const detections = postprocess(cfg.family, output, meta, options);

  return { detections, inferenceMs, imageSize: { width: meta.origW, height: meta.origH } };
}
