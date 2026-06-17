import express from "express";
import cors from "cors";
import { MODELS, detect } from "./detectors/index.js";

const PORT = process.env.PORT || 3001;

const app = express();
app.use(cors());
app.use(express.json({ limit: "25mb" })); // JSON with base64 image
app.use(express.raw({ type: ["image/*", "application/octet-stream"], limit: "25mb" })); // binary image

// Liveness + capabilities
app.get("/health", (req, res) => {
  res.json({ status: "ok", gpu: false, models: Object.keys(MODELS) });
});

// Available models + display metadata
app.get("/models", (req, res) => {
  res.json(Object.values(MODELS).map(({ id, name, family }) => ({ id, name, family })));
});

// Single-image detection
app.post("/detect", async (req, res) => {
  try {
    let buffer;
    let opts = {};
    let modelId;

    if (Buffer.isBuffer(req.body)) {
      // binary image; options come from query string
      buffer = req.body;
      modelId = req.query.modelId;
      opts = {
        confidenceThreshold: parseFloat(req.query.confidenceThreshold) || 0.25,
        nmsThreshold: parseFloat(req.query.nmsThreshold) || 0.55,
        classFilter: req.query.classFilter || "all",
      };
    } else {
      // JSON body with a base64 (data URL) image
      const { image, modelId: id, confidenceThreshold, nmsThreshold, classFilter } = req.body || {};
      modelId = id;
      const b64 = (image || "").replace(/^data:image\/\w+;base64,/, "");
      buffer = Buffer.from(b64, "base64");
      opts = {
        confidenceThreshold: confidenceThreshold ?? 0.25,
        nmsThreshold: nmsThreshold ?? 0.55,
        classFilter: classFilter ?? "all",
      };
    }

    if (!modelId || !MODELS[modelId]) {
      return res.status(400).json({ error: `Unknown or missing modelId. Available: ${Object.keys(MODELS).join(", ")}` });
    }
    if (!buffer || buffer.length === 0) {
      return res.status(400).json({ error: "No image data received" });
    }

    const result = await detect(modelId, buffer, opts);
    res.json(result);
  } catch (err) {
    console.error("❌ /detect error:", err);
    res.status(500).json({ error: err.message });
  }
});

app.listen(PORT, () => {
  console.log(`🚀 Inference server listening on http://localhost:${PORT}`);
  console.log(`   Models: ${Object.keys(MODELS).join(", ")}`);
});
