# my-thesis-server

Inference backend for the Object Detection Platform — Node + Express + `onnxruntime-node`.
Runs YOLO models natively (CPU locally, CUDA GPU on the deployment server) and serves
detections to the [`my-thesis-client`](https://github.com/SarwarMorshad/my-thesis-client) UI.

## Requirements
- Node 18+
- Model files in `models/` (not in git — too large):
  - `models/yolov8x.onnx`
  - `models/yolov5l.onnx`

## Setup
```bash
npm install
# place the .onnx files in models/  (copy from the client export, or download)
npm run dev        # http://localhost:3001  (auto-restart)
# or: npm start
```

## API
| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | `{ status, gpu, models }` |
| GET | `/models` | available models |
| POST | `/detect` | detect objects in one image |

**POST /detect** — two ways to send the image:
- **Binary:** body = raw image bytes, `Content-Type: image/jpeg`; options as query params
  (`?modelId=yolov8&confidenceThreshold=0.25&nmsThreshold=0.55&classFilter=all`).
- **JSON:** `{ "image": "data:image/jpeg;base64,...", "modelId": "yolov8", "confidenceThreshold": 0.25, "nmsThreshold": 0.55, "classFilter": "all" }`

**Response:**
```json
{
  "detections": [
    { "classId": 0, "className": "person", "confidence": 0.93, "bbox": [x, y, w, h] }
  ],
  "inferenceMs": 312.4,
  "imageSize": { "width": 1080, "height": 810 }
}
```
`bbox` is `[x, y, width, height]` in **original image pixels**.

## Roadmap
1. ✅ Image detection (YOLOv8x, YOLOv5l) — CPU
2. WebSocket frame streaming (real-time video)
3. SSD models (ONNX)
4. GPU (CUDA) execution provider + Docker
5. Deploy to the university GPU server
