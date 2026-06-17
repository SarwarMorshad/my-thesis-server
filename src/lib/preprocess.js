import sharp from "sharp";

const INPUT_SIZE = 640;
const PAD = 114; // YOLO letterbox padding colour

/**
 * Letterbox an image into a 640x640 CHW float32 tensor (normalized 0..1).
 * Returns the tensor data plus the geometry meta needed to map boxes back.
 */
export async function preprocess(buffer, inputSize = INPUT_SIZE) {
  const meta0 = await sharp(buffer).metadata();
  const origW = meta0.width;
  const origH = meta0.height;

  const scale = Math.min(inputSize / origW, inputSize / origH);
  const newW = Math.round(origW * scale);
  const newH = Math.round(origH * scale);
  const padX = Math.floor((inputSize - newW) / 2);
  const padY = Math.floor((inputSize - newH) / 2);

  // Resize preserving aspect, then pad to a square with the YOLO grey.
  const raw = await sharp(buffer)
    .resize(newW, newH, { fit: "fill" })
    .removeAlpha()
    .extend({
      top: padY,
      bottom: inputSize - newH - padY,
      left: padX,
      right: inputSize - newW - padX,
      background: { r: PAD, g: PAD, b: PAD },
    })
    .raw()
    .toBuffer(); // RGB, length = inputSize*inputSize*3

  // HWC uint8 -> CHW float32 normalized
  const hw = inputSize * inputSize;
  const data = new Float32Array(3 * hw);
  for (let i = 0; i < hw; i++) {
    data[i] = raw[i * 3] / 255; // R
    data[hw + i] = raw[i * 3 + 1] / 255; // G
    data[2 * hw + i] = raw[i * 3 + 2] / 255; // B
  }

  return { data, meta: { origW, origH, inputSize, scale, padX, padY } };
}
