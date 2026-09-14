// ML-based AI-image detector using a real pretrained classifier, run
// entirely client-side via onnxruntime-web (WebAssembly). This is the
// piece the pure heuristics in heuristics.js could not provide: a model
// trained on large labeled real/AI-generated datasets, as opposed to
// hand-tuned pixel statistics.
//
// Model: LPX55/detection-model-1-ONNX (ONNX conversion of
// haywoodsloan/ai-image-detector-deploy)
//   - Base: SwinV2 (~0.2B params), trained generically on "artificial" vs
//     "real" images.
//   - Reported validation accuracy: ~98.1% (F1 0.988, AUC 0.995).
//   - Labels (id2label): 0 = "artificial" (AI-generated/manipulated),
//     1 = "real".
//   - Input: "pixel_values", float32 [1,3,256,256], ImageNet
//     normalization (mean [0.485,0.456,0.406], std [0.229,0.224,0.225]).
//   - Output: "logits", float32 [1,2] (apply softmax for probabilities).
//
// IMPLEMENTATION NOTE: this uses onnxruntime-web directly rather than the
// higher-level transformers.js `pipeline()` API, because transformers.js
// (as of v4.2.0) does not have "Swinv2ForImageClassification" registered
// in its architecture map - loading this exact model through
// `pipeline('image-classification', ...)` throws "Unsupported model
// type: swinv2" (confirmed by testing, not assumed). onnxruntime-web has
// no such limitation since it just executes whatever computation graph is
// in the .onnx file - so the model's pre/post-processing is implemented
// by hand here, matching preprocessor_config.json exactly, and verified
// end-to-end against the real hosted model file before shipping.
//
// Switched 2026-09-14 from onnx-community/Deep-Fake-Detector-v2-Model-ONNX
// (a ViT model trained specifically on real-vs-deepfake FACES - a
// face-swap detector, not a general AI-art detector) after user testing
// showed it performing poorly on full AI-generated portraits/scenes from
// modern generators (Midjourney/Flux-style), which is a different task
// than the face-swap deepfakes it was trained on.
//
// Because this runs the actual model in-browser via WebAssembly, the
// image never leaves the device - same privacy guarantee as the
// heuristics.

import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.min.mjs';

ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/';

const MODEL_URL = 'https://huggingface.co/LPX55/detection-model-1-ONNX/resolve/main/onnx/model_q4f16.onnx';
const INPUT_SIZE = 256;
const IMAGE_MEAN = [0.485, 0.456, 0.406];
const IMAGE_STD = [0.229, 0.224, 0.225];

let sessionPromise = null;

export function isLoaded() {
  return sessionPromise !== null;
}

async function fetchWithProgress(url, onProgress) {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`Model download failed: HTTP ${resp.status}`);
  const total = Number(resp.headers.get('content-length')) || 0;
  const reader = resp.body.getReader();
  const chunks = [];
  let received = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    if (onProgress) onProgress({ status: 'progress', progress: total ? (received / total) * 100 : 0 });
  }
  const buf = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) { buf.set(chunk, offset); offset += chunk.length; }
  return buf.buffer;
}

export function loadDetector(onProgress) {
  if (!sessionPromise) {
    sessionPromise = (async () => {
      const buf = await fetchWithProgress(MODEL_URL, onProgress);
      return await ort.InferenceSession.create(buf, { executionProviders: ['wasm'] });
    })().catch(err => {
      sessionPromise = null; // allow retry on next call
      throw err;
    });
  }
  return sessionPromise;
}

function preprocess(imgEl) {
  const canvas = document.createElement('canvas');
  canvas.width = INPUT_SIZE; canvas.height = INPUT_SIZE;
  const ctx = canvas.getContext('2d');
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(imgEl, 0, 0, INPUT_SIZE, INPUT_SIZE);
  const data = ctx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE).data;

  const chw = new Float32Array(3 * INPUT_SIZE * INPUT_SIZE);
  const plane = INPUT_SIZE * INPUT_SIZE;
  for (let y = 0; y < INPUT_SIZE; y++) {
    for (let x = 0; x < INPUT_SIZE; x++) {
      const i = (y * INPUT_SIZE + x) * 4;
      const p = y * INPUT_SIZE + x;
      chw[0 * plane + p] = (data[i] / 255 - IMAGE_MEAN[0]) / IMAGE_STD[0];
      chw[1 * plane + p] = (data[i + 1] / 255 - IMAGE_MEAN[1]) / IMAGE_STD[1];
      chw[2 * plane + p] = (data[i + 2] / 255 - IMAGE_MEAN[2]) / IMAGE_STD[2];
    }
  }
  return new ort.Tensor('float32', chw, [1, 3, INPUT_SIZE, INPUT_SIZE]);
}

function softmax(logits) {
  const max = Math.max(...logits);
  const exps = logits.map(l => Math.exp(l - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(e => e / sum);
}

// imgEl: an HTMLImageElement (already loaded).
export async function runDetector(session, imgEl) {
  const tensor = preprocess(imgEl);
  const results = await session.run({ pixel_values: tensor });
  const logits = Array.from(results.logits.data);
  const [deepfakeProb, realismProb] = softmax(logits); // id2label: 0=artificial, 1=real
  return { deepfakeProb, realismProb };
}
