// ML-based AI-image detector using a real pretrained Vision Transformer,
// run entirely client-side via transformers.js (onnxruntime-web under the
// hood). This is the piece the pure heuristics in heuristics.js could not
// provide: a model trained on large labeled real/AI-generated datasets, as
// opposed to hand-tuned pixel statistics.
//
// Model: onnx-community/Deep-Fake-Detector-v2-Model-ONNX
//   - Base: google/vit-base-patch16-224 (ViT), fine-tuned for real vs.
//     deepfake/AI-generated classification.
//   - Reported test accuracy: ~92% (Realism precision ~97%, Deepfake
//     precision ~88%) - strong, but not infallible; see README.md.
//   - Labels: "Realism" (real photo) / "Deepfake" (AI-generated or
//     manipulated).
//   - Loaded at 4-bit quantization (~40MB) to keep the one-time download
//     reasonable; the browser caches it after first load (Cache Storage,
//     handled internally by transformers.js).
//
// Because this runs the actual model in-browser via WebAssembly/WebGPU, the
// image never leaves the device - same privacy guarantee as the heuristics.

import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0';

env.allowLocalModels = false;

const MODEL_ID = 'onnx-community/Deep-Fake-Detector-v2-Model-ONNX';

let classifierPromise = null;

export function isLoaded() {
  return classifierPromise !== null;
}

export function loadDetector(onProgress) {
  if (!classifierPromise) {
    classifierPromise = pipeline('image-classification', MODEL_ID, {
      dtype: 'q4',
      progress_callback: onProgress
    }).catch(err => {
      classifierPromise = null; // allow retry on next call
      throw err;
    });
  }
  return classifierPromise;
}

// dataUrl: a data:image/... URL (from canvas.toDataURL), or any URL/blob
// transformers.js's RawImage.read supports.
export async function runDetector(classifier, imageInput) {
  const results = await classifier(imageInput, { top_k: 2 });
  const byLabel = {};
  for (const r of results) byLabel[String(r.label).toLowerCase()] = r.score;

  const deepfakeProb = byLabel['deepfake'] ?? (1 - (byLabel['realism'] ?? 0.5));
  const realismProb = byLabel['realism'] ?? (1 - deepfakeProb);

  return { deepfakeProb, realismProb, raw: results };
}
