# Aperture

AI image forensics that runs entirely in your browser — a real pretrained
ML classifier plus transparent metadata/pixel heuristics, with no server
and no image ever leaving the device.

## What changed from the Claude Artifact version

The original Aperture (a Claude Artifact) was heuristics-only: file
metadata, resolution/chroma fingerprints, error-level analysis, color
profile, and frequency-domain checks. Real-world testing showed those
heuristics hit a hard ceiling once a file's metadata is stripped (which
happens to almost every image downloaded or shared through a chat/social
app) and against modern photoreal AI generators, which are specifically
trained to avoid the visual tells the heuristics were built to catch.

This version adds a **real pretrained SwinV2 classifier** (`LPX55/detection-model-1-ONNX`,
an ONNX conversion of `haywoodsloan/ai-image-detector-deploy`, ~98%
reported validation accuracy) that runs client-side via
[onnxruntime-web](https://github.com/microsoft/onnxruntime) (WebAssembly).
Claude Artifacts cap total page size at 16MB and block fetching external
binary files at runtime — a model this size (~175MB even at its smallest
quantization) can't live there, so this had to move to a normal static
website instead. The heuristics are kept as a secondary, fully transparent
cross-check alongside the model's verdict.

**Model choice, revised 2026-09-14**: the first version of this used
`onnx-community/Deep-Fake-Detector-v2-Model-ONNX`, a smaller (~40MB) ViT
model — but that model is trained specifically on real-vs-deepfake
**faces** (a face-swap detector), not general AI-generated images. Testing
against real Midjourney/Flux-style AI portraits showed it performing
poorly, since that's a different task than what it was trained for.
`LPX55/detection-model-1-ONNX` uses generic "artificial"/"real" labels and
was validated more broadly, at the cost of a much bigger download (~175MB
vs ~40MB).

**Implementation note**: `detector.js` calls onnxruntime-web directly
rather than going through the higher-level `@huggingface/transformers`
`pipeline()` API, because that library doesn't have this model's SwinV2
architecture registered (confirmed by testing - it throws "Unsupported
model type: swinv2"). Preprocessing (resize to 256×256, ImageNet
normalization) is implemented by hand in `detector.js` to exactly match
the model's `preprocessor_config.json`, verified end-to-end against the
real hosted model file before shipping.

## Running it locally

Because the app uses ES modules (`<script type="module">`) and loads the
model over `fetch`, it must be served over `http://`, not opened directly
as a `file://` URL. If you don't have Node or Python installed, this repo
includes a zero-install option that uses only what Windows already has:

```powershell
powershell -File serve.ps1
```

Then open `http://localhost:8792/index.html`. (If you do have Node or
Python: `npx serve .` or `python -m http.server 8000` work too.)

## Deploying it for real

This is a plain static site (`index.html`, `style.css`, `app.js`,
`detector.js`, `heuristics.js`) — no build step, no backend. Any of these
work and are free:

- **GitHub Pages**: push this folder to a GitHub repo, then in the repo's
  Settings → Pages, set the source to your branch/`root`. Your site will be
  live at `https://<username>.github.io/<repo>/`.
- **Netlify**: drag-and-drop this folder onto [app.netlify.com/drop](https://app.netlify.com/drop).
- **Vercel**: `vercel deploy` from inside this folder (via the Vercel CLI),
  or connect the GitHub repo in the Vercel dashboard.

No environment variables or server configuration are needed.

## How the score is put together

- **ML model** (75% weight in the combined score): the artificial/real
  probability from the SwinV2 classifier. This is the strongest signal —
  trained on labeled real/AI-generated image pairs, not hand-tuned pixel
  statistics.
- **Heuristics** (25% weight, shown in full in the breakdown): metadata &
  AI-signature scan, resolution/chroma-subsampling fingerprints, color/
  saturation profile, frequency-domain (FFT) artifact scan, error-level
  analysis, and texture/noise uniformity. Each heuristic result carries its
  own confidence label (low/medium/high) based on how much real structural
  evidence (not just weak pixel signals) was found.
- If the model and the heuristics disagree by a wide margin, the report
  flags that explicitly rather than silently picking one side.
- If the model fails to load (offline, blocked download), the app falls
  back to heuristics-only and says so.

## Known limitations

- ~98% reported validation accuracy is measured on the model's own
  validation set, not a guarantee for every image or every generator —
  real-world images (unusual crops, heavy compression, novel generators
  released after the model's training cutoff) can do worse. Treat every
  result as evidence, not proof.
- The first analysis in a fresh browser profile downloads the ~175MB
  model; subsequent visits reuse the browser's cache. This is a
  meaningfully bigger one-time download than a typical web page — a
  deliberate trade for higher accuracy on general AI-generated images
  (see "Model choice" above).
- Heuristic checks alone (when the model is unavailable) inherit the
  known ceiling described above — most informative when the file still
  carries its original metadata.
