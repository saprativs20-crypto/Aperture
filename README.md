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

This version adds a **real pretrained Vision Transformer** (`onnx-community/Deep-Fake-Detector-v2-Model-ONNX`,
~92% reported test accuracy) that runs client-side via
[transformers.js](https://huggingface.co/docs/transformers.js) /
onnxruntime-web + WebAssembly. That model needed ~40MB even at 4-bit
quantization, and Claude Artifacts cap total page size at 16MB and block
fetching external binary files at runtime — so this had to move to a
normal static website instead. The heuristics are kept as a secondary,
fully transparent cross-check alongside the model's verdict.

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

- **ML model** (75% weight in the combined score): the deepfake/real
  probability from the Vision Transformer classifier. This is the strongest
  signal — trained on labeled real/AI-generated image pairs, not hand-tuned
  pixel statistics.
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

- ~92% test accuracy means the model is wrong roughly 1 in 12 times on its
  own test distribution — real-world images (unusual crops, heavy
  compression, novel generators released after the model's training cutoff)
  can do worse. Treat every result as evidence, not proof.
- The first analysis in a fresh browser profile downloads the ~40MB model;
  subsequent visits reuse the browser's cache.
- Heuristic checks alone (when the model is unavailable) inherit the
  known ceiling described above — most informative when the file still
  carries its original metadata.
