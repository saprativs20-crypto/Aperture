// Aperture heuristic forensics engine — metadata/structural, color, ELA,
// texture, and frequency-domain checks. Ported from the original Aperture
// Claude Artifact. Kept as a transparent, explainable secondary signal
// alongside the ML model in detector.js, since these heuristics alone have a
// known ceiling against modern photoreal AI once file metadata is stripped
// (see README.md).

function clamp(v, a, b) { return Math.max(a, Math.min(b, v)); }
function lerpScore(x, x0, x1, y0, y1) {
  if (x1 === x0) return y0;
  const t = clamp((x - x0) / (x1 - x0), 0, 1);
  return y0 + t * (y1 - y0);
}

function loadImageEl(url) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = url;
  });
}

function drawToCanvas(img, maxDim) {
  let w = img.naturalWidth, h = img.naturalHeight;
  if (Math.max(w, h) > maxDim) {
    const s = maxDim / Math.max(w, h);
    w = Math.round(w * s); h = Math.round(h * s);
  }
  const canvas = document.createElement('canvas');
  canvas.width = w; canvas.height = h;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(img, 0, 0, w, h);
  return { canvas, ctx, w, h };
}

// ---- Metadata, AI-signature & file-structure scan (byte-level) ----
const AI_SIGNATURES = [
  'stable diffusion', 'stable-diffusion', 'stablediffusionpipeline', 'comfyui',
  'automatic1111', 'invokeai', 'fooocus', 'sdxl', 'flux.1', 'flux schnell', 'midjourney',
  'dall-e', 'dall·e', 'dalle', 'novelai', 'leonardo.ai', 'leonardo ai', 'runwayml',
  'runway ml', 'adobe firefly', 'firefly generative', 'bing image creator', 'imagen',
  'playground ai', 'ideogram', 'nightcafe', 'artbreeder', 'wombo', 'starryai',
  'dreamstudio', 'krea', 'civitai', 'grok imagine', 'cfg scale', 'sampler:',
  'model hash:', 'negative_prompt', 'denoising strength', 'clip skip', 'hires fix',
  'lora:', '"prompt":', 'trainedalgorithmicmedia', 'compositewithtrainedalgorithmicmedia',
  'digitalsourcetype', 'ai_generated', 'ai-generated', 'generative fill', 'generativefill',
  'neural filters', 'c2pa', 'jumbf', 'urn:c2pa', 'content credentials'
];
const EDIT_SOFTWARE = [
  'adobe photoshop', 'photoshop', 'gimp', 'lightroom', 'affinity photo',
  'pixelmator', 'capture one', 'luminar', 'snapseed', 'picsart'
];
const CAMERA_MAKERS = [
  'canon', 'nikon', 'sony', 'fujifilm', 'apple', 'samsung', 'xiaomi', 'redmi', 'huawei',
  'oppo', 'vivo', 'oneplus', 'google', 'pixel', 'leica', 'panasonic', 'lumix', 'olympus',
  'pentax', 'ricoh', 'hasselblad', 'dji', 'gopro', 'motorola', 'nokia'
];

function detectFormat(bytes) {
  if (bytes[0] === 0xFF && bytes[1] === 0xD8 && bytes[2] === 0xFF) return 'jpeg';
  if (bytes[0] === 0x89 && bytes[1] === 0x50 && bytes[2] === 0x4E && bytes[3] === 0x47) return 'png';
  if (bytes.length > 12 && bytes[0] === 0x52 && bytes[1] === 0x49 && bytes[2] === 0x46 && bytes[3] === 0x46 &&
      bytes[8] === 0x57 && bytes[9] === 0x45 && bytes[10] === 0x42 && bytes[11] === 0x50) return 'webp';
  if (bytes[0] === 0x47 && bytes[1] === 0x49 && bytes[2] === 0x46) return 'gif';
  return 'unknown';
}

// Real camera JPEGs almost always carry a small embedded thumbnail JPEG
// inside their EXIF block (IFD1) - a second SOI (FF D8 FF) sitting in the
// header region before the actual image scan begins. AI-tool exports,
// screenshots, and re-saved/re-compressed JPEGs essentially never have this.
function detectEmbeddedThumbnail(bytes) {
  if (!(bytes[0] === 0xFF && bytes[1] === 0xD8)) return false;
  const scanLimit = Math.min(bytes.length, 2000000);
  let sosIndex = -1;
  for (let i = 2; i < scanLimit - 1; i++) {
    if (bytes[i] === 0xFF && bytes[i + 1] === 0xDA) { sosIndex = i; break; }
  }
  const searchEnd = sosIndex === -1 ? scanLimit : sosIndex;
  for (let i = 4; i < searchEnd - 2; i++) {
    if (bytes[i] === 0xFF && bytes[i + 1] === 0xD8 && bytes[i + 2] === 0xFF) return true;
  }
  return false;
}

function readU16BE(bytes, i) { return (bytes[i] << 8) | bytes[i + 1]; }
function readU32BE(bytes, i) { return ((bytes[i] << 24) | (bytes[i + 1] << 16) | (bytes[i + 2] << 8) | bytes[i + 3]) >>> 0; }

// Extract ONLY the text-bearing metadata regions of the file - EXIF/XMP/
// comment segments in JPEG, tEXt/iTXt chunks in PNG - never the compressed
// pixel data itself (which is high-entropy and can coincidentally match a
// short marker string).
function extractMetadataText(bytes, format) {
  const dec = new TextDecoder('latin1');
  const parts = [];
  try {
    if (format === 'png') {
      let i = 8;
      while (i + 8 <= bytes.length) {
        const len = readU32BE(bytes, i);
        const type = dec.decode(bytes.subarray(i + 4, i + 8));
        const dataStart = i + 8;
        const dataEnd = dataStart + len;
        if (dataEnd > bytes.length) break;
        if (type === 'tEXt' || type === 'iTXt') parts.push(dec.decode(bytes.subarray(dataStart, dataEnd)));
        if (type === 'IEND') break;
        i = dataEnd + 4;
      }
    } else if (format === 'jpeg') {
      let i = 2;
      while (i + 4 <= bytes.length) {
        if (bytes[i] !== 0xFF) break;
        const marker = bytes[i + 1];
        if (marker === 0xD8 || marker === 0xD9) { i += 2; continue; }
        if (marker >= 0xD0 && marker <= 0xD7) { i += 2; continue; }
        if (marker === 0xDA) break;
        const len = readU16BE(bytes, i + 2);
        const segStart = i + 4, segEnd = i + 2 + len;
        if (segEnd > bytes.length) break;
        if ((marker >= 0xE0 && marker <= 0xEF) || marker === 0xFE) parts.push(dec.decode(bytes.subarray(segStart, segEnd)));
        i = segEnd;
      }
    }
  } catch (e) { /* malformed segment - use whatever was collected */ }
  return parts.join(' ');
}

// Real camera/phone JPEG encoders overwhelmingly use 4:2:0 chroma
// subsampling; many AI export pipelines and generic re-encoders default to
// 4:4:4 or 4:2:2 instead. Parsed straight from the JPEG SOF marker.
function readJpegSubsampling(bytes) {
  let i = 2;
  while (i + 4 <= bytes.length) {
    if (bytes[i] !== 0xFF) break;
    const marker = bytes[i + 1];
    if (marker === 0xD8 || marker === 0xD9) { i += 2; continue; }
    if (marker >= 0xD0 && marker <= 0xD7) { i += 2; continue; }
    const isSOF = marker >= 0xC0 && marker <= 0xCF && marker !== 0xC4 && marker !== 0xC8 && marker !== 0xCC;
    const len = readU16BE(bytes, i + 2);
    const segStart = i + 4, segEnd = i + 2 + len;
    if (segEnd > bytes.length) break;
    if (isSOF && segEnd - segStart >= 6) {
      const numComponents = bytes[segStart + 5];
      if (numComponents >= 3 && segEnd - segStart >= 6 + numComponents * 3) {
        const lumaSampling = bytes[segStart + 6 + 1];
        const lh = lumaSampling >> 4, lv = lumaSampling & 0x0F;
        const cSampling = bytes[segStart + 6 + 3 + 1];
        const ch = cSampling >> 4, cv = cSampling & 0x0F;
        if (lh === 2 && lv === 2 && ch === 1 && cv === 1) return '4:2:0';
        if (lh === 2 && lv === 1 && ch === 1 && cv === 1) return '4:2:2';
        if (lh === 1 && lv === 1 && ch === 1 && cv === 1) return '4:4:4';
        return 'other';
      }
      return null;
    }
    if (marker === 0xDA) break;
    i = segEnd;
  }
  return null;
}

function scanMetadata(bytes) {
  const format = detectFormat(bytes);
  const text = extractMetadataText(bytes, format).toLowerCase();
  const hasAISig = AI_SIGNATURES.some(s => text.includes(s));
  const hasEditSoftware = EDIT_SOFTWARE.some(s => text.includes(s));
  const hasCameraMaker = CAMERA_MAKERS.some(s => text.includes(s));
  const hasEmbeddedThumb = format === 'jpeg' && detectEmbeddedThumbnail(bytes);
  const subsampling = format === 'jpeg' ? readJpegSubsampling(bytes) : null;
  return { hasAISig, hasEditSoftware, hasCameraMaker, format, hasEmbeddedThumb, subsampling };
}

// Diffusion models sample on a small fixed grid of resolutions, all
// multiples of 64px. Real camera sensors produce fixed, odd native
// resolutions dictated by megapixel count, not this grid.
const GENERATOR_RESOLUTION_BUCKETS = [512, 576, 640, 704, 768, 832, 896, 960, 1024,
  1088, 1152, 1216, 1280, 1344, 1408, 1472, 1536, 1600, 1664, 1728, 1792, 1856, 1920, 1984, 2048];
function checkResolutionSignal(width, height) {
  const inBucket = v => GENERATOR_RESOLUTION_BUCKETS.includes(v);
  const bothBucket = inBucket(width) && inBucket(height);
  const cameraLike = (Math.max(width, height) > 2200) && (width % 64 !== 0 || height % 64 !== 0);
  return { bothBucket, cameraLike };
}

// ---- Error Level Analysis ----
async function runELA(canvas, ctx, w, h) {
  const original = ctx.getImageData(0, 0, w, h);
  const jpegURL = canvas.toDataURL('image/jpeg', 0.9);
  const recImg = await loadImageEl(jpegURL);
  const c2 = document.createElement('canvas');
  c2.width = w; c2.height = h;
  const ctx2 = c2.getContext('2d', { willReadFrequently: true });
  ctx2.drawImage(recImg, 0, 0, w, h);
  const recompressed = ctx2.getImageData(0, 0, w, h);

  const AMP = 12;
  const ela = new Uint8ClampedArray(w * h * 4);
  for (let i = 0; i < w * h * 4; i += 4) {
    const dr = Math.abs(original.data[i] - recompressed.data[i]) * AMP;
    const dg = Math.abs(original.data[i + 1] - recompressed.data[i + 1]) * AMP;
    const db = Math.abs(original.data[i + 2] - recompressed.data[i + 2]) * AMP;
    const v = clamp((dr + dg + db) / 3, 0, 255);
    ela[i] = v; ela[i + 1] = v; ela[i + 2] = v; ela[i + 3] = 255;
  }

  const BLOCK = 16;
  const bw = Math.ceil(w / BLOCK), bh = Math.ceil(h / BLOCK);
  const means = new Float32Array(bw * bh);
  for (let by = 0; by < bh; by++) {
    for (let bx = 0; bx < bw; bx++) {
      let sum = 0, count = 0;
      const x0 = bx * BLOCK, y0 = by * BLOCK;
      const x1 = Math.min(x0 + BLOCK, w), y1 = Math.min(y0 + BLOCK, h);
      for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) { sum += ela[(y * w + x) * 4]; count++; }
      means[by * bw + bx] = count ? sum / count : 0;
    }
  }
  let mean = 0;
  for (let i = 0; i < means.length; i++) mean += means[i];
  mean /= means.length || 1;
  let variance = 0;
  for (let i = 0; i < means.length; i++) variance += (means[i] - mean) ** 2;
  variance /= means.length || 1;
  const std = Math.sqrt(variance);
  const cv = std / (mean + 1e-3);

  const hotThreshold = mean + 2 * std;
  let hotBlocks = 0;
  for (let i = 0; i < means.length; i++) if (means[i] > hotThreshold) hotBlocks++;
  const hotspotRatio = means.length ? hotBlocks / means.length : 0;

  const preview = document.createElement('canvas');
  const pw = 320, ph = Math.round(320 * h / w);
  preview.width = pw; preview.height = ph;
  const pctx = preview.getContext('2d');
  const tmp = document.createElement('canvas');
  tmp.width = w; tmp.height = h;
  tmp.getContext('2d').putImageData(new ImageData(ela, w, h), 0, 0);
  pctx.drawImage(tmp, 0, 0, pw, ph);

  return { cv, hotspotRatio, thumb: preview.toDataURL('image/png') };
}

// ---- Texture & noise uniformity ----
function runTextureNoise(ctx, w, h) {
  const data = ctx.getImageData(0, 0, w, h).data;
  const gray = new Float32Array(w * h);
  for (let i = 0, p = 0; i < data.length; i += 4, p++) gray[p] = data[i] * 0.299 + data[i + 1] * 0.587 + data[i + 2] * 0.114;

  const blurred = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let sum = 0, count = 0;
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
        const nx = x + dx, ny = y + dy;
        if (nx >= 0 && nx < w && ny >= 0 && ny < h) { sum += gray[ny * w + nx]; count++; }
      }
      blurred[y * w + x] = sum / count;
    }
  }
  const residual = new Float32Array(w * h);
  for (let i = 0; i < gray.length; i++) residual[i] = gray[i] - blurred[i];

  const BLOCK = 32;
  const bw = Math.ceil(w / BLOCK), bh = Math.ceil(h / BLOCK);
  const blockVar = [];
  for (let by = 0; by < bh; by++) {
    for (let bx = 0; bx < bw; bx++) {
      const x0 = bx * BLOCK, y0 = by * BLOCK, x1 = Math.min(x0 + BLOCK, w), y1 = Math.min(y0 + BLOCK, h);
      let sum = 0, n = 0;
      for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) { sum += residual[y * w + x]; n++; }
      const m = n ? sum / n : 0;
      let ss = 0;
      for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) ss += (residual[y * w + x] - m) ** 2;
      blockVar.push(n ? ss / n : 0);
    }
  }
  let meanVar = 0;
  for (const v of blockVar) meanVar += v;
  meanVar /= blockVar.length || 1;
  let varOfVar = 0;
  for (const v of blockVar) varOfVar += (v - meanVar) ** 2;
  varOfVar /= blockVar.length || 1;
  const uniformity = Math.sqrt(varOfVar) / (meanVar + 1e-3);

  return { meanVar, uniformity };
}

// ---- Color & saturation profile ----
// CAVEAT: built against 2022-2023-era Stable Diffusion output which really
// was more saturated/"designed-looking" than ordinary snapshots. It breaks
// down against modern photoreal generators (trained to produce muted,
// naturalistic tones) and against real professionally color-graded
// photography (often MORE saturated than a deliberately-muted AI image).
// Kept as a modest-weight signal, not a strong one - see README.md.
function colorStats(ctx, w, h) {
  const data = ctx.getImageData(0, 0, w, h).data;
  let satSum = 0, highSatCount = 0, n = 0;
  let rgSum = 0, ybSum = 0, rgSqSum = 0, ybSqSum = 0;
  for (let i = 0; i < data.length; i += 4) {
    const r = data[i], g = data[i + 1], b = data[i + 2];
    const mx = Math.max(r, g, b), mn = Math.min(r, g, b);
    const sat = mx > 0 ? (mx - mn) / mx : 0;
    satSum += sat;
    if (sat > 0.5) highSatCount++;
    const rg = r - g, yb = 0.5 * (r + g) - b;
    rgSum += rg; ybSum += yb;
    rgSqSum += rg * rg; ybSqSum += yb * yb;
    n++;
  }
  const satMean = satSum / (n || 1);
  const highSatFrac = highSatCount / (n || 1);
  const rgMean = rgSum / n, ybMean = ybSum / n;
  const rgStd = Math.sqrt(Math.max(0, rgSqSum / n - rgMean * rgMean));
  const ybStd = Math.sqrt(Math.max(0, ybSqSum / n - ybMean * ybMean));
  const colorfulness = Math.sqrt(rgStd * rgStd + ybStd * ybStd) + 0.3 * Math.sqrt(rgMean * rgMean + ybMean * ybMean);
  return { satMean, highSatFrac, colorfulness };
}

// ---- Frequency-domain (FFT) periodic-artifact scan ----
function fft1d(re, im, invert) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      let t = re[i]; re[i] = re[j]; re[j] = t;
      t = im[i]; im[i] = im[j]; im[j] = t;
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (invert ? 1 : -1) * 2 * Math.PI / len;
    const wRe = Math.cos(ang), wIm = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let curRe = 1, curIm = 0;
      const half = len >> 1;
      for (let k = 0; k < half; k++) {
        const uRe = re[i + k], uIm = im[i + k];
        const vRe = re[i + k + half] * curRe - im[i + k + half] * curIm;
        const vIm = re[i + k + half] * curIm + im[i + k + half] * curRe;
        re[i + k] = uRe + vRe; im[i + k] = uIm + vIm;
        re[i + k + half] = uRe - vRe; im[i + k + half] = uIm - vIm;
        const nextRe = curRe * wRe - curIm * wIm;
        const nextIm = curRe * wIm + curIm * wRe;
        curRe = nextRe; curIm = nextIm;
      }
    }
  }
}

function fft2dMagnitude(gray, size) {
  const re = new Float32Array(size * size);
  const im = new Float32Array(size * size);
  re.set(gray);
  for (let y = 0; y < size; y++) fft1d(re.subarray(y * size, y * size + size), im.subarray(y * size, y * size + size), false);
  const colRe = new Float32Array(size), colIm = new Float32Array(size);
  for (let x = 0; x < size; x++) {
    for (let y = 0; y < size; y++) { colRe[y] = re[y * size + x]; colIm[y] = im[y * size + x]; }
    fft1d(colRe, colIm, false);
    for (let y = 0; y < size; y++) { re[y * size + x] = colRe[y]; im[y * size + x] = colIm[y]; }
  }
  const mag = new Float32Array(size * size);
  for (let i = 0; i < size * size; i++) mag[i] = Math.sqrt(re[i] * re[i] + im[i] * im[i]);
  return mag;
}

function runSpectralScan(ctx, w, h) {
  const SIZE = 256;
  const cropW = Math.min(SIZE, w), cropH = Math.min(SIZE, h);
  const x0 = Math.floor((w - cropW) / 2), y0 = Math.floor((h - cropH) / 2);
  const data = ctx.getImageData(x0, y0, cropW, cropH).data;
  const gray = new Float32Array(SIZE * SIZE);
  for (let y = 0; y < SIZE; y++) {
    const sy = Math.min(cropH - 1, y);
    for (let x = 0; x < SIZE; x++) {
      const sx = Math.min(cropW - 1, x);
      const i = (sy * cropW + sx) * 4;
      gray[y * SIZE + x] = data[i] * 0.299 + data[i + 1] * 0.587 + data[i + 2] * 0.114;
    }
  }
  for (let y = 0; y < SIZE; y++) {
    const wy = 0.5 - 0.5 * Math.cos(2 * Math.PI * y / (SIZE - 1));
    for (let x = 0; x < SIZE; x++) {
      const wx = 0.5 - 0.5 * Math.cos(2 * Math.PI * x / (SIZE - 1));
      gray[y * SIZE + x] *= wx * wy;
    }
  }
  const mag = fft2dMagnitude(gray, SIZE);
  const maxR = SIZE / 2;
  const ringSum = new Float64Array(maxR + 1);
  const ringMax = new Float64Array(maxR + 1);
  const ringCount = new Float64Array(maxR + 1);
  for (let y = 0; y < SIZE; y++) {
    const fy = (y + SIZE / 2) % SIZE - SIZE / 2;
    for (let x = 0; x < SIZE; x++) {
      const fx = (x + SIZE / 2) % SIZE - SIZE / 2;
      const r = Math.round(Math.sqrt(fx * fx + fy * fy));
      if (r < 4 || r > maxR) continue;
      const v = mag[y * SIZE + x];
      ringSum[r] += v; ringCount[r]++;
      if (v > ringMax[r]) ringMax[r] = v;
    }
  }
  let spikeRings = 0, ringsChecked = 0;
  const ratios = [];
  for (let r = 4; r <= maxR; r++) {
    if (ringCount[r] < 8) continue;
    const mean = ringSum[r] / ringCount[r];
    const ratio = mean > 1e-3 ? ringMax[r] / mean : 0;
    ringsChecked++;
    ratios.push(ratio);
    if (ratio > 6) spikeRings++;
  }
  const spikeFraction = ringsChecked ? spikeRings / ringsChecked : 0;
  ratios.sort((a, b) => a - b);
  const medianRatio = ratios.length ? ratios[Math.floor(ratios.length / 2)] : 1;
  return { spikeFraction, medianRatio };
}

// ---- Provenance: additive points system, not an if/else priority chain,
// so weak structural clues can combine. The "no evidence" case lands
// exactly on neutral (50) - real photos routinely have zero metadata once
// re-shared/downloaded, and that must not read as "leaning AI".
function computeProvenance(meta, res) {
  if (meta.hasAISig) return { gen: 97, edit: 50, confidence: 0.95 };
  if (meta.hasCameraMaker && meta.hasEmbeddedThumb) return { gen: 4, edit: 6, confidence: 0.85 };

  let genPoints = 0, editPoints = 0, evidence = 0;
  if (meta.hasCameraMaker) { genPoints -= 30; editPoints -= 25; evidence++; }
  if (meta.hasEditSoftware) { genPoints += 5; editPoints += 18; evidence++; }
  if (meta.format === 'jpeg' && meta.hasEmbeddedThumb) { genPoints -= 22; editPoints -= 15; evidence++; }
  if (meta.subsampling === '4:2:0') { genPoints -= 8; evidence++; }
  else if (meta.subsampling === '4:4:4') { genPoints += 12; evidence++; }
  else if (meta.subsampling === '4:2:2') { genPoints += 4; evidence++; }
  if (res.bothBucket) { genPoints += 14; evidence++; }
  if (res.cameraLike) { genPoints -= 16; editPoints -= 8; evidence++; }

  return {
    gen: clamp(50 + genPoints, 4, 90),
    edit: clamp(50 + editPoints, 4, 85),
    confidence: clamp(evidence * 0.12, 0, 0.55)
  };
}

function computeScores(meta, ela, tex, color, res, spectral) {
  const prov = computeProvenance(meta, res);

  const satScore = clamp(lerpScore(color.satMean, 0.20, 0.55, 15, 85), 0, 100);
  const highSatScore = clamp(lerpScore(color.highSatFrac, 0.15, 0.55, 15, 85), 0, 100);
  const colorfulScore = clamp(lerpScore(color.colorfulness, 20, 55, 20, 80), 0, 100);
  const stylizationScore = satScore * 0.55 + highSatScore * 0.30 + colorfulScore * 0.15;

  const spectralScore = clamp(
    lerpScore(spectral.spikeFraction, 0.03, 0.22, 15, 85) * 0.6 +
    lerpScore(spectral.medianRatio, 2, 8, 15, 85) * 0.4, 0, 100);

  const elaUniformityScore = clamp(lerpScore(ela.cv, 0.15, 1.0, 88, 10), 0, 100);
  const elaHotspotScore = clamp(ela.hotspotRatio * 260, 0, 100);
  const noiseEnergyScore = clamp(lerpScore(tex.meanVar, 0, 90, 80, 20), 0, 100);
  const noiseUniformityScore = clamp(lerpScore(tex.uniformity, 0.2, 2.6, 70, 25), 0, 100);
  const textureScore = noiseEnergyScore * 0.5 + noiseUniformityScore * 0.5;
  const pixelModifier = elaUniformityScore * 0.5 + textureScore * 0.5;

  const metaWeight = 0.15 + 0.80 * prov.confidence;
  const remaining = 1 - metaWeight;
  const genRaw = prov.gen * metaWeight
    + stylizationScore * (remaining * 0.30)
    + spectralScore * (remaining * 0.40)
    + pixelModifier * (remaining * 0.30);

  const genContrast = 1.0 + 1.2 * prov.confidence;
  const genLo = 2 + (1 - prov.confidence) * 28;
  const genHi = 98 - (1 - prov.confidence) * 28;
  const generated = Math.round(clamp(50 + (genRaw - 50) * genContrast, genLo, genHi));

  const editRaw = prov.edit * 0.4 + elaHotspotScore * 0.6;
  const editContrast = 1.0 + 0.7 * prov.confidence;
  const editLo = 4 + (1 - prov.confidence) * 26;
  const editHi = 96 - (1 - prov.confidence) * 26;
  const edited = Math.round(clamp(50 + (editRaw - 50) * editContrast, editLo, editHi));

  const overall = Math.round(clamp(
    0.8 * Math.max(generated, edited) + 0.2 * Math.min(generated, edited), genLo, genHi));

  const confidenceLevel = prov.confidence >= 0.65 ? 'high' : prov.confidence >= 0.3 ? 'medium' : 'low';

  const resNote = res.bothBucket
    ? ' Its resolution also lands exactly on a generator-typical size grid (both dimensions are multiples of 64 within the common diffusion-model output range).'
    : res.cameraLike
      ? ' Its resolution is large and off the 64px grid generators sample on - typical of an unresized camera photo.'
      : '';
  const subsamplingNote = meta.subsampling === '4:4:4'
    ? ' Chroma subsampling is 4:4:4, which real camera JPEGs almost never use (they use 4:2:0).'
    : meta.subsampling === '4:2:0'
      ? ' Chroma subsampling is the standard 4:2:0 used by virtually all camera/phone JPEG encoders.'
      : '';

  const notes = {
    metadata: (meta.hasAISig
      ? 'Found an AI-generator signature (parameter block or content-credential tag) embedded in the file.'
      : meta.hasCameraMaker && meta.hasEmbeddedThumb
        ? 'Camera make/model tag plus an embedded camera-style thumbnail found - a strong signal of a genuine camera photo.'
        : meta.hasCameraMaker
          ? 'Camera make/model metadata detected, though without the embedded thumbnail real camera JPEGs usually carry.'
          : meta.hasEditSoftware
            ? 'Editing-software tag found (e.g. Photoshop/Lightroom); no explicit AI-generator signature.'
            : meta.format === 'jpeg' && meta.hasEmbeddedThumb
              ? 'No maker tag, but the file carries an embedded camera-style thumbnail - typically only present in camera-originated JPEGs.'
              : 'No camera metadata, AI signature, or embedded thumbnail found - common for both AI output and ordinary re-exported or shared photos, so treated as neutral rather than suspicious on its own.'
    ) + resNote + subsamplingNote,
    color: color.satMean > 0.42
      ? `Saturation is high and broadly distributed (${Math.round(color.highSatFrac * 100)}% of pixels strongly saturated). This can indicate generated/stylized art, but richly color-graded real photography triggers this too.`
      : 'Saturation and color spread fall within the range typical of ordinary photography. This check does not reliably catch modern photoreal AI generators.',
    spectral: spectral.spikeFraction > 0.1
      ? `Frequency spectrum shows off-axis spikes in ${Math.round(spectral.spikeFraction * 100)}% of checked frequency bands - a pattern associated with periodic upsampling artifacts from GAN/diffusion decoders.`
      : 'Frequency spectrum falls off smoothly with no unusual periodic spikes - consistent with a natural optical/sensor pipeline.',
    compression: ela.hotspotRatio > 0.12
      ? `Error-level map shows localized hotspots (${Math.round(ela.hotspotRatio * 100)}% of blocks) - sometimes seen where part of an image was inpainted or AI-edited.`
      : 'No strong localized compression anomalies found. Low-confidence, secondary signal.',
    texture: 'Texture and noise levels recorded, but weighted lightly - modern AI art can be as textured as, or more textured than, ordinary photos.'
  };

  return {
    generated, edited, overall, confidenceLevel,
    checks: [
      { name: 'Metadata, resolution & subsampling', score: Math.round(prov.gen), note: notes.metadata },
      { name: 'Color & saturation profile', score: Math.round(stylizationScore), note: notes.color },
      { name: 'Frequency-domain (FFT) artifacts', score: Math.round(spectralScore), note: notes.spectral },
      { name: 'Compression consistency (ELA)', score: Math.round(elaUniformityScore), note: notes.compression },
      { name: 'Texture & noise (low confidence)', score: Math.round(textureScore), note: notes.texture }
    ]
  };
}

export async function runHeuristics(bytes, url) {
  const img = await loadImageEl(url);
  const { canvas, ctx, w, h } = drawToCanvas(img, 1024);
  const meta = scanMetadata(bytes);
  const res = checkResolutionSignal(img.naturalWidth, img.naturalHeight);
  const ela = await runELA(canvas, ctx, w, h);
  const tex = runTextureNoise(ctx, w, h);
  const color = colorStats(ctx, w, h);
  const spectral = runSpectralScan(ctx, w, h);
  const scores = computeScores(meta, ela, tex, color, res, spectral);
  return { width: img.naturalWidth, height: img.naturalHeight, elaThumb: ela.thumb, ...scores };
}
