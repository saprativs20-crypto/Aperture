import { runHeuristics } from './heuristics.js';
import { loadDetector, runDetector, isLoaded } from './detector.js';

const $ = (sel, root) => (root || document).querySelector(sel);
const chat = $('#chat');
const chipRow = $('#chipRow');
const textInput = $('#textInput');
const sendBtn = $('#sendBtn');
const fileInput = $('#fileInput');
const attachBtn = $('#attachBtn');
const composerForm = $('#composerForm');

function escapeHtml(str) {
  return String(str).replace(/[&<>"']/g, c => ({ '&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;' }[c]));
}
function fmtBytes(n) {
  if (n < 1024) return n + ' B';
  if (n < 1024 * 1024) return (n / 1024).toFixed(0) + ' KB';
  return (n / 1024 / 1024).toFixed(1) + ' MB';
}
function band(score) {
  if (score >= 50) return 'bad';
  if (score >= 25) return 'warn';
  return 'good';
}
function scrollToEnd() {
  requestAnimationFrame(() => { chat.scrollTop = chat.scrollHeight; });
}

// ---- Composer: attach / paste / drag-drop ----
let pending = null; // { file, url, bytes, name, size }

function renderChip() {
  if (!pending) { chipRow.innerHTML = ''; return; }
  chipRow.innerHTML = `
    <div class="pending-chip">
      <img src="${pending.url}" alt="">
      <div class="meta">
        <span class="fname">${escapeHtml(pending.name)}</span>
        <span class="fsize">${fmtBytes(pending.size)}</span>
      </div>
      <button type="button" id="removeChip" aria-label="Remove image">✕</button>
    </div>`;
  $('#removeChip').addEventListener('click', () => {
    if (pending) URL.revokeObjectURL(pending.url);
    pending = null;
    renderChip();
    updateSendState();
  });
}

function updateSendState() {
  sendBtn.disabled = !pending && textInput.value.trim() === '';
}

async function handleFile(file) {
  if (!file) return;
  if (!file.type.startsWith('image/')) {
    addBotBubble(`That file (<span class="cmd">${escapeHtml(file.name || 'file')}</span>) isn't an image.`);
    return;
  }
  if (file.size > 25 * 1024 * 1024) {
    addBotBubble(`That image is a little large (${fmtBytes(file.size)}). Try one under 25 MB.`);
    return;
  }
  if (pending) URL.revokeObjectURL(pending.url);
  const bytes = new Uint8Array(await file.arrayBuffer());
  pending = { file, url: URL.createObjectURL(file), bytes, name: file.name || 'pasted-image', size: file.size };
  renderChip();
  updateSendState();
}

attachBtn.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', (e) => {
  if (e.target.files && e.target.files[0]) handleFile(e.target.files[0]);
  fileInput.value = '';
});
textInput.addEventListener('paste', (e) => {
  const items = e.clipboardData && e.clipboardData.items;
  if (!items) return;
  for (const item of items) {
    if (item.type && item.type.startsWith('image/')) {
      e.preventDefault();
      handleFile(item.getAsFile());
      return;
    }
  }
});
['dragover', 'dragenter'].forEach(evt => composerForm.addEventListener(evt, (e) => { e.preventDefault(); composerForm.classList.add('drag'); }));
['dragleave', 'drop'].forEach(evt => composerForm.addEventListener(evt, (e) => { if (evt === 'drop') e.preventDefault(); composerForm.classList.remove('drag'); }));
composerForm.addEventListener('drop', (e) => { const f = e.dataTransfer?.files?.[0]; if (f) handleFile(f); });
chat.addEventListener('dragover', e => e.preventDefault());
chat.addEventListener('drop', (e) => { e.preventDefault(); const f = e.dataTransfer?.files?.[0]; if (f) handleFile(f); });
textInput.addEventListener('input', () => {
  textInput.style.height = 'auto';
  textInput.style.height = Math.min(textInput.scrollHeight, 120) + 'px';
  updateSendState();
});
textInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); composerForm.requestSubmit(); }
});

// ---- Chat rendering ----
function addUserBubble(text, imgMeta) {
  const div = document.createElement('div');
  div.className = 'msg user';
  div.innerHTML = `
    <div class="avatar">You</div>
    <div class="bubble">
      ${imgMeta ? `
      <div class="thumb-chip">
        <img src="${imgMeta.url}" alt="">
        <div class="meta"><span class="fname">${escapeHtml(imgMeta.name)}</span><span class="fsize">${fmtBytes(imgMeta.size)}</span></div>
      </div>` : ''}
      <p>${escapeHtml(text)}</p>
    </div>`;
  chat.appendChild(div);
  scrollToEnd();
}
function addBotBubble(html) {
  const div = document.createElement('div');
  div.className = 'msg bot';
  div.innerHTML = `<div class="avatar">AI</div><div class="bubble"><p>${html}</p></div>`;
  chat.appendChild(div);
  scrollToEnd();
  return div;
}
function addProgressBubble() {
  const div = document.createElement('div');
  div.className = 'msg bot';
  div.innerHTML = `<div class="avatar">AI</div><div class="bubble">
    <p class="progress-status">Loading the AI model (first time only, ~175MB)…</p>
    <div class="progress-track"><div class="progress-fill" style="width:0%"></div></div>
  </div>`;
  chat.appendChild(div);
  scrollToEnd();
  return div;
}
function addTyping() {
  const div = document.createElement('div');
  div.className = 'msg bot';
  div.innerHTML = `<div class="avatar">AI</div><div class="bubble"><div class="typing"><span></span><span></span><span></span></div></div>`;
  chat.appendChild(div);
  scrollToEnd();
  return div;
}

function verdictHeadline(overall, mlAvailable, disagree) {
  if (disagree) return 'Mixed signals — the ML model and the pixel/metadata checks disagree. Treat this one with extra caution.';
  if (overall >= 65) return mlAvailable ? 'Strong signs of AI generation, per both the model and supporting checks.' : 'Signs of AI generation detected in the pixel/metadata checks.';
  if (overall >= 40) return 'Mixed or inconclusive signals — nothing definitive either way.';
  return mlAvailable ? 'Looks like a real photo — the model and supporting checks agree.' : 'No strong indicators of AI generation found.';
}

// ---- Combine ML + heuristic scores ----
function combineScores(ml, heuristic) {
  if (!ml) {
    return {
      generated: heuristic.generated,
      edited: heuristic.edited,
      overall: heuristic.overall,
      mlAvailable: false,
      disagree: false
    };
  }
  const mlGenerated = Math.round(ml.deepfakeProb * 100);
  // ML gets most of the weight - it's the stronger, trained signal; the
  // heuristics remain a transparent, independent cross-check.
  const combinedGenerated = Math.round(mlGenerated * 0.75 + heuristic.generated * 0.25);
  const disagree = Math.abs(mlGenerated - heuristic.generated) >= 35;
  const overall = Math.round(combinedGenerated * 0.85 + heuristic.overall * 0.15);
  return { generated: combinedGenerated, mlGenerated, edited: heuristic.edited, overall, mlAvailable: true, disagree };
}

function renderReport(container, r) {
  const overallBand = band(r.overall);
  const modelPanel = r.mlAvailable ? `
    <div class="model-panel">
      <div class="mp-title">ML model verdict <span class="model-badge">SwinV2 AI-image detector</span></div>
      <div>Artificial-image probability: <strong>${r.mlGenerated}%</strong> (~98% reported validation accuracy — not infallible, see disclaimer below)</div>
      ${r.disagree ? '<div class="disagree-note">⚠ The model and the heuristic checks disagree by a wide margin — weigh this result with extra caution.</div>' : ''}
    </div>` : `
    <div class="confidence-banner low">
      <span class="cb-label">ML MODEL UNAVAILABLE</span>
      <span>${r.mlError ? escapeHtml(r.mlError) : 'Couldn\'t load the AI classifier (offline, or the download failed).'} Falling back to the heuristic-only estimate below — treat it as a rough lean, not a verdict. Sending another message will retry the download.</span>
    </div>`;

  container.innerHTML = `
    <div class="report-card">
      <div class="report-head">
        <span class="verdict-dot ${overallBand}"></span>
        <div>
          <div class="report-title">${escapeHtml(r.headline)}</div>
          <div class="report-sub">${escapeHtml(r.fileName)} · ${r.width}×${r.height}px · ${r.elapsed}ms</div>
        </div>
      </div>
      <div class="metric-row">
        <div class="metric">
          <div class="metric-label">AI-GENERATED LIKELIHOOD</div>
          <div class="meter"><div class="meter-fill ${band(r.generated)}" style="width:${r.generated}%"></div></div>
          <div class="metric-value">${r.generated}%</div>
        </div>
        <div class="metric">
          <div class="metric-label">AI-EDITED LIKELIHOOD (heuristic)</div>
          <div class="meter"><div class="meter-fill ${band(r.edited)}" style="width:${r.edited}%"></div></div>
          <div class="metric-value">${r.edited}%</div>
        </div>
      </div>
      <div class="overall">
        <span class="overall-label">Overall AI involvement</span>
        <span class="overall-value ${overallBand}">${r.overall}%</span>
      </div>
      ${modelPanel}
      <div class="confidence-banner ${r.heuristic.confidenceLevel}">
        <span class="cb-label">HEURISTIC ${r.heuristic.confidenceLevel.toUpperCase()} CONFIDENCE</span>
        <span>${r.heuristic.confidenceLevel === 'low'
          ? 'No reliable file metadata survived (normal after downloading/re-sharing). The heuristic cross-check leans on pixel/color patterns alone here.'
          : 'Some structural/metadata evidence backs the heuristic cross-check.'}</span>
      </div>
      <details class="breakdown">
        <summary>Show heuristic analysis breakdown</summary>
        <ul class="check-list">
          ${r.heuristic.checks.map(c => `
            <li>
              <div class="check-row"><span class="cname">${escapeHtml(c.name)}</span><span class="cval">${c.score}%</span></div>
              <div class="meter small"><div class="meter-fill ${band(c.score)}" style="width:${c.score}%"></div></div>
              <p class="check-note">${escapeHtml(c.note)}</p>
            </li>`).join('')}
        </ul>
        <div class="ela-wrap">
          <img src="${r.heuristic.elaThumb}" alt="Error-level-analysis heatmap of the uploaded image">
          <p class="ela-caption">Error-level-analysis heatmap — brighter regions indicate compression inconsistencies.</p>
        </div>
      </details>
      <p class="disclaimer">The ML model is a real pretrained classifier (~98% reported validation accuracy) — much stronger evidence than pixel heuristics alone, but still not proof, and validation accuracy on its own test set doesn't guarantee the same accuracy on every image or every generator. The heuristic checks are shown alongside it as an independent, fully transparent cross-check.</p>
    </div>`;
}

// ---- Model loading (lazy, on first analysis) ----
let classifier = null;
let modelLoadError = null;

// Note: a failed load is intentionally NOT a permanent latch. Each
// loadDetector() call already retries several times internally
// (detector.js's fetchWithProgress), but if the whole attempt still fails
// (e.g. a real network drop), the user can trigger a fresh attempt simply
// by sending another image - a transient failure shouldn't lock out the ML
// model for the rest of the session.
async function ensureDetector(progressBubble) {
  if (classifier) return classifier;
  try {
    classifier = await loadDetector((info) => {
      if (!progressBubble) return;
      const fill = $('.progress-fill', progressBubble);
      const status = $('.progress-status', progressBubble);
      if (info.status === 'progress' && typeof info.progress === 'number') {
        if (fill) fill.style.width = Math.round(info.progress) + '%';
        if (status) status.textContent = `Loading the AI model (first time only, ~175MB)… ${Math.round(info.progress)}%`;
      } else if (info.status === 'retrying') {
        if (status) status.textContent = `Connection interrupted — retrying download (attempt ${info.attempt})… ${Math.round(info.progress)}%`;
      }
    });
    return classifier;
  } catch (err) {
    console.error('Detector load failed:', err);
    modelLoadError = err.message || 'Unknown error';
    return null;
  }
}

async function analyzeImage(pendingImage, progressBubble) {
  const t0 = performance.now();
  const heuristic = await runHeuristics(pendingImage.bytes, pendingImage.url);

  const clf = await ensureDetector(progressBubble);
  let ml = null;
  if (clf) {
    try {
      const img = new Image();
      await new Promise((res, rej) => { img.onload = res; img.onerror = rej; img.src = pendingImage.url; });
      ml = await runDetector(clf, img);
    } catch (err) {
      console.error('Detector inference failed:', err);
      ml = null;
    }
  }

  const combined = combineScores(ml, heuristic);
  const elapsed = Math.round(performance.now() - t0);
  return {
    fileName: pendingImage.name,
    width: heuristic.width,
    height: heuristic.height,
    elapsed,
    headline: verdictHeadline(combined.overall, combined.mlAvailable, combined.disagree),
    heuristic,
    mlError: modelLoadError,
    ...combined
  };
}

// ---- Submit flow ----
composerForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  const text = textInput.value.trim();
  if (!pending && !text) return;

  if (!pending) {
    addUserBubble(text);
    textInput.value = ''; textInput.style.height = 'auto';
    updateSendState();
    addBotBubble('Attach or paste an image to analyze — click the paperclip, drag a file in, or Ctrl/Cmd-V a copied image.');
    return;
  }

  const img = pending;
  const displayText = text || 'Check the AI';
  addUserBubble(displayText, img);
  textInput.value = ''; textInput.style.height = 'auto';
  pending = null; renderChip(); updateSendState();

  // isLoaded() reflects whether a model attempt is currently in-flight or
  // has completed; detector.js resets it to "not loaded" internally when an
  // attempt fails, so this naturally shows the progress bar again on retry.
  const needsModelLoad = !isLoaded();
  const progressBubble = needsModelLoad ? addProgressBubble() : addTyping();

  let report;
  try {
    await new Promise(r => setTimeout(r, 200));
    report = await analyzeImage(img, needsModelLoad ? progressBubble : null);
  } catch (err) {
    console.error(err);
    progressBubble.remove();
    addBotBubble('Something went wrong reading that image — it may be corrupted or in an unsupported format. Try a different file.');
    URL.revokeObjectURL(img.url);
    return;
  }

  progressBubble.className = 'msg bot';
  progressBubble.innerHTML = `<div class="avatar">AI</div><div class="report"></div>`;
  renderReport($('.report', progressBubble), report);
  scrollToEnd();
  URL.revokeObjectURL(img.url);
});

updateSendState();

// Warm up the model in the background shortly after load, so the first
// real analysis doesn't have to wait for the full download.
setTimeout(() => { ensureDetector(null); }, 1500);

// Registers the service worker that makes the app installable ("Add to
// Home Screen") and caches the app shell for offline use. Requires a
// secure context (https://, or localhost for local testing) - browsers
// refuse to register service workers on plain http:// or file://.
if ('serviceWorker' in navigator) {
  window.addEventListener('load', () => {
    navigator.serviceWorker.register('sw.js').catch(err => {
      console.warn('Service worker registration failed (expected on non-HTTPS hosts):', err);
    });
  });
}
