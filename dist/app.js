const photos = new Map();
const allowedTypes = new Set(['image/jpeg', 'image/png', 'image/webp']);
const status = document.querySelector('.status');
const generate = document.querySelector('.generate');
const generateHint = generate.querySelector('small');
const result = document.querySelector('.result');
const resultImage = result.querySelector('.result-image');
const warningsList = result.querySelector('.warnings');
const download = result.querySelector('.download');
const names = result.querySelector('.names');
let keepsake = null;
// Only two pairings make sense: elder sister + younger brother, or younger sister + elder brother.
const NAME_PAIRS = { 'didi-bhai': { sister: 'didi', brother: 'bhai' }, 'bahini-dai': { sister: 'bahini', brother: 'dai' } };
const chosenLabels = () => NAME_PAIRS[document.querySelector('input[name="sibling-names"]:checked').value];
async function showKeepsake(canvas) {
  const blob = await new Promise((resolve, reject) => canvas.toBlob(b => b ? resolve(b) : reject(new Error('export failed')), 'image/png'));
  if (resultUrl) URL.revokeObjectURL(resultUrl);
  resultUrl = URL.createObjectURL(blob);
  resultImage.src = resultUrl;
  download.href = resultUrl;
}
names.addEventListener('change', async () => {
  if (!keepsake) return;
  if (!editor.hidden) buildTabs();
  try { await showKeepsake(keepsake.render(chosenLabels())); }
  catch (err) { console.error(err); status.textContent = 'We couldn’t update the names. Please try again.'; }
});
let mode = 'separate';
let busy = false;
let resultUrl = null;
const isReady = () => mode === 'together' ? photos.has('together') : photos.has('sister') && photos.has('brother');
// ---------- mala editor ----------
const editor = result.querySelector('.editor');
const editorCanvas = editor.querySelector('.editor-canvas');
const editorCtx = editorCanvas.getContext('2d');
const editorTabs = editor.querySelector('.editor-tabs');
const sizeInput = editor.querySelector('input[type="range"]');
const sizeOutput = editor.querySelector('.size-value');
const sizeButtons = editor.querySelectorAll('.size-btn');
const resultActions = result.querySelector('.result-actions');
let sceneIndex = 0, selectedMala = 0, drag = null, editorScale = 1, frame = 0;
const currentScene = () => keepsake && keepsake.scenes[sceneIndex];
const hypot = (p, q) => Math.hypot(p.x - q.x, p.y - q.y);
function redrawEditor() {
  cancelAnimationFrame(frame);
  frame = requestAnimationFrame(() => {
    const scene = currentScene();
    if (!scene) return;
    window.Tika.drawEditor(editorCtx, scene, editorScale, selectedMala);
    const m = scene.malas[selectedMala];
    sizeInput.disabled = !m;
    sizeButtons.forEach(b => { b.disabled = !m; });
    if (m) {
      const pct = Math.round((m.unit / m.unit0) * 100);
      if (!drag) sizeInput.value = pct;
      sizeOutput.textContent = `${pct}%`;
    }
  });
}
function layoutEditor() {
  const scene = currentScene();
  if (!scene || editor.hidden) return;
  const dpr = window.devicePixelRatio || 1;
  const cssScale = Math.min((editor.clientWidth || 600) / scene.base.width, (window.innerHeight * 0.7) / scene.base.height, 1.5);
  editorCanvas.style.width = `${Math.round(scene.base.width * cssScale)}px`;
  editorCanvas.style.height = `${Math.round(scene.base.height * cssScale)}px`;
  editorCanvas.width = Math.round(scene.base.width * cssScale * dpr);
  editorCanvas.height = Math.round(scene.base.height * cssScale * dpr);
  editorScale = cssScale * dpr;
  redrawEditor();
}
function buildTabs() {
  const scenes = keepsake.scenes;
  editorTabs.hidden = scenes.length < 2;
  const names = chosenLabels();
  editorTabs.replaceChildren(...scenes.map((scene, i) => {
    const tab = document.createElement('button');
    tab.type = 'button';
    tab.setAttribute('role', 'tab');
    tab.setAttribute('aria-selected', String(i === sceneIndex));
    const label = window.Tika.LABELS[names[scene.id]];
    tab.textContent = label ? `${label.word}’s photo` : scene.id;
    tab.addEventListener('click', () => { sceneIndex = i; selectedMala = 0; buildTabs(); layoutEditor(); });
    return tab;
  }));
}
function openEditor() {
  if (!keepsake) return;
  sceneIndex = 0; selectedMala = 0;
  editor.hidden = false;
  resultImage.hidden = true;
  resultActions.hidden = true;
  buildTabs();
  layoutEditor();
  editorCanvas.focus({ preventScroll: true });
  editor.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}
function closeEditor() {
  editor.hidden = true;
  resultImage.hidden = false;
  resultActions.hidden = false;
  drag = null;
}
function toImagePoint(e) {
  const rect = editorCanvas.getBoundingClientRect();
  const scene = currentScene();
  return { x: ((e.clientX - rect.left) / rect.width) * scene.base.width, y: ((e.clientY - rect.top) / rect.height) * scene.base.height };
}
const imagePxPerCss = () => currentScene().base.width / editorCanvas.getBoundingClientRect().width;
function handleUnder(p) {
  const m = currentScene().malas[selectedMala];
  if (!m) return null;
  const reach = 24 * imagePxPerCss();
  const da = hypot(p, m.a), db = hypot(p, m.b);
  if (Math.min(da, db) > reach) return null;
  return da <= db ? 'a' : 'b';
}
const touches = new Map();
let pinch = null;
const touchSpread = () => { const [p, q] = [...touches.values()]; return hypot(p, q); };
editorCanvas.addEventListener('pointerdown', e => {
  const scene = currentScene();
  if (!scene) return;
  const p = toImagePoint(e);
  if (e.pointerType === 'touch') {
    touches.set(e.pointerId, { x: e.clientX, y: e.clientY });
    const m = scene.malas[selectedMala];
    if (touches.size === 2 && m) {
      // Second finger down: switch from dragging to pinch-resizing the selected mala.
      drag = null;
      pinch = { spread: touchSpread() || 1, size: m.unit / m.unit0 };
      editorCanvas.setPointerCapture(e.pointerId);
      e.preventDefault();
      return;
    }
  }
  const end = handleUnder(p);
  if (end) drag = { type: 'end', which: end };
  else {
    const i = window.Tika.hitTest(scene.malas, p, 6 * imagePxPerCss());
    if (i >= 0) { selectedMala = i; drag = { type: 'move', last: p }; }
  }
  if (drag) { editorCanvas.setPointerCapture(e.pointerId); editorCanvas.style.cursor = 'grabbing'; e.preventDefault(); }
  redrawEditor();
});
editorCanvas.addEventListener('pointermove', e => {
  const scene = currentScene();
  if (!scene) return;
  if (touches.has(e.pointerId)) touches.set(e.pointerId, { x: e.clientX, y: e.clientY });
  if (pinch) {
    if (touches.size === 2) setSize(scene.malas[selectedMala], pinch.size * (touchSpread() / pinch.spread));
    return;
  }
  const p = toImagePoint(e);
  if (!drag) {
    editorCanvas.style.cursor = handleUnder(p) ? 'grab' : window.Tika.hitTest(scene.malas, p, 6 * imagePxPerCss()) >= 0 ? 'move' : 'default';
    return;
  }
  const m = scene.malas[selectedMala];
  if (drag.type === 'end') window.Tika.moveEndpoint(m, drag.which, p);
  else { window.Tika.moveMala(m, p.x - drag.last.x, p.y - drag.last.y); drag.last = p; }
  redrawEditor();
});
const endDrag = e => {
  touches.delete(e.pointerId);
  if (touches.size < 2) pinch = null;
  if (drag) { drag = null; editorCanvas.style.cursor = 'default'; redrawEditor(); }
};
editorCanvas.addEventListener('pointerup', endDrag);
editorCanvas.addEventListener('pointercancel', endDrag);
editorCanvas.addEventListener('keydown', e => {
  const scene = currentScene();
  const m = scene && scene.malas[selectedMala];
  const step = 3 * imagePxPerCss() * (e.shiftKey ? 4 : 1);
  const moves = { ArrowLeft: [-step, 0], ArrowRight: [step, 0], ArrowUp: [0, -step], ArrowDown: [0, step] };
  if (e.key === 'Tab' || !m) return;
  if (moves[e.key]) { window.Tika.moveMala(m, ...moves[e.key]); e.preventDefault(); redrawEditor(); }
  else if (e.key === '+' || e.key === '=') { nudgeSize(1); e.preventDefault(); }
  else if (e.key === '-' || e.key === '_') { nudgeSize(-1); e.preventDefault(); }
  else if (e.key === 'n' && scene.malas.length > 1) { selectedMala = (selectedMala + 1) % scene.malas.length; redrawEditor(); }
});
// Bigger/smaller: everything funnels through setSize (a fraction of the original size).
const selected = () => currentScene() && currentScene().malas[selectedMala];
const SIZE_MIN = +sizeInput.min / 100, SIZE_MAX = +sizeInput.max / 100;
function setSize(m, f) {
  if (!m) return;
  window.Tika.resizeMala(m, Math.min(SIZE_MAX, Math.max(SIZE_MIN, f)));
  redrawEditor();
}
const nudgeSize = dir => { const m = selected(); if (m) setSize(m, (m.unit / m.unit0) * (dir > 0 ? 1.08 : 1 / 1.08)); };
sizeInput.addEventListener('input', () => setSize(selected(), sizeInput.value / 100));
editor.querySelector('.size-btn.bigger').addEventListener('click', () => nudgeSize(1));
editor.querySelector('.size-btn.smaller').addEventListener('click', () => nudgeSize(-1));
// Trackpad pinch (and ctrl + wheel) over the editor resizes the selected mala.
editorCanvas.addEventListener('wheel', e => {
  if (!e.ctrlKey) return;
  const m = selected();
  if (!m) return;
  e.preventDefault();
  setSize(m, (m.unit / m.unit0) * Math.exp(-e.deltaY * 0.01));
}, { passive: false });
editor.querySelector('.reset').addEventListener('click', () => {
  const m = currentScene() && currentScene().malas[selectedMala];
  if (m) { window.Tika.resetMala(m); redrawEditor(); }
});
editor.querySelector('.done').addEventListener('click', async () => {
  closeEditor();
  try { await showKeepsake(keepsake.render(chosenLabels())); }
  catch (err) { console.error(err); status.textContent = 'We couldn’t update the keepsake. Please try again.'; }
  result.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
});
result.querySelector('.adjust').addEventListener('click', openEditor);
window.addEventListener('resize', layoutEditor);

function clearResult() {
  if (resultUrl) URL.revokeObjectURL(resultUrl);
  resultUrl = null;
  result.hidden = true;
  keepsake = null;
  closeEditor();
  resultImage.removeAttribute('src');
  download.setAttribute('href', '#');
}
function updateStatus() {
  generate.disabled = busy || !isReady();
  generateHint.textContent = busy ? 'PLACING YOUR TIKA…' : isReady() ? 'MADE ON YOUR DEVICE' : 'ADD YOUR PHOTOS FIRST';
  if (busy) return;
  if (mode === 'together') {
    status.textContent = photos.has('together') ? 'Your photo is ready. Let’s add the tika!' : 'Add one photo with both of you in it.';
    return;
  }
  const count = ['sister', 'brother'].filter(role => photos.has(role)).length;
  status.textContent = count === 2
    ? 'Both photos are ready. Let’s add the tika!'
    : count === 1
      ? `Looking lovely. Add the ${photos.has('sister') ? 'brother' : 'sister'}’s photo next.`
      : 'Your story starts with two photos.';
}
document.querySelectorAll('input[name="mode"]').forEach(radio => radio.addEventListener('change', () => {
  if (!radio.checked) return;
  mode = radio.value;
  document.querySelectorAll('.upload-card').forEach(card => { card.hidden = card.dataset.mode !== mode; });
  clearResult();
  updateStatus();
  if (window.Tika) window.Tika.preload();
}));
generate.addEventListener('click', async () => {
  if (busy || !isReady() || !window.Tika) return;
  busy = true;
  clearResult();
  updateStatus();
  status.textContent = 'Finding faces and placing the tika…';
  try {
    keepsake = mode === 'together'
      ? await window.Tika.createTogether(photos.get('together').url)
      : await window.Tika.createSeparate(photos.get('sister').url, photos.get('brother').url);
    await showKeepsake(keepsake.render(chosenLabels()));
    names.hidden = !keepsake.labelled;
    warningsList.replaceChildren(...keepsake.warnings.map(text => Object.assign(document.createElement('li'), { textContent: text })));
    result.hidden = false;
    busy = false;
    updateStatus();
    status.textContent = 'शुभ भाइटीका! Your keepsake is ready.';
    result.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  } catch (err) {
    console.error(err);
    busy = false;
    updateStatus();
    status.textContent = err instanceof window.Tika.TikaError ? err.message : 'Something went wrong while placing the tika. Please try again.';
  }
});
const cards = new Map();
let activeRole = null;
document.querySelectorAll('.upload-card').forEach(card => {
  const role = card.dataset.role;
  cards.set(role, { card, select: files => selectFiles(files) });
  card.addEventListener('pointerenter', () => { activeRole = role; });
  card.addEventListener('focusin', () => { activeRole = role; });
  const input = card.querySelector('input');
  const zone = card.querySelector('.dropzone');
  const preview = card.querySelector('.preview');
  const error = card.querySelector('.error');
  let selection = 0;
  async function selectFiles(files) {
    const current = ++selection;
    error.textContent = '';
    input.value = '';
    if (!files.length) return;
    if (files.length !== 1) { error.textContent = 'Please choose one photo at a time.'; return; }
    const file = files[0];
    if (!allowedTypes.has(file.type)) { error.textContent = 'Choose a JPG, PNG or WebP image.'; return; }
    if (file.size > 10 * 1024 * 1024) { error.textContent = 'This photo is too large. Choose one under 10 MB.'; return; }
    const url = URL.createObjectURL(file);
    const image = new Image();
    image.src = url;
    try { await image.decode(); }
    catch { URL.revokeObjectURL(url); if (current === selection) error.textContent = 'This image could not be opened. Try a different photo.'; return; }
    if (current !== selection) { URL.revokeObjectURL(url); return; }
    if (photos.has(role)) URL.revokeObjectURL(photos.get(role).url);
    photos.set(role, { file, url });
    clearResult();
    if (window.Tika) window.Tika.preload();
    preview.src = url;
    preview.hidden = false;
    card.querySelector('.empty-state').hidden = true;
    card.querySelector('.replace').hidden = false;
    card.querySelector('.file-details').hidden = false;
    card.querySelector('.filename').textContent = file.name;
    updateStatus();
  }
  input.addEventListener('change', () => selectFiles(Array.from(input.files)));
  ['dragenter', 'dragover'].forEach(event => zone.addEventListener(event, e => { e.preventDefault(); zone.classList.add('dragover'); }));
  zone.addEventListener('dragleave', e => { if (!zone.contains(e.relatedTarget)) zone.classList.remove('dragover'); });
  zone.addEventListener('drop', e => { e.preventDefault(); zone.classList.remove('dragover'); selectFiles(Array.from(e.dataTransfer.files)); });
  card.querySelector('.remove').addEventListener('click', () => {
    selection++;
    if (photos.has(role)) URL.revokeObjectURL(photos.get(role).url);
    photos.delete(role);
    clearResult();
    input.value = '';
    preview.removeAttribute('src');
    preview.hidden = true;
    card.querySelector('.empty-state').hidden = false;
    card.querySelector('.replace').hidden = true;
    card.querySelector('.file-details').hidden = true;
    error.textContent = '';
    updateStatus();
    input.focus();
  });
});
// Prevent accidental browser navigation when a file misses its upload target.
// Paste a photo (⌘V / Ctrl+V). Goes to the card you're hovering or focused on,
// otherwise the first empty card in the current mode.
function pasteTarget() {
  const visible = [...cards.keys()].filter(role => !cards.get(role).card.hidden);
  if (activeRole && visible.includes(activeRole)) return activeRole;
  return visible.find(role => !photos.has(role)) || null;
}
document.addEventListener('paste', e => {
  if (e.target.closest && e.target.closest('input:not([type=file]), textarea, [contenteditable]')) return;
  const files = Array.from(e.clipboardData?.files || []).filter(file => file.type.startsWith('image/'));
  if (!files.length) {
    if (e.clipboardData?.types.includes('Files')) status.textContent = 'That doesn’t look like a photo. Copy an image and paste again.';
    return;
  }
  e.preventDefault();
  const role = pasteTarget();
  if (!role) { status.textContent = 'Both photos are in. Point at the one you want to replace, then paste again.'; return; }
  const ext = (files[0].type.split('/')[1] || 'png').replace('jpeg', 'jpg');
  const file = files[0].name && files[0].name !== 'image.png' ? files[0] : new File([files[0]], `pasted-photo.${ext}`, { type: files[0].type });
  cards.get(role).select([file]);
  cards.get(role).card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
});
window.addEventListener('dragover', e => { if (e.dataTransfer.types.includes('Files')) e.preventDefault(); });
window.addEventListener('drop', e => { if (e.dataTransfer.types.includes('Files')) e.preventDefault(); });
if (!/Mac|iPhone|iPad/.test(navigator.platform)) document.querySelectorAll('.dropzone kbd').forEach(k => { k.textContent = 'Ctrl+V'; });
