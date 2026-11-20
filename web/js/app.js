// Somnibeat UI: hash routing, inputs, worker orchestration and report rendering.
// Every number shown comes from data/results.json, data/samples.json or the report
// computed from the loaded file. Nothing numeric about results is typed here.

import {
  esc, fmtNum, fmtHM, fmtClock, hypnogramSvg, blandAltmanSvg, comparisonBarsSvg,
  STAGE_MODES, COLORS, computeBA, alignReference, compareNight,
} from './charts.js';

// Link to the source repository. Empty string hides every repository link.
const REPO_URL = 'https://github.com/handue/somnibeat';

// Development only: replace the real worker with a local stub. Keep false.
const USE_STUB_WORKER = false;

// Contract thresholds (docs/web_contract.md), not results.
const INVALID_WARN_PCT = 10;
const DAY_S = 86400;

const STEP_ORDER = ['read', 'beats', 'features', 'report'];
const ROUTES = ['start', 'processing', 'report', 'method', 'problem'];
const ROUTE_TITLES = {
  start: 'Somnibeat',
  processing: 'Computing… · Somnibeat',
  report: 'Night report · Somnibeat',
  method: 'Method & evaluation · Somnibeat',
  problem: 'File problem · Somnibeat',
};

const state = {
  results: null,
  resultsError: null,
  samples: null,
  samplesError: null,
  report: null,
  reportClockStartS: null,
  reportClockNote: '',
  reportIsStub: false,
  // Sleep lab comparison: the sample the report came from (null for uploads) and its
  // expert reference labels (data/samples/*_reference.json), or an error message.
  reportSample: null,
  reportReference: null,
  reportReferenceError: null,
  mode: 'sleep_wake',
  animateNext: false,
  // Inputs of the last run, kept so the report can be recomputed with the other model.
  lastRun: null,
  // Method page: which v2 test analysis the results table shows.
  v2View: 'hmc:primary',
  job: null,
  error: null,
  minHours: null,
};

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

const reducedMotion = window.matchMedia ? window.matchMedia('(prefers-reduced-motion: reduce)') : { matches: false };

// ---------------------------------------------------------------------------
// results.json accessors (tolerant of {point, ci_low, ci_high} or plain numbers)

function stat(v) {
  if (v === null || v === undefined) return null;
  if (typeof v === 'number') return { point: v, ci_low: NaN, ci_high: NaN };
  if (typeof v === 'object' && 'point' in v) {
    return { point: Number(v.point), ci_low: Number(v.ci_low), ci_high: Number(v.ci_high) };
  }
  return null;
}

function taskOf(task) {
  return state.results && state.results.tasks ? state.results.tasks[task] || null : null;
}

function primaryModel() {
  return (state.results && state.results.primary_model) || 'gradient_boosting';
}

function modelNames(task) {
  const t = taskOf(task);
  if (!t || !t.models) return [];
  const primary = primaryModel();
  const names = Object.keys(t.models);
  return [primary, ...names.filter((n) => n !== primary)].filter((n) => names.includes(n));
}

function externalOf(task, model) {
  const t = taskOf(task);
  const m = t && t.models && t.models[model];
  return (m && m.external_ucddb) || null;
}

function metricOf(task, model, key) {
  const ext = externalOf(task, model);
  return ext && ext.metrics ? stat(ext.metrics[key]) : null;
}

const BASELINE_PREFERENCE = ['asleep_from_lights_off', 'no_apnea'];

function baselineName(task) {
  const t = taskOf(task);
  if (!t || !t.baselines) return null;
  for (const name of BASELINE_PREFERENCE) if (t.baselines[name]) return name;
  const keys = Object.keys(t.baselines);
  return keys.length ? keys[0] : null;
}

function baselineMetric(task, key) {
  const t = taskOf(task);
  const name = baselineName(task);
  if (!t || !name) return null;
  const b = t.baselines[name];
  const metrics = (b && (b.metrics || (b.external_ucddb && b.external_ucddb.metrics))) || b;
  return metrics ? stat(metrics[key]) : null;
}

function nTestRecords(task) {
  const ext = externalOf(task, primaryModel());
  if (ext && Number.isFinite(ext.n_records)) return ext.n_records;
  const pr = state.results && state.results.per_record && state.results.per_record[task];
  return Array.isArray(pr) ? pr.length : null;
}

function prettyName(key) {
  const known = {
    gradient_boosting: 'Gradient boosting',
    logistic_regression: 'Logistic regression',
    asleep_from_lights_off: 'Asleep from lights off',
    no_apnea: 'No apnea minutes',
    majority_class: 'Majority class',
  };
  if (known[key]) return known[key];
  const s = String(key).replace(/_/g, ' ');
  return s.charAt(0).toUpperCase() + s.slice(1);
}

// ---------------------------------------------------------------------------
// results.json "v2" block accessors (v2 and v1 on HMC / UCDDB, docs/web_contract.md)

const MODEL_IDS = ['v2', 'v1'];

function v2Block() {
  return (state.results && state.results.v2) || null;
}

function v2Set(ds) {
  const v = v2Block();
  return (v && v.test_sets && v.test_sets[ds]) || null;
}

/** Entry of a v2 test analysis: ds 'hmc' | 'ucddb', analysis 'primary' | 'all_records', model 'v2' | 'v1' | baseline. */
function v2Entry(ds, task, analysis, model) {
  const set = v2Set(ds);
  const t = set && set.tasks && set.tasks[task];
  return (t && t[analysis] && t[analysis][model]) || null;
}

function v2Metric(ds, task, analysis, model, key) {
  const e = v2Entry(ds, task, analysis, model);
  return e && e.metrics ? stat(e.metrics[key]) : null;
}

function v2Apnea(model, analysis = 'primary') {
  const v = v2Block();
  const a = v && v.apnea_ucddb && v.apnea_ucddb[analysis];
  return (a && a[model]) || null;
}

function v2ApneaMetric(model, key) {
  const e = v2Apnea(model);
  return e && e.metrics ? stat(e.metrics[key]) : null;
}

function v2StageBaseline() {
  const v = v2Block();
  return (v && v.baselines && v.baselines.stages) || 'asleep_from_lights_off';
}

function v2ApneaBaseline() {
  const v = v2Block();
  return (v && v.baselines && v.baselines.apnea) || 'no_apnea';
}

function frozenDate() {
  const v = v2Block();
  return (v && v.frozen && v.frozen.date) || null;
}

function modelLabel(model) {
  if (model === 'v2') {
    const d = frozenDate();
    return d ? `v2 (frozen ${d})` : 'v2';
  }
  if (model === 'v1') return 'v1 (first version)';
  return String(model || 'unknown');
}

function datasetShort(id) {
  const ds = (state.results && state.results.datasets) || [];
  const hit = ds.find((d) => d.id === id);
  return hit ? (hit.short || hit.name) : String(id).toUpperCase();
}

/**
 * Compare model a with model b on one metric using the 95% intervals.
 * Returns 'better' | 'worse' (intervals do not overlap) or 'overlap'; null without data.
 */
function compareStats(a, b, higherIsBetter) {
  if (!a || !b) return null;
  if (![a.ci_low, a.ci_high, b.ci_low, b.ci_high].every(Number.isFinite)) return null;
  if (a.ci_low > b.ci_high) return higherIsBetter ? 'better' : 'worse';
  if (a.ci_high < b.ci_low) return higherIsBetter ? 'worse' : 'better';
  return 'overlap';
}

function ciText(s, digits, opts = {}) {
  if (!s || !Number.isFinite(s.ci_low) || !Number.isFinite(s.ci_high)) return '';
  if (s.ci_low === s.ci_high) return ''; // a fixed rule (for example kappa 0 of the trivial rule) has no interval
  return `${fmtNum(s.ci_low, digits, opts)} to ${fmtNum(s.ci_high, digits, opts)}`;
}

// ---------------------------------------------------------------------------
// Data loading

async function fetchJson(path) {
  const res = await fetch(path, { cache: 'no-cache' });
  if (!res.ok) throw new Error(`${path}: HTTP ${res.status}`);
  return res.json();
}

async function loadData() {
  const [results, samples] = await Promise.allSettled([
    fetchJson('data/results.json'),
    fetchJson('data/samples.json'),
  ]);
  if (results.status === 'fulfilled') state.results = results.value;
  else state.resultsError = results.reason;
  if (samples.status === 'fulfilled') state.samples = Array.isArray(samples.value) ? samples.value : (samples.value.samples || []);
  else state.samplesError = samples.reason;
}

async function loadMinHours() {
  // MIN_HOURS lives in rr.js (owned by the parser). Optional: the page works without it.
  try {
    const rr = await import('./rr.js');
    if (Number.isFinite(rr.MIN_HOURS)) state.minHours = rr.MIN_HOURS;
  } catch (_) { /* rr.js unavailable: the requirement text stays generic */ }
}

// ---------------------------------------------------------------------------
// Routing

function currentRoute() {
  const h = (location.hash || '#/').replace(/^#\/?/, '').split(/[?/]/)[0];
  return h === '' ? 'start' : h;
}

function go(route) {
  const hash = route === 'start' ? '#/' : `#/${route}`;
  if (location.hash === hash) onRoute();
  else location.hash = hash;
}

let lastRoute = null;

function onRoute() {
  let route = currentRoute();
  if (!ROUTES.includes(route)) route = 'start';
  if (route === 'report' && !state.report) route = 'start';
  if (route === 'processing' && !state.job) route = state.report ? 'report' : 'start';
  if (route === 'problem' && !state.error) route = 'start';
  if (lastRoute === 'processing' && route !== 'processing' && state.job) cancelJob();

  const expected = route === 'start' ? ['', '#', '#/'] : [`#/${route}`];
  if (!expected.includes(location.hash)) {
    history.replaceState(null, '', route === 'start' ? '#/' : `#/${route}`);
  }

  for (const section of $$('[data-route]')) section.hidden = section.dataset.route !== route;
  for (const link of $$('[data-nav]')) {
    if (link.dataset.nav === route) link.setAttribute('aria-current', 'page');
    else link.removeAttribute('aria-current');
  }
  document.title = ROUTE_TITLES[route] || ROUTE_TITLES.start;

  if (route === 'report') renderReport();
  if (route === 'method') renderMethod();
  if (route === 'problem') renderProblem();
  $('#back-to-report').hidden = !state.report;

  if (lastRoute !== null && lastRoute !== route) {
    window.scrollTo(0, 0);
    const heading = $(`[data-route="${route}"] [tabindex="-1"]`);
    if (heading) heading.focus({ preventScroll: true });
  }
  lastRoute = route;
}

// ---------------------------------------------------------------------------
// Start screen: samples, upload, clock inputs

function renderSamples() {
  const grid = $('#sample-grid');
  if (!state.samples) {
    grid.innerHTML = `<p class="body-muted small">Samples could not be loaded${state.samplesError ? ` (${esc(state.samplesError.message)})` : ''}.</p>`;
    return;
  }
  grid.innerHTML = state.samples.map((s, i) => {
    const src = s.source || {};
    const lic = src.license ? (src.license_url ? `<a href="${esc(src.license_url)}" rel="noopener" target="_blank">${esc(src.license)}</a>` : esc(src.license)) : '';
    const pieces = [src.dataset, src.record ? `record ${src.record}` : ''].filter(Boolean).map(esc).concat(lic ? [lic] : []).join(', ');
    const link = src.url ? ` <a href="${esc(src.url)}" rel="noopener" target="_blank">source</a>` : '';
    return `<article class="card sample-card">
      <h3>${esc(s.title || s.id || 'Sample')}</h3>
      <p class="body-muted small">${esc(s.summary || '')}</p>
      <p class="caption">Source: ${pieces || 'unknown'}${link}${src.note ? `. ${esc(src.note)}` : ''}</p>
      <button type="button" class="btn btn-secondary btn-small sample-btn" data-sample="${i}" aria-label="Open sample: ${esc(s.title || s.id || '')}">Open this sample</button>
    </article>`;
  }).join('');
  for (const btn of $$('.sample-btn', grid)) {
    btn.addEventListener('click', () => openSample(Number(btn.dataset.sample)));
  }
}

function focusSamples() {
  go('start');
  requestAnimationFrame(() => {
    const section = $('#samples');
    section.scrollIntoView({ behavior: reducedMotion.matches ? 'auto' : 'smooth', block: 'start' });
    const first = $('.sample-btn', section);
    (first || section).focus({ preventScroll: true });
  });
}

async function openSample(index) {
  const s = state.samples && state.samples[index];
  if (!s) return;
  const clocks = readClocks();
  if (!clocks) return;
  const name = s.file ? s.file.split('/').pop() : s.id;
  const label = s.title ? `${s.title} (${name})` : name;
  // The reference labels load alongside the CSV; a failure only hides the comparison.
  const refPromise = s.reference
    ? fetchJson(`data/${s.reference}`).then((value) => ({ value }), (err) => ({ error: err.message }))
    : Promise.resolve({ value: null });
  try {
    const res = await fetch(`data/${s.file}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const text = await res.text();
    const ref = await refPromise;
    runAnalysis(text, label, clocks, { sample: s, reference: ref.value || null, referenceError: ref.error || null });
  } catch (err) {
    showProblem({ code: 'sample_unavailable', message: `The sample file could not be loaded (${err.message}).` }, label);
  }
}

async function openFile(file) {
  if (!file) return;
  const clocks = readClocks();
  if (!clocks) return;
  try {
    const text = await file.text();
    runAnalysis(text, file.name, clocks);
  } catch (err) {
    showProblem({ code: 'read_failed', message: `The file could not be read (${err.message}).` }, file.name);
  }
}

function parseClock(value) {
  if (!value) return null;
  const m = /^(\d{1,2}):(\d{2})(?::(\d{2}))?$/.exec(value);
  if (!m) return null;
  return Number(m[1]) * 3600 + Number(m[2]) * 60 + Number(m[3] || 0);
}

function setClockMessage(text) {
  const el = $('#clock-message');
  el.textContent = text || '';
  el.hidden = !text;
}

/**
 * Read the clock inputs. Returns {clockStartS, lightsOffS, lightsOnS} or null when
 * the run must not start. Offsets follow the contract: (clock - start) mod 24 h.
 */
function readClocks() {
  const start = parseClock($('#clock-start').value);
  const off = parseClock($('#clock-off').value);
  const on = parseClock($('#clock-on').value);
  if (start === null) {
    if (off !== null || on !== null) {
      const note = 'Lights off and lights on need the recording start time, so they were ignored. The whole recording is analysed.';
      setClockMessage(note);
      return { clockStartS: null, lightsOffS: null, lightsOnS: null, note };
    }
    setClockMessage('');
    return { clockStartS: null, lightsOffS: null, lightsOnS: null, note: '' };
  }
  const lightsOffS = off === null ? null : (((off - start) % DAY_S) + DAY_S) % DAY_S;
  const lightsOnS = on === null ? null : (((on - start) % DAY_S) + DAY_S) % DAY_S;
  if (lightsOffS !== null && lightsOnS !== null && lightsOnS <= lightsOffS) {
    setClockMessage('Lights on must come after lights off (within 24 hours of the recording start). Please check the times.');
    $('#clock-on').focus();
    return null;
  }
  setClockMessage('');
  return { clockStartS: start, lightsOffS, lightsOnS, note: '' };
}

function wireStart() {
  const input = $('#file-input');
  $('#btn-upload').addEventListener('click', () => input.click());
  $('#btn-drop-browse').addEventListener('click', () => input.click());
  $('#btn-problem-upload').addEventListener('click', () => { go('start'); input.click(); });
  $('#btn-samples').addEventListener('click', focusSamples);
  $('#btn-problem-samples').addEventListener('click', focusSamples);
  input.addEventListener('change', () => {
    const file = input.files && input.files[0];
    input.value = '';
    openFile(file);
  });
  for (const id of ['#clock-start', '#clock-off', '#clock-on']) {
    $(id).addEventListener('change', () => {
      const start = $('#clock-start').value;
      if (start || (!$('#clock-off').value && !$('#clock-on').value)) setClockMessage('');
    });
  }

  const zone = $('#drop-zone');
  let depth = 0;
  const hasFiles = (e) => e.dataTransfer && Array.from(e.dataTransfer.types || []).includes('Files');
  document.addEventListener('dragenter', (e) => {
    if (!hasFiles(e) || $('[data-route="start"]').hidden) return;
    depth++;
    zone.classList.add('is-dragover');
  });
  document.addEventListener('dragleave', () => {
    depth = Math.max(0, depth - 1);
    if (depth === 0) zone.classList.remove('is-dragover');
  });
  document.addEventListener('dragover', (e) => { if (hasFiles(e)) e.preventDefault(); });
  document.addEventListener('drop', (e) => {
    if (!hasFiles(e)) return;
    e.preventDefault();
    depth = 0;
    zone.classList.remove('is-dragover');
    if (!$('[data-route="start"]').hidden) openFile(e.dataTransfer.files[0]);
  });
}

// ---------------------------------------------------------------------------
// Worker orchestration

function createWorker() {
  if (USE_STUB_WORKER) return createStubWorker();
  return new Worker('js/worker.js', { type: 'module' });
}

// Text of the "features" progress step: v2 also computes 10-minute context windows.
const FEATURE_STEP_TEXT = {
  v2: 'Computing 30-second and 10-minute features',
  v1: 'Computing 30-second features',
};

function runAnalysis(csvText, fileName, clocks, origin = null, model = 'v2') {
  cancelJob();
  state.error = null;
  state.lastRun = { csvText, fileName, clocks, origin };
  const job = { fileName, clocks, origin, model, steps: {}, worker: null };
  state.job = job;
  for (const el of $$('[data-bind="job-file"]')) el.textContent = fileName;
  const featureStep = $('#steps li[data-step="features"] > span:first-child');
  if (featureStep) featureStep.textContent = FEATURE_STEP_TEXT[model] || FEATURE_STEP_TEXT.v2;
  resetSteps();
  go('processing');

  let worker;
  try {
    worker = createWorker();
  } catch (err) {
    failJob(job, { code: 'worker_failed', message: `The analysis could not start in this browser (${err.message}).` });
    return;
  }
  job.worker = worker;
  worker.onmessage = (ev) => onWorkerMessage(job, ev.data || {});
  worker.onerror = (ev) => {
    if (ev && ev.preventDefault) ev.preventDefault();
    failJob(job, { code: 'worker_failed', message: `The analysis stopped unexpectedly${ev && ev.message ? ` (${ev.message})` : ''}.` });
  };
  worker.onmessageerror = () => failJob(job, { code: 'worker_failed', message: 'The analysis returned unreadable data.' });
  worker.postMessage({
    type: 'run',
    csvText,
    fileName,
    lightsOffS: clocks.lightsOffS,
    lightsOnS: clocks.lightsOnS,
    model,
  });
}

/** Recompute the current report with the other model (same file, clock times and sample). */
function rerunWithModel(model) {
  const run = state.lastRun;
  if (!run || !state.report || state.report.model === model) return;
  const mode = state.mode;
  runAnalysis(run.csvText, run.fileName, run.clocks, run.origin, model);
  if (state.job) state.job.keepMode = mode;
}

function onWorkerMessage(job, msg) {
  if (state.job !== job) return;
  if (msg.type === 'progress') {
    job.steps[msg.step] = msg.status;
    updateSteps(job);
  } else if (msg.type === 'error') {
    failJob(job, msg);
  } else if (msg.type === 'result') {
    finishJob(job);
    state.report = msg.report;
    state.reportClockStartS = job.clocks.clockStartS;
    state.reportClockNote = job.clocks.note || '';
    state.reportIsStub = USE_STUB_WORKER;
    state.reportSample = (job.origin && job.origin.sample) || null;
    state.reportReference = (job.origin && job.origin.reference) || null;
    state.reportReferenceError = (job.origin && job.origin.referenceError) || null;
    state.mode = job.keepMode || 'sleep_wake';
    state.animateNext = true;
    go('report');
  }
}

function finishJob(job) {
  if (job.worker) job.worker.terminate();
  if (state.job === job) state.job = null;
}

function failJob(job, err) {
  if (state.job !== job) return;
  finishJob(job);
  showProblem(err, job.fileName);
}

function cancelJob() {
  if (state.job) {
    const job = state.job;
    state.job = null;
    if (job.worker) job.worker.terminate();
  }
}

function showProblem(err, fileName) {
  state.error = { ...err, fileName };
  go('problem');
}

function resetSteps() {
  for (const li of $$('#steps li')) {
    li.dataset.status = 'waiting';
    $('.step-status', li).textContent = 'Waiting';
  }
  setProgress(0);
}

function updateSteps(job) {
  let done = 0;
  for (const step of STEP_ORDER) {
    const li = $(`#steps li[data-step="${step}"]`);
    const status = job.steps[step];
    if (!li) continue;
    li.dataset.status = status === 'done' ? 'done' : status === 'running' ? 'running' : 'waiting';
    $('.step-status', li).textContent = status === 'done' ? 'Done' : status === 'running' ? 'In progress' : 'Waiting';
    if (status === 'done') done++;
  }
  setProgress(done);
}

function setProgress(done) {
  const bar = $('#progress');
  bar.setAttribute('aria-valuenow', String(done));
  bar.setAttribute('aria-valuetext', `${done} of ${STEP_ORDER.length} steps done`);
  $('#progress-fill').style.width = `${(100 * done) / STEP_ORDER.length}%`;
}

// ---------------------------------------------------------------------------
// Report screen

function unitHTML(value, unit) {
  return `${esc(value)}<span class="unit">${esc(unit)}</span>`;
}

function tstHTML(minutes) {
  if (!Number.isFinite(minutes)) return '—';
  const total = Math.round(minutes);
  const h = Math.floor(total / 60);
  const m = total % 60;
  return `${h}<span class="unit">h</span> ${m}<span class="unit">m</span>`;
}

function warningText(w) {
  if (typeof w === 'string') return w;
  if (w && typeof w === 'object') return w.message || w.text || w.code || JSON.stringify(w);
  return String(w);
}

function renderReport() {
  const r = state.report;
  if (!r) return;
  const m = r.metrics || {};
  const q = r.quality || {};
  for (const el of $$('[data-bind="report-file"]')) el.textContent = r.fileName || '';

  $('#tile-tst').innerHTML = tstHTML(m.tst_min);
  $('#tile-se').innerHTML = Number.isFinite(m.se_pct) ? unitHTML(fmtNum(m.se_pct, 0), '%') : '—';
  $('#tile-wake').textContent = Number.isFinite(m.awakenings) ? fmtNum(m.awakenings, 0) : '—';
  $('#tile-apnea').innerHTML = Number.isFinite(m.apnea_min) ? unitHTML(fmtNum(m.apnea_min, 0), 'min') : '—';
  const tib = Number.isFinite(m.tib_min) ? ` (${fmtHM(m.tib_min)} in bed)` : '';
  const grid = r.grid || {};
  const clock = state.reportClockStartS;
  let spanText = 'Time judged asleep between lights off and lights on';
  if (clock !== null && Number.isFinite(grid.start_s) && Number.isFinite(grid.end_s)) {
    spanText = `Time judged asleep between ${fmtClock(clock + grid.start_s)} and ${fmtClock(clock + grid.start_s + (grid.n_epochs || 0) * (grid.epoch_s || 30))}`;
  }
  $('#tile-tst-help').textContent = spanText + tib;

  // Data-quality banner.
  const banner = $('#quality-banner');
  const warnings = (q.warnings || []).map(warningText).filter(Boolean);
  const invalid = Number(q.invalid_pct);
  const showInvalid = Number.isFinite(invalid) && invalid > INVALID_WARN_PCT;
  if (showInvalid || warnings.length) {
    const headline = showInvalid ? `${Math.round(invalid)}% of beats were missing or unusable` : 'Check the data quality of this file';
    const extra = warnings.filter((w) => !(showInvalid && /missing or unusable/i.test(w)));
    banner.innerHTML = `<p class="banner-title">${esc(headline)}</p>` +
      `<p class="body-muted small">Results for this file may be unreliable. Stretches with too little usable data are drawn hatched on the chart; estimates there are less certain.</p>` +
      (extra.length ? `<ul class="bullets small">${extra.map((w) => `<li>${esc(w)}</li>`).join('')}</ul>` : '');
    banner.hidden = false;
  } else {
    banner.hidden = true;
    banner.innerHTML = '';
  }
  $('#stub-banner').hidden = !state.reportIsStub;

  const parts = [];
  if (Number.isFinite(q.duration_h)) parts.push(`${fmtNum(q.duration_h, 1)} h recorded`);
  if (Number.isFinite(q.n_beats)) parts.push(`${Number(q.n_beats).toLocaleString('en-US')} beats`);
  if (Number.isFinite(q.valid_fraction)) parts.push(`${fmtNum(100 * q.valid_fraction, 0)}% usable intervals`);
  const lowCount = (r.lowQuality || []).filter((v) => Number(v) === 1).length;
  if ((r.lowQuality || []).length) parts.push(`${lowCount} of ${r.lowQuality.length} epochs with not enough usable data`);
  $('#quality-line').textContent = (parts.length ? `Data quality: ${parts.join(' · ')}.` : '') +
    (state.reportClockNote ? ` ${state.reportClockNote}` : '');

  renderModelLine();
  syncModeButtons();
  renderChart();
  renderTrust();
}

function syncModeButtons() {
  for (const btn of $$('.seg[data-mode]')) btn.setAttribute('aria-pressed', String(btn.dataset.mode === state.mode));
}

function renderChart() {
  const r = state.report;
  const box = $('#hypnogram');
  if (!r || !box) return;
  const width = Math.max(280, Math.floor(box.clientWidth || 800));
  const animate = state.animateNext && !reducedMotion.matches;
  state.animateNext = false;
  box.innerHTML = hypnogramSvg({
    report: r,
    mode: state.mode,
    width,
    clockStartS: state.reportClockStartS,
    animate,
    idPrefix: `hyp-${state.mode}`,
  });

  const aligned = renderCompare(width, animate);

  const def = STAGE_MODES[state.mode];
  const items = Object.values(def.stages).map((s) => `<li><span class="swatch" style="background:${s.color}"></span>${esc(s.name === 'Light' ? 'Light sleep' : s.name === 'Deep' ? 'Deep sleep' : s.name)}</li>`);
  items.push(`<li><span class="swatch" style="background:${COLORS.apnea}"></span>${aligned ? 'Suspected apnea (estimate) / scored apnea or hypopnea (lab)' : 'Suspected apnea'}</li>`);
  items.push('<li><span class="swatch swatch-hatch"></span>not enough usable data</li>');
  if (aligned && aligned.stages[state.mode].some((v) => v === null)) {
    items.push('<li><span class="swatch swatch-gap"></span>gap: epoch not scored by the lab</li>');
  }
  $('#legend').innerHTML = items.join('');

  const note = $('#chart-note');
  let text = 'Stages are shown by row position and text labels as well as color.';
  if (state.mode !== 'sleep_wake') {
    const model = reportModel();
    const k = v2Metric('hmc', state.mode, 'primary', model, 'kappa_pooled');
    text += ' Experimental: separating sleep stages from heartbeats alone is much less accurate than wake versus sleep';
    text += k ? ` (${model} agreement on the ${datasetShort('hmc')} test set: kappa ${fmtNum(k.point, 2)}).` : '.';
  }
  if (clockStartIsKnown()) text += ' Times are clock times.';
  else text += ' Times are hours from the start of the analysed window.';
  note.textContent = text;
}

// ---------------------------------------------------------------------------
// Compare with the sleep lab (UCDDB samples only)

function compareHidden() {
  $('#series-model').hidden = true;
  $('#compare-lab').hidden = true;
  $('#compare-summary').hidden = true;
  $('#hypnogram-lab').innerHTML = '';
}

/** Draw the lab row and the agreement summary. Returns the aligned reference or null. */
function renderCompare(width, animate) {
  const r = state.report;
  const sample = state.reportSample;
  const note = $('#compare-unavailable');
  compareHidden();
  note.hidden = true;
  note.textContent = '';
  if (!r || !sample) return null; // uploads: no reference exists, the view is hidden
  if (!sample.has_stage_reference) {
    note.textContent = 'Compare with the sleep lab is not available for this sample: the Apnea-ECG database has no expert sleep-stage scoring, so there is no sleep lab hypnogram to compare with.';
    note.hidden = false;
    return null;
  }
  if (!state.reportReference) {
    note.textContent = `Compare with the sleep lab is not available: the reference labels could not be loaded${state.reportReferenceError ? ` (${state.reportReferenceError})` : ''}.`;
    note.hidden = false;
    return null;
  }
  const aligned = alignReference(state.reportReference, r);
  if (!aligned) {
    note.textContent = 'Compare with the sleep lab is not available for this analysis window: lights off must fall on a whole minute of the recording.';
    note.hidden = false;
    return null;
  }

  const labReport = {
    grid: r.grid,
    stages: { [state.mode]: aligned.stages[state.mode] },
    lowQuality: [],
    apnea: { minute_s: (r.apnea && r.apnea.minute_s) || 60, labels: aligned.apnea },
  };
  $('#hypnogram-lab').innerHTML = hypnogramSvg({
    report: labReport,
    mode: state.mode,
    width,
    clockStartS: state.reportClockStartS,
    animate,
    idPrefix: `lab-${state.mode}`,
    subject: 'Sleep lab reference hypnogram (polysomnography, scored by experts)',
    apneaTitle: 'Scored apnea or hypopnea',
    apneaPhrase: 'minutes with scored apnea or hypopnea',
  });
  $('#series-model').hidden = false;
  $('#compare-lab').hidden = false;

  const c = compareNight(r, aligned, state.mode);
  const modeName = state.mode === 'sleep_wake' ? 'wake / sleep' : STAGE_MODES[state.mode].title;
  const pair = (estimate, lab) => `<span class="compare-pair"><span class="who">Estimate</span><span class="val">${estimate}</span><span class="who">Sleep lab</span><span class="val">${lab}</span></span>`;
  const unscored = c.nEpochs - c.nScored;
  $('#compare-stats').innerHTML = `
    <div class="compare-stat">
      <dt>Epochs where both agree (${esc(modeName)})</dt>
      <dd class="compare-value">${Number.isFinite(c.agreementPct) ? `${fmtNum(c.agreementPct, 0)}<span class="unit">%</span>` : '—'}</dd>
      <dd class="help">${fmtNum(c.nScored, 0)} epochs of 30 s scored by the lab${unscored ? `; ${fmtNum(unscored, 0)} not scored are left out` : ''}.</dd>
    </div>
    <div class="compare-stat">
      <dt>Total sleep time</dt>
      <dd>${pair(esc(fmtHM(c.tstModelMin)), esc(fmtHM(c.tstLabMin)))}</dd>
      <dd class="help">Difference ${fmtNum(c.tstModelMin - c.tstLabMin, 0, { sign: true })} min (estimate minus lab).</dd>
    </div>
    <div class="compare-stat">
      <dt>Apnea minutes</dt>
      <dd>${pair(`${fmtNum(c.apneaModelMin, 0)} min`, `${fmtNum(c.apneaLabMin, 0)} min`)}</dd>
      <dd class="help">Estimate: suspected apnea minutes. Lab: minutes with a scored apnea or hypopnea.</dd>
    </div>`;
  const rec = sample.source && sample.source.record ? ` (UCDDB record ${sample.source.record})` : '';
  $('#compare-caption').textContent = `This is one night of the external test set${rec}, which the model never saw during training. ` +
    'The numbers are computed in your browser from the two charts above. One night says little on its own: ' +
    'the overall error over all test nights is in “How far can you trust this?” below.';
  $('#compare-summary').hidden = false;
  return aligned;
}

function clockStartIsKnown() {
  return state.reportClockStartS !== null;
}

function chartWidth(el, fallback = 560) {
  return Math.max(260, Math.floor((el && el.clientWidth) || fallback));
}

function noResults(el) {
  el.innerHTML = `<p class="body-muted small">Evaluation results could not be loaded${state.resultsError ? ` (${esc(state.resultsError.message)})` : ''}.</p>`;
}

// Rows of the v1 results table (UCDDB, first look).
const ERROR_ROWS = [
  { task: 'sleep_wake', label: 'Total sleep time error', unit: 'min', bias: 'tst_bias_min', mae: 'tst_mae_min', digits: 0 },
  { task: 'sleep_wake', label: 'Sleep efficiency error', unit: 'pct. points', bias: 'se_bias_pctpt', mae: 'se_mae_pctpt', digits: 1 },
  { task: 'apnea', label: 'Suspected apnea minutes error', unit: 'min', bias: 'apnea_min_bias', mae: 'apnea_min_mae', digits: 0 },
];

// Night-level error rows of the v2 test results (task sleep_wake).
const NIGHT_ROWS = [
  { label: 'Total sleep time error', short: 'total sleep time error', unit: 'min', bias: 'tst_bias_min', mae: 'tst_mae_min', digits: 0 },
  { label: 'Sleep efficiency error', short: 'sleep efficiency error', unit: 'pct. points', bias: 'se_bias_pctpt', mae: 'se_mae_pctpt', digits: 1 },
  { label: 'Wake bouts error', short: 'wake-bout error', unit: 'bouts', bias: 'awakenings_bias', mae: 'awakenings_mae', digits: 1 },
];

// Epoch agreement rows (kappa) of the v2 test results.
const KAPPA_TASK_ROWS = [
  { task: 'sleep_wake', label: 'Epoch agreement, wake/sleep', short: 'wake/sleep kappa' },
  { task: 'wake_nrem_rem', label: '3-stage agreement (experimental)', short: '3-stage kappa' },
  { task: 'four_class', label: '4-stage agreement (experimental)', short: '4-stage kappa' },
];

function valueInner(s, digits, opts = {}) {
  if (!s) return '—';
  const ci = ciText(s, digits, opts);
  return `<span class="cell-stack"><span class="cell-value">${fmtNum(s.point, digits, opts)}</span>${ci ? `<span class="cell-ci">95% CI ${ci}</span>` : ''}</span>`;
}

function reportModel() {
  const m = state.report && state.report.model;
  return MODEL_IDS.includes(m) ? m : 'v2';
}

function renderModelLine() {
  const r = state.report;
  const model = r && MODEL_IDS.includes(r.model) ? r.model : null;
  $('#model-label').textContent = model ? `Model: ${modelLabel(model)}` : 'Model: unknown';
  for (const btn of $$('.seg-model')) {
    btn.setAttribute('aria-pressed', String(btn.dataset.model === model));
    btn.disabled = !state.lastRun;
  }
}

function renderTrust() {
  const tableBox = $('#trust-table');
  const cmpBox = $('#trust-compare');
  const honestBox = $('#trust-honest');
  const otherBox = $('#trust-other');
  const baBox = $('#trust-ba');
  const barsBox = $('#trust-bars');
  renderLimitations($('#report-limitations'));
  if (!state.results || !v2Block()) {
    noResults(tableBox);
    for (const el of [cmpBox, honestBox, otherBox, baBox, barsBox]) el.innerHTML = '';
    $('#trust-bars-note').textContent = '';
    return;
  }
  const model = reportModel();
  const other = model === 'v2' ? 'v1' : 'v2';
  const base = v2StageBaseline();
  const hmc = v2Set('hmc') || {};
  const hmcName = datasetShort('hmc');
  const nP = hmc.records_primary;
  const nAll = hmc.records_all;
  $('#trust-intro').textContent = `Errors against hospital sleep studies (estimate minus reference) on ${nP ?? '—'} ${hmcName} recordings ` +
    `(primary analysis${Number.isFinite(nAll) && Number.isFinite(nP) && nAll !== nP ? `; ${nAll - nP} of ${nAll} left out for too little usable heartbeat data` : ''}). ` +
    `${hmcName} was never used before v2 was frozen, so it is a fresh test. Model behind this report: ${modelLabel(model)}.`;

  // Headline: the report's model on HMC primary.
  const rows = NIGHT_ROWS.map((row) => {
    const bias = v2Metric('hmc', 'sleep_wake', 'primary', model, row.bias);
    const mae = v2Metric('hmc', 'sleep_wake', 'primary', model, row.mae);
    return `<tr><th scope="row">${esc(row.label)} <span class="unit-note">(${esc(row.unit)})</span></th>` +
      `<td data-label="Bias" class="num">${valueInner(bias, row.digits, { sign: true })}</td>` +
      `<td data-label="Mean absolute error" class="num">${valueInner(mae, row.digits)}</td></tr>`;
  });
  const kappa = v2Metric('hmc', 'sleep_wake', 'primary', model, 'kappa_pooled');
  rows.push(`<tr><th scope="row">Epoch agreement, wake/sleep <span class="unit-note">(kappa)</span></th>` +
    `<td data-label="Kappa" class="num" colspan="2">${valueInner(kappa, 2)}</td></tr>`);
  tableBox.innerHTML = `<div class="table-wrap"><table class="data-table stack-on-mobile">
    <caption class="visually-hidden">${esc(modelLabel(model))} on ${esc(hmcName)}, primary analysis</caption>
    <thead><tr><th scope="col">Metric</th><th scope="col">Bias (mean error)</th><th scope="col">Mean absolute error</th></tr></thead>
    <tbody>${rows.join('')}</tbody></table></div>`;

  // Compact comparison on the same nights: v2, v1, trivial rule.
  const cols = [...MODEL_IDS, base];
  const colName = (c) => (c === base ? 'Trivial rule' : c);
  const cmpRow = (label, get, digits) => `<tr><th scope="row">${esc(label)}</th>${cols.map((c) => {
    const s = get(c);
    return `<td data-label="${esc(colName(c))}" class="num${c === model ? ' is-current' : ''}">${valueInner(s, digits)}</td>`;
  }).join('')}</tr>`;
  const cmpRows = NIGHT_ROWS.map((row) => cmpRow(`${row.label}, mean absolute (${row.unit})`, (c) => v2Metric('hmc', 'sleep_wake', 'primary', c, row.mae), row.digits))
    .concat(KAPPA_TASK_ROWS.map((row) => cmpRow(`${row.label} (kappa)`, (c) => v2Metric('hmc', row.task, 'primary', c, 'kappa_pooled'), 2)));
  cmpBox.innerHTML = `<div class="table-wrap"><table class="data-table compare-table stack-on-mobile">
    <caption class="visually-hidden">v2, v1 and the trivial rule on the same ${esc(hmcName)} nights</caption>
    <thead><tr><th scope="col">Metric</th>${cols.map((c) => `<th scope="col">${esc(colName(c))}</th>`).join('')}</tr></thead>
    <tbody>${cmpRows.join('')}</tbody></table></div>`;

  honestBox.innerHTML = honestySummary();
  otherBox.innerHTML = otherEvidence();

  baBox.innerHTML = v2TstBlandAltman(chartWidth(baBox), 'hmc');
  $('#trust-ba-title').textContent = `Total sleep time: v2 estimate vs. sleep study (${hmcName}, primary analysis)`;

  const items = [];
  const barColors = { v2: COLORS.accent, v1: COLORS.muted };
  for (const c of cols) {
    const s = v2Metric('hmc', 'sleep_wake', 'primary', c, 'tst_mae_min');
    if (s) items.push({ label: c === base ? 'Trivial rule' : c, value: s.point, ciLow: s.ci_low, ciHigh: s.ci_high, color: barColors[c] || COLORS.hatch });
  }
  barsBox.innerHTML = items.length
    ? comparisonBarsSvg({ items, width: chartWidth(barsBox), unit: 'min', digits: 0, title: `Mean absolute error of total sleep time on ${hmcName}` })
    : '<p class="body-muted small">No comparison available.</p>';
  $('#trust-bars-note').textContent = items.length
    ? `Mean absolute error of total sleep time on the same ${hmcName} nights; whiskers show the 95% interval. The trivial rule counts every epoch from lights off to lights on as sleep.`
    : '';
}

/** Plain statement of where v2 improved on HMC, where it did not, and where it got worse (from the 95% intervals). */
function honestySummary() {
  const hmcName = datasetShort('hmc');
  const better = [];
  const notBetter = [];
  const worse = [];
  for (const row of KAPPA_TASK_ROWS) {
    const a = v2Metric('hmc', row.task, 'primary', 'v2', 'kappa_pooled');
    const b = v2Metric('hmc', row.task, 'primary', 'v1', 'kappa_pooled');
    const cmp = compareStats(a, b, true);
    if (!cmp) continue;
    const text = `${row.short} ${fmtNum(b.point, 2)} → ${fmtNum(a.point, 2)}`;
    if (cmp === 'better') better.push(text);
    else if (cmp === 'worse') worse.push(text);
    else notBetter.push(`${text} (${a.point > b.point ? 'higher, but the 95% intervals overlap' : 'the 95% intervals overlap'})`);
  }
  // Night-level errors, plus the 3-stage REM-minute error (where v2 overestimates REM).
  const errorRows = NIGHT_ROWS.map((row) => ({ task: 'sleep_wake', key: row.mae, short: row.short, unit: row.unit, digits: row.digits }))
    .concat([{ task: 'wake_nrem_rem', key: 'minutes_mae_rem', short: 'REM minutes error (3-stage)', unit: 'min', digits: 0 }]);
  for (const row of errorRows) {
    const a = v2Metric('hmc', row.task, 'primary', 'v2', row.key);
    const b = v2Metric('hmc', row.task, 'primary', 'v1', row.key);
    const cmp = compareStats(a, b, false);
    if (!cmp) continue;
    const text = `${row.short} ${fmtNum(b.point, row.digits)} → ${fmtNum(a.point, row.digits)} ${row.unit}`;
    if (cmp === 'better') better.push(text);
    else if (cmp === 'worse') worse.push(text);
    else notBetter.push(`${text} (not clearly different: the 95% intervals overlap)`);
  }
  const parts = [];
  if (better.length) parts.push(`<p><strong>Better with v2</strong> on ${esc(hmcName)} (v1 → v2, 95% intervals do not overlap): ${esc(better.join('; '))}.</p>`);
  if (notBetter.length) parts.push(`<p><strong>Not improved by v2</strong> on ${esc(hmcName)} (v1 → v2): ${esc(notBetter.join('; '))}.</p>`);
  if (worse.length) parts.push(`<p><strong>Worse with v2</strong> on ${esc(hmcName)} (v1 → v2, 95% intervals do not overlap): ${esc(worse.join('; '))}.</p>`);
  return parts.length ? `<div class="honest-box">${parts.join('')}</div>` : '';
}

/** One line for UCDDB (second look) and one for apnea (UCDDB only). */
function otherEvidence() {
  const v = v2Block();
  const ucd = v2Set('ucddb') || {};
  const name = datasetShort('ucddb');
  const lines = [];
  const pair = (task, key, digits, higherIsBetter) => {
    const a = v2Metric('ucddb', task, 'primary', 'v2', key);
    const b = v2Metric('ucddb', task, 'primary', 'v1', key);
    if (!a || !b) return null;
    const cmp = compareStats(a, b, higherIsBetter);
    const note = cmp === 'better' ? 'better with v2' : cmp === 'worse' ? 'worse with v2' : 'intervals overlap';
    return { text: `v1 ${fmtNum(b.point, digits)} → v2 ${fmtNum(a.point, digits)}`, note };
  };
  const k = pair('sleep_wake', 'kappa_pooled', 2, true);
  const tst = pair('sleep_wake', 'tst_mae_min', 0, false);
  const wb = pair('sleep_wake', 'awakenings_mae', 1, false);
  if (k || tst || wb) {
    const bits = [
      k && `wake/sleep kappa ${k.text} (${k.note})`,
      tst && `total sleep time error ${tst.text} min (${tst.note})`,
      wb && `wake-bout error ${wb.text} bouts (${wb.note})`,
    ].filter(Boolean);
    lines.push(`<li><strong>${esc(name)}, second look</strong> (${esc(String(ucd.records_primary ?? '—'))} nights; v1 was first tested there, so these numbers are less independent): ${esc(bits.join('; '))}.</li>`);
  }
  const a2 = v2ApneaMetric('v2', 'apnea_min_mae');
  const a1 = v2ApneaMetric('v1', 'apnea_min_mae');
  const a0 = v2ApneaMetric(v2ApneaBaseline(), 'apnea_min_mae');
  const k2 = v2ApneaMetric('v2', 'kappa_pooled');
  const k1 = v2ApneaMetric('v1', 'kappa_pooled');
  const s2 = v2ApneaMetric('v2', 'sensitivity');
  const s1 = v2ApneaMetric('v1', 'sensitivity');
  const bias2 = v2ApneaMetric('v2', 'apnea_min_bias');
  if (a2 && a1) {
    lines.push(`<li><strong>Suspected apnea minutes</strong> (tested on ${esc(name)} only): mean absolute error v1 ${fmtNum(a1.point, 0)} → v2 ${fmtNum(a2.point, 0)} min (95% CI ${esc(ciText(a2, 0))})` +
      `${a0 ? `, trivial rule (${esc(prettyName(v2ApneaBaseline()).toLowerCase())}) ${fmtNum(a0.point, 0)} min` : ''}` +
      `${k1 && k2 ? `; minute agreement (kappa) ${fmtNum(k1.point, 2)} → ${fmtNum(k2.point, 2)}` : ''}` +
      `${s1 && s2 ? `; share of apnea minutes found (sensitivity) ${fmtNum(s1.point, 2)} → ${fmtNum(s2.point, 2)}` : ''}` +
      `${bias2 ? `; v2 bias ${fmtNum(bias2.point, 0, { sign: true })} min${bias2.point < 0 ? ' (v2 reports fewer apnea minutes than the reference on average)' : ''}` : ''}.</li>`);
  }
  return lines.length && v ? `<ul class="bullets small">${lines.join('')}</ul>` : '';
}

function v2TstBlandAltman(width, ds) {
  const v = v2Block();
  const pr = (v && v.per_record && v.per_record[`${ds}_primary_sleep_wake`]) || [];
  const e = v2Entry(ds, 'sleep_wake', 'primary', 'v2');
  return blandAltmanSvg({
    points: pr.map((p) => ({ ref: Number(p.ref_tst_min), pred: Number(p.pred_tst_min), record: p.record })),
    stats: (e && e.bland_altman && e.bland_altman.tst_min) || null,
    width,
    unit: 'min',
    quantity: 'total sleep time',
    digits: 0,
  });
}

function tstBlandAltman(width, model) {
  const pr = (state.results.per_record && state.results.per_record.sleep_wake) || [];
  const ext = externalOf('sleep_wake', model);
  const ba = ext && ext.bland_altman && ext.bland_altman.tst_min;
  return blandAltmanSvg({
    points: pr.map((p) => ({ ref: Number(p.ref_tst_min), pred: Number(p.pred_tst_min), record: p.record })),
    stats: ba || null,
    width,
    unit: 'min',
    quantity: 'total sleep time',
    digits: 0,
  });
}

function renderLimitations(ul) {
  const v = v2Block();
  const src = v && Array.isArray(v.limitations) ? v.limitations : (state.results && state.results.limitations);
  const lims = Array.isArray(src) ? src : [];
  ul.innerHTML = lims.length
    ? lims.map((l) => `<li>${esc(typeof l === 'string' ? l : warningText(l))}</li>`).join('')
    : '<li>Limitations could not be loaded.</li>';
}

// ---------------------------------------------------------------------------
// Method & evaluation screen

const KAPPA_ROWS = [
  { task: 'sleep_wake', label: 'Epoch agreement (kappa), wake/sleep' },
  { task: 'wake_nrem_rem', label: '3-stage kappa (experimental)' },
  { task: 'four_class', label: '4-stage kappa (experimental)' },
  { task: 'apnea', label: 'Apnea-minute agreement (kappa)' },
];

function methodRows(model) {
  const rows = [];
  for (const row of ERROR_ROWS) {
    rows.push({ label: `${row.label} (${row.unit}), mean absolute error`, m: metricOf(row.task, model, row.mae), b: baselineMetric(row.task, row.mae), digits: row.digits, sign: false, task: row.task });
    rows.push({ label: `${row.label} (${row.unit}), bias`, m: metricOf(row.task, model, row.bias), b: baselineMetric(row.task, row.bias), digits: row.digits, sign: true, task: row.task });
  }
  for (const row of KAPPA_ROWS) {
    if (!taskOf(row.task)) continue;
    rows.push({ label: row.label, m: metricOf(row.task, model, 'kappa_pooled'), b: baselineMetric(row.task, 'kappa_pooled'), digits: 2, sign: false, task: row.task });
  }
  return rows;
}

function renderMethod() {
  renderLimitations($('#method-limitations'));
  for (const id of ['#method-datasets', '#method-results', '#method-rpeak', '#method-ba-tst', '#method-ba-apnea', '#method-v2-story', '#method-v2-results', '#method-v2-apnea', '#method-v2-excluded', '#method-v2-ba-tst', '#method-v2-ba-apnea']) {
    if (!state.results) noResults($(id));
  }
  if (!state.results) return;

  // Data table.
  const ds = Array.isArray(state.results.datasets) ? state.results.datasets : [];
  const roleOf = (d, v) => (d.roles && d.roles[v]) || (v === 'v1' ? d.role || '' : '');
  $('#method-datasets').innerHTML = ds.length ? `<div class="table-wrap"><table class="data-table stack-on-mobile">
    <thead><tr><th scope="col">Dataset</th><th scope="col">Role for v1</th><th scope="col">Role for v2</th><th scope="col">Records / subjects</th><th scope="col">Population</th><th scope="col">License</th></tr></thead>
    <tbody>${ds.map((d) => `<tr>
      <th scope="row">${d.url ? `<a href="${esc(d.url)}" rel="noopener" target="_blank">${esc(d.short || d.name)}</a>` : esc(d.short || d.name)}${d.short && d.name && d.short !== d.name ? `<span class="cell-ci">${esc(d.name)}</span>` : ''}</th>
      <td data-label="Role for v1">${esc(roleOf(d, 'v1'))}</td>
      <td data-label="Role for v2">${esc(roleOf(d, 'v2'))}</td>
      <td data-label="Records / subjects" class="num">${esc(recordsText(d))}</td>
      <td data-label="Population">${esc(d.population || '')}</td>
      <td data-label="License">${d.license_url ? `<a href="${esc(d.license_url)}" rel="noopener" target="_blank">${esc(d.license_short || d.license || 'license')}</a>` : esc(d.license || '')}</td></tr>`).join('')}</tbody></table></div>` : '<p class="body-muted small">No dataset information.</p>';

  renderMethodV2();

  // v1 results table (UCDDB, first look): primary model group, then the other models.
  const models = modelNames('sleep_wake');
  const swBase = baselineName('sleep_wake');
  const apBase = baselineName('apnea');
  const n = nTestRecords('sleep_wake');
  const baseNote = [swBase && `stages: ${prettyName(swBase).toLowerCase()}`, apBase && `apnea: ${prettyName(apBase).toLowerCase()}`].filter(Boolean).join('; ');
  $('#method-results-intro').textContent = `v1, first look: external test on UCDDB${n !== null ? ` (${n} recordings)` : ''}, the first and only test of v1 before v2 existed. Errors are estimate minus reference. Lower error and higher kappa are better. Trivial rule${baseNote ? ` (${baseNote})` : ''} is scored on the same epochs.`;
  const groups = models.map((model, gi) => {
    const rows = methodRows(model).map((row) => `<tr>
      <th scope="row">${esc(row.label)}</th>
      <td data-label="${esc(prettyName(model))}" class="num">${row.m ? fmtNum(row.m.point, row.digits, { sign: row.sign }) : '—'}</td>
      <td data-label="Trivial rule" class="num">${row.b ? fmtNum(row.b.point, row.digits, { sign: row.sign }) : '—'}</td>
      <td data-label="95% interval (model)" class="num">${row.m ? (ciText(row.m, row.digits, { sign: row.sign }) || '—') : '—'}</td></tr>`).join('');
    const title = gi === 0 ? `v1: ${prettyName(model)} (primary model)` : `v1: ${prettyName(model)} (shown alongside)`;
    return `<tbody class="${gi === 0 ? 'group-primary' : 'group-secondary'}"><tr class="group-row"><th scope="rowgroup" colspan="4">${esc(title)}</th></tr>${rows}</tbody>`;
  }).join('');
  $('#method-results').innerHTML = `<div class="table-wrap"><table class="data-table results-table stack-on-mobile">
    <thead><tr><th scope="col">Metric</th><th scope="col">Model</th><th scope="col">Trivial rule</th><th scope="col">95% interval (model)</th></tr></thead>
    ${groups}</table></div>`;

  const model = primaryModel();
  const tstBox = $('#method-ba-tst');
  tstBox.innerHTML = tstBlandAltman(chartWidth(tstBox), model);
  const apBox = $('#method-ba-apnea');
  const prA = (state.results.per_record && state.results.per_record.apnea) || [];
  const extA = externalOf('apnea', model);
  apBox.innerHTML = blandAltmanSvg({
    points: prA.map((p) => ({ ref: Number(p.ref_apnea_min), pred: Number(p.pred_apnea_min), record: p.record })),
    stats: (extA && extA.bland_altman && extA.bland_altman.apnea_min) || null,
    width: chartWidth(apBox),
    unit: 'min',
    quantity: 'suspected apnea minutes',
    digits: 0,
  });

  renderRpeak();
}

// v2 views of the method results table: test set and analysis.
function v2Views() {
  const hmc = v2Set('hmc') || {};
  const ucd = v2Set('ucddb') || {};
  const h = datasetShort('hmc');
  const u = datasetShort('ucddb');
  return [
    { id: 'hmc:primary', ds: 'hmc', analysis: 'primary', label: `${h}, primary (${hmc.records_primary ?? '—'})`, short: `${h} primary` },
    { id: 'hmc:all_records', ds: 'hmc', analysis: 'all_records', label: `${h}, all records (${hmc.records_all ?? '—'})`, short: `${h} all` },
    { id: 'ucddb:primary', ds: 'ucddb', analysis: 'primary', label: `${u}, second look (${ucd.records_primary ?? '—'})`, short: u },
  ];
}

const V2_TABLE_ROWS = [
  ...NIGHT_ROWS.flatMap((row) => [
    { task: 'sleep_wake', key: row.mae, label: `${row.label} (${row.unit}), mean absolute`, digits: row.digits, sign: false, higher: false },
    { task: 'sleep_wake', key: row.bias, label: `${row.label} (${row.unit}), bias`, digits: row.digits, sign: true, higher: null },
  ]),
  ...KAPPA_TASK_ROWS.map((row) => ({ task: row.task, key: 'kappa_pooled', label: `${row.label} (kappa)`, digits: 2, sign: false, higher: true })),
  { task: 'wake_nrem_rem', key: 'minutes_mae_rem', label: 'REM minutes error, 3-stage (min), mean absolute', digits: 0, sign: false, higher: false },
  { task: 'wake_nrem_rem', key: 'minutes_bias_rem', label: 'REM minutes error, 3-stage (min), bias', digits: 0, sign: true, higher: null },
];

function renderMethodV2() {
  const v = v2Block();
  const ids = ['#method-v2-story', '#method-v2-results', '#method-v2-apnea', '#method-v2-excluded', '#method-v2-ba-tst', '#method-v2-ba-apnea'];
  if (!v) {
    for (const id of ids) $(id).innerHTML = '<p class="body-muted small">v2 results could not be loaded.</p>';
    return;
  }
  const sel = v.selection || {};
  const st = sel.staging || {};
  const ap = sel.apnea || {};
  const folds = sel.folds || {};
  const hmc = v2Set('hmc') || {};
  const hName = datasetShort('hmc');
  const uName = datasetShort('ucddb');
  const trainNames = (list) => (list || []).map(datasetShort).join(' and ');
  const story = [];
  story.push(`Candidates were compared on training data only (stages: ${esc(trainNames(sel.training_datasets_only && sel.training_datasets_only.staging))}; apnea: ${esc(trainNames(sel.training_datasets_only && sel.training_datasets_only.apnea))}) with cross-dataset tests: train on one database, test on the other, then swap${folds.staging ? ` (${esc(folds.staging.a)}; ${esc(folds.staging.b)})` : ''}. No test set was looked at.`);
  if (st.chosen && st.v1_reference) {
    const guard = /<=\s*([0-9.]+)\s*x/.exec((st.rule && st.rule.guard) || '');
    const guardText = guard ? `at most ${fmtNum(100 * (Number(guard[1]) - 1), 0)}% worse than v1's` : "not much worse than v1's";
    story.push(`Sleep staging: ${esc(String(st.n_candidates ?? '—'))} candidates; the rule picks the highest mean cross-dataset wake/sleep kappa among candidates whose total sleep time error is ${guardText}, preferring the simpler model on near-ties. Chosen: <code>${esc(st.chosen.name)}</code>. Cross-dataset kappa: v1 reference ${fmtNum(st.v1_reference.xds_sw_kappa, 2)}, chosen ${fmtNum(st.chosen.xds_sw_kappa, 2)}.`);
  }
  if (ap.chosen && ap.v1_reference) {
    story.push(`Apnea minutes: ${esc(String(ap.n_candidates ?? '—'))} candidates, same kind of rule on minute kappa. Chosen: <code>${esc(ap.chosen.name)}</code>. Cross-dataset kappa: v1 reference ${fmtNum(ap.v1_reference.xds_kappa, 2)}, chosen ${fmtNum(ap.chosen.xds_kappa, 2)}.`);
  }
  const fz = v.frozen || {};
  story.push(`The choice was frozen (git tag <code>${esc(fz.tag || 'v2-frozen')}</code>${fz.date ? `, ${esc(fz.date)}` : ''}) before any test-set prediction. Then v2 and v1 predicted the test sets once, side by side, with the trivial rule scored on the same epochs.`);
  story.push(`${esc(hName)} (${esc(String(hmc.records_all ?? '—'))} recordings) had never been used and is the main test. ${esc(v.exclusion_rule && v.exclusion_rule.text ? v.exclusion_rule.text : '')}`);
  story.push(`${esc(uName)} is a second look: v1 was already evaluated on it before v2 was built, so it is less independent. Apnea is tested on ${esc(uName)} only.`);
  $('#method-v2-story').innerHTML = `<ul class="bullets">${story.map((s) => `<li>${s}</li>`).join('')}</ul>`;

  // View switch + results table.
  const views = v2Views();
  if (!views.some((x) => x.id === state.v2View)) state.v2View = views[0].id;
  const view = views.find((x) => x.id === state.v2View);
  const base = v2StageBaseline();
  const cols = [...MODEL_IDS, base];
  const colName = (c) => (c === base ? 'Trivial rule' : c);
  const seg = `<div class="segmented" role="group" aria-label="Test set shown in the v2 table">${views.map((x) => `<button type="button" class="seg seg-v2view" data-view="${esc(x.id)}" aria-pressed="${x.id === view.id}" aria-label="${esc(x.label)}"><span class="lbl-long">${esc(x.label)}</span><span class="lbl-short">${esc(x.short)}</span></button>`).join('')}</div>`;
  const body = V2_TABLE_ROWS.map((row) => {
    const vals = cols.map((c) => v2Metric(view.ds, row.task, view.analysis, c, row.key));
    const cmp = row.higher === null ? null : compareStats(vals[0], vals[1], row.higher);
    const cmpText = cmp === 'better' ? 'v2 better' : cmp === 'worse' ? 'v2 worse' : cmp === 'overlap' ? 'intervals overlap' : '—';
    return `<tr><th scope="row">${esc(row.label)}</th>${cols.map((c, i) => `<td data-label="${esc(colName(c))}" class="num">${valueInner(vals[i], row.digits, { sign: row.sign })}</td>`).join('')}` +
      `<td data-label="v2 vs v1">${esc(cmpText)}</td></tr>`;
  }).join('');
  const e2 = v2Entry(view.ds, 'sleep_wake', view.analysis, 'v2');
  $('#method-v2-results').innerHTML = `${seg}
    <p class="body-muted small">${esc(view.label.replace(/\s*\(.*\)$/, ''))}: ${esc(String(e2 && e2.n_records != null ? e2.n_records : '—'))} recordings, ${esc(e2 && e2.n_epochs != null ? Number(e2.n_epochs).toLocaleString('en-US') : '—')} epochs of 30 s. Errors are estimate minus reference, with record-level bootstrap 95% intervals. “v2 vs v1” says whether the two 95% intervals overlap.</p>
    <div class="table-wrap"><table class="data-table stack-on-mobile">
    <thead><tr><th scope="col">Metric</th>${cols.map((c) => `<th scope="col">${esc(colName(c))}</th>`).join('')}<th scope="col">v2 vs v1</th></tr></thead>
    <tbody>${body}</tbody></table></div>`;
  for (const btn of $$('.seg-v2view')) {
    btn.addEventListener('click', () => {
      if (state.v2View === btn.dataset.view) return;
      state.v2View = btn.dataset.view;
      renderMethodV2();
    });
  }

  // Apnea (UCDDB only).
  const aBase = v2ApneaBaseline();
  const aCols = [...MODEL_IDS, aBase];
  const aName = (c) => (c === aBase ? 'Trivial rule' : c);
  const aRows = [
    { key: 'apnea_min_mae', label: 'Suspected apnea minutes error (min), mean absolute', digits: 0, sign: false, higher: false },
    { key: 'apnea_min_bias', label: 'Suspected apnea minutes error (min), bias', digits: 0, sign: true, higher: null },
    { key: 'kappa_pooled', label: 'Apnea-minute agreement (kappa)', digits: 2, sign: false, higher: true },
    { key: 'sensitivity', label: 'Share of apnea minutes found (sensitivity)', digits: 2, sign: false, higher: true },
  ];
  const a2 = v2Apnea('v2');
  $('#method-v2-apnea').innerHTML = `<p class="body-muted small">Apnea minutes on ${esc(uName)} (${esc(String(a2 && a2.n_records != null ? a2.n_records : '—'))} nights, second look); ${esc(hName)} was not used for apnea. Trivial rule: ${esc(prettyName(aBase).toLowerCase())}.</p>
    <div class="table-wrap"><table class="data-table stack-on-mobile">
    <thead><tr><th scope="col">Metric</th>${aCols.map((c) => `<th scope="col">${esc(aName(c))}</th>`).join('')}<th scope="col">v2 vs v1</th></tr></thead>
    <tbody>${aRows.map((row) => {
      const vals = aCols.map((c) => v2ApneaMetric(c, row.key));
      const cmp = row.higher === null ? null : compareStats(vals[0], vals[1], row.higher);
      const cmpText = cmp === 'better' ? 'v2 better' : cmp === 'worse' ? 'v2 worse' : cmp === 'overlap' ? 'intervals overlap' : '—';
      return `<tr><th scope="row">${esc(row.label)}</th>${aCols.map((c, i) => `<td data-label="${esc(aName(c))}" class="num">${valueInner(vals[i], row.digits, { sign: row.sign })}</td>`).join('')}<td data-label="v2 vs v1">${esc(cmpText)}</td></tr>`;
    }).join('')}</tbody></table></div>`;

  // Excluded HMC records.
  const exc = Array.isArray(hmc.excluded_low_coverage) ? hmc.excluded_low_coverage : [];
  $('#method-v2-excluded').innerHTML = exc.length
    ? `<p class="body-muted small">Left out of the ${esc(hName)} primary analysis (${exc.length} of ${esc(String(hmc.records_all ?? '—'))}) because too few epochs had enough usable heartbeats; they are included in “all records”:</p>
      <ul class="bullets small">${exc.map((x) => `<li>${esc(x.record)}: ${fmtNum(100 * Number(x.adequate_fraction), 0)}% of epochs usable</li>`).join('')}</ul>`
    : `<p class="body-muted small">No ${esc(hName)} recording was left out.</p>`;

  const baT = $('#method-v2-ba-tst');
  baT.innerHTML = v2TstBlandAltman(chartWidth(baT), 'hmc');
  const baA = $('#method-v2-ba-apnea');
  const prA = (v.per_record && v.per_record.ucddb_primary_apnea) || [];
  baA.innerHTML = blandAltmanSvg({
    points: prA.map((p) => ({ ref: Number(p.ref_apnea_min), pred: Number(p.pred_apnea_min), record: p.record })),
    stats: (a2 && a2.bland_altman && a2.bland_altman.apnea_min) || null,
    width: chartWidth(baA),
    unit: 'min',
    quantity: 'suspected apnea minutes',
    digits: 0,
  });
}

function recordsText(d) {
  const r = d.records;
  const s = d.subjects;
  if (r !== undefined && s !== undefined && r !== null && s !== null) return `${r} / ${s}`;
  if (r !== undefined && r !== null) return String(r);
  if (s !== undefined && s !== null) return String(s);
  return '';
}

function renderRpeak() {
  const box = $('#method-rpeak');
  const rp = state.results.rpeak_validation;
  if (!rp) { box.innerHTML = '<p class="body-muted small">No detector validation data.</p>'; return; }
  const sets = rp.datasets && typeof rp.datasets === 'object' ? rp.datasets : { pooled: rp };
  const pctCell = (v) => (Number.isFinite(Number(v)) ? `${fmtNum(100 * Number(v), 2)}%` : '—');
  const rows = Object.entries(sets).map(([name, d]) => `<tr>
    <th scope="row">${esc(datasetLabel(name))}</th>
    <td data-label="Reference">${esc(d.reference || '')}</td>
    <td data-label="Records" class="num">${esc(d.records ?? '')}</td>
    <td data-label="Sensitivity" class="num">${pctCell(d.pooled_sensitivity)}</td>
    <td data-label="Positive predictive value" class="num">${pctCell(d.pooled_ppv)}</td>
    <td data-label="Lowest record (sens. / PPV)" class="num">${pctCell(d.min_record_sensitivity)} / ${pctCell(d.min_record_ppv)}</td></tr>`).join('');
  const tol = Number(rp.tolerance_ms);
  box.innerHTML = `<p class="body-muted small">Detected R-peaks compared with reference beat annotations${Number.isFinite(tol) ? `, matched within ${fmtNum(tol, 0)} ms` : ''}. Pooled over all beats.</p>
    <div class="table-wrap"><table class="data-table stack-on-mobile">
    <thead><tr><th scope="col">Dataset</th><th scope="col">Reference</th><th scope="col">Records</th><th scope="col">Sensitivity</th><th scope="col">Positive predictive value</th><th scope="col">Lowest record (sens. / PPV)</th></tr></thead>
    <tbody>${rows}</tbody></table></div>`;
}

function datasetLabel(key) {
  const ds = (state.results && state.results.datasets) || [];
  const k = String(key).toLowerCase();
  const hit = ds.find((d) => d.id === key || String(d.short || '').toLowerCase() === k);
  if (hit) return hit.short || hit.name;
  const known = { slpdb: 'MIT-BIH Polysomnographic', 'apnea-ecg': 'Apnea-ECG', ucddb: 'UCDDB', cap: 'CAP Sleep Database' };
  return known[k] || key;
}

// ---------------------------------------------------------------------------
// Problem screen

function renderProblem() {
  const e = state.error;
  if (!e) return;
  const name = e.fileName ? `File: ${e.fileName}. ` : '';
  let reason = e.message || 'The file could not be processed.';
  if (typeof e.details === 'string' && e.details) reason += ` ${e.details}`;
  $('#problem-reason').textContent = name + reason;

  // Failures of the app itself (worker, models, unexpected errors) are not a problem with the file.
  const appFailure = ['worker_failed', 'MODELS_UNAVAILABLE', 'INTERNAL', 'sample_unavailable'].includes(e.code);
  $('.problem-label').textContent = appFailure ? 'Analysis problem' : 'File problem';
  $('#problem-title').textContent = appFailure ? 'The analysis could not run' : "We couldn't use this file";
  document.title = appFailure ? 'Analysis problem · Somnibeat' : ROUTE_TITLES.problem;

  const hours = state.minHours ?? (e.details && Number.isFinite(e.details.min_hours) ? e.details.min_hours : null);
  $('#problem-min-hours').textContent = hours !== null
    ? `The recording needs at least ${hours} hours of beats to make a report.`
    : 'The recording needs enough hours of beats to make a report.';
  renderExample();
}

let exampleLoaded = false;

async function renderExample() {
  if (exampleLoaded) return;
  const s = state.samples && state.samples[0];
  if (!s || !s.file) return;
  try {
    const res = await fetch(`data/${s.file}`);
    if (!res.ok) return;
    const text = await res.text();
    const lines = text.split(/\r?\n/).filter((l) => l.trim() !== '').slice(0, 5);
    if (lines.length < 2) return;
    $('#problem-example').innerHTML = `<code>${esc(lines.join('\n'))}\n…</code>`;
    $('#problem-example-caption').textContent = `Expected format (first rows of the sample “${s.title || s.id}”)`;
    exampleLoaded = true;
  } catch (_) { /* keep the header-only example */ }
}

// ---------------------------------------------------------------------------
// Footer attribution and repository links

function renderFooter() {
  const box = $('#attribution');
  const parts = [];
  const ds = state.results && Array.isArray(state.results.datasets) ? state.results.datasets : [];
  if (ds.length) {
    parts.push('<p>Data sources: ' + ds.map((d) => {
      const label = d.short || d.name;
      const name = d.url ? `<a href="${esc(d.url)}" rel="noopener" target="_blank">${esc(label)}</a>` : esc(label);
      const lic = d.license ? (d.license_url ? `<a href="${esc(d.license_url)}" rel="noopener" target="_blank">${esc(d.license)}</a>` : esc(d.license)) : '';
      return `${name}${lic ? ` (${lic})` : ''}`;
    }).join(' · ') + (ds.every((d) => /physionet\.org/.test(d.url || '')) ? '. Distributed by PhysioNet.' : '.') + '</p>');
    const cites = ds.map((d) => (d.citation ? `${d.short || d.name}: ${d.citation}` : '')).filter(Boolean);
    if (state.results.physionet_citation) cites.push(`PhysioNet: ${state.results.physionet_citation}`);
    if (cites.length) {
      parts.push(`<details class="citations"><summary>Citations</summary><ul>${cites.map((c) => `<li>${esc(c)}</li>`).join('')}</ul></details>`);
    }
  }
  if (state.samples && state.samples.length) {
    parts.push('<p>Sample files: ' + state.samples.map((s) => {
      const src = s.source || {};
      return `${esc(s.title || s.id)} from ${esc(src.dataset || 'unknown')}${src.record ? ` record ${esc(src.record)}` : ''}${src.license ? `, ${esc(src.license)}` : ''}`;
    }).join(' · ') + '.</p>');
  }
  box.innerHTML = parts.join('');

  for (const el of $$('.repo-link')) el.hidden = !REPO_URL;
  for (const a of $$('[data-repo-link]')) {
    if (REPO_URL) { a.href = REPO_URL; a.rel = 'noopener'; a.textContent = REPO_URL.replace(/^https?:\/\//, ''); }
  }
}

// ---------------------------------------------------------------------------
// Resizing: re-render width-dependent SVGs (no animation).

let resizeTimer = 0;
function onResize() {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => {
    const route = currentRoute();
    if (route === 'report' && state.report) { renderChart(); renderTrust(); }
    if (route === 'method') renderMethod();
  }, 120);
}

// ---------------------------------------------------------------------------
// Development stub worker (only when USE_STUB_WORKER is true).

function createStubWorker() {
  const listeners = { onmessage: null };
  let cancelled = false;
  const stub = {
    set onmessage(fn) { listeners.onmessage = fn; },
    set onerror(_) {},
    set onmessageerror(_) {},
    terminate() { cancelled = true; },
    postMessage(msg) {
      const emit = (data, delay) => setTimeout(() => { if (!cancelled && listeners.onmessage) listeners.onmessage({ data }); }, delay);
      const lines = String(msg.csvText || '').split(/\r?\n/).filter(Boolean);
      const header = (lines.shift() || '').split(',').map((h) => h.trim().toLowerCase());
      const ti = header.indexOf('time_s');
      const ri = header.indexOf('rr_ms');
      if (ri < 0) { emit({ type: 'error', code: 'missing_rr_column', message: 'Column rr_ms not found.' }, 200); return; }
      const t = [];
      const rr = [];
      let acc = 0;
      for (const line of lines) {
        const cells = line.split(',');
        const v = Number(cells[ri]);
        if (!Number.isFinite(v)) continue;
        acc += v / 1000;
        rr.push(v);
        t.push(ti >= 0 ? Number(cells[ti]) : acc);
      }
      const startS = msg.lightsOffS ?? 0;
      const endS = msg.lightsOnS ?? (t[t.length - 1] || 0);
      const nE = Math.max(0, Math.floor((endS - startS) / 30));
      const nM = Math.max(0, Math.floor((endS - startS) / 60));
      const means = new Array(nE).fill(0).map((_, k) => {
        let s = 0; let c = 0;
        for (let i = 0; i < t.length; i++) if (t[i] >= startS + 30 * k && t[i] < startS + 30 * (k + 1)) { s += rr[i]; c++; }
        return c ? s / c : NaN;
      });
      const finite = means.filter(Number.isFinite).sort((a, b) => a - b);
      const med = finite[Math.floor(finite.length / 2)] || 1000;
      const sw = means.map((x) => (Number.isFinite(x) && x < med * 0.92 ? 'wake' : 'sleep'));
      const w3 = means.map((x, k) => (sw[k] === 'wake' ? 'wake' : x < med ? 'rem' : 'nrem'));
      const w4 = means.map((x, k) => (w3[k] === 'nrem' ? (x > med * 1.05 ? 'deep' : 'light') : w3[k]));
      const sleepN = sw.filter((s) => s === 'sleep').length;
      const report = {
        fileName: msg.fileName,
        grid: { start_s: startS, end_s: endS, epoch_s: 30, n_epochs: nE },
        stages: { sleep_wake: sw, wake_nrem_rem: w3, four_class: w4 },
        lowQuality: means.map((x) => (Number.isFinite(x) ? 0 : 1)),
        apnea: { minute_s: 60, labels: new Array(nM).fill(0).map((_, m) => (m % 97 < 4 ? 1 : 0)) },
        metrics: { tib_min: nE * 0.5, tst_min: sleepN * 0.5, se_pct: nE ? (100 * sleepN) / nE : NaN, awakenings: 0, apnea_min: 0 },
        quality: { valid_fraction: 1, invalid_pct: 0, n_beats: rr.length, duration_h: (t[t.length - 1] || 0) / 3600, warnings: ['Development stub output'] },
      };
      report.metrics.apnea_min = report.apnea.labels.reduce((a, b) => a + b, 0);
      STEP_ORDER.forEach((step, i) => {
        emit({ type: 'progress', step, status: 'running' }, 150 + i * 250);
        emit({ type: 'progress', step, status: 'done' }, 300 + i * 250);
      });
      emit({ type: 'result', report }, 1300);
    },
  };
  return stub;
}

// ---------------------------------------------------------------------------
// Boot

function wireReport() {
  for (const btn of $$('.seg[data-mode]')) {
    btn.addEventListener('click', () => {
      if (state.mode === btn.dataset.mode) return;
      state.mode = btn.dataset.mode;
      state.animateNext = true;
      syncModeButtons();
      renderChart();
    });
  }
  for (const btn of $$('.seg-model')) {
    btn.addEventListener('click', () => rerunWithModel(btn.dataset.model));
  }
  $('#btn-cancel').addEventListener('click', () => {
    cancelJob();
    go('start');
  });
}

async function boot() {
  wireStart();
  wireReport();
  window.addEventListener('hashchange', onRoute);
  window.addEventListener('resize', onResize);
  onRoute();
  await Promise.all([loadData(), loadMinHours()]);
  renderSamples();
  renderFooter();
  onRoute();
}

// Exposed for debugging in the console only.
export const _debug = { state, computeBA };

boot();
