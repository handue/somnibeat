// Somnibeat charts: pure functions that return inline SVG markup strings.
// No DOM access here, so the same code can be checked in Node.
// Every number drawn comes from the arguments (the computed report or results.json).

export const COLORS = {
  text: '#F2F4F7',
  muted: '#969FAC',
  grid: '#1A1F27',
  connector: '#2A313C',
  accent: '#4FD1C5',
  wake: '#F5A524',
  rem: '#A78BFA',
  light: '#5B7FA6',
  deep: '#3B5BDB',
  apnea: '#FF5D5D',
  hatch: '#4A5361',
};

// Row slots are fixed so a stage keeps its vertical position across modes.
const SLOT_NAMES = ['Wake', 'REM', 'Light', 'Deep'];

// Per mode: model label -> {slot, name, color}.
export const STAGE_MODES = {
  sleep_wake: {
    title: 'Wake / sleep',
    stages: {
      wake: { slot: 0, name: 'Wake', color: COLORS.wake },
      sleep: { slot: 2, name: 'Sleep', color: COLORS.light },
    },
  },
  wake_nrem_rem: {
    title: '3 stages (experimental)',
    stages: {
      wake: { slot: 0, name: 'Wake', color: COLORS.wake },
      rem: { slot: 1, name: 'REM', color: COLORS.rem },
      nrem: { slot: 2, name: 'NREM', color: COLORS.light },
    },
  },
  four_class: {
    title: '4 stages (experimental)',
    stages: {
      wake: { slot: 0, name: 'Wake', color: COLORS.wake },
      rem: { slot: 1, name: 'REM', color: COLORS.rem },
      light: { slot: 2, name: 'Light', color: COLORS.light },
      deep: { slot: 3, name: 'Deep', color: COLORS.deep },
    },
  },
};

export function esc(value) {
  return String(value)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

const MINUS = '−';

/** Format a number with fixed digits and a typographic minus sign. */
export function fmtNum(x, digits = 0, { sign = false } = {}) {
  if (x === null || x === undefined || !Number.isFinite(Number(x))) return '—';
  const v = Number(x);
  const rounded = Number(v.toFixed(digits));
  const s = Math.abs(rounded).toFixed(digits);
  if (rounded < 0) return MINUS + s;
  if (sign && rounded > 0) return '+' + s;
  return s;
}

/** Minutes -> "6 h 12 m" (plain text). */
export function fmtHM(minutes) {
  if (!Number.isFinite(minutes)) return '—';
  const total = Math.round(minutes);
  const h = Math.floor(total / 60);
  const m = total % 60;
  return `${h} h ${m} m`;
}

/** Seconds of day -> "HH:MM". */
export function fmtClock(secondsOfDay) {
  const s = ((Math.round(secondsOfDay) % 86400) + 86400) % 86400;
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}`;
}

/** Collapse an array into runs [{value, start, end}] (end exclusive). */
export function runs(values) {
  const out = [];
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    const last = out[out.length - 1];
    if (last && last.value === v) last.end = i + 1;
    else out.push({ value: v, start: i, end: i + 1 });
  }
  return out;
}

/** "Nice" axis ticks covering [lo, hi]. */
export function niceTicks(lo, hi, target = 5) {
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return [];
  if (lo === hi) { lo -= 1; hi += 1; }
  const span = hi - lo;
  const raw = span / Math.max(1, target);
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw / mag;
  const step = (norm >= 5 ? 10 : norm >= 2 ? 5 : norm >= 1 ? 2 : 1) * mag;
  const first = Math.ceil(lo / step) * step;
  const ticks = [];
  for (let v = first; v <= hi + step * 1e-9; v += step) ticks.push(Number(v.toFixed(10)));
  return ticks;
}

function pct(part, whole) {
  return whole > 0 ? Math.round((100 * part) / whole) : 0;
}

/**
 * Night chart: stage rows, low-quality hatching, apnea strip and time axis in one SVG.
 * opts: {report, mode, width, clockStartS (seconds of day of the recording start, or null),
 *        animate, idPrefix}
 */
export function hypnogramSvg(opts) {
  const { report, mode = 'sleep_wake', width = 800, clockStartS = null, animate = false, idPrefix = 'hyp' } = opts;
  const modeDef = STAGE_MODES[mode] || STAGE_MODES.sleep_wake;
  const labels = (report.stages && report.stages[mode]) || [];
  const lowQ = report.lowQuality || [];
  const apnea = (report.apnea && report.apnea.labels) || [];
  const epochS = (report.grid && report.grid.epoch_s) || 30;
  const minuteS = (report.apnea && report.apnea.minute_s) || 60;
  const startS = (report.grid && report.grid.start_s) || 0;
  const nEpochs = labels.length;
  const durationS = Math.max(nEpochs * epochS, apnea.length * minuteS, 1);

  const compact = width < 520;
  const padL = compact ? 52 : 72;
  const padR = compact ? 8 : 12;
  const rowH = compact ? 36 : 48;
  const barH = compact ? 18 : 24;
  const topPad = 10;
  const stageH = rowH * SLOT_NAMES.length;
  const apneaGap = 14;
  const apneaH = compact ? 14 : 18;
  const apneaY = topPad + stageH + apneaGap;
  const axisY = apneaY + apneaH + 10;
  const height = axisY + 22;
  const plotW = Math.max(10, width - padL - padR);
  const xOf = (s) => padL + (s / durationS) * plotW;
  const slotY = (slot) => topPad + slot * rowH + rowH / 2;

  const usedSlots = new Map();
  for (const def of Object.values(modeDef.stages)) usedSlots.set(def.slot, def);

  const parts = [];
  const hatchId = `${idPrefix}-hatch`;
  parts.push(`<defs><pattern id="${hatchId}" width="8" height="8" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">` +
    `<rect width="8" height="8" fill="#000000" fill-opacity="0.55"/>` +
    `<line x1="0" y1="0" x2="0" y2="8" stroke="${COLORS.hatch}" stroke-width="3"/></pattern></defs>`);

  // Row grid lines and labels (text labels, not color alone).
  for (let slot = 0; slot < SLOT_NAMES.length; slot++) {
    const y = slotY(slot);
    const def = usedSlots.get(slot);
    parts.push(`<line x1="${padL}" y1="${y}" x2="${padL + plotW}" y2="${y}" stroke="${COLORS.grid}" stroke-width="1"/>`);
    if (def) {
      parts.push(`<text x="${padL - 10}" y="${y + 4}" text-anchor="end" font-size="12" fill="${COLORS.muted}">${esc(def.name)}</text>`);
    }
  }
  parts.push(`<text x="${padL - 10}" y="${apneaY + apneaH / 2 + 4}" text-anchor="end" font-size="12" fill="${COLORS.muted}">Apnea</text>`);
  parts.push(`<line x1="${padL}" y1="${apneaY + apneaH / 2}" x2="${padL + plotW}" y2="${apneaY + apneaH / 2}" stroke="${COLORS.grid}" stroke-width="1"/>`);

  // Data layer (revealed left to right when animated).
  const data = [];
  const stageRuns = runs(labels);
  const counts = {};
  for (let i = 0; i < stageRuns.length; i++) {
    const r = stageRuns[i];
    const def = modeDef.stages[r.value];
    counts[r.value] = (counts[r.value] || 0) + (r.end - r.start);
    if (!def) continue;
    const x0 = xOf(r.start * epochS);
    const x1 = xOf(r.end * epochS);
    const y = slotY(def.slot);
    const next = stageRuns[i + 1];
    if (next && modeDef.stages[next.value]) {
      const y2 = slotY(modeDef.stages[next.value].slot);
      data.push(`<line x1="${x1.toFixed(2)}" y1="${y}" x2="${x1.toFixed(2)}" y2="${y2}" stroke="${COLORS.connector}" stroke-width="1"/>`);
    }
    data.push(`<rect x="${x0.toFixed(2)}" y="${y - barH / 2}" width="${Math.max(0.75, x1 - x0).toFixed(2)}" height="${barH}" rx="${compact ? 2 : 3}" fill="${def.color}"><title>${esc(def.name)}</title></rect>`);
  }

  let lowCount = 0;
  for (const r of runs(lowQ)) {
    if (Number(r.value) !== 1) continue;
    lowCount += r.end - r.start;
    const x0 = xOf(r.start * epochS);
    const x1 = xOf(r.end * epochS);
    const w = Math.max(1.5, x1 - x0);
    data.push(`<rect x="${x0.toFixed(2)}" y="${topPad}" width="${w.toFixed(2)}" height="${stageH}" fill="url(#${hatchId})"><title>not enough usable data</title></rect>`);
    if (w >= 150) {
      data.push(`<text x="${(x0 + w / 2).toFixed(2)}" y="${topPad + 14}" text-anchor="middle" font-size="11" fill="${COLORS.text}">not enough usable data</text>`);
    }
  }

  let apneaMinutes = 0;
  for (const r of runs(apnea)) {
    if (Number(r.value) !== 1) continue;
    apneaMinutes += r.end - r.start;
    const x0 = xOf(r.start * minuteS);
    const x1 = xOf(r.end * minuteS);
    data.push(`<rect x="${x0.toFixed(2)}" y="${apneaY}" width="${Math.max(1, x1 - x0).toFixed(2)}" height="${apneaH}" rx="1.5" fill="${COLORS.apnea}"><title>Suspected apnea</title></rect>`);
  }

  // Time axis: hourly ticks from the grid start (clock times when known).
  const hours = durationS / 3600;
  let stepH = hours <= 1.5 ? 0.5 : 1;
  while ((stepH * 3600 / durationS) * plotW < (clockStartS !== null ? 48 : 36)) stepH *= 2;
  const axis = [];
  for (let t = 0; t <= durationS + 1e-6; t += stepH * 3600) {
    const x = xOf(t);
    const label = clockStartS !== null
      ? fmtClock(clockStartS + startS + t)
      : `${Number.isInteger(t / 3600) ? t / 3600 : (t / 3600).toFixed(1)} h`;
    const anchor = t === 0 ? 'start' : (x > padL + plotW - 20 ? 'end' : 'middle');
    axis.push(`<line x1="${x.toFixed(2)}" y1="${axisY - 6}" x2="${x.toFixed(2)}" y2="${axisY - 2}" stroke="${COLORS.muted}" stroke-width="1"/>`);
    axis.push(`<text x="${x.toFixed(2)}" y="${axisY + 12}" text-anchor="${anchor}" font-size="11" fill="${COLORS.muted}">${esc(label)}</text>`);
  }

  // Accessible summary computed from the drawn data.
  const totalMin = (nEpochs * epochS) / 60;
  const shares = Object.entries(modeDef.stages)
    .map(([key, def]) => `${def.name} ${pct(counts[key] || 0, nEpochs)}%`)
    .join(', ');
  const axisNote = clockStartS !== null
    ? `from ${fmtClock(clockStartS + startS)} to ${fmtClock(clockStartS + startS + nEpochs * epochS)}`
    : `over ${fmtHM(totalMin)}`;
  const summary = `Sleep stage chart, ${modeDef.title}, ${axisNote}. ${shares}. ` +
    `${apneaMinutes} minutes of suspected apnea. ` +
    `${lowCount} of ${nEpochs} epochs marked as not enough usable data.`;

  return `<svg xmlns="http://www.w3.org/2000/svg" role="img" aria-label="${esc(summary)}" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" class="chart-svg">` +
    parts.join('') +
    `<g class="${animate ? 'reveal' : ''}">${data.join('')}</g>` +
    axis.join('') +
    `</svg>`;
}

/**
 * Bland-Altman plot. points: [{ref, pred, record}]. stats (optional): {bias, loa_low, loa_high};
 * computed from the points when missing.
 */
export function blandAltmanSvg({ points, stats = null, width = 560, height = 260, unit = 'min', quantity = 'Total sleep time', digits = 0 }) {
  const pts = (points || [])
    .filter((p) => Number.isFinite(p.ref) && Number.isFinite(p.pred))
    .map((p) => ({ mean: (p.ref + p.pred) / 2, diff: p.pred - p.ref, record: p.record }));
  const s = stats && Number.isFinite(stats.bias) ? stats : computeBA(pts.map((p) => p.diff));
  const compact = width < 420;
  const padL = 52;
  const padR = compact ? 12 : 92;
  const padT = 14;
  const padB = 44;
  const w = Math.max(10, width - padL - padR);
  const h = Math.max(10, height - padT - padB);

  if (!pts.length) {
    return `<svg xmlns="http://www.w3.org/2000/svg" role="img" aria-label="No per-record data available" width="${width}" height="60" viewBox="0 0 ${width} 60"><text x="0" y="30" font-size="13" fill="${COLORS.muted}">No per-record data available.</text></svg>`;
  }

  let xLo = Math.min(...pts.map((p) => p.mean));
  let xHi = Math.max(...pts.map((p) => p.mean));
  const xPad = (xHi - xLo || 1) * 0.08;
  xLo -= xPad; xHi += xPad;
  const ys = pts.map((p) => p.diff).concat([0, s.bias, s.loa_low, s.loa_high].filter(Number.isFinite));
  let yLo = Math.min(...ys);
  let yHi = Math.max(...ys);
  const yPad = (yHi - yLo || 1) * 0.1;
  yLo -= yPad; yHi += yPad;
  const xOf = (v) => padL + ((v - xLo) / (xHi - xLo)) * w;
  const yOf = (v) => padT + (1 - (v - yLo) / (yHi - yLo)) * h;

  const parts = [];
  for (const t of niceTicks(yLo, yHi, compact ? 4 : 5)) {
    const y = yOf(t);
    parts.push(`<line x1="${padL}" y1="${y.toFixed(2)}" x2="${padL + w}" y2="${y.toFixed(2)}" stroke="${COLORS.grid}" stroke-width="1"/>`);
    parts.push(`<text x="${padL - 8}" y="${(y + 4).toFixed(2)}" text-anchor="end" font-size="11" fill="${COLORS.muted}">${fmtNum(t, Math.abs(t) < 10 && !Number.isInteger(t) ? 1 : 0)}</text>`);
  }
  for (const t of niceTicks(xLo, xHi, compact ? 3 : 5)) {
    const x = xOf(t);
    parts.push(`<line x1="${x.toFixed(2)}" y1="${padT + h}" x2="${x.toFixed(2)}" y2="${padT + h + 4}" stroke="${COLORS.muted}" stroke-width="1"/>`);
    parts.push(`<text x="${x.toFixed(2)}" y="${padT + h + 16}" text-anchor="middle" font-size="11" fill="${COLORS.muted}">${fmtNum(t, 0)}</text>`);
  }
  parts.push(`<line x1="${padL}" y1="${padT + h}" x2="${padL + w}" y2="${padT + h}" stroke="${COLORS.connector}" stroke-width="1"/>`);
  const y0 = yOf(0);
  parts.push(`<line x1="${padL}" y1="${y0.toFixed(2)}" x2="${padL + w}" y2="${y0.toFixed(2)}" stroke="${COLORS.muted}" stroke-width="1" stroke-opacity="0.6"/>`);

  const lines = [
    { v: s.loa_high, label: '+1.96 SD', dash: '5 4', color: COLORS.muted },
    { v: s.bias, label: 'Bias', dash: '', color: COLORS.accent },
    { v: s.loa_low, label: '−1.96 SD', dash: '5 4', color: COLORS.muted },
  ];
  for (const l of lines) {
    if (!Number.isFinite(l.v)) continue;
    const y = yOf(l.v);
    parts.push(`<line x1="${padL}" y1="${y.toFixed(2)}" x2="${padL + w}" y2="${y.toFixed(2)}" stroke="${l.color}" stroke-width="1.5"${l.dash ? ` stroke-dasharray="${l.dash}"` : ''}/>`);
    if (!compact) {
      parts.push(`<text x="${padL + w + 8}" y="${(y - 2).toFixed(2)}" font-size="11" fill="${l.color}">${esc(l.label)}</text>`);
      parts.push(`<text x="${padL + w + 8}" y="${(y + 11).toFixed(2)}" font-size="11" fill="${COLORS.text}" class="num">${fmtNum(l.v, digits, { sign: true })}</text>`);
    }
  }
  for (const p of pts) {
    parts.push(`<circle cx="${xOf(p.mean).toFixed(2)}" cy="${yOf(p.diff).toFixed(2)}" r="4" fill="${COLORS.accent}" fill-opacity="0.55" stroke="${COLORS.accent}" stroke-width="1"><title>${esc(p.record || '')}: ${fmtNum(p.diff, digits, { sign: true })} ${esc(unit)}</title></circle>`);
  }
  parts.push(`<text x="${padL + w / 2}" y="${height - 6}" text-anchor="middle" font-size="11" fill="${COLORS.muted}">Mean of reference and estimate (${esc(unit)})</text>`);
  parts.push(`<text transform="translate(12 ${padT + h / 2}) rotate(-90)" text-anchor="middle" font-size="11" fill="${COLORS.muted}">Estimate − reference (${esc(unit)})</text>`);

  const summary = `Bland-Altman plot of ${quantity} for ${pts.length} test recordings: ` +
    `bias ${fmtNum(s.bias, digits, { sign: true })} ${unit}, 95% limits of agreement ` +
    `${fmtNum(s.loa_low, digits, { sign: true })} to ${fmtNum(s.loa_high, digits, { sign: true })} ${unit}.`;
  return `<svg xmlns="http://www.w3.org/2000/svg" role="img" aria-label="${esc(summary)}" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" class="chart-svg">${parts.join('')}</svg>`;
}

export function computeBA(diffs) {
  const d = diffs.filter(Number.isFinite);
  if (!d.length) return { bias: NaN, loa_low: NaN, loa_high: NaN };
  const mean = d.reduce((a, b) => a + b, 0) / d.length;
  const sd = d.length > 1 ? Math.sqrt(d.reduce((a, b) => a + (b - mean) ** 2, 0) / (d.length - 1)) : 0;
  return { bias: mean, loa_low: mean - 1.96 * sd, loa_high: mean + 1.96 * sd };
}

/**
 * Horizontal comparison bars. items: [{label, value, color, ciLow, ciHigh}].
 */
export function comparisonBarsSvg({ items, width = 560, unit = 'min', digits = 0, title = 'Comparison' }) {
  const valid = (items || []).filter((it) => Number.isFinite(it.value));
  const labelW = width < 420 ? 104 : 150;
  const valueW = 72;
  const rowH = 40;
  const barH = 12;
  const height = Math.max(1, valid.length) * rowH + 4;
  const plotW = Math.max(10, width - labelW - valueW);
  const maxV = Math.max(1e-9, ...valid.map((it) => (Number.isFinite(it.ciHigh) ? it.ciHigh : it.value)));
  const parts = [];
  valid.forEach((it, i) => {
    const yMid = i * rowH + rowH / 2;
    const bw = Math.max(2, (it.value / maxV) * plotW);
    parts.push(`<text x="0" y="${yMid + 4}" font-size="13" fill="${COLORS.text}">${esc(it.label)}</text>`);
    parts.push(`<rect x="${labelW}" y="${yMid - barH / 2}" width="${plotW}" height="${barH}" rx="${barH / 2}" fill="${COLORS.grid}"/>`);
    parts.push(`<rect x="${labelW}" y="${yMid - barH / 2}" width="${bw.toFixed(2)}" height="${barH}" rx="${barH / 2}" fill="${it.color || COLORS.accent}"/>`);
    if (Number.isFinite(it.ciLow) && Number.isFinite(it.ciHigh) && it.ciHigh > it.ciLow) {
      const x0 = labelW + (it.ciLow / maxV) * plotW;
      const x1 = labelW + (it.ciHigh / maxV) * plotW;
      parts.push(`<line x1="${x0.toFixed(2)}" y1="${yMid}" x2="${x1.toFixed(2)}" y2="${yMid}" stroke="${COLORS.text}" stroke-width="1.5"/>`);
      parts.push(`<line x1="${x0.toFixed(2)}" y1="${yMid - 6}" x2="${x0.toFixed(2)}" y2="${yMid + 6}" stroke="${COLORS.text}" stroke-width="1.5"/>`);
      parts.push(`<line x1="${x1.toFixed(2)}" y1="${yMid - 6}" x2="${x1.toFixed(2)}" y2="${yMid + 6}" stroke="${COLORS.text}" stroke-width="1.5"/>`);
    }
    parts.push(`<text x="${width}" y="${yMid + 4}" text-anchor="end" font-size="13" fill="${COLORS.text}" class="num">${fmtNum(it.value, digits)} ${esc(unit)}</text>`);
  });
  const summary = `${title}: ` + valid.map((it) => `${it.label} ${fmtNum(it.value, digits)} ${unit}`).join(', ') + '.';
  return `<svg xmlns="http://www.w3.org/2000/svg" role="img" aria-label="${esc(summary)}" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" class="chart-svg">${parts.join('')}</svg>`;
}
