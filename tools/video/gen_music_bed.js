// Calm ambient bed for the Somnibeat demo video.
// Generated with the ls-clad build-music engine (fully algorithmic: no samples,
// no model weights; Apache-2.0 engine), so the audio is license-clean.
// Usage: node tools/video/gen_music_bed.js <seconds> <out.wav>
const path = require('path');
const fs = require('fs');

const ENGINE = '/Users/anjung/.claude/plugins/cache/ls-extensions/ls-clad/1.1.0/skills/build-music/tools';
const m = require(ENGINE);

const seconds = Number(process.argv[2] || 150);
const out = path.resolve(process.argv[3] || 'tools/video/out/music_bed.wav');
fs.mkdirSync(path.dirname(out), { recursive: true });

const genre = 'ambient';
const bpm = m.suggestTempo(genre);
const barS = 4 * 60 / bpm;
const BARS = Math.ceil(seconds / barS) + 2;

const { chords, meta } = m.composeChords({ genre, voice: 'pad', extension: 'add9' });
console.log('harmony', JSON.stringify(meta), 'bpm', bpm, 'bars', BARS);

// Soft piano chords every two bars, pad underneath the whole time.
const pianoEv = m.chordEvents(chords, { voice: 'piano', bars: BARS, barsPerChord: 2, velocity: 52, stagger: 0.2 });
const padEv = m.chordEvents(chords, { voice: 'pad', bars: BARS, barsPerChord: 2, velocity: 46 });

// A sparse vibraphone line only in the middle of the piece.
const km = meta.key.match(/^([A-Ga-g][#b]?)(-?\d+)$/);
const scaleRoot = km[1] + (parseInt(km[2], 10) + 1);
const melodyEv = m.composeMelody({ chords, bars: BARS, notesPerBar: 2, octaveShift: 1, contour: 'arch',
                                   scale: meta.scale, scaleRoot, restProbability: 0.45 });
const active = Array.from({ length: BARS }, (_, i) => i >= Math.floor(BARS * 0.3) && i < Math.floor(BARS * 0.75));
const melodyMasked = m.mask(melodyEv, m.arrangement.maskFromCycles(active), BARS, 4);

const tracks = [
  m.track('piano', 'piano', pianoEv, { fx: { hpf: 180, lpf: 4200, reverb: 'cathedral', gain: 0.34 } }),
  m.track('pad', 'pad', padEv, { fx: { hpf: 160, lpf: 2600, reverb: 'largeHall', gain: 0.2,
                                       chorus: { voices: 3, mix: 0.3 } }, width: 1.3 }),
  m.track('vibes', 'vibraphone', melodyMasked, { fx: { hpf: 450, lpf: 5200, reverb: 'largeHall', gain: 0.18 } }),
];
const rendered = m.render(tracks, { bpm, duration: seconds + 4, master: { normalize: 'peak', glue: true, width: 1.1 } });
m.WavBuilder.write(rendered, out);
console.log('wrote', out);
