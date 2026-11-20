// Unit tests for web/js/smoothing.js: the tiny HMM examples of tests/test_v2_select.py.
// Run: node --test tests/parity/smoothing_unit.mjs   (also run by tests/test_parity_model_v2.py)
import assert from "node:assert/strict";
import { test } from "node:test";
import { emissions, EPS, forwardBackward, segments, smoothRecord, viterbi } from "../../web/js/smoothing.js";

const flat = (rows) => Float64Array.from(rows.flat());

// Joint score of every path: best path and posterior marginals (as brute_force in Python).
function bruteForce(emis, n, k, trans, init) {
  let best = null;
  let bestS = -1;
  const post = new Float64Array(n * k);
  const path = new Array(n).fill(0);
  const total = k ** n;
  for (let code = 0; code < total; code++) {
    let c = code;
    for (let t = n - 1; t >= 0; t--) {
      path[t] = c % k;
      c = Math.floor(c / k);
    }
    let s = init[path[0]] * emis[path[0]];
    for (let t = 1; t < n; t++) s *= trans[path[t - 1]][path[t]] * emis[t * k + path[t]];
    if (s > bestS) {
      best = path.slice();
      bestS = s;
    }
    for (let t = 0; t < n; t++) post[t * k + path[t]] += s;
  }
  for (let t = 0; t < n; t++) {
    let s = 0;
    for (let j = 0; j < k; j++) s += post[t * k + j];
    for (let j = 0; j < k; j++) post[t * k + j] /= s;
  }
  return { best, post };
}

// Small deterministic generator (tests only need variety, not Python's random stream).
function rng(seed) {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 4294967296;
  };
}
const dirichletLike = (r, k) => {
  const v = Array.from({ length: k }, () => r() + 0.05);
  const s = v.reduce((a, b) => a + b, 0);
  return v.map((x) => x / s);
};

test("viterbi hand computed", () => {
  const trans = [[0.9, 0.1], [0.1, 0.9]];
  const init = [0.5, 0.5];
  const emis = flat([[0.9, 0.1], [0.4, 0.6], [0.9, 0.1]]);
  assert.deepEqual(Array.from(viterbi(emis, 3, 2, trans, init)), [0, 0, 0]);
  const flatTrans = [[0.5, 0.5], [0.5, 0.5]];
  assert.deepEqual(Array.from(viterbi(emis, 3, 2, flatTrans, init)), [0, 1, 0]);
});

test("viterbi tie rule: lowest class index wins", () => {
  const trans = [[0.5, 0.5], [0.5, 0.5]];
  const init = [0.5, 0.5];
  const emis = flat([[0.5, 0.5], [0.5, 0.5]]);
  assert.deepEqual(Array.from(viterbi(emis, 2, 2, trans, init)), [0, 0]);
  const k3 = [[1 / 3, 1 / 3, 1 / 3], [1 / 3, 1 / 3, 1 / 3], [1 / 3, 1 / 3, 1 / 3]];
  assert.deepEqual(Array.from(viterbi(flat([[0.2, 0.4, 0.4]]), 1, 3, k3, [1 / 3, 1 / 3, 1 / 3])), [1]);
});

test("viterbi matches brute force", () => {
  const r = rng(1);
  for (let rep = 0; rep < 20; rep++) {
    const k = 3;
    const n = 5;
    const trans = Array.from({ length: k }, () => dirichletLike(r, k));
    const init = dirichletLike(r, k);
    const emis = Float64Array.from({ length: n * k }, () => r() + 0.01);
    const { best } = bruteForce(emis, n, k, trans, init);
    assert.deepEqual(Array.from(viterbi(emis, n, k, trans, init)), best);
  }
});

test("forward-backward sums to one and matches brute force", () => {
  const r = rng(2);
  for (const [k, n] of [[2, 6], [3, 5], [4, 4]]) {
    const trans = Array.from({ length: k }, () => dirichletLike(r, k));
    const init = dirichletLike(r, k);
    const emis = Float64Array.from({ length: n * k }, () => r() * 5 + 0.01);
    const post = forwardBackward(emis, n, k, trans, init);
    const { post: bf } = bruteForce(emis, n, k, trans, init);
    for (let t = 0; t < n; t++) {
      let s = 0;
      for (let j = 0; j < k; j++) s += post[t * k + j];
      assert.ok(Math.abs(s - 1) < 1e-12);
    }
    for (let i = 0; i < n * k; i++) assert.ok(Math.abs(post[i] - bf[i]) < 1e-10);
  }
});

test("emissions clip to [EPS, 1] and divide by the prior", () => {
  const e = emissions(flat([[0, 1.5], [0.25, 0.75]]), 2, 2, [0.5, 0.25]);
  assert.deepEqual(Array.from(e), [EPS / 0.5, 1 / 0.25, 0.25 / 0.5, 0.75 / 0.25]);
});

test("segments split at gaps", () => {
  assert.deepEqual(segments([0, 1, 2, 10, 11, 13]), [[0, 3], [3, 5], [5, 6]]);
  assert.deepEqual(segments([]), []);
  assert.deepEqual(segments([4]), [[0, 1]]);
});

test("smoothRecord segments and none", () => {
  const p = flat([[0.8, 0.2], [0.3, 0.7], [0.8, 0.2], [0.2, 0.8]]);
  const trans = [[0.99, 0.01], [0.01, 0.99]];
  const prior = [0.5, 0.5];
  assert.deepEqual(Array.from(smoothRecord(p, [0, 1, 2, 3], trans, prior, prior, "none")), Array.from(p));
  const v = smoothRecord(p, [0, 1, 2, 10], trans, prior, prior, "hmm_viterbi");
  const argmax = [0, 1, 2, 3].map((t) => (v[t * 2 + 1] > v[t * 2] ? 1 : 0));
  assert.deepEqual(argmax, [0, 0, 0, 1]);
  const fb = smoothRecord(p, [0, 1, 2, 10], trans, prior, prior, "hmm_forward_backward");
  for (let t = 0; t < 4; t++) assert.ok(Math.abs(fb[t * 2] + fb[t * 2 + 1] - 1) < 1e-12);
  assert.ok(Math.abs(fb[6] - 0.2) < 1e-12 && Math.abs(fb[7] - 0.8) < 1e-12);
  assert.throws(() => smoothRecord(p, [0, 1, 2, 3], trans, prior, prior, "bogus"));
});
