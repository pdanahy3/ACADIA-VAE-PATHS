import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const BATCH_MODE =
  typeof window !== "undefined" &&
  new URLSearchParams(window.location.search).get("batch") === "1";

/** Verlet + distance constraints on GPU (Chrome/Edge). Plasticity and impulses stay on CPU. */
const USE_WEBGPU =
  typeof window !== "undefined" &&
  new URLSearchParams(window.location.search).get("webgpu") === "1";

/**
 * 512 x 512 position-based mesh spring simulation.
 *
 * Coordinate convention:
 * u direction -> x axis
 * v direction -> z axis
 * vertical displacement -> y axis
 *
 * Anchors:
 * u = 0 row is fixed
 * u = U - 1 row is fixed
 */

const U_COUNT = 512;
const V_COUNT = 512;

const VERTEX_COUNT = U_COUNT * V_COUNT;
const CELL_COUNT_U = U_COUNT - 1;
const CELL_COUNT_V = V_COUNT - 1;

const PLANE_WIDTH = 24.0;
const PLANE_DEPTH = 24.0;

const DX = PLANE_WIDTH / (U_COUNT - 1);
const DZ = PLANE_DEPTH / (V_COUNT - 1);

const DT = 1.0 / 60.0;
const DT2 = DT * DT;

const STRETCH_U_COUNT = (U_COUNT - 1) * V_COUNT;
const STRETCH_V_COUNT = U_COUNT * (V_COUNT - 1);
const BEND_U_COUNT = (U_COUNT - 2) * V_COUNT;
const BEND_V_COUNT = U_COUNT * (V_COUNT - 2);

const bendRestU = new Float32Array(BEND_U_COUNT);
const bendRestV = new Float32Array(BEND_V_COUNT);

const SIM_POS_BYTES = VERTEX_COUNT * 3 * 4;
const SIM_PREV_BYTES = SIM_POS_BYTES;
const SIM_INV_BYTES = VERTEX_COUNT * 4;
const SIM_SU_BYTES = STRETCH_U_COUNT * 4;
const SIM_SV_BYTES = STRETCH_V_COUNT * 4;
const SIM_SAB_TOTAL = SIM_POS_BYTES + SIM_PREV_BYTES + SIM_INV_BYTES + SIM_SU_BYTES + SIM_SV_BYTES;
const OFF_INV = SIM_POS_BYTES + SIM_PREV_BYTES;
const OFF_SU = OFF_INV + SIM_INV_BYTES;
const OFF_SV = OFF_SU + SIM_SU_BYTES;

let positions;
let previousPositions;
let renderPositions;
let inverseMass;
let stretchRestU;
let stretchRestV;
let simSharedBuffer = null;
/** Web Workers + SharedArrayBuffer stretch waves (requires crossOriginIsolated + serve COOP/COEP). */
let parallelStretchWorkersReady = false;
const stretchWaveWorkers = [];
let stretchWaveJobToken = 1;
/** When true, mesh BufferAttribute reads `positions` directly (no per-frame copy). */
let meshPositionUsesSharedPositions = false;

function splitRange(parts, index, total) {
  const a = Math.floor((total * index) / parts);
  const b = Math.floor((total * (index + 1)) / parts);
  return [a, b];
}

function initStretchWaveWorkers() {
  if (!simSharedBuffer || typeof Worker === "undefined" || typeof import.meta.url === "undefined") {
    return;
  }
  const n = Math.min(
    4,
    Math.max(2, Math.floor((navigator.hardwareConcurrency || 4) / 2))
  );
  try {
    const url = new URL("./sim-wave-worker.js", import.meta.url);
    for (let i = 0; i < n; i++) {
      stretchWaveWorkers.push(new Worker(url, { type: "module" }));
    }
    parallelStretchWorkersReady = stretchWaveWorkers.length >= 2;
  } catch {
    while (stretchWaveWorkers.length) {
      stretchWaveWorkers.pop().terminate();
    }
    parallelStretchWorkersReady = false;
  }
}

function initParallelSimulationMemory() {
  parallelStretchWorkersReady = false;
  meshPositionUsesSharedPositions = false;
  while (stretchWaveWorkers.length) {
    stretchWaveWorkers.pop().terminate();
  }
  const coi =
    typeof crossOriginIsolated !== "undefined" && crossOriginIsolated === true;
  const hasSab = typeof SharedArrayBuffer !== "undefined";
  if (!coi || !hasSab) {
    simSharedBuffer = null;
    positions = new Float32Array(VERTEX_COUNT * 3);
    previousPositions = new Float32Array(VERTEX_COUNT * 3);
    renderPositions = new Float32Array(VERTEX_COUNT * 3);
    inverseMass = new Float32Array(VERTEX_COUNT);
    stretchRestU = new Float32Array(STRETCH_U_COUNT);
    stretchRestV = new Float32Array(STRETCH_V_COUNT);
    return;
  }
  try {
    simSharedBuffer = new SharedArrayBuffer(SIM_SAB_TOTAL);
    let byteOff = 0;
    positions = new Float32Array(simSharedBuffer, byteOff, VERTEX_COUNT * 3);
    byteOff += SIM_POS_BYTES;
    previousPositions = new Float32Array(simSharedBuffer, byteOff, VERTEX_COUNT * 3);
    byteOff += SIM_PREV_BYTES;
    inverseMass = new Float32Array(simSharedBuffer, byteOff, VERTEX_COUNT);
    byteOff += SIM_INV_BYTES;
    stretchRestU = new Float32Array(simSharedBuffer, byteOff, STRETCH_U_COUNT);
    byteOff += SIM_SU_BYTES;
    stretchRestV = new Float32Array(simSharedBuffer, byteOff, STRETCH_V_COUNT);
    byteOff += SIM_SV_BYTES;
    void byteOff;
    renderPositions = new Float32Array(VERTEX_COUNT * 3);
    meshPositionUsesSharedPositions = true;
    initStretchWaveWorkers();
  } catch {
    simSharedBuffer = null;
    positions = new Float32Array(VERTEX_COUNT * 3);
    previousPositions = new Float32Array(VERTEX_COUNT * 3);
    renderPositions = new Float32Array(VERTEX_COUNT * 3);
    inverseMass = new Float32Array(VERTEX_COUNT);
    stretchRestU = new Float32Array(STRETCH_U_COUNT);
    stretchRestV = new Float32Array(STRETCH_V_COUNT);
    meshPositionUsesSharedPositions = false;
  }
}

initParallelSimulationMemory();

// ------------------------------------------------------------
// Optional WebGPU physics (full cloth solve on GPU)
// ------------------------------------------------------------

let webGpuSim = null;
let webGpuInitPromise = null;

const gpuPos4 = new Float32Array(VERTEX_COUNT * 4);
const gpuPrev4 = new Float32Array(VERTEX_COUNT * 4);

function packPosPrevForGpu(pos, prev, outPos, outPrev) {
  for (let i = 0; i < VERTEX_COUNT; i++) {
    const s = i * 3;
    const d = i * 4;
    outPos[d] = pos[s];
    outPos[d + 1] = pos[s + 1];
    outPos[d + 2] = pos[s + 2];
    outPos[d + 3] = 0;
    outPrev[d] = prev[s];
    outPrev[d + 1] = prev[s + 1];
    outPrev[d + 2] = prev[s + 2];
    outPrev[d + 3] = 0;
  }
}

function unpackGpuPosPrevToCpu(outPos, outPrev, pos, prev) {
  for (let i = 0; i < VERTEX_COUNT; i++) {
    const s = i * 3;
    const d = i * 4;
    pos[s] = outPos[d];
    pos[s + 1] = outPos[d + 1];
    pos[s + 2] = outPos[d + 2];
    prev[s] = outPrev[d];
    prev[s + 1] = outPrev[d + 1];
    prev[s + 2] = outPrev[d + 2];
  }
}

async function ensureWebGpuSim() {
  if (!USE_WEBGPU) return null;
  if (webGpuSim) return webGpuSim;
  if (!webGpuInitPromise) {
    webGpuInitPromise = (async () => {
      try {
        const { createWebGpuClothSimulator, isWebGpuAvailable } = await import("./webgpu-cloth-sim.js");
        if (!isWebGpuAvailable()) {
          console.warn("[WebGPU] navigator.gpu not available; using CPU solver.");
          return null;
        }
        const sim = await createWebGpuClothSimulator(U_COUNT, V_COUNT);
        console.info("[WebGPU] GPU cloth physics enabled.");
        return sim;
      } catch (e) {
        console.warn("[WebGPU] init failed, using CPU solver:", e);
        return null;
      }
    })();
  }
  webGpuSim = await webGpuInitPromise;
  return webGpuSim;
}

/** 0 = elastic (fixed rest lengths), 1 = strong plastic drift toward current edge lengths after yield. */
let plasticity = 0;

/** Base engineering-strain yield; current threshold is 4× this (plastic creep + edge “unweld”). */
const PLASTIC_STRAIN_YIELD_BASE = 0.0015;
const PLASTIC_STRAIN_YIELD = PLASTIC_STRAIN_YIELD_BASE * 4;
/** Stretch-edge visualization: full red at this engineering strain; edges beyond are unwelded (hidden). */
const EDGE_UNWELD_YIELD_RATIO = 100;
const EDGE_STRAIN_COLOR_CAP = PLASTIC_STRAIN_YIELD_BASE * EDGE_UNWELD_YIELD_RATIO;
const PLASTIC_CREEP_SCALE = 8.0;

let stretchStiffness = 5;
let bendStiffness = 10;
let damping = 0.463;
let gravity = -0.5;
let solverIterations = 1;

let frameCounter = 0;

/** Shared with center pulse and boid kicks. */
let impulseRadius = 40;
let impulseForce = 0.02;

const BOID_MAX = 64;
const boidU = new Float32Array(BOID_MAX);
const boidV = new Float32Array(BOID_MAX);
const boidVu = new Float32Array(BOID_MAX);
const boidVv = new Float32Array(BOID_MAX);
let boidCount = 8;
let boidsEnabled = false;

/** Per-vertex mean |engineering strain| on incident stretch edges (0–1 typical). */
const vertexStretchStress = new Float32Array(VERTEX_COUNT);

let captureFramesEnabled = false;
/** Separate filename indices for view vs displacement captures. */
let captureViewExportIndex = 0;
let captureDispExportIndex = 0;
/** Successful POST /api/save-capture calls this session (max CAPTURE_LIMIT for flat writes). */
let capturesSavedCount = 0;
let capturePipelineBusy = false;

const CAPTURE_LIMIT = 100;
const CAPTURE_API = "/api/save-capture";

/**
 * View / displacement save cadence (simulation steps).
 * Override with URL query: `view_stride` and `disp_stride` (integers ≥ 1; defaults 50 and 2).
 */
function captureStridesFromUrl() {
  if (typeof window === "undefined") return { viewStride: 50, dispStride: 2 };
  const q = new URLSearchParams(window.location.search);
  let v = parseInt(q.get("view_stride") ?? "50", 10);
  let d = parseInt(q.get("disp_stride") ?? "2", 10);
  if (!Number.isFinite(v) || v < 1) v = 50;
  if (!Number.isFinite(d) || d < 1) d = 2;
  return { viewStride: v, dispStride: d };
}
const { viewStride: CAPTURE_VIEW_STRIDE, dispStride: CAPTURE_DISP_STRIDE } = captureStridesFromUrl();
/** View capture only: ~4× focal length vs interactive (narrower vertical FOV). */
const VIEW_CAPTURE_FOCAL_LENGTH_MULT = 4;
/** Multiply window pixel size for view PNG only (displacement stays UV grid resolution). */
const VIEW_CAPTURE_PIXEL_SCALE = 2;
const VIEW_CAPTURE_MAX_DIMENSION = 4096;

const boidScratch = new THREE.Vector3();
const restScratch = new THREE.Vector3();
const stressGradScratch = new THREE.Vector2();

const boidSepDx = new Float32Array(BOID_MAX);
const boidSepDy = new Float32Array(BOID_MAX);
/** Phase (rad) for local circular wobble in UV; orbit radius scales as 2× impulseRadius. */
const boidCirclePhase = new Float32Array(BOID_MAX);

/** Seeded RNG for batch runs; when null use Math.random(). */
let batchRng = null;

function mulberry32(a) {
  return function () {
    let t = (a += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function rnd() {
  return batchRng ? batchRng() : Math.random();
}

// ------------------------------------------------------------
// Utility
// ------------------------------------------------------------

function vertexIndex(u, v) {
  return v * U_COUNT + u;
}

function arrayIndex(i) {
  return i * 3;
}

function isAnchor(u) {
  return u === 0 || u === U_COUNT - 1;
}

// ------------------------------------------------------------
// Initialize simulation data
// ------------------------------------------------------------

function resetPlasticRestLengths() {
  stretchRestU.fill(DX);
  stretchRestV.fill(DZ);
  bendRestU.fill(DX * 2.0);
  bendRestV.fill(DZ * 2.0);
}

function resetSimulation() {
  resetPlasticRestLengths();
  for (let v = 0; v < V_COUNT; v++) {
    for (let u = 0; u < U_COUNT; u++) {
      const i = vertexIndex(u, v);
      const a = arrayIndex(i);

      const x = (u / (U_COUNT - 1) - 0.5) * PLANE_WIDTH;
      const y = 0.0;
      const z = (v / (V_COUNT - 1) - 0.5) * PLANE_DEPTH;

      positions[a + 0] = x;
      positions[a + 1] = y;
      positions[a + 2] = z;

      previousPositions[a + 0] = x;
      previousPositions[a + 1] = y;
      previousPositions[a + 2] = z;

      renderPositions[a + 0] = x;
      renderPositions[a + 1] = y;
      renderPositions[a + 2] = z;

      inverseMass[i] = isAnchor(u) ? 0.0 : 1.0;
    }
  }
}

// ------------------------------------------------------------
// Verlet integration
// ------------------------------------------------------------

function integrate() {
  for (let i = 0; i < VERTEX_COUNT; i++) {
    if (inverseMass[i] === 0.0) continue;

    const a = arrayIndex(i);

    const px = positions[a + 0];
    const py = positions[a + 1];
    const pz = positions[a + 2];

    const vx = (positions[a + 0] - previousPositions[a + 0]) * damping;
    const vy = (positions[a + 1] - previousPositions[a + 1]) * damping;
    const vz = (positions[a + 2] - previousPositions[a + 2]) * damping;

    previousPositions[a + 0] = px;
    previousPositions[a + 1] = py;
    previousPositions[a + 2] = pz;

    positions[a + 0] += vx;
    positions[a + 1] += vy + gravity * DT2;
    positions[a + 2] += vz;
  }
}

function edgeLength(i0, i1) {
  const a0 = arrayIndex(i0);
  const a1 = arrayIndex(i1);
  const dx = positions[a1 + 0] - positions[a0 + 0];
  const dy = positions[a1 + 1] - positions[a0 + 1];
  const dz = positions[a1 + 2] - positions[a0 + 2];
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

/**
 * After constraints, move each spring's rest length toward the current span when strain exceeds yield.
 * Scaled by plasticity slider (0 = no drift).
 */
function applyPlasticCreep() {
  if (plasticity <= 0.0) return;

  const creep = plasticity * PLASTIC_CREEP_SCALE * DT;

  let idx = 0;
  for (let v = 0; v < V_COUNT; v++) {
    for (let u = 0; u < U_COUNT - 1; u++) {
      const i0 = vertexIndex(u, v);
      const i1 = vertexIndex(u + 1, v);
      const L = edgeLength(i0, i1);
      let Lr = stretchRestU[idx];
      if (Lr < 1e-9) Lr = DX;
      const strain = Math.abs(L - Lr) / Lr;
      if (strain > PLASTIC_STRAIN_YIELD) {
        stretchRestU[idx] += creep * (L - Lr);
      }
      idx++;
    }
  }

  idx = 0;
  for (let v = 0; v < V_COUNT - 1; v++) {
    for (let u = 0; u < U_COUNT; u++) {
      const i0 = vertexIndex(u, v);
      const i1 = vertexIndex(u, v + 1);
      const L = edgeLength(i0, i1);
      let Lr = stretchRestV[idx];
      if (Lr < 1e-9) Lr = DZ;
      const strain = Math.abs(L - Lr) / Lr;
      if (strain > PLASTIC_STRAIN_YIELD) {
        stretchRestV[idx] += creep * (L - Lr);
      }
      idx++;
    }
  }

  idx = 0;
  for (let v = 0; v < V_COUNT; v++) {
    for (let u = 0; u < U_COUNT - 2; u++) {
      const i0 = vertexIndex(u, v);
      const i1 = vertexIndex(u + 2, v);
      const L = edgeLength(i0, i1);
      const nominal = DX * 2.0;
      let Lr = bendRestU[idx];
      if (Lr < 1e-9) Lr = nominal;
      const strain = Math.abs(L - Lr) / Lr;
      if (strain > PLASTIC_STRAIN_YIELD) {
        bendRestU[idx] += creep * (L - Lr);
      }
      idx++;
    }
  }

  idx = 0;
  for (let v = 0; v < V_COUNT - 2; v++) {
    for (let u = 0; u < U_COUNT; u++) {
      const i0 = vertexIndex(u, v);
      const i1 = vertexIndex(u, v + 2);
      const L = edgeLength(i0, i1);
      const nominal = DZ * 2.0;
      let Lr = bendRestV[idx];
      if (Lr < 1e-9) Lr = nominal;
      const strain = Math.abs(L - Lr) / Lr;
      if (strain > PLASTIC_STRAIN_YIELD) {
        bendRestV[idx] += creep * (L - Lr);
      }
      idx++;
    }
  }
}

// ------------------------------------------------------------
// Distance spring constraint
// ------------------------------------------------------------

function solveDistanceConstraint(i0, i1, restLength, stiffness) {
  const w0 = inverseMass[i0];
  const w1 = inverseMass[i1];
  const wSum = w0 + w1;

  if (wSum === 0.0) return;

  const a0 = arrayIndex(i0);
  const a1 = arrayIndex(i1);

  const dx = positions[a1 + 0] - positions[a0 + 0];
  const dy = positions[a1 + 1] - positions[a0 + 1];
  const dz = positions[a1 + 2] - positions[a0 + 2];

  const lenSq = dx * dx + dy * dy + dz * dz;
  if (lenSq < 1e-12) return;

  const len = Math.sqrt(lenSq);
  const diff = (len - restLength) / len;

  const cx = stiffness * diff * dx;
  const cy = stiffness * diff * dy;
  const cz = stiffness * diff * dz;

  if (w0 > 0.0) {
    const s0 = w0 / wSum;
    positions[a0 + 0] += cx * s0;
    positions[a0 + 1] += cy * s0;
    positions[a0 + 2] += cz * s0;
  }

  if (w1 > 0.0) {
    const s1 = w1 / wSum;
    positions[a1 + 0] -= cx * s1;
    positions[a1 + 1] -= cy * s1;
    positions[a1 + 2] -= cz * s1;
  }
}

// ------------------------------------------------------------
// Constraint passes
// ------------------------------------------------------------

function solveStretchSprings(stiffness) {
  let idx = 0;
  // Springs along u direction.
  for (let v = 0; v < V_COUNT; v++) {
    for (let u = 0; u < U_COUNT - 1; u++) {
      const i0 = vertexIndex(u, v);
      const i1 = vertexIndex(u + 1, v);
      solveDistanceConstraint(i0, i1, stretchRestU[idx], stiffness);
      idx++;
    }
  }

  idx = 0;
  // Springs along v direction.
  for (let v = 0; v < V_COUNT - 1; v++) {
    for (let u = 0; u < U_COUNT; u++) {
      const i0 = vertexIndex(u, v);
      const i1 = vertexIndex(u, v + 1);
      solveDistanceConstraint(i0, i1, stretchRestV[idx], stiffness);
      idx++;
    }
  }
}

function postStretchWaveWorker(w, payload) {
  return new Promise((resolve, reject) => {
    const onMsg = (e) => {
      if (e.data.token === payload.token) {
        w.removeEventListener("message", onMsg);
        if (e.data.ok) resolve();
        else reject(new Error("sim-wave-worker failed"));
      }
    };
    w.addEventListener("message", onMsg);
    w.postMessage(payload);
  });
}

async function solveStretchSpringsParallelAsync(stiffness) {
  if (!parallelStretchWorkersReady || !simSharedBuffer) {
    solveStretchSprings(stiffness);
    return;
  }
  const sab = simSharedBuffer;
  const n = stretchWaveWorkers.length;
  const base = {
    sab,
    offPos: 0,
    offInv: OFF_INV,
    offStretchU: OFF_SU,
    offStretchV: OFF_SV,
    stiffness
  };
  for (let parity = 0; parity < 2; parity++) {
    const token = ++stretchWaveJobToken;
    const jobs = [];
    for (let wi = 0; wi < n; wi++) {
      const [v0, v1] = splitRange(n, wi, V_COUNT);
      jobs.push(
        postStretchWaveWorker(stretchWaveWorkers[wi], {
          ...base,
          token,
          kind: "stretchU",
          parity,
          v0,
          v1
        })
      );
    }
    await Promise.all(jobs);
  }
  for (let parity = 0; parity < 2; parity++) {
    const token = ++stretchWaveJobToken;
    const jobs = [];
    for (let wi = 0; wi < n; wi++) {
      const [u0, u1] = splitRange(n, wi, U_COUNT);
      jobs.push(
        postStretchWaveWorker(stretchWaveWorkers[wi], {
          ...base,
          token,
          kind: "stretchV",
          parity,
          u0,
          u1
        })
      );
    }
    await Promise.all(jobs);
  }
}

function solveBendingSprings(stiffness) {
  let idx = 0;
  // Second-neighbor springs along u direction.
  for (let v = 0; v < V_COUNT; v++) {
    for (let u = 0; u < U_COUNT - 2; u++) {
      const i0 = vertexIndex(u, v);
      const i1 = vertexIndex(u + 2, v);
      solveDistanceConstraint(i0, i1, bendRestU[idx], stiffness);
      idx++;
    }
  }

  idx = 0;
  // Second-neighbor springs along v direction.
  for (let v = 0; v < V_COUNT - 2; v++) {
    for (let u = 0; u < U_COUNT; u++) {
      const i0 = vertexIndex(u, v);
      const i1 = vertexIndex(u, v + 2);
      solveDistanceConstraint(i0, i1, bendRestV[idx], stiffness);
      idx++;
    }
  }
}

/** Multiplier on bend stiffness for second-neighbor (bending) constraints only. */
const BEND_RESISTANCE_SCALE = 2;

function stretchStiffnessPassesAndK() {
  const s = stretchStiffness;
  if (s <= 1) return { passes: 1, k: s };
  return { passes: Math.ceil(s), k: 1 };
}

function bendStiffnessPassesAndK() {
  const s = bendStiffness * BEND_RESISTANCE_SCALE;
  if (s <= 1) return { passes: 1, k: s };
  return { passes: Math.ceil(s), k: 1 };
}

async function solveConstraints() {
  const { passes: stretchPasses, k: stretchK } = stretchStiffnessPassesAndK();
  const { passes: bendPasses, k: bendK } = bendStiffnessPassesAndK();

  for (let k = 0; k < solverIterations; k++) {
    for (let ps = 0; ps < stretchPasses; ps++) {
      if (parallelStretchWorkersReady && !USE_WEBGPU) {
        await solveStretchSpringsParallelAsync(stretchK);
      } else {
        solveStretchSprings(stretchK);
      }
    }
    for (let pb = 0; pb < bendPasses; pb++) {
      solveBendingSprings(bendK);
    }
  }
}

// ------------------------------------------------------------
// Optional disturbance
// ------------------------------------------------------------

function restPositionForVertex(u, v, out) {
  out.set(
    (u / (U_COUNT - 1) - 0.5) * PLANE_WIDTH,
    0.0,
    (v / (V_COUNT - 1) - 0.5) * PLANE_DEPTH
  );
}

function sampleMeshPosition(uf, vf, out) {
  const u0 = Math.max(0, Math.min(U_COUNT - 2, Math.floor(uf)));
  const v0 = Math.max(0, Math.min(V_COUNT - 2, Math.floor(vf)));
  const u1 = u0 + 1;
  const v1 = v0 + 1;
  const fu = Math.max(0, Math.min(1, uf - u0));
  const fv = Math.max(0, Math.min(1, vf - v0));

  const i00 = vertexIndex(u0, v0);
  const i10 = vertexIndex(u1, v0);
  const i01 = vertexIndex(u0, v1);
  const i11 = vertexIndex(u1, v1);

  const a00 = arrayIndex(i00);
  const a10 = arrayIndex(i10);
  const a01 = arrayIndex(i01);
  const a11 = arrayIndex(i11);

  let x = 0;
  let y = 0;
  let z = 0;
  for (let c = 0; c < 3; c++) {
    const p00 = positions[a00 + c];
    const p10 = positions[a10 + c];
    const p01 = positions[a01 + c];
    const p11 = positions[a11 + c];
    const p0 = p00 * (1 - fu) + p10 * fu;
    const p1 = p01 * (1 - fu) + p11 * fu;
    const pc = p0 * (1 - fv) + p1 * fv;
    if (c === 0) x = pc;
    else if (c === 1) y = pc;
    else z = pc;
  }
  out.set(x, y, z);
}

function computeVertexStretchStress() {
  for (let v = 0; v < V_COUNT; v++) {
    for (let u = 0; u < U_COUNT; u++) {
      let sum = 0;
      let cnt = 0;
      const i = vertexIndex(u, v);
      const a = arrayIndex(i);

      if (u < U_COUNT - 1) {
        const i1 = vertexIndex(u + 1, v);
        const a1 = arrayIndex(i1);
        const dx = positions[a1 + 0] - positions[a + 0];
        const dy = positions[a1 + 1] - positions[a + 1];
        const dz = positions[a1 + 2] - positions[a + 2];
        const L = Math.sqrt(dx * dx + dy * dy + dz * dz);
        const eu = v * (U_COUNT - 1) + u;
        let Lr = stretchRestU[eu];
        if (Lr < 1e-9) Lr = DX;
        sum += Math.abs(L - Lr) / Lr;
        cnt++;
      }
      if (u > 0) {
        const i0 = vertexIndex(u - 1, v);
        const a0 = arrayIndex(i0);
        const dx = positions[a + 0] - positions[a0 + 0];
        const dy = positions[a + 1] - positions[a0 + 1];
        const dz = positions[a + 2] - positions[a0 + 2];
        const L = Math.sqrt(dx * dx + dy * dy + dz * dz);
        const eu = v * (U_COUNT - 1) + (u - 1);
        let Lr = stretchRestU[eu];
        if (Lr < 1e-9) Lr = DX;
        sum += Math.abs(L - Lr) / Lr;
        cnt++;
      }
      if (v < V_COUNT - 1) {
        const i1 = vertexIndex(u, v + 1);
        const a1 = arrayIndex(i1);
        const dx = positions[a1 + 0] - positions[a + 0];
        const dy = positions[a1 + 1] - positions[a + 1];
        const dz = positions[a1 + 2] - positions[a + 2];
        const L = Math.sqrt(dx * dx + dy * dy + dz * dz);
        const ev = v * U_COUNT + u;
        let Lr = stretchRestV[ev];
        if (Lr < 1e-9) Lr = DZ;
        sum += Math.abs(L - Lr) / Lr;
        cnt++;
      }
      if (v > 0) {
        const i0 = vertexIndex(u, v - 1);
        const a0 = arrayIndex(i0);
        const dx = positions[a + 0] - positions[a0 + 0];
        const dy = positions[a + 1] - positions[a0 + 1];
        const dz = positions[a + 2] - positions[a0 + 2];
        const L = Math.sqrt(dx * dx + dy * dy + dz * dz);
        const ev = (v - 1) * U_COUNT + u;
        let Lr = stretchRestV[ev];
        if (Lr < 1e-9) Lr = DZ;
        sum += Math.abs(L - Lr) / Lr;
        cnt++;
      }

      vertexStretchStress[i] = cnt > 0 ? sum / cnt : 0;
    }
  }
}

function sampleScalarFieldBilinear(field, uf, vf) {
  const maxU = U_COUNT - 1;
  const maxV = V_COUNT - 1;
  let uu = uf;
  let vv = vf;
  if (uu <= 0) uu = 0.0001;
  if (vv <= 0) vv = 0.0001;
  if (uu >= maxU) uu = maxU - 0.0001;
  if (vv >= maxV) vv = maxV - 0.0001;

  const u0 = Math.floor(uu);
  const v0 = Math.floor(vv);
  const u1 = Math.min(u0 + 1, maxU);
  const v1 = Math.min(v0 + 1, maxV);
  const fu = uu - u0;
  const fv = vv - v0;

  const f00 = field[vertexIndex(u0, v0)];
  const f10 = field[vertexIndex(u1, v0)];
  const f01 = field[vertexIndex(u0, v1)];
  const f11 = field[vertexIndex(u1, v1)];

  const f0 = f00 * (1 - fu) + f10 * fu;
  const f1 = f01 * (1 - fu) + f11 * fu;
  return f0 * (1 - fv) + f1 * fv;
}

function stressGradientUv(uf, vf, outVec2) {
  const h = 1.75;
  const du =
    (sampleScalarFieldBilinear(vertexStretchStress, uf + h, vf) -
      sampleScalarFieldBilinear(vertexStretchStress, uf - h, vf)) /
    (2 * h);
  const dv =
    (sampleScalarFieldBilinear(vertexStretchStress, uf, vf + h) -
      sampleScalarFieldBilinear(vertexStretchStress, uf, vf - h)) /
    (2 * h);
  outVec2.set(du, dv);
}

/**
 * Radial falloff impulse in +Y (same model as the original center pulse).
 * centerU/centerV are float indices in the vertex grid (u,v).
 */
function addRadialImpulse(centerU, centerV, radius, forceAmount) {
  if (radius <= 0 || forceAmount === 0) return;

  const r0 = Math.ceil(radius);
  const uMin = Math.max(0, Math.floor(centerU - r0) - 1);
  const uMax = Math.min(U_COUNT - 1, Math.ceil(centerU + r0) + 1);
  const vMin = Math.max(0, Math.floor(centerV - r0) - 1);
  const vMax = Math.min(V_COUNT - 1, Math.ceil(centerV + r0) + 1);

  const rInv = 1.0 / radius;

  for (let v = vMin; v <= vMax; v++) {
    for (let u = uMin; u <= uMax; u++) {
      if (isAnchor(u)) continue;

      const du = u - centerU;
      const dv = v - centerV;
      const d = Math.sqrt(du * du + dv * dv);
      if (d > radius) continue;

      const falloff = 1.0 - d * rInv;
      const i = vertexIndex(u, v);
      const a = arrayIndex(i);
      positions[a + 1] += falloff * forceAmount;
    }
  }
}

function addCenterImpulse() {
  addRadialImpulse((U_COUNT - 1) * 0.5, (V_COUNT - 1) * 0.5, impulseRadius, impulseForce);
}

function clampBoidUv() {
  const uLo = 2.5;
  const uHi = U_COUNT - 1 - 2.5;
  const vLo = 1.5;
  const vHi = V_COUNT - 1 - 1.5;
  for (let b = 0; b < boidCount; b++) {
    boidU[b] = Math.max(uLo, Math.min(uHi, boidU[b]));
    boidV[b] = Math.max(vLo, Math.min(vHi, boidV[b]));
  }
}

function seedBoids() {
  const n = Math.max(1, Math.min(BOID_MAX, boidCount));
  boidCount = n;
  const uLo = 4;
  const uHi = U_COUNT - 5;
  const vLo = 4;
  const vHi = V_COUNT - 5;
  for (let b = 0; b < boidCount; b++) {
    boidU[b] = uLo + rnd() * (uHi - uLo);
    boidV[b] = vLo + rnd() * (vHi - vLo);
    const ang = rnd() * Math.PI * 2;
    const sp = 8 + rnd() * 12;
    boidVu[b] = Math.cos(ang) * sp;
    boidVv[b] = Math.sin(ang) * sp;
    boidCirclePhase[b] = rnd() * Math.PI * 2;
  }
  clampBoidUv();
}

function updateBoids(dt) {
  if (!boidsEnabled || boidCount < 1) return;

  const perception = 20;
  const separationDist = 10;
  const separationW = 1.65;
  /** Agents drift up the stretch-stress field (toward higher mean edge strain). */
  const stressFollow = 520;
  const maxSpeed = 40;

  for (let i = 0; i < boidCount; i++) {
    boidSepDx[i] = 0;
    boidSepDy[i] = 0;
  }

  for (let i = 0; i < boidCount; i++) {
    for (let j = 0; j < boidCount; j++) {
      if (i === j) continue;
      const du = boidU[j] - boidU[i];
      const dv = boidV[j] - boidV[i];
      const distSq = du * du + dv * dv;
      if (distSq < 1e-6 || distSq > perception * perception) continue;
      const dist = Math.sqrt(distSq);
      if (dist < separationDist) {
        const push = separationW / (dist * dist + 0.01);
        boidSepDx[i] -= (du / dist) * push;
        boidSepDy[i] -= (dv / dist) * push;
      }
    }
  }

  for (let i = 0; i < boidCount; i++) {
    stressGradientUv(boidU[i], boidV[i], stressGradScratch);
    let ax = stressGradScratch.x * stressFollow + boidSepDx[i];
    let ay = stressGradScratch.y * stressFollow + boidSepDy[i];

    const gLen = Math.sqrt(ax * ax + ay * ay);
    if (gLen < 1e-5) {
      ax += (rnd() - 0.5) * 28;
      ay += (rnd() - 0.5) * 28;
    }

    boidVu[i] += ax * dt;
    boidVv[i] += ay * dt;

    const sp = Math.sqrt(boidVu[i] * boidVu[i] + boidVv[i] * boidVv[i]);
    if (sp > maxSpeed) {
      const s = maxSpeed / sp;
      boidVu[i] *= s;
      boidVv[i] *= s;
    }

    boidU[i] += boidVu[i] * dt;
    boidV[i] += boidVv[i] * dt;
  }

  // Local concentric motion in (u,v): orbit radius 2× impulseRadius (grid cells), faster spin.
  const R = 2 * Math.max(0.25, impulseRadius);
  const omega = 24 / (impulseRadius + 1.5);
  for (let i = 0; i < boidCount; i++) {
    const ph = boidCirclePhase[i];
    const ph2 = ph + omega * dt;
    boidU[i] += R * Math.cos(ph2) - R * Math.cos(ph);
    boidV[i] += R * Math.sin(ph2) - R * Math.sin(ph);
    let wrapped = ph2 % (Math.PI * 2);
    if (wrapped < 0) wrapped += Math.PI * 2;
    boidCirclePhase[i] = wrapped;
  }

  clampBoidUv();
}

function applyBoidImpulses() {
  if (!boidsEnabled || boidCount < 1) return;
  // All boids steer and render; only the first applies radial kicks to the mesh.
  addRadialImpulse(boidU[0], boidV[0], impulseRadius, impulseForce);
}

function updateBoidMarkers(pointsGeom) {
  const posAttr = pointsGeom.attributes.position;
  const arr = posAttr.array;
  for (let b = 0; b < boidCount; b++) {
    sampleMeshPosition(boidU[b], boidV[b], boidScratch);
    arr[b * 3 + 0] = boidScratch.x;
    arr[b * 3 + 1] = boidScratch.y + 0.08;
    arr[b * 3 + 2] = boidScratch.z;
  }
  posAttr.needsUpdate = true;
  pointsGeom.setDrawRange(0, boidCount);
}

function blobToBase64(blob) {
  return new Promise((resolve, reject) => {
    const fr = new FileReader();
    fr.onloadend = () => {
      const s = fr.result;
      const i = s.indexOf(",");
      resolve(i >= 0 ? s.slice(i + 1) : s);
    };
    fr.onerror = () => reject(new Error("FileReader failed"));
    fr.readAsDataURL(blob);
  });
}

/**
 * Axis-aligned bbox of current mesh positions; longest edge L; domain half-extent D = L
 * so each displacement component maps in [-D, D] → [0, 255] with fixed scale (comparable across exports).
 */
function displacementMapDomainHalfExtent() {
  let minx = Infinity;
  let miny = Infinity;
  let minz = Infinity;
  let maxx = -Infinity;
  let maxy = -Infinity;
  let maxz = -Infinity;

  for (let i = 0; i < VERTEX_COUNT; i++) {
    const a = arrayIndex(i);
    const x = positions[a + 0];
    const y = positions[a + 1];
    const z = positions[a + 2];
    minx = Math.min(minx, x);
    maxx = Math.max(maxx, x);
    miny = Math.min(miny, y);
    maxy = Math.max(maxy, y);
    minz = Math.min(minz, z);
    maxz = Math.max(maxz, z);
  }

  const sx = maxx - minx;
  const sy = maxy - miny;
  const sz = maxz - minz;
  const L = Math.max(sx, sy, sz, 1e-9);
  return L;
}

function displacementComponentToByte(value, D) {
  if (D < 1e-12) return 127;
  const t = ((value + D) / (2 * D)) * 255;
  return THREE.MathUtils.clamp(Math.round(t), 0, 255);
}

/** 512×512 displacement PNG as Blob (Z-up RGB mapping, same as manual export). */
function buildDisplacementMapBlob() {
  return new Promise((resolve) => {
    const canvas = document.createElement("canvas");
    canvas.width = U_COUNT;
    canvas.height = V_COUNT;
    const ctx = canvas.getContext("2d");
    const imageData = ctx.createImageData(U_COUNT, V_COUNT);
    const data = imageData.data;

    /** Fixed domain from mesh bbox: same D for x,y,z; R=Δx, G=Δz, B=Δy (Three → Z-up). */
    const D = displacementMapDomainHalfExtent();

    let p = 0;
    for (let v = 0; v < V_COUNT; v++) {
      for (let u = 0; u < U_COUNT; u++) {
        restPositionForVertex(u, v, restScratch);
        const i = vertexIndex(u, v);
        const a = arrayIndex(i);
        const dx = positions[a + 0] - restScratch.x;
        const dy = positions[a + 1] - restScratch.y;
        const dz = positions[a + 2] - restScratch.z;
        const dX = dx;
        const dY = dz;
        const dZ = dy;
        data[p++] = displacementComponentToByte(dX, D);
        data[p++] = displacementComponentToByte(dY, D);
        data[p++] = displacementComponentToByte(dZ, D);
        data[p++] = 255;
      }
    }

    ctx.putImageData(imageData, 0, 0);
    canvas.toBlob((blob) => resolve(blob), "image/png");
  });
}

/**
 * POST view and/or displacement PNGs. Optional blobs use viewIndex / dispIndex for filenames.
 */
async function postCapturesToServer({ dataFolder, viewIndex, dispIndex, viewBlob, dispBlob }) {
  if (!viewBlob && !dispBlob) {
    throw new Error("postCapturesToServer: need viewBlob and/or dispBlob");
  }
  const payload = {};
  if (dataFolder) {
    payload.folder = dataFolder;
  }
  if (viewBlob) {
    payload.viewPng = await blobToBase64(viewBlob);
    payload.viewIndex = viewIndex;
  }
  if (dispBlob) {
    payload.dispPng = await blobToBase64(dispBlob);
    payload.dispIndex = dispIndex;
  }
  const body = JSON.stringify(payload);
  const res = await fetch(CAPTURE_API, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body
  });
  const text = await res.text();
  let parsed = null;
  try {
    parsed = JSON.parse(text);
  } catch {
    /* ignore */
  }
  if (!res.ok) {
    const msg = parsed?.error || text || res.statusText;
    throw new Error(msg);
  }
  return parsed;
}

function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 2000);
}

function exportCurrentViewPng() {
  captureViewBlobFromRenderer().then((blob) => {
    if (blob) downloadBlob(blob, `simulation-view-${Date.now()}.png`);
  });
}

function exportDisplacementMapPng() {
  buildDisplacementMapBlob().then((blob) => {
    if (blob) downloadBlob(blob, `vertex-displacement-rgb-${Date.now()}.png`);
  });
}

// ------------------------------------------------------------
// Three.js scene
// ------------------------------------------------------------

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x111111);

/** Vertical FOV (deg) for live view; view PNG uses higher pixel resolution and VIEW_CAPTURE_FOCAL_LENGTH_MULT. */
let interactiveCameraFovDeg = 45;

const camera = new THREE.PerspectiveCamera(
  interactiveCameraFovDeg,
  window.innerWidth / window.innerHeight,
  0.1,
  1000
);

camera.position.set(0, 48, 82);

/** Narrower vertical FOV for same film gate: focalLengthMult times longer focal vs interactive. */
function viewCaptureFovFromInteractive(fovDeg, focalLengthMult) {
  const half = (fovDeg * Math.PI) / 360;
  const newHalf = Math.atan(Math.tan(half) / focalLengthMult);
  return (newHalf * 360) / Math.PI;
}

const renderer = new THREE.WebGLRenderer({
  antialias: false,
  powerPreference: "high-performance",
  preserveDrawingBuffer: true
});

renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
document.body.appendChild(renderer.domElement);

let _viewCaptureSurface = null;

function enterViewCaptureSurface() {
  const pr = renderer.getPixelRatio();
  const w = Math.max(1, window.innerWidth);
  const h = Math.max(1, window.innerHeight);
  let capW = Math.floor(Math.min(w * VIEW_CAPTURE_PIXEL_SCALE, VIEW_CAPTURE_MAX_DIMENSION));
  let capH = Math.floor(Math.min(h * VIEW_CAPTURE_PIXEL_SCALE, VIEW_CAPTURE_MAX_DIMENSION));
  const aspect = w / h;
  const capAspect = capW / capH;
  if (capAspect > aspect) {
    capW = Math.floor(capH * aspect);
  } else {
    capH = Math.floor(capW / aspect);
  }
  _viewCaptureSurface = { pr, w, h, fov: camera.fov };
  renderer.setPixelRatio(1);
  renderer.setSize(capW, capH, false);
  camera.aspect = capW / capH;
  camera.fov = viewCaptureFovFromInteractive(
    interactiveCameraFovDeg,
    VIEW_CAPTURE_FOCAL_LENGTH_MULT
  );
  camera.updateProjectionMatrix();
}

function leaveViewCaptureSurface() {
  if (!_viewCaptureSurface) return;
  const { pr, w, h, fov } = _viewCaptureSurface;
  _viewCaptureSurface = null;
  camera.fov = fov;
  renderer.setPixelRatio(pr);
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}

/** Renders at higher resolution + telephoto FOV; restores interactive renderer state before resolving. */
function captureViewBlobFromRenderer() {
  enterViewCaptureSurface();
  renderer.render(scene, camera);
  return new Promise((resolve) => {
    renderer.domElement.toBlob((blob) => {
      leaveViewCaptureSurface();
      renderer.render(scene, camera);
      resolve(blob);
    }, "image/png");
  });
}

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.target.set(0, -2, 0);

/** Batch camera look-at (sheet center). */
const BATCH_TARGET = new THREE.Vector3(0, 0, 0);

/**
 * Y-up Three.js: azimuth in xz from +x toward +z; inclination = elevation above xz plane (deg).
 */
function setBatchCamera(azimuthDeg, inclinationDeg, distance = 144) {
  const toRad = Math.PI / 180;
  const az = azimuthDeg * toRad;
  const el = inclinationDeg * toRad;
  const R = distance;
  const xh = R * Math.cos(el) * Math.cos(az);
  const yh = R * Math.sin(el);
  const zh = R * Math.cos(el) * Math.sin(az);
  camera.position.set(
    BATCH_TARGET.x + xh,
    BATCH_TARGET.y + yh,
    BATCH_TARGET.z + zh
  );
  camera.lookAt(BATCH_TARGET);
  camera.updateProjectionMatrix();
}

const lightA = new THREE.DirectionalLight(0xffffff, 1.8);
lightA.position.set(8, 12, 8);
scene.add(lightA);

const lightB = new THREE.AmbientLight(0xffffff, 0.45);
scene.add(lightB);

// ------------------------------------------------------------
// Geometry
// ------------------------------------------------------------

function buildIndices() {
  const triangleCount = CELL_COUNT_U * CELL_COUNT_V * 2;
  const indices = new Uint32Array(triangleCount * 3);

  let p = 0;

  for (let v = 0; v < CELL_COUNT_V; v++) {
    for (let u = 0; u < CELL_COUNT_U; u++) {
      const i00 = vertexIndex(u, v);
      const i10 = vertexIndex(u + 1, v);
      const i01 = vertexIndex(u, v + 1);
      const i11 = vertexIndex(u + 1, v + 1);

      indices[p++] = i00;
      indices[p++] = i01;
      indices[p++] = i10;

      indices[p++] = i10;
      indices[p++] = i01;
      indices[p++] = i11;
    }
  }

  return indices;
}

resetSimulation();

const geometry = new THREE.BufferGeometry();
geometry.setAttribute(
  "position",
  new THREE.BufferAttribute(
    meshPositionUsesSharedPositions ? positions : renderPositions,
    3
  ).setUsage(THREE.DynamicDrawUsage)
);
geometry.setIndex(new THREE.BufferAttribute(buildIndices(), 1));
geometry.computeVertexNormals();

const material = new THREE.MeshStandardMaterial({
  color: 0x9fb7ff,
  roughness: 0.75,
  metalness: 0.05,
  side: THREE.DoubleSide,
  wireframe: false
});

const mesh = new THREE.Mesh(geometry, material);
mesh.visible = false;
scene.add(mesh);

/** Stretch-edge segments: vertex colors white (low strain) → red (at EDGE_STRAIN_COLOR_CAP); omitted if unwelded. */
const MESH_STRETCH_EDGE_SEGMENTS = STRETCH_U_COUNT + STRETCH_V_COUNT;
const EDGE_LINE_MAX_VERTICES = MESH_STRETCH_EDGE_SEGMENTS * 2;
const edgeLinePositions = new Float32Array(EDGE_LINE_MAX_VERTICES * 3);
const edgeLineColors = new Float32Array(EDGE_LINE_MAX_VERTICES * 3);
const edgeLineGeom = new THREE.BufferGeometry();
edgeLineGeom.setAttribute(
  "position",
  new THREE.BufferAttribute(edgeLinePositions, 3).setUsage(THREE.DynamicDrawUsage)
);
edgeLineGeom.setAttribute(
  "color",
  new THREE.BufferAttribute(edgeLineColors, 3).setUsage(THREE.DynamicDrawUsage)
);
edgeLineGeom.setDrawRange(0, 0);
const edgeLines = new THREE.LineSegments(
  edgeLineGeom,
  new THREE.LineBasicMaterial({ vertexColors: true, color: 0xffffff })
);
edgeLines.frustumCulled = false;
scene.add(edgeLines);

function pushEdgeLineSegment(rp, arr, col, w, c, a0, a1, strain) {
  if (strain > EDGE_STRAIN_COLOR_CAP) return { w, c };
  const u = Math.min(1, strain / EDGE_STRAIN_COLOR_CAP);
  const g = 1 - u;
  const b = 1 - u;
  arr[w++] = rp[a0 + 0];
  arr[w++] = rp[a0 + 1];
  arr[w++] = rp[a0 + 2];
  arr[w++] = rp[a1 + 0];
  arr[w++] = rp[a1 + 1];
  arr[w++] = rp[a1 + 2];
  col[c++] = 1;
  col[c++] = g;
  col[c++] = b;
  col[c++] = 1;
  col[c++] = g;
  col[c++] = b;
  return { w, c };
}

function updateMeshEdgeLines() {
  const arr = edgeLinePositions;
  const col = edgeLineColors;
  const rp = positions;
  let w = 0;
  let c = 0;
  let idx = 0;
  for (let v = 0; v < V_COUNT; v++) {
    for (let u = 0; u < U_COUNT - 1; u++) {
      const i0 = vertexIndex(u, v);
      const i1 = vertexIndex(u + 1, v);
      const L = edgeLength(i0, i1);
      let Lr = stretchRestU[idx];
      if (Lr < 1e-9) Lr = DX;
      const strain = Math.abs(L - Lr) / Lr;
      const a0 = arrayIndex(i0);
      const a1 = arrayIndex(i1);
      const p = pushEdgeLineSegment(rp, arr, col, w, c, a0, a1, strain);
      w = p.w;
      c = p.c;
      idx++;
    }
  }
  idx = 0;
  for (let v = 0; v < V_COUNT - 1; v++) {
    for (let u = 0; u < U_COUNT; u++) {
      const i0 = vertexIndex(u, v);
      const i1 = vertexIndex(u, v + 1);
      const L = edgeLength(i0, i1);
      let Lr = stretchRestV[idx];
      if (Lr < 1e-9) Lr = DZ;
      const strain = Math.abs(L - Lr) / Lr;
      const a0 = arrayIndex(i0);
      const a1 = arrayIndex(i1);
      const p = pushEdgeLineSegment(rp, arr, col, w, c, a0, a1, strain);
      w = p.w;
      c = p.c;
      idx++;
    }
  }
  edgeLineGeom.setDrawRange(0, w / 3);
  edgeLineGeom.attributes.position.needsUpdate = true;
  edgeLineGeom.attributes.color.needsUpdate = true;
}

const boidMarkerPositions = new Float32Array(BOID_MAX * 3);
const boidPointsGeom = new THREE.BufferGeometry();
boidPointsGeom.setAttribute(
  "position",
  new THREE.BufferAttribute(boidMarkerPositions, 3).setUsage(THREE.DynamicDrawUsage)
);
boidPointsGeom.setDrawRange(0, 0);
const boidPoints = new THREE.Points(
  boidPointsGeom,
  new THREE.PointsMaterial({
    color: 0xffcc44,
    size: 0.42,
    sizeAttenuation: true,
    depthTest: true,
    transparent: true,
    opacity: 0.95
  })
);
boidPoints.visible = false;
boidPoints.frustumCulled = false;
scene.add(boidPoints);

// Anchor visualization.
const anchorMaterial = new THREE.LineBasicMaterial({ color: 0xff5555 });

function createAnchorLine(u) {
  const points = [];

  for (let v = 0; v < V_COUNT; v++) {
    const i = vertexIndex(u, v);
    const a = arrayIndex(i);

    points.push(
      new THREE.Vector3(
        positions[a + 0],
        positions[a + 1] + 0.025,
        positions[a + 2]
      )
    );
  }

  const lineGeometry = new THREE.BufferGeometry().setFromPoints(points);
  const line = new THREE.Line(lineGeometry, anchorMaterial);
  return line;
}

const anchorLineA = createAnchorLine(0);
const anchorLineB = createAnchorLine(U_COUNT - 1);

scene.add(anchorLineA);
scene.add(anchorLineB);

// ------------------------------------------------------------
// UI
// ------------------------------------------------------------

if (!BATCH_MODE) {
  document.getElementById("stretch").addEventListener("input", (e) => {
    stretchStiffness = Number(e.target.value);
  });

  document.getElementById("bend").addEventListener("input", (e) => {
    bendStiffness = Number(e.target.value);
  });

  document.getElementById("damping").addEventListener("input", (e) => {
    damping = Number(e.target.value);
  });

  document.getElementById("gravity").addEventListener("input", (e) => {
    gravity = Number(e.target.value);
  });

  document.getElementById("iterations").addEventListener("input", (e) => {
    solverIterations = Number(e.target.value);
  });

  document.getElementById("plastic").addEventListener("input", (e) => {
    plasticity = Number(e.target.value);
  });

  document.getElementById("impulse-radius").addEventListener("input", (e) => {
    impulseRadius = Number(e.target.value);
  });

  document.getElementById("impulse-force").addEventListener("input", (e) => {
    impulseForce = Number(e.target.value);
  });

  document.getElementById("boid-count").addEventListener("input", (e) => {
    boidCount = Math.max(1, Math.min(BOID_MAX, Math.floor(Number(e.target.value))));
  });

  document.getElementById("boids-enabled").addEventListener("change", (e) => {
    boidsEnabled = e.target.checked;
    boidPoints.visible = boidsEnabled;
    if (boidsEnabled) {
      boidCount = Math.max(
        1,
        Math.min(BOID_MAX, Math.floor(Number(document.getElementById("boid-count").value)))
      );
      seedBoids();
    }
  });

  document.getElementById("capture-frames").addEventListener("change", (e) => {
    captureFramesEnabled = e.target.checked;
    if (captureFramesEnabled) {
      captureViewExportIndex = 0;
      captureDispExportIndex = 0;
      capturesSavedCount = 0;
      capturePipelineBusy = false;
    }
  });

  document.getElementById("boids-seed").addEventListener("click", () => {
    syncUiParametersFromDom();
    seedBoids();
  });

  document.getElementById("export-view").addEventListener("click", () => {
    exportCurrentViewPng();
  });

  document.getElementById("export-displacement").addEventListener("click", () => {
    exportDisplacementMapPng();
  });

  function syncUiParametersFromDom() {
    stretchStiffness = Number(document.getElementById("stretch").value);
    bendStiffness = Number(document.getElementById("bend").value);
    damping = Number(document.getElementById("damping").value);
    gravity = Number(document.getElementById("gravity").value);
    solverIterations = Number(document.getElementById("iterations").value);
    plasticity = Number(document.getElementById("plastic").value);
    impulseRadius = Number(document.getElementById("impulse-radius").value);
    impulseForce = Number(document.getElementById("impulse-force").value);
    boidCount = Math.max(
      1,
      Math.min(BOID_MAX, Math.floor(Number(document.getElementById("boid-count").value)))
    );
    boidsEnabled = document.getElementById("boids-enabled").checked;
    captureFramesEnabled = document.getElementById("capture-frames").checked;
  }

  syncUiParametersFromDom();
  boidPoints.visible = boidsEnabled;

  document.getElementById("reset").addEventListener("click", () => {
    resetSimulation();
    updateRenderGeometry(true);
  });

  document.getElementById("pulse").addEventListener("click", () => {
    addCenterImpulse();
  });
}

// ------------------------------------------------------------
// Render updates
// ------------------------------------------------------------

function updateRenderGeometry(updateNormals = false) {
  if (!meshPositionUsesSharedPositions) {
    renderPositions.set(positions);
  }

  const posAttr = geometry.attributes.position;
  posAttr.needsUpdate = true;

  // Normal computation on a 512 x 512 mesh is expensive.
  // Updating every few frames is a useful compromise.
  if (updateNormals) {
    geometry.computeVertexNormals();
    geometry.attributes.normal.needsUpdate = true;
  }

  updateMeshEdgeLines();
}

async function simulationTick() {
  if (boidsEnabled) {
    computeVertexStretchStress();
  }
  updateBoids(DT);
  applyBoidImpulses();

  const gpu = await ensureWebGpuSim();
  if (gpu) {
    packPosPrevForGpu(positions, previousPositions, gpuPos4, gpuPrev4);
    gpu.uploadInv(inverseMass);
    gpu.uploadRests(stretchRestU, stretchRestV, bendRestU, bendRestV);
    gpu.uploadPositionsAndPrev(gpuPos4, gpuPrev4);
    const { passes: stretchPasses, k: stretchK } = stretchStiffnessPassesAndK();
    const { passes: bendPasses, k: bendK } = bendStiffnessPassesAndK();
    gpu.submitPhysics({
      dt: DT,
      dt2: DT2,
      gravity,
      damping,
      dx: DX,
      dz: DZ,
      stretchK,
      bendK,
      stretchPasses,
      bendPasses,
      solverIterations
    });
    await gpu.readPositionsVec4(gpuPos4);
    await gpu.readPrevToVec4(gpuPrev4);
    unpackGpuPosPrevToCpu(gpuPos4, gpuPrev4, positions, previousPositions);
  } else {
    integrate();
    await solveConstraints();
  }

  applyPlasticCreep();

  frameCounter++;

  const updateNormals = frameCounter % 8 === 0;
  updateRenderGeometry(updateNormals);

  if (boidsEnabled) {
    updateBoidMarkers(boidPointsGeom);
  }

  controls.update();
  renderer.render(scene, camera);
}

function applySimulationDefaultsForBatch(nBoids) {
  stretchStiffness = 10;
  bendStiffness = 40;
  damping = 0.4995;
  gravity = 0;
  solverIterations = 1;
  plasticity = 1;
  impulseRadius = 40;
  impulseForce = 0.02;
  boidCount = Math.max(6, Math.min(32, Math.floor(nBoids)));
  boidsEnabled = true;
  captureFramesEnabled = false;
}

async function runHeadlessBatch() {
  const q = new URLSearchParams(location.search);
  const steps = Math.max(1, parseInt(q.get("steps") || "500", 10));
  let folder = (q.get("folder") || "run").replace(/[^a-zA-Z0-9_.-]/g, "_").slice(0, 120);
  if (!folder) folder = "run";
  const seed = parseInt(q.get("seed") || "1", 10) >>> 0;
  const boids = Math.max(6, Math.min(32, parseInt(q.get("boids") || "16", 10)));

  batchRng = mulberry32(seed);
  applySimulationDefaultsForBatch(boids);
  resetSimulation();
  seedBoids();
  boidPoints.visible = true;
  controls.target.copy(BATCH_TARGET);
  setBatchCamera(45, 45);
  controls.enabled = false;

  frameCounter = 0;
  let viewIdx = 0;
  let dispIdx = 0;
  let posts = 0;
  /** Keep headless logs readable (~20 lines max per run). */
  const progressEvery = Math.max(1, Math.floor(steps / 20));
  try {
    for (let t = 1; t <= steps; t++) {
      if (t === 1 || t === 2 || t === steps || t % progressEvery === 0) {
        console.log(`[batch] ${folder} sim step ${t}/${steps} (captures start at step ${CAPTURE_DISP_STRIDE}; view every ${CAPTURE_VIEW_STRIDE})`);
      }
      await simulationTick();
      const doView = t % CAPTURE_VIEW_STRIDE === 0;
      const doDisp = t % CAPTURE_DISP_STRIDE === 0;
      if (!doView && !doDisp) continue;

      let viewBlob = null;
      if (doView) {
        viewBlob = await captureViewBlobFromRenderer();
        if (!viewBlob) throw new Error("View capture toBlob failed");
      }
      let dispBlob = null;
      if (doDisp) {
        dispBlob = await buildDisplacementMapBlob();
        if (!dispBlob) throw new Error("Displacement toBlob failed");
      }

      await postCapturesToServer({
        dataFolder: folder,
        viewIndex: viewIdx,
        dispIndex: dispIdx,
        viewBlob: doView ? viewBlob : null,
        dispBlob: doDisp ? dispBlob : null
      });
      if (doView) viewIdx++;
      if (doDisp) dispIdx++;
      posts++;
      if (posts === 1 || posts % 25 === 0) {
        console.log(`[batch] ${folder} posted capture #${posts} (viewIdx=${viewIdx} dispIdx=${dispIdx})`);
      }
    }
    window.__BATCH_DONE__ = {
      ok: true,
      posts,
      viewCount: viewIdx,
      dispCount: dispIdx,
      folder,
      steps
    };
  } catch (e) {
    console.error(e);
    window.__BATCH_DONE__ = { ok: false, error: String(e) };
  }
}

// ------------------------------------------------------------
// Main loop
// ------------------------------------------------------------

async function animate() {
  await simulationTick();

  if (captureFramesEnabled && capturesSavedCount < CAPTURE_LIMIT && !capturePipelineBusy) {
    const doView = frameCounter > 0 && frameCounter % CAPTURE_VIEW_STRIDE === 0;
    const doDisp = frameCounter > 0 && frameCounter % CAPTURE_DISP_STRIDE === 0;
    if (doView || doDisp) {
      const vIdx = captureViewExportIndex;
      const dIdx = captureDispExportIndex;
      capturePipelineBusy = true;
      (async () => {
        try {
          let viewBlob = null;
          if (doView) {
            viewBlob = await captureViewBlobFromRenderer();
            if (!viewBlob) throw new Error("View capture toBlob failed");
          }
          let dispBlob = null;
          if (doDisp) {
            dispBlob = await buildDisplacementMapBlob();
            if (!dispBlob) throw new Error("Displacement toBlob failed");
          }
          await postCapturesToServer({
            dataFolder: null,
            viewIndex: vIdx,
            dispIndex: dIdx,
            viewBlob: doView ? viewBlob : null,
            dispBlob: doDisp ? dispBlob : null
          });
          if (doView) captureViewExportIndex++;
          if (doDisp) captureDispExportIndex++;
          capturesSavedCount++;
          if (capturesSavedCount >= CAPTURE_LIMIT) {
            captureFramesEnabled = false;
            const el = document.getElementById("capture-frames");
            if (el) el.checked = false;
          }
        } catch (err) {
          console.error("Capture save failed:", err);
          captureFramesEnabled = false;
          const el = document.getElementById("capture-frames");
          if (el) el.checked = false;
        } finally {
          capturePipelineBusy = false;
        }
      })();
    }
  }

  requestAnimationFrame(() => void animate());
}

if (BATCH_MODE) {
  runHeadlessBatch().catch((e) => {
    console.error(e);
    window.__BATCH_DONE__ = { ok: false, error: String(e) };
  });
} else {
  void animate();
}

// ------------------------------------------------------------
// Resize
// ------------------------------------------------------------

if (!BATCH_MODE) {
  window.addEventListener("resize", () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();

    renderer.setSize(window.innerWidth, window.innerHeight);
  });
}