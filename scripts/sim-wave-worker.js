/**
 * Parallel stretch-distance waves: disjoint edge sets per parity so workers can
 * write different vertices without races (SharedArrayBuffer positions).
 */
import {
  U_COUNT,
  V_COUNT,
  VERTEX_COUNT,
  STRETCH_U_COUNT,
  arrayIndex,
  vertexIndex
} from "./sim-parallel-shared.js";

function solveDistanceEdge(pos, inv, i0, i1, restLength, stiffness) {
  const w0 = inv[i0];
  const w1 = inv[i1];
  const wSum = w0 + w1;
  if (wSum === 0.0) return;

  const a0 = arrayIndex(i0);
  const a1 = arrayIndex(i1);

  const dx = pos[a1 + 0] - pos[a0 + 0];
  const dy = pos[a1 + 1] - pos[a0 + 1];
  const dz = pos[a1 + 2] - pos[a0 + 2];

  const lenSq = dx * dx + dy * dy + dz * dz;
  if (lenSq < 1e-12) return;

  const len = Math.sqrt(lenSq);
  const diff = (len - restLength) / len;

  const cx = stiffness * diff * dx;
  const cy = stiffness * diff * dy;
  const cz = stiffness * diff * dz;

  if (w0 > 0.0) {
    const s0 = w0 / wSum;
    pos[a0 + 0] += cx * s0;
    pos[a0 + 1] += cy * s0;
    pos[a0 + 2] += cz * s0;
  }

  if (w1 > 0.0) {
    const s1 = w1 / wSum;
    pos[a1 + 0] -= cx * s1;
    pos[a1 + 1] -= cy * s1;
    pos[a1 + 2] -= cz * s1;
  }
}

self.addEventListener("message", (e) => {
  const d = e.data;
  const {
    token,
    kind,
    sab,
    offPos,
    offInv,
    offStretchU,
    offStretchV,
    parity,
    stiffness
  } = d;

  const pos = new Float32Array(sab, offPos, VERTEX_COUNT * 3);
  const inv = new Float32Array(sab, offInv, VERTEX_COUNT);
  const stretchU = new Float32Array(sab, offStretchU, STRETCH_U_COUNT);
  const stretchV = new Float32Array(sab, offStretchV, STRETCH_V_COUNT);

  if (kind === "stretchU") {
    const v0 = d.v0;
    const v1 = d.v1;
    for (let v = v0; v < v1; v++) {
      let idx = v * (U_COUNT - 1);
      for (let u = 0; u < U_COUNT - 1; u++) {
        if ((u & 1) === parity) {
          const i0 = vertexIndex(u, v);
          const i1 = vertexIndex(u + 1, v);
          solveDistanceEdge(pos, inv, i0, i1, stretchU[idx], stiffness);
        }
        idx++;
      }
    }
  } else if (kind === "stretchV") {
    const u0 = d.u0;
    const u1 = d.u1;
    for (let u = u0; u < u1; u++) {
      for (let v = 0; v < V_COUNT - 1; v++) {
        if ((v & 1) === parity) {
          const idx = v * U_COUNT + u;
          const i0 = vertexIndex(u, v);
          const i1 = vertexIndex(u, v + 1);
          solveDistanceEdge(pos, inv, i0, i1, stretchV[idx], stiffness);
        }
      }
    }
  }

  self.postMessage({ token, ok: true });
});
