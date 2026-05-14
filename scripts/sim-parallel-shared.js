/** Shared constants for main thread + sim-wave-worker (no Three.js). */
export const U_COUNT = 512;
export const V_COUNT = 512;
export const VERTEX_COUNT = U_COUNT * V_COUNT;
export const STRETCH_U_COUNT = (U_COUNT - 1) * V_COUNT;
export const STRETCH_V_COUNT = U_COUNT * (V_COUNT - 1);

export function vertexIndex(u, v) {
  return v * U_COUNT + u;
}

export function arrayIndex(i) {
  return i * 3;
}
