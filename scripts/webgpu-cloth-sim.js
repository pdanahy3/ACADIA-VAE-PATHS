/**
 * WebGPU cloth: Verlet integration + Jacobi distance constraints (stretch + bend).
 * Each constraint wave is preceded by a full buffer copy so untouched vertices stay valid.
 * Plasticity / boids / impulses stay on CPU — upload state each step after CPU kicks.
 */
const WGSL = /* wgsl */ `
struct SimParams {
  U: u32,
  V: u32,
  stretchK: f32,
  bendK: f32,
  dt: f32,
  dt2: f32,
  gravity: f32,
  damping: f32,
  dx: f32,
  dz: f32,
}

@group(0) @binding(0) var<uniform> P: SimParams;
@group(0) @binding(1) var<storage, read> invMass: array<f32>;
@group(0) @binding(2) var<storage, read> stretchU: array<f32>;
@group(0) @binding(3) var<storage, read> stretchV: array<f32>;
@group(0) @binding(4) var<storage, read> bendU: array<f32>;
@group(0) @binding(5) var<storage, read> bendV: array<f32>;
@group(0) @binding(6) var<storage, read> posRead: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read_write> posWrite: array<vec4<f32>>;
@group(0) @binding(8) var<storage, read_write> prev: array<vec4<f32>>;

fn vid(u: u32, v: u32) -> u32 { return v * P.U + u; }
fn anchor(u: u32) -> bool { return u == 0u || u == P.U - 1u; }

@compute @workgroup_size(256)
fn copy_r_to_w(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= P.U * P.V) { return; }
  posWrite[i] = posRead[i];
}

@compute @workgroup_size(256)
fn integrate(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  let n = P.U * P.V;
  if (i >= n) { return; }
  let u = i % P.U;
  let v = i / P.U;
  if (anchor(u)) {
    posWrite[i] = posRead[i];
    return;
  }
  if (invMass[i] <= 0.0) {
    posWrite[i] = posRead[i];
    return;
  }
  let px = posRead[i].xyz;
  let prv = prev[i].xyz;
  let vx = (px.x - prv.x) * P.damping;
  let vy = (px.y - prv.y) * P.damping;
  let vz = (px.z - prv.z) * P.damping;
  prev[i] = vec4<f32>(px, 0.0);
  posWrite[i] = vec4<f32>(px.x + vx, px.y + vy + P.gravity * P.dt2, px.z + vz, 0.0);
}

fn solve_edge(i0: u32, i1: u32, rest: f32, k: f32) {
  let w0 = invMass[i0];
  let w1 = invMass[i1];
  let wSum = w0 + w1;
  if (wSum < 1e-9) { return; }
  let b0 = posWrite[i0].xyz;
  let b1 = posWrite[i1].xyz;
  let d = posRead[i1].xyz - posRead[i0].xyz;
  let lenSq = dot(d, d);
  if (lenSq < 1e-12) { return; }
  let len = sqrt(lenSq);
  let diff = (len - rest) / len;
  let c = k * diff * d;
  if (w0 > 0.0) {
    posWrite[i0] = vec4<f32>(b0 + c * (w0 / wSum), 0.0);
  }
  if (w1 > 0.0) {
    posWrite[i1] = vec4<f32>(b1 - c * (w1 / wSum), 0.0);
  }
}

@compute @workgroup_size(256)
fn stretch_u_even(@builtin(global_invocation_id) gid: vec3<u32>) {
  let eid = gid.x;
  let perRow = (P.U - 1u + 1u) / 2u;
  let v = eid / perRow;
  let eu = eid % perRow;
  if (v >= P.V) { return; }
  let u = eu * 2u;
  if (u + 1u >= P.U) { return; }
  let i0 = vid(u, v);
  let i1 = vid(u + 1u, v);
  solve_edge(i0, i1, stretchU[v * (P.U - 1u) + u], P.stretchK);
}

@compute @workgroup_size(256)
fn stretch_u_odd(@builtin(global_invocation_id) gid: vec3<u32>) {
  let eid = gid.x;
  let perRow = (P.U - 1u) / 2u;
  if (perRow == 0u) { return; }
  let v = eid / perRow;
  let ou = eid % perRow;
  if (v >= P.V) { return; }
  let u = ou * 2u + 1u;
  if (u + 1u >= P.U) { return; }
  let i0 = vid(u, v);
  let i1 = vid(u + 1u, v);
  solve_edge(i0, i1, stretchU[v * (P.U - 1u) + u], P.stretchK);
}

@compute @workgroup_size(256)
fn stretch_v_even(@builtin(global_invocation_id) gid: vec3<u32>) {
  let eid = gid.x;
  let perCol = (P.V - 1u + 1u) / 2u;
  let u = eid / perCol;
  let ev = eid % perCol;
  if (u >= P.U) { return; }
  let v = ev * 2u;
  if (v + 1u >= P.V) { return; }
  let i0 = vid(u, v);
  let i1 = vid(u, v + 1u);
  solve_edge(i0, i1, stretchV[v * P.U + u], P.stretchK);
}

@compute @workgroup_size(256)
fn stretch_v_odd(@builtin(global_invocation_id) gid: vec3<u32>) {
  let eid = gid.x;
  let perCol = (P.V - 1u) / 2u;
  if (perCol == 0u) { return; }
  let u = eid / perCol;
  let ov = eid % perCol;
  if (u >= P.U) { return; }
  let v = ov * 2u + 1u;
  if (v + 1u >= P.V) { return; }
  let i0 = vid(u, v);
  let i1 = vid(u, v + 1u);
  solve_edge(i0, i1, stretchV[v * P.U + u], P.stretchK);
}

@compute @workgroup_size(256)
fn bend_u_p0(@builtin(global_invocation_id) gid: vec3<u32>) {
  let stride = 4u;
  let perRow = (P.U - 2u + stride - 1u) / stride;
  let eid = gid.x;
  let v = eid / perRow;
  let k = eid % perRow;
  if (v >= P.V) { return; }
  let u = k * stride;
  if (u + 2u >= P.U) { return; }
  solve_edge(vid(u, v), vid(u + 2u, v), bendU[v * (P.U - 2u) + u], P.bendK);
}
@compute @workgroup_size(256)
fn bend_u_p1(@builtin(global_invocation_id) gid: vec3<u32>) {
  let stride = 4u;
  let perRow = (P.U - 2u + stride - 1u) / stride;
  let eid = gid.x;
  let v = eid / perRow;
  let k = eid % perRow;
  if (v >= P.V) { return; }
  let u = k * stride + 1u;
  if (u + 2u >= P.U) { return; }
  solve_edge(vid(u, v), vid(u + 2u, v), bendU[v * (P.U - 2u) + u], P.bendK);
}
@compute @workgroup_size(256)
fn bend_u_p2(@builtin(global_invocation_id) gid: vec3<u32>) {
  let stride = 4u;
  let perRow = (P.U - 2u + stride - 1u) / stride;
  let eid = gid.x;
  let v = eid / perRow;
  let k = eid % perRow;
  if (v >= P.V) { return; }
  let u = k * stride + 2u;
  if (u + 2u >= P.U) { return; }
  solve_edge(vid(u, v), vid(u + 2u, v), bendU[v * (P.U - 2u) + u], P.bendK);
}
@compute @workgroup_size(256)
fn bend_u_p3(@builtin(global_invocation_id) gid: vec3<u32>) {
  let stride = 4u;
  let perRow = (P.U - 2u + stride - 1u) / stride;
  let eid = gid.x;
  let v = eid / perRow;
  let k = eid % perRow;
  if (v >= P.V) { return; }
  let u = k * stride + 3u;
  if (u + 2u >= P.U) { return; }
  solve_edge(vid(u, v), vid(u + 2u, v), bendU[v * (P.U - 2u) + u], P.bendK);
}

@compute @workgroup_size(256)
fn bend_v_p0(@builtin(global_invocation_id) gid: vec3<u32>) {
  let stride = 4u;
  let perCol = (P.V - 2u + stride - 1u) / stride;
  let eid = gid.x;
  let u = eid / perCol;
  let k = eid % perCol;
  if (u >= P.U) { return; }
  let v = k * stride;
  if (v + 2u >= P.V) { return; }
  solve_edge(vid(u, v), vid(u, v + 2u), bendV[v * P.U + u], P.bendK);
}
@compute @workgroup_size(256)
fn bend_v_p1(@builtin(global_invocation_id) gid: vec3<u32>) {
  let stride = 4u;
  let perCol = (P.V - 2u + stride - 1u) / stride;
  let eid = gid.x;
  let u = eid / perCol;
  let k = eid % perCol;
  if (u >= P.U) { return; }
  let v = k * stride + 1u;
  if (v + 2u >= P.V) { return; }
  solve_edge(vid(u, v), vid(u, v + 2u), bendV[v * P.U + u], P.bendK);
}
@compute @workgroup_size(256)
fn bend_v_p2(@builtin(global_invocation_id) gid: vec3<u32>) {
  let stride = 4u;
  let perCol = (P.V - 2u + stride - 1u) / stride;
  let eid = gid.x;
  let u = eid / perCol;
  let k = eid % perCol;
  if (u >= P.U) { return; }
  let v = k * stride + 2u;
  if (v + 2u >= P.V) { return; }
  solve_edge(vid(u, v), vid(u, v + 2u), bendV[v * P.U + u], P.bendK);
}
@compute @workgroup_size(256)
fn bend_v_p3(@builtin(global_invocation_id) gid: vec3<u32>) {
  let stride = 4u;
  let perCol = (P.V - 2u + stride - 1u) / stride;
  let eid = gid.x;
  let u = eid / perCol;
  let k = eid % perCol;
  if (u >= P.U) { return; }
  let v = k * stride + 3u;
  if (v + 2u >= P.V) { return; }
  solve_edge(vid(u, v), vid(u, v + 2u), bendV[v * P.U + u], P.bendK);
}
`;

function ceildiv(a, b) {
  return Math.floor((a + b - 1) / b);
}

export async function createWebGpuClothSimulator(U, V) {
  const adapter = await navigator.gpu?.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("No WebGPU adapter");
  const device = await adapter.requestDevice();
  const shader = device.createShaderModule({ code: WGSL });
  const bindLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }
    ]
  });
  const pl = device.createPipelineLayout({ bindGroupLayouts: [bindLayout] });
  const mk = (entry) =>
    device.createComputePipeline({ layout: pl, compute: { module: shader, entryPoint: entry } });

  const pipes = {
    copy: mk("copy_r_to_w"),
    integrate: mk("integrate"),
    su0: mk("stretch_u_even"),
    su1: mk("stretch_u_odd"),
    sv0: mk("stretch_v_even"),
    sv1: mk("stretch_v_odd"),
    bu0: mk("bend_u_p0"),
    bu1: mk("bend_u_p1"),
    bu2: mk("bend_u_p2"),
    bu3: mk("bend_u_p3"),
    bv0: mk("bend_v_p0"),
    bv1: mk("bend_v_p1"),
    bv2: mk("bend_v_p2"),
    bv3: mk("bend_v_p3")
  };

  const n = U * V;
  const szV = n * 16;
  const szF = n * 4;
  const nSU = (U - 1) * V;
  const nSV = U * (V - 1);
  const nBU = (U - 2) * V;
  const nBV = U * (V - 2);

  const bufPosA = device.createBuffer({ size: szV, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
  const bufPosB = device.createBuffer({ size: szV, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
  const bufPrev = device.createBuffer({ size: szV, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const bufInv = device.createBuffer({ size: szF, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const bufSU = device.createBuffer({ size: nSU * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const bufSV = device.createBuffer({ size: nSV * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const bufBU = device.createBuffer({ size: nBU * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const bufBV = device.createBuffer({ size: nBV * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const ubo = device.createBuffer({ size: 256, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

  const bind = (read, write) =>
    device.createBindGroup({
      layout: bindLayout,
      entries: [
        { binding: 0, resource: { buffer: ubo } },
        { binding: 1, resource: { buffer: bufInv } },
        { binding: 2, resource: { buffer: bufSU } },
        { binding: 3, resource: { buffer: bufSV } },
        { binding: 4, resource: { buffer: bufBU } },
        { binding: 5, resource: { buffer: bufBV } },
        { binding: 6, resource: { buffer: read } },
        { binding: 7, resource: { buffer: write } },
        { binding: 8, resource: { buffer: bufPrev } }
      ]
    });

  const suE = Math.ceil((U - 1) / 2) * V;
  const suO = Math.floor((U - 1) / 2) * V;
  const svE = Math.ceil((V - 1) / 2) * U;
  const svO = Math.floor((V - 1) / 2) * U;
  const stride = 4;
  const bendUThr = ceildiv(U - 2, stride) * V;
  const bendVThr = ceildiv(V - 2, stride) * U;

  function setUbo(stretchK, bendK, dt, dt2, gravity, damping, dx, dz) {
    const b = new ArrayBuffer(256);
    const d = new DataView(b);
    d.setUint32(0, U, true);
    d.setUint32(4, V, true);
    d.setFloat32(8, stretchK, true);
    d.setFloat32(12, bendK, true);
    d.setFloat32(16, dt, true);
    d.setFloat32(20, dt2, true);
    d.setFloat32(24, gravity, true);
    d.setFloat32(28, damping, true);
    d.setFloat32(32, dx, true);
    d.setFloat32(36, dz, true);
    device.queue.writeBuffer(ubo, 0, b);
  }

  function dispatch(enc, pipe, bg, threads) {
    const p = enc.beginComputePass();
    p.setPipeline(pipe);
    p.setBindGroup(0, bg);
    p.dispatchWorkgroups(ceildiv(threads, 256));
    p.end();
  }

  /** One full copy + constraint wave: read<-src, write<-dst, then run pipe (reads read, updates write) */
  function wave(enc, src, dst, pipe, threads) {
    dispatch(enc, pipes.copy, bind(src, dst), n);
    dispatch(enc, pipe, bind(src, dst), Math.max(threads, 1));
  }

  const sim = {
    device,
    U,
    V,
    n,
    bufPosA,
    bufPosB,
    bufPrev,
    bufInv,
    bufSU,
    bufSV,
    bufBU,
    bufBV,
    ubo,
    readIsA: true,

    uploadRests(su, sv, bu, bv) {
      device.queue.writeBuffer(bufSU, 0, su.buffer, su.byteOffset, su.byteLength);
      device.queue.writeBuffer(bufSV, 0, sv.buffer, sv.byteOffset, sv.byteLength);
      device.queue.writeBuffer(bufBU, 0, bu.buffer, bu.byteOffset, bu.byteLength);
      device.queue.writeBuffer(bufBV, 0, bv.buffer, bv.byteOffset, bv.byteLength);
    },

    uploadInv(inv) {
      device.queue.writeBuffer(bufInv, 0, inv.buffer, inv.byteOffset, inv.byteLength);
    },

    /** vec4 per vertex xyzw (w unused) */
    uploadPositionsAndPrev(pos4, prev4) {
      device.queue.writeBuffer(bufPosA, 0, pos4.buffer, pos4.byteOffset, pos4.byteLength);
      device.queue.writeBuffer(bufPrev, 0, prev4.buffer, prev4.byteOffset, prev4.byteLength);
      sim.readIsA = true;
    },

    /**
     * One frame: Verlet integrate once, then solverIterations × (stretch × stretchPasses + bend × bendPasses).
     * Matches CPU order: integrate(); for each solver iteration, run stretch passes then bend passes.
     */
    encodeSimulationStep(
      enc,
      {
        dt,
        dt2,
        gravity,
        damping,
        dx,
        dz,
        stretchK,
        bendK,
        stretchPasses,
        bendPasses,
        solverIterations
      }
    ) {
      let src = bufPosA;
      let dst = bufPosB;
      setUbo(stretchK, bendK, dt, dt2, gravity, damping, dx, dz);
      dispatch(enc, pipes.integrate, bind(src, dst), n);
      src = bufPosB;
      dst = bufPosA;

      const runStretch = (k) => {
        setUbo(k, 0, dt, dt2, gravity, damping, dx, dz);
        wave(enc, src, dst, pipes.su0, suE);
        src = dst;
        dst = src === bufPosA ? bufPosB : bufPosA;
        wave(enc, src, dst, pipes.su1, suO);
        src = dst;
        dst = src === bufPosA ? bufPosB : bufPosA;
        wave(enc, src, dst, pipes.sv0, svE);
        src = dst;
        dst = src === bufPosA ? bufPosB : bufPosA;
        wave(enc, src, dst, pipes.sv1, svO);
        src = dst;
        dst = src === bufPosA ? bufPosB : bufPosA;
      };
      const runBend = (k) => {
        setUbo(0, k, dt, dt2, gravity, damping, dx, dz);
        for (const key of ["bu0", "bu1", "bu2", "bu3"]) {
          wave(enc, src, dst, pipes[key], bendUThr);
          src = dst;
          dst = src === bufPosA ? bufPosB : bufPosA;
        }
        for (const key of ["bv0", "bv1", "bv2", "bv3"]) {
          wave(enc, src, dst, pipes[key], bendVThr);
          src = dst;
          dst = src === bufPosA ? bufPosB : bufPosA;
        }
      };

      const iters = Math.max(1, solverIterations | 0);
      for (let iter = 0; iter < iters; iter++) {
        for (let i = 0; i < stretchPasses; i++) runStretch(stretchK);
        for (let i = 0; i < bendPasses; i++) runBend(bendK);
      }

      sim.readIsA = src === bufPosA;
    },

    /** @deprecated Prefer encodeSimulationStep with solverIterations. */
    encodePhysicsPass(
      enc,
      { dt, dt2, gravity, damping, dx, dz, stretchK, bendK, stretchPasses, bendPasses, solverIterations = 1 }
    ) {
      sim.encodeSimulationStep(enc, {
        dt,
        dt2,
        gravity,
        damping,
        dx,
        dz,
        stretchK,
        bendK,
        stretchPasses,
        bendPasses,
        solverIterations
      });
    },

    async readPrevToVec4(outF32) {
      const staging = sim.device.createBuffer({
        size: outF32.byteLength,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
      });
      const enc = sim.device.createCommandEncoder();
      enc.copyBufferToBuffer(bufPrev, 0, staging, 0, outF32.byteLength);
      sim.device.queue.submit([enc.finish()]);
      await staging.mapAsync(GPUMapMode.READ);
      outF32.set(new Float32Array(staging.getMappedRange()));
      staging.unmap();
      staging.destroy();
    },

    submitPhysics(params) {
      const enc = sim.device.createCommandEncoder();
      sim.encodeSimulationStep(enc, params);
      sim.device.queue.submit([enc.finish()]);
    },

    async readPositionsVec4(outF32) {
      const readBuf = sim.readIsA ? bufPosA : bufPosB;
      const staging = sim.device.createBuffer({
        size: outF32.byteLength,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
      });
      const enc = sim.device.createCommandEncoder();
      enc.copyBufferToBuffer(readBuf, 0, staging, 0, outF32.byteLength);
      sim.device.queue.submit([enc.finish()]);
      await staging.mapAsync(GPUMapMode.READ);
      outF32.set(new Float32Array(staging.getMappedRange()));
      staging.unmap();
      staging.destroy();
    }
  };

  return sim;
}

export function isWebGpuAvailable() {
  return typeof navigator !== "undefined" && !!navigator.gpu;
}
