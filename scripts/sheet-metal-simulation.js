/**
 * Sheet metal–style mass–spring shell on a rectangular grid (12" × 24" sheet, 512×512 cells).
 * - Edge springs: structural (axis-aligned) + shear (diagonals) resist stretching.
 * - Bending: discrete bilaplacian of displacement from rest (Kirchhoff thin-plate / mean-curvature
 *   analogue; resists creasing similarly to dihedral resistance on a regular grid).
 * - Colliders: sphere or axis-aligned box push vertices to the free side of the surface.
 * - Grippers: sphere or box volumes; vertices inside are kinematically fixed (pinned).
 *
 * Run from sheet-metal-simulation.html or instantiate SheetMetalSimulation in your own page.
 */
(function (global) {
  'use strict';

  const INCHES_WIDE = 24;
  const INCHES_DEEP = 12;
  const DEFAULT_DIV = 512;

  function clamp(x, a, b) {
    return Math.max(a, Math.min(b, x));
  }

  function sub(out, a, b) {
    out[0] = a[0] - b[0];
    out[1] = a[1] - b[1];
    out[2] = a[2] - b[2];
    return out;
  }

  function len3(v) {
    return Math.hypot(v[0], v[1], v[2]);
  }

  function pointInBox(p, minB, maxB) {
    return (
      p[0] >= minB[0] &&
      p[0] <= maxB[0] &&
      p[1] >= minB[1] &&
      p[1] <= maxB[1] &&
      p[2] >= minB[2] &&
      p[2] <= maxB[2]
    );
  }

  class SheetMetalSimulation {
    /**
     * @param {object} opts
     * @param {number} [opts.cellsX=512] - number of quads along 24" (wide) axis
     * @param {number} [opts.cellsZ=512] - number of quads along 12" (deep) axis
     * @param {number} [opts.inchesWide=24]
     * @param {number} [opts.inchesDeep=12]
     * @param {number} [opts.kStretch=8e4] - axial / shear spring stiffness (per unit rest length)
     * @param {number} [opts.kBend=2e3] - bilaplacian bending scale
     * @param {number} [opts.damping=0.12] - velocity damping per second (linear factor ~ exp)
     * @param {number} [opts.vertexMass=0.02] - mass per vertex (lb scale arbitrary; tune with k)
     * @param {number[]} [opts.gravity=[0,-386,0]] - in/s² (Earth ~ 386 in/s² along -Y)
     */
    constructor(opts) {
      opts = opts || {};
      this.cellsX = opts.cellsX != null ? opts.cellsX | 0 : DEFAULT_DIV;
      this.cellsZ = opts.cellsZ != null ? opts.cellsZ | 0 : DEFAULT_DIV;
      this.inchesWide = opts.inchesWide != null ? +opts.inchesWide : INCHES_WIDE;
      this.inchesDeep = opts.inchesDeep != null ? +opts.inchesDeep : INCHES_DEEP;

      this.kStretch = opts.kStretch != null ? +opts.kStretch : 8e4;
      this.kBend = opts.kBend != null ? +opts.kBend : 2e3;
      this.damping = opts.damping != null ? +opts.damping : 0.12;
      this.vertexMass = opts.vertexMass != null ? +opts.vertexMass : 0.02;
      this.gravity = opts.gravity
        ? new Float32Array(opts.gravity)
        : new Float32Array([0, -386, 0]);

      this.nx = this.cellsX + 1;
      this.nz = this.cellsZ + 1;
      this.nv = this.nx * this.nz;

      this.dx = this.inchesWide / this.cellsX;
      this.dz = this.inchesDeep / this.cellsZ;

      this.pos = new Float32Array(this.nv * 3);
      this.rest = new Float32Array(this.nv * 3);
      this.vel = new Float32Array(this.nv * 3);
      this.force = new Float32Array(this.nv * 3);
      this.pinned = new Uint8Array(this.nv);
      this.pinPos = new Float32Array(this.nv * 3);

      this._tmp0 = new Float32Array(this.nv * 3);
      this._tmp1 = new Float32Array(this.nv * 3);

      this.springs = [];
      this._buildSprings();

      /** @type {null | { type:'sphere', center:Float32Array, radius:number } | { type:'box', min:Float32Array, max:Float32Array }} */
      this.collider = null;

      /** @type {Array<{ type:'sphere'|'box', min?:Float32Array, max?:Float32Array, center?:Float32Array, radius?:number, pinToRest:boolean }>} */
      this.grippers = [];

      this._p = new Float32Array(3);
      this._e = new Float32Array(3);
      this._restA = new Float32Array(3);
      this._restB = new Float32Array(3);

      this.reset();
    }

    _buildSprings() {
      const { nx, nz } = this;
      const springs = this.springs;
      springs.length = 0;
      const ra = this._restA;
      const rb = this._restB;
      const push = (a, b) => {
        this._restAt(a, ra);
        this._restAt(b, rb);
        springs.push({ a, b, rest: len3(sub(this._e, ra, rb)) });
      };

      for (let iz = 0; iz < nz; iz++) {
        for (let ix = 0; ix < nx; ix++) {
          const i = iz * nx + ix;
          if (ix + 1 < nx) push(i, i + 1);
          if (iz + 1 < nz) push(i, i + nx);
          if (ix + 1 < nx && iz + 1 < nz) {
            push(i, i + 1 + nx);
            push(i + 1, i + nx);
          }
        }
      }
    }

    _restAt(vertIdx, out) {
      if (!out) out = this._restA;
      const ix = vertIdx % this.nx;
      const iz = (vertIdx / this.nx) | 0;
      out[0] = ix * this.dx;
      out[1] = 0;
      out[2] = iz * this.dz;
      return out;
    }

    reset() {
      const { nv, nx, pos, rest, vel, pinned } = this;
      for (let i = 0; i < nv; i++) {
        const ix = i % nx;
        const iz = (i / nx) | 0;
        const x = ix * this.dx;
        const z = iz * this.dz;
        const b = i * 3;
        pos[b] = rest[b] = x;
        pos[b + 1] = rest[b + 1] = 0;
        pos[b + 2] = rest[b + 2] = z;
        vel[b] = vel[b + 1] = vel[b + 2] = 0;
        pinned[i] = 0;
      }
      this.rebuildSpringRestLengths();
      this.applyGrippers();
    }

    rebuildSpringRestLengths() {
      const { springs, pos, rest } = this;
      for (let s = 0; s < springs.length; s++) {
        const sp = springs[s];
        const ra = sp.a * 3;
        const rb = sp.b * 3;
        const ax = rest[ra] - rest[rb];
        const ay = rest[ra + 1] - rest[rb + 1];
        const az = rest[ra + 2] - rest[rb + 2];
        sp.rest = Math.hypot(ax, ay, az);
      }
    }

    /**
     * @param {object|null} c
     * - { type:'sphere', center:[x,y,z], radius }
     * - { type:'box', min:[x,y,z], max:[x,y,z] }
     */
    setCollider(c) {
      if (c == null) {
        this.collider = null;
        return;
      }
      if (c.type === 'sphere') {
        this.collider = {
          type: 'sphere',
          center: new Float32Array(c.center),
          radius: +c.radius,
        };
      } else if (c.type === 'box') {
        this.collider = {
          type: 'box',
          min: new Float32Array(c.min),
          max: new Float32Array(c.max),
        };
      } else {
        this.collider = null;
      }
    }

    /**
     * @param {object} g
     * @param {'box'|'sphere'} g.type
     * @param {number[]} [g.min] [g.max] for box
     * @param {number[]} [g.center] g.radius for sphere
     * @param {boolean} [g.pinToRest=true] if true, pin to rest pose; else pin to current pose when applied
     */
    addGripper(g) {
      const pinToRest = g.pinToRest !== false;
      let rec;
      if (g.type === 'box') {
        rec = {
          type: 'box',
          min: new Float32Array(g.min),
          max: new Float32Array(g.max),
          pinToRest,
        };
      } else if (g.type === 'sphere') {
        rec = {
          type: 'sphere',
          center: new Float32Array(g.center),
          radius: +g.radius,
          pinToRest,
        };
      } else {
        return;
      }
      this.grippers.push(rec);
      this.applyGrippers();
    }

    clearGrippers() {
      this.grippers.length = 0;
      for (let i = 0; i < this.nv; i++) this.pinned[i] = 0;
    }

    /** Re-scan all grippers and update pin flags / pin positions. */
    applyGrippers() {
      const { nv, pos, rest, pinPos, pinned } = this;
      for (let i = 0; i < nv; i++) pinned[i] = 0;
      for (let g = 0; g < this.grippers.length; g++) {
        const gr = this.grippers[g];
        for (let i = 0; i < nv; i++) {
          const b = i * 3;
          this._p[0] = pos[b];
          this._p[1] = pos[b + 1];
          this._p[2] = pos[b + 2];
          let inside = false;
          if (gr.type === 'box') inside = pointInBox(this._p, gr.min, gr.max);
          else {
            sub(this._e, this._p, gr.center);
            inside = len3(this._e) <= gr.radius;
          }
          if (inside) {
            pinned[i] = 1;
            if (gr.pinToRest) {
              pinPos[b] = rest[b];
              pinPos[b + 1] = rest[b + 1];
              pinPos[b + 2] = rest[b + 2];
            } else {
              pinPos[b] = pos[b];
              pinPos[b + 1] = pos[b + 1];
              pinPos[b + 2] = pos[b + 2];
            }
          }
        }
      }
    }

    /** Discrete 5-point Laplacian of a 3-component field (no rest subtraction). */
    _laplacianField(out3, field) {
      const { nx, nz, nv } = this;
      for (let i = 0; i < nv * 3; i++) out3[i] = 0;
      for (let iz = 1; iz < nz - 1; iz++) {
        for (let ix = 1; ix < nx - 1; ix++) {
          const i = iz * nx + ix;
          const b = i * 3;
          const bL = b - 3;
          const bR = b + 3;
          const bD = b - nx * 3;
          const bU = b + nx * 3;
          for (let c = 0; c < 3; c++) {
            out3[b + c] =
              field[bL + c] +
              field[bR + c] +
              field[bD + c] +
              field[bU + c] -
              4 * field[b + c];
          }
        }
      }
    }

    _laplacianDisp(out3, srcPos) {
      const { nx, nz, nv, rest } = this;
      for (let i = 0; i < nv * 3; i++) out3[i] = 0;
      for (let iz = 1; iz < nz - 1; iz++) {
        for (let ix = 1; ix < nx - 1; ix++) {
          const i = iz * nx + ix;
          const b = i * 3;
          const bL = b - 3;
          const bR = b + 3;
          const bD = b - nx * 3;
          const bU = b + nx * 3;
          for (let c = 0; c < 3; c++) {
            const u =
              srcPos[bL + c] -
              rest[bL + c] +
              (srcPos[bR + c] - rest[bR + c]) +
              (srcPos[bD + c] - rest[bD + c]) +
              (srcPos[bU + c] - rest[bU + c]) -
              4 * (srcPos[b + c] - rest[b + c]);
            out3[b + c] = u;
          }
        }
      }
    }

    _addBendingForces() {
      const { nx, nz, pos, force, kBend, dx, dz } = this;
      const h2 = (dx + dz) * 0.5;
      const h4 = h2 * h2 * h2 * h2;
      const scale = kBend / Math.max(h4, 1e-12);

      const L1 = this._tmp0;
      const L2 = this._tmp1;
      this._laplacianDisp(L1, pos);
      this._laplacianField(L2, L1);

      const i0 = 2;
      const i1 = nx - 2;
      const j0 = 2;
      const j1 = nz - 2;
      for (let iz = j0; iz < j1; iz++) {
        for (let ix = i0; ix < i1; ix++) {
          const i = iz * nx + ix;
          if (this.pinned[i]) continue;
          const b = i * 3;
          force[b] -= scale * L2[b];
          force[b + 1] -= scale * L2[b + 1];
          force[b + 2] -= scale * L2[b + 2];
        }
      }
    }

    _springForces() {
      const { springs, pos, force, kStretch, pinned } = this;
      const e = this._e;
      for (let s = 0; s < springs.length; s++) {
        const sp = springs[s];
        const ia = sp.a;
        const ib = sp.b;
        if (pinned[ia] && pinned[ib]) continue;
        const ra = ia * 3;
        const rb = ib * 3;
        e[0] = pos[rb] - pos[ra];
        e[1] = pos[rb + 1] - pos[ra + 1];
        e[2] = pos[rb + 2] - pos[ra + 2];
        const L = len3(e) || 1e-12;
        const invL = 1 / L;
        const strain = L - sp.rest;
        const mag = kStretch * strain;
        const sx = (e[0] * invL) * mag;
        const sy = (e[1] * invL) * mag;
        const sz = (e[2] * invL) * mag;
        if (!pinned[ib]) {
          force[rb] -= sx;
          force[rb + 1] -= sy;
          force[rb + 2] -= sz;
        }
        if (!pinned[ia]) {
          force[ra] += sx;
          force[ra + 1] += sy;
          force[ra + 2] += sz;
        }
      }
    }

    _resolveCollider() {
      const c = this.collider;
      if (!c) return;
      const { pos, vel, pinned } = this;
      for (let i = 0; i < this.nv; i++) {
        if (pinned[i]) continue;
        const b = i * 3;
        this._p[0] = pos[b];
        this._p[1] = pos[b + 1];
        this._p[2] = pos[b + 2];
        if (c.type === 'sphere') {
          sub(this._e, this._p, c.center);
          const d = len3(this._e);
          if (d < c.radius && d > 1e-12) {
            const pen = c.radius - d;
            const nx = this._e[0] / d;
            const ny = this._e[1] / d;
            const nz = this._e[2] / d;
            pos[b] += nx * pen;
            pos[b + 1] += ny * pen;
            pos[b + 2] += nz * pen;
            const vn = vel[b] * nx + vel[b + 1] * ny + vel[b + 2] * nz;
            if (vn < 0) {
              vel[b] -= vn * nx;
              vel[b + 1] -= vn * ny;
              vel[b + 2] -= vn * nz;
            }
          }
        } else {
          const p = this._p;
          if (pointInBox(p, c.min, c.max)) {
            const dxL = p[0] - c.min[0];
            const dxR = c.max[0] - p[0];
            const dyB = p[1] - c.min[1];
            const dyT = c.max[1] - p[1];
            const dzF = p[2] - c.min[2];
            const dzBk = c.max[2] - p[2];
            let pen = dxL;
            let nx = -1;
            let ny = 0;
            let nz = 0;
            if (dxR < pen) {
              pen = dxR;
              nx = 1;
              ny = nz = 0;
            }
            if (dyB < pen) {
              pen = dyB;
              nx = 0;
              ny = -1;
              nz = 0;
            }
            if (dyT < pen) {
              pen = dyT;
              nx = 0;
              ny = 1;
              nz = 0;
            }
            if (dzF < pen) {
              pen = dzF;
              nx = ny = 0;
              nz = -1;
            }
            if (dzBk < pen) {
              pen = dzBk;
              nx = ny = 0;
              nz = 1;
            }
            pos[b] += nx * pen;
            pos[b + 1] += ny * pen;
            pos[b + 2] += nz * pen;
            const vn = vel[b] * nx + vel[b + 1] * ny + vel[b + 2] * nz;
            if (vn < 0) {
              vel[b] -= vn * nx;
              vel[b + 1] -= vn * ny;
              vel[b + 2] -= vn * nz;
            }
          }
        }
      }
    }

    _enforcePins() {
      const { nv, pos, vel, pinned, pinPos } = this;
      for (let i = 0; i < nv; i++) {
        if (!pinned[i]) continue;
        const b = i * 3;
        pos[b] = pinPos[b];
        pos[b + 1] = pinPos[b + 1];
        pos[b + 2] = pinPos[b + 2];
        vel[b] = vel[b + 1] = vel[b + 2] = 0;
      }
    }

    /**
     * @param {number} dt - seconds
     * @param {number} [substeps=2]
     */
    step(dt, substeps) {
      substeps = substeps == null ? 2 : Math.max(1, substeps | 0);
      const h = dt / substeps;
      const { nv, pos, vel, force, pinned, vertexMass, gravity, damping } = this;
      const damp = Math.exp(-damping * h);

      for (let sub = 0; sub < substeps; sub++) {
        for (let i = 0; i < nv * 3; i++) force[i] = 0;

        for (let i = 0; i < nv; i++) {
          if (pinned[i]) continue;
          const b = i * 3;
          force[b] += gravity[0] * vertexMass;
          force[b + 1] += gravity[1] * vertexMass;
          force[b + 2] += gravity[2] * vertexMass;
        }

        this._springForces();
        this._addBendingForces();

        for (let i = 0; i < nv; i++) {
          if (pinned[i]) continue;
          const b = i * 3;
          vel[b] = (vel[b] + (force[b] / vertexMass) * h) * damp;
          vel[b + 1] = (vel[b + 1] + (force[b + 1] / vertexMass) * h) * damp;
          vel[b + 2] = (vel[b + 2] + (force[b + 2] / vertexMass) * h) * damp;
          pos[b] += vel[b] * h;
          pos[b + 1] += vel[b + 1] * h;
          pos[b + 2] += vel[b + 2] * h;
        }

        this._resolveCollider();
        this._enforcePins();
      }
    }

    getVertexCount() {
      return this.nv;
    }

    getGridSize() {
      return { nx: this.nx, nz: this.nz, cellsX: this.cellsX, cellsZ: this.cellsZ };
    }

    /** @returns {Float32Array} flattened xyz positions (length 3 * nv) */
    getPositions() {
      return this.pos;
    }

    getRestPositions() {
      return this.rest;
    }
  }

  global.SheetMetalSimulation = SheetMetalSimulation;
})(typeof window !== 'undefined' ? window : globalThis);
